# coding:utf-8
from logging import log
import torch 
import pandas as pd
import numpy as np
import os
import random
from torch._C import dtype
from torch.nn.modules import normalization

from torchtext.vocab import Vocab
from transformers.utils.dummy_pt_objects import XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
torch.backends.cudnn.enabled=False

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2
csv_path = '/home/zhangyuhan/NLPcodes/poetryFromTang/train.csv'

# data preprocess
import nltk
from torchtext.legacy import data
#from tensorflow.keras.preprocessing.text  import Tokenizer

def tokenizer(text):
    return list(text)

DATA = data.Field(sequential=True, 
                tokenize=tokenizer, 
                init_token='<start>', 
                eos_token='<end>', 
                batch_first= True)
dataset = data.TabularDataset(path=csv_path, 
                            format='csv', 
                            skip_header=True, 
                            fields=[('poetry', DATA)])
DATA.build_vocab(dataset)
vocab = DATA.vocab

PAD_ID = vocab.stoi['<pad>']
START_ID = vocab.stoi['start']
END_ID = vocab.stoi['<end>']

train_data, valid_data = dataset.split(split_ratio=0.7, random_state=random.seed(SEED))
train_iterator, valid_iterator = data.BucketIterator.splits((train_data, valid_data), 
                                                            batch_size=BATCH_SIZE, 
                                                            sort_key=lambda x: len(x.poetry), 
                                                            shuffle=False, 
                                                            device=DEVICE)

# Create model
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, drop_out=0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional= True, dropout= drop_out, batch_first= True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first= True)
        self.L1 = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.L2 = nn.Linear(self.hidden_dim, input_dim)

    def forward(self, x, pre_hidden, context):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, pre_hidden)

        output = output.view(-1, self.hidden_dim)
        concat = torch.cat([output, context], dim=-1)

        FF1_out = self.L1(concat)
        FF2_out = self.L2(FF1_out)
        out = F.softmax(FF2_out, dim=1)

        return out, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # feed-forward layer
        self.Wh = nn.Linear(hidden_dim*2, hidden_dim*2, bias= False)
        self.Ws = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.v = nn.Linear(hidden_dim*2, 1, bias= False)

    def forward(self, states, output, pad_masks, coverage=None):
        hidden, cell = states
        s_t = torch.cat([hidden, cell], dim=2)
        s_t = s_t.transpose(0, 1)
        s_t = s_t.expand_as(output).contiguous()

        en_features = self.Wh(output.contiguous())
        de_features = self.Ws(s_t)
        att_input = en_features + de_features
        score = self.v(torch.tanh(att_input))
        att_weights = F.softmax(score, dim=1).squeeze(2)
        att_weights = att_weights*pad_masks     # mask <pad>
        # normalize attention weights 
        normalization_factor = att_weights.sum(1, keepdim= True)
        att_weights = att_weights/normalization_factor
        context = torch.bmm(att_weights.unsqueeze(1), output)
        context = context.squeeze(1)

        return context, att_weights

# Encoder bidirection=True, Decoder bidirection=False
class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim= True)
        c_reduced = torch.sum(c, dim=0, keepdim= True)
        return (h_reduced, c_reduced)

class Beam(object):
    def __init__(self, tokens, log_probs, de_hidden) -> None:
        self.tokens = tokens
        self.log_probs = log_probs
        self.de_hidden = de_hidden
    
    def extend(self, token, log_prob, de_hidden):
        return Beam(tokens= self.tokens+[token], 
                    log_probs= self.log_probs + [log_prob],
                    de_hidden= de_hidden)
    
    def score(self):
        return np.sum(self.log_probs)

class Seq2seq(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(Seq2seq, self).__init__()
        self.attention = Attention(hidden_dim)
        self.encoder = Encoder(input_dim, embedding_dim, hidden_dim)
        self.decoder = Decoder(input_dim, embedding_dim, hidden_dim)
        self.reduce = ReduceState()

    def forward(self, x, y):
        x_pad_masks = torch.ne(x, PAD_ID).byte().float()
        en_output, en_hidden = self.encoder(x)
        de_hidden = self.reduce(en_hidden)
        
        step_loss = []
        x_t = y[:, 0]
        for t in range(y.shape[1]-1):
            y_t = y[:, t+1]

            context, att_weights = self.attention(de_hidden, en_output, x_pad_masks)
            de_output, de_hidden = self.decoder(x_t.unsqueeze(1), de_hidden, context)
            
            x_t = torch.argmax(de_output, dim=1).to(DEVICE)
            
            probs = torch.gather(de_output, 1, y_t.unsqueeze(1))
            probs = probs.squeeze(1)
            # mask <pad>
            mask = torch.ne(y_t, PAD_ID).byte()
            loss = -torch.log(probs+1e-15)
            mask = mask.float()
            loss = loss*mask
            step_loss.append(loss)
        
        sample_loss = torch.sum(torch.stack(step_loss, 1), 1)
        # non-padded length of each sequence
        len_mask = torch.ne(y, PAD_ID).byte().float()
        batch_len = torch.sum(len_mask, dim=1)
        batch_loss = torch.mean(sample_loss/batch_len)
        
        return batch_loss

# define model
INPUT_DIM = len(vocab)
EMBEDDING_DIM = len(vocab)
HIDDEN_DIM = 128
DROP_OUT = 0.5

model = Seq2seq(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM)
model.to(DEVICE)

# optimizer
from torch.optim import Adam
LEARNING_RATE = 1e-3
optimizer = Adam(model.parameters(), LEARNING_RATE)

def train(model, dataloader, optimizer):
    epoch_loss = []
    epoch_ppl = []
    model.train()

    for batch in dataloader:
        optimizer.zero_grad()

        tokens = batch.poetry.to(DEVICE)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        batch_loss = model(x, y)
        batch_ppl = torch.exp(batch_loss)

        epoch_loss.append(batch_loss.item())
        epoch_ppl.append(batch_ppl.item())

        batch_loss.backward()
        # Do gradient clipping to prevent gradient explosion.
        nn.utils.clip_grad_norm_(model.encoder.parameters(), 1)
        nn.utils.clip_grad_norm_(model.decoder.parameters(), 1)
        nn.utils.clip_grad_norm_(model.attention.parameters(), 1)
        optimizer.step()

    return np.mean(epoch_loss), np.mean(epoch_ppl)

# evaluate
def evaluate(model, iterator):
    valid_loss = []
    valid_ppl = []

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            tokens = batch.poetry.to(DEVICE)
            x = tokens[:, :-1]
            y = tokens[:, 1:]
            try:
                loss = model(x, y)
                ppl = torch.exp(loss)
                valid_loss.append(loss.item())
                valid_ppl.append(ppl.item())
            except:
                continue

    return np.mean(valid_loss), np.mean(valid_ppl)

# time format
import time

def time_form(time):
    minute = int(time/60)
    second = int(time - minute*60)
    return minute, second

# train model
N_EPOCHS = 20
#best_valid_loss = float('inf')

print("Start Training...")
for epoch in range(N_EPOCHS):
    # record running time
    start = time.time()
    # train and evaluate model
    train_loss, train_ppl_mean = train(model, train_iterator, optimizer)
    valid_loss, valid_ppl_mean = evaluate(model, valid_iterator)
    
    end = time.time()
    epoch_time = end-start

    minute, second = time_form(epoch_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {minute}m {second}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Train ppl: {train_ppl_mean:.2f}')
    print(f'\tValid Loss: {valid_loss:.4f} | Valid ppl: {valid_ppl_mean:.2f}')
    
print('Training Complete!') 

# Generate
def greedy_generate(txt, model, vocab, type= 'start', max_len= 32, gamma = 1):
    model.eval()

    results = list(txt)
    tokens = list(vocab.stoi[word] for word in results)
    pad_mask = torch.ne(torch.tensor(tokens), PAD_ID).byte().float().to(DEVICE )
    length = len(results)

    tokens = torch.tensor(tokens).to(DEVICE)
    tokens = tokens.reshape(1, length)
    en_output, en_hidden = model.encoder(tokens)
    de_hidden = model.reduce(en_hidden)

    punc_ids = [PAD_ID, END_ID, vocab.stoi['，'], vocab.stoi['。']]

    if type=='start':
        x_t = torch.tensor([START_ID]).to(DEVICE)
        print(x_t)
        while (x_t.item() != END_ID and len(results)<max_len):
            context, att_weight = model.attention(de_hidden, en_output, pad_mask)
            de_output, de_hidden = model.decoder(x_t.unsqueeze(0), de_hidden, context)
            log_probs = torch.log(de_output.squeeze())
            #log_probs[punc_ids] = log_probs[punc_ids]*gamma

            topk_probs, topk_ids = torch.topk(log_probs, 10)
            if topk_ids[0] in punc_ids:
                x_t = topk_ids[random.randint(0, 9)].reshape(1)
            else:
                x_t = topk_ids[0].reshape(1)
            results.append(vocab.itos[x_t.item()]) 
            
    return ''.join(results)

def best_k(model, beam, k, en_output, pad_mask):
    x_t = torch.tensor(beam.tokens[-1]).reshape(1, 1).to(DEVICE)

    context, att_weights = model.attention(beam.de_hidden, en_output, pad_mask)
    de_output, de_hidden = model.decoder(x_t, beam.de_hidden, context)
    log_probs = torch.log(de_output.squeeze())
    _, topk_idx = torch.topk(log_probs, k)

    best_k = []
    for topi in topk_idx.tolist():
        best_k.append(beam.extend(topi, log_probs[topi], de_hidden))
    return best_k


def beam_generate(txt, model, vocab, beam_width=3, max_len= 16):
    model.eval()
    
    results = list(txt)
    tokens = list(vocab.stoi[word] for word in results)
    pad_masks = torch.ne(torch.tensor(tokens), PAD_ID).byte().float().to(DEVICE)
    tokens = torch.tensor(tokens).to(DEVICE)
    tokens = tokens.reshape(1, len(results))
    en_output, en_hidden = model.encoder(tokens)
    de_hidden = model.reduce(en_hidden)
        
    beam = Beam(tokens= [START_ID], log_probs= [0], de_hidden= de_hidden)
    k = beam_width
    curr, completed = [beam], []

    for i in range(max_len):
        topk = []

        for beam in curr:
            if beam.tokens[-1] == END_ID:
                completed.append(beam)
                k -= 1
                continue
            candidators = best_k(model, beam, k, en_output, pad_masks)
            for candi in candidators:
                topk.append(candi)
            
        topk = sorted(topk, key= lambda x: x.score(), reverse= True)
        curr = topk[: k]
        if len(completed) == beam_width:
            break
        
    completed += curr
    completed = sorted(completed, key= lambda x: x.score(), reverse= True)
    results += list(vocab.itos[s] for s in completed[0].tokens)
    return ''.join(results)

txt = "花开无情"
output = greedy_generate(txt, model, vocab)
print(output)
output = beam_generate(txt, model, vocab)
print(output)