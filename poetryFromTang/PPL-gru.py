# coding:utf-8
import torch 
import pandas as pd
import numpy as np
import os
import random
from torch.nn.modules import padding

from torchtext.vocab import Vocab

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read in datafile
file_name = '/home/zhangyuhan/NLPcodes/poetryFromTang/poetryFromTang.txt'
csv_path = '/home/zhangyuhan/NLPcodes/poetryFromTang/train.csv'
with open(file_name, encoding='utf-8') as f:
    lines = f.read().split('\n\n')
lines = list(map(lambda x: x.replace('\n', ''), lines))

df = pd.DataFrame()
df['poetry'] = lines
df.to_csv(csv_path, index=False, encoding='utf_8_sig')

# data preprocess
import nltk
from torchtext.legacy import data
#from tensorflow.keras.preprocessing.text  import Tokenizer

BATCH_SIZE = 32

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

train_data, valid_data = dataset.split(split_ratio=0.7, random_state=random.seed(SEED))
train_iterator, valid_iterator = data.BucketIterator.splits((train_data, valid_data), 
                                                            batch_size=BATCH_SIZE, 
                                                            sort_key=lambda x: len(x.poetry), 
                                                            shuffle=False, 
                                                            device=DEVICE)

# Create model
import torch.nn as nn
import torch.nn.functional as F

class lstmModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, ouput_dim, bidirectional=True):
        super(lstmModel, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, batch_first= True)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, ouput_dim)
        else:
            self.fc = nn.Linear(hidden_dim, ouput_dim)
    
    def forward(self, X):
        embedded = self.embedding(X)

        output, (hidden, cell) =  self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]),dim=1)
        out = self.fc(hidden)
        return out

class gruModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, ouput_dim, drop_out):
        super(gruModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first= True, dropout= drop_out)
        self.fc = nn.Linear(hidden_dim, ouput_dim)
    
    def forward(self, X, pre_hidden= None):
        embedded = self.embedding(X)
        
        output, hidden =  self.gru(embedded, pre_hidden)
        out = self.fc(output)
        return out, hidden
    
    def generate(self, txt, vocab, type='begin', sent_num=4, max_len=16):
        tokens = list(vocab.stoi[word] for word in txt)
        tokens = torch.tensor(tokens).unsqueeze(0).cuda()
        x = self.embedding(tokens)
        pre_hidden = torch.zeros(1, 1, self.hidden_dim).cuda()
        output = []

        if (type=='hidden head' and x.shape[1]!=sent_num):
            print("ERROR: 字数不等于句数！")
            return
        
        for i in range(sent_num):
            if i==0 and type=='begin' :
                out, hidden = self.gru(x, pre_hidden)
                out = out[:, -1, :].unsqueeze(1)
                output.append(txt)
                pre_hidden = hidden
            if type=='hidden head':
                out, hidden = self.gru(x[:, i, :].unsqueeze(1), pre_hidden)
                out = out[:, -1, :].unsqueeze(1)
                output.append(txt[i])
                pre_hidden = hidden
            for j in range(max_len):
                _, top = self.fc(out).data.topk(10)
                print(top)
                topi = top[0, 0, 0]
                #topi = top[0, 0, random.randint(0, 9)]
                if vocab.itos[topi]=='<pad>' :    # <pad>
                    #topi = vocab.stoi['。']
                    topi = top[0, 0, random.randint(1, 9)]
                xi_from_out = torch.zeros(1, 1, x.shape[-1]).cuda()
                xi_from_out[0][0][topi] = 1
                output.append(vocab.itos[topi])
                out, hidden =self.gru(xi_from_out, pre_hidden)
                pre_hidden = hidden
                if topi == vocab.stoi['。']:
                    break
        return ''.join(output)

# define model
INPUT_DIM = len(vocab)
EMBEDDING_DIM = len(vocab)
HIDDEN_DIM = 128
OUTPUT_DIM = len(vocab)
DROP_OUT = 0.5

model = gruModel(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROP_OUT)
model.to(DEVICE)

# optimizer
from torch.optim import Adam
LEARNING_RATE = 1e-3
optimizer = Adam(model.parameters(), LEARNING_RATE)

# criterion
from torch.nn import CrossEntropyLoss
loss_fn = CrossEntropyLoss()
loss_fn.to(DEVICE)

# train
def train(model, iterator, optimizer):
    train_loss = 0
    ppls = []

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        tokens = batch.poetry.to(DEVICE)
        x = tokens[:, :-1]
        y = tokens[:, 1:]
        pad_mask = torch.ne(y, vocab.stoi['<pad>']).byte().float()
        
        logits, _ = model(x)
        #probs = F.softmax(logits)
        #probs = torch.gather(probs, 2, y.unsqueeze(1)).squeeze(1)
        logits = logits.reshape(-1, logits.shape[-1])
        loss = loss_fn(logits, y.flatten())
        #probs = probs*pad_mask
        #loss = torch.mean(-torch.log(probs+1e-10))
        ppl = torch.exp(loss)

        train_loss += loss.item()
        ppls.append(ppl.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
   
    return train_loss/len(iterator), ppls, np.mean(ppls)

# evaluate
def evaluate(model, iterator):
    valid_loss = 0
    ppls = []

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            tokens = batch.poetry.to(DEVICE)
            x = tokens[:, :-1]
            y = tokens[:, 1:].flatten()
            try:
                logits, _ = model(x)
                logits = logits.reshape(-1, logits.shape[-1])
                loss = loss_fn(logits, y)
                ppl = torch.exp(loss)

                valid_loss += loss.item()
                ppls.append(ppl.item())
                preds = torch.argmax(logits, dim=1).flatten()
            except:
                continue

    return valid_loss/len(iterator), ppls, np.mean(ppls)

# time format
import time

def time_form(time):
    minute = int(time/60)
    second = int(time - minute*60)
    return minute, second

# train model
N_EPOCHS = 100
#best_valid_loss = float('inf')

print("Start Training...")
for epoch in range(N_EPOCHS):
    # record running time
    start = time.time()
    # train and evaluate model
    train_loss, train_ppls, train_ppl_mean = train(model, train_iterator, optimizer)
    valid_loss, valid_ppls, valid_ppl_mean = evaluate(model, valid_iterator)
    
    end = time.time()
    epoch_time = end-start

    minute, second = time_form(epoch_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {minute}m {second}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Train ppl: {train_ppl_mean:.2f}')
    print(f'\tValid Loss: {valid_loss:.4f} | Valid ppl: {valid_ppl_mean:.2f}')
    
print('Training Complete!') 

txt = "花红柳绿"
#output = model.generate(txt, vocab)
output = model.generate(txt, vocab, type='hidden head')
print(output)

MAX_GEN_len = 16
sentence_num = 4

def generate(model, starter, vocab):
    model.eval()
    results = list(starter)

    starter_tokens = list(vocab.stoi[word] for word in starter)
    starter_len = len(results)

    input = torch.tensor([starter_tokens[0]]).view(1, 1).cuda()
    pre_hidden = None

    for s in range(sentence_num):
        for i in range(1, MAX_GEN_len):
            logits, hidden = model(input, pre_hidden)

            if s==0 and i< starter_len:
                input = input.data.new([starter_tokens[i]]).view(1, 1)
                pre_hidden = hidden.cuda()
            else:
                top_token = logits.data[0].topk(1)[1][0].item()
                top_word = vocab.itos[top_token]
                results.append(top_word)
                input = input.data.new([top_token]).view(1, 1)
                _, pre_hidden = model(input, hidden)
                if top_word == '。':
                    break
    return ''.join(results)

print("Start generating...")
poetry = generate(model, "春江花月夜", vocab)
print(poetry) 
