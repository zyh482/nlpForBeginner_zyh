# coding:utf-8
from numpy.core.fromnumeric import shape
import tokenizers
import torch 
import pandas as pd
import numpy as np
import os
import random
import debugpy
from torch.utils.data.sampler import RandomSampler

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#debugpy.listen(('202.121.138.160'))
#debugpy.wait_for_client()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 128
BATCH_SIZE = 32
N_EPOCHS = 200

# data preprocess
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import TextGenerationPipeline
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
from torchtext.legacy.data import BucketIterator
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

csv_path = '/home/zhangyuhan/NLPcodes/poetryFromTang/train.csv'
json_path = '/home/zhangyuhan/NLPcodes/poetryFromTang/poetry1000.json'
config_path = '/home/zhangyuhan/NLPcodes/poetryFromTang/gpt-2/config.json'
vocab_path = '/home/zhangyuhan/NLPcodes/poetryFromTang/gpt-2/vocab.txt'
best_model_path = '/home/zhangyuhan/NLPcodes/poetryFromTang/best_model.bin'
pre_tokenizer_name = 'bert-base-chinese'
pre_model_name = 'ckiplab/gpt2-base-chinese'

#tokenizer = BertTokenizer(vocab_file= vocab_path)
#model_config = GPT2Config.from_json_file(config_path)
#model = GPT2LMHeadModel(config= model_config)

tokenizer = BertTokenizer.from_pretrained(pre_tokenizer_name)
model = GPT2LMHeadModel.from_pretrained(pre_model_name)
vocab = tokenizer.vocab
PAD_ID = vocab['[PAD]']
FULL_STOP_ID = vocab['。']
START_ID = vocab['[CLS]']
SEP_ID = vocab['[SEP]']
UNK_ID = vocab['[UNK]']
special = [PAD_ID, START_ID, SEP_ID, UNK_ID]

model.to(DEVICE)
optimizer = Adam(model.parameters(), lr= 2.5e-5)

def dataset(data):
    input_ids = []
    attention_masks = []
    type_ids = []
    for txt in data:
        tokens = tokenizer.encode_plus(text = txt,
                                            add_special_tokens = True, # add [CLS], [SEP]
                                            max_length = MAX_LEN,
                                            padding = 'max_length',
                                            truncation= True,
                                            return_attention_mask = True)
        input_ids.append(tokens.get('input_ids'))
        attention_masks.append(tokens.get('attention_mask'))
        type_ids.append(tokens.get('token_type_ids'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    type_ids = torch.tensor(type_ids)
    dataset = TensorDataset(input_ids, attention_masks, type_ids)
    return dataset

from sklearn.model_selection import train_test_split

#data = pd.read_csv(csv_path)
#data = data.poetry.values
data = pd.read_json(json_path)
data.paragraphs = data.paragraphs.apply(lambda po: ''.join(po))
data = data.paragraphs.values
train_data, valid_data = train_test_split(data, test_size=0.3, random_state= SEED, shuffle= True)
train_data, valid_data = dataset(train_data), dataset(valid_data)
train_sampler, valid_sampler = RandomSampler(train_data), RandomSampler(valid_data)
train_iterator, valid_iterator = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE), DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

def train(model, iterator, optimizer):
    model.train()
    epoch_loss = 0
    epoch_ppl = 0
    for batch in iterator:
        optimizer.zero_grad()

        in_ids, attn_mask, type_ids =tuple(t.to(DEVICE) for t in batch)
        #mask_labels = torch.zeros_like(in_ids).copy_(in_ids).cpu()
        #mask_labels = torch.where(mask_labels==PAD_ID, torch.tensor(-100), mask_labels)
        #mask_labels.to(DEVICE)
        output = model.forward(input_ids= in_ids,attention_mask= attn_mask, token_type_ids= type_ids, labels= in_ids, return_dict= True)
        loss = output.loss
        ppl = torch.exp(loss)
        epoch_loss += loss.item()
        epoch_ppl += ppl.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return epoch_loss/len(iterator), epoch_ppl/len(iterator)

def evaluate(model, iterator):
    model.eval()
    epoch_loss = 0
    epoch_ppl = 0
    
    
    with torch.no_grad():
        for batch in iterator:
            in_ids, attn_mask, type_ids =tuple(t.to(DEVICE) for t in batch)
            output = model.forward(input_ids= in_ids,attention_mask= attn_mask, token_type_ids= type_ids, labels= in_ids, return_dict= True)
            loss = output.loss
            ppl = torch.exp(loss)
            epoch_loss += loss.item()
            epoch_ppl += ppl.item()
    return epoch_loss/len(iterator), epoch_ppl/len(iterator)

# time format
import time
def time_form(time):
    minute = int(time/60)
    second = int(time - minute*60)
    return minute, second

# train model
print("Start Training...")
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    # record running time
    start = time.time()
    # train and evaluate model
    train_loss, train_ppl = train(model, train_iterator, optimizer)
    valid_loss, valid_ppl = evaluate(model, valid_iterator)
    
    end = time.time()
    epoch_time = end-start

    if valid_loss<best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), best_model_path)

    minute, second = time_form(epoch_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {minute}m {second}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Train ppl: {train_ppl:.2f}')
    print(f'\tValid Loss: {valid_loss:.4f} | Valid ppl: {valid_ppl:.2f}')
    
print('Training Complete!') 

def top_k_filtering(logits, top_k=15, filter_value = -float('Inf')):
    top_k = min(top_k, logits.size(-1))
    remove_idx = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[remove_idx] = filter_value
    logits[special] = filter_value
    return logits
'''
def generate(model, tokens):
    inputs = torch.LongTensor(tokens).view(1, -1).to(DEVICE)
    generated = []+ tokens
    _, past = model(inputs[:, :-1])[:2]
    prev = inputs[:, -1].view(1, -1)

    with torch.no_grad():
        for i in range(MAX_LEN):
            output = model(prev, past_key_values= past)
            logits, past = output[:2]
            #next_token = torch.argmax(logits)
            #_, top_k = torch.topk(logits, 5)
            #next_token = top_k[:, :, random.randint(0, 4)]
            logits = logits[-1].squeeze(0)
            filtered_logits = top_k_filtering(logits)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples= 1)
            if next_token==PAD_ID :
                break
            generated.append(next_token.item())
            prev = next_token.view(1, 1)
    return generated
'''
def generate(model, tokens, sen_num= 4):
    inputs = torch.LongTensor(tokens).view(1, -1).to(DEVICE)
    generated = []+ tokens
    _, past = model(inputs[:, :-1])[:2]
    prev = inputs[:, -1].view(1, -1)

    with torch.no_grad():
        for sen in range(sen_num):
            while True:
                output = model(prev, past_key_values= past)
                logits, past = output[:2]
                logits = logits[-1].squeeze(0)
                filtered_logits = top_k_filtering(logits)
                _, next_token = torch.topk(torch.softmax(filtered_logits, dim=-1), k= 1)
                generated.append(next_token.item())
                prev = next_token.view(1, 1)
                if next_token== FULL_STOP_ID:
                    break
    return generated

def hidden_head_generate(model, txt):
    inputs = torch.LongTensor(txt).view(1, -1).to(DEVICE)
    length = len(txt)
    generated = []

    with torch.no_grad():
        for i in range(length):
            _, past = model(inputs[:, i])[:2]
            prev = inputs[:, i].view(1, -1)
            generated.append(txt[i])

            while True:
                output = model(prev, past_key_values= past)
                logits, past = output[:2]
                #next_token = torch.argmax(logits)
                #_, top_k = torch.topk(logits, 5)
                #next_token = top_k[:, :, random.randint(0, 4)]
                logits = logits[-1].squeeze(0)
                filtered_logits = top_k_filtering(logits)
                _, next_token = torch.topk(torch.softmax(filtered_logits, dim=-1), k= 1)
                generated.append(next_token.item())
                prev = next_token.view(1, 1)
                if next_token == FULL_STOP_ID:
                    break
        return generated

# beam-search
class Beam(object):
    def __init__(self, tokens, probs, past, sen_count= 0) -> None:
        self.tokens = tokens
        self.probs = probs
        self.past = past
        self.sen_count = sen_count
    
    def extend(self, token, prob, past, sen_count=0):
        return Beam(tokens= self.tokens+[token], 
                    probs= self.probs + [prob],
                    past= past,
                    sen_count= self.sen_count+sen_count)
    
    def score(self):
        return np.sum(self.probs)

def best_k(model, beam, k):
    x_t = torch.tensor(beam.tokens[-1]).reshape(1, 1).to(DEVICE)

    output = model(x_t, past_key_values= beam.past)
    logits, past = output[:2]
    filtered_logits = top_k_filtering(logits[-1].squeeze(0))
    probs = F.softmax(filtered_logits, dim=-1)
    #topk_idx = torch.multinomial(probs, num_samples= k)
    _, topk_idx = torch.topk(probs, k)
    
    best_k = []
    for topi in topk_idx.tolist():
        if topi == FULL_STOP_ID:
            best_k.append(beam.extend(topi, probs[topi], past, 1))
        else:
            best_k.append(beam.extend(topi, probs[topi], past))
    return best_k


def beam_generate(txt, model, beam_width=3, sen_num=4):
    model.eval()
    
    tokens = tokenizer.encode(txt, add_special_tokens= False)
    results = []+ tokens[: -1]
    tokens = torch.tensor(tokens).to(DEVICE)
    tokens = tokens.reshape(1, -1)

    if tokens.size(-1)>1:
        _, past = model(tokens[:, :-1], past_key_values= None)[:2]
    else:
        _, past = model(tokens[:, -1], past_key_values= None)[:2]
    prev = tokens[:, -1].item()
    
    beam = Beam(tokens= [prev], probs= [0], past=past)
    k = beam_width
    curr, completed = [beam], []

    while True:
        topk = []

        for beam in curr:
            if beam.sen_count == sen_num:
                completed.append(beam)
                k -= 1
                continue
            candidators = best_k(model, beam, k)
            for candi in candidators:
                topk.append(candi)
            
        topk = sorted(topk, key= lambda x: x.score(), reverse= True)
        curr = topk[: k]
        if len(completed) == beam_width:
            break
        
    completed += curr
    completed = sorted(completed, key= lambda x: x.score(), reverse= True)
    results += [token for token in completed[0].tokens]
    return results      

def beam_hidden_head_generate(txt, model, beam_width=3):
    model.eval()
    
    txt = list(txt)
    sen_num = len(txt)
    results = []
    for start in range(sen_num):
        sen = beam_generate(txt[start], model, beam_width, sen_num= 1)
        results += sen
    
    return results 

print('Start Generating...')
model.load_state_dict(torch.load(best_model_path))
txt = '梅兰竹菊'
print(txt)
tokens = tokenizer.encode_plus(txt, add_special_tokens= False)['input_ids']
generated = generate(model, tokens, sen_num=8)
out = tokenizer.decode(generated)
print("开头生成...")
print(out)
hidden_head_generated = hidden_head_generate(model, tokens)
out = tokenizer.decode(hidden_head_generated)
print("藏头诗生成...")
print(out)
beam_generated = beam_generate(txt, model, beam_width= 5, sen_num= 8)
out = tokenizer.decode(beam_generated)
print("beam search...")
print(out)
beam_hidden_head_generated = beam_hidden_head_generate(txt, model, beam_width= 5)
out = tokenizer.decode(beam_hidden_head_generated)
print("beam search hidden head...")
print(out)
