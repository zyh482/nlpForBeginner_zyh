#读取数据集
import pandas as pd

train_csv = pd.read_csv("/home/zhangyuhan/NLPcodes/sentiment-analysis-on-movie-reviews/train.tsv", sep="\t")

import torch
import random
# 随机数
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 设置设备     
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义Field
from torchtext.legacy import data, datasets
from torchtext.utils import download_from_url
import os
import spacy


TEXT = data.Field(sequential = True, tokenize= 'spacy',lower=True)
LABEL = data.LabelField(sequential = False, dtype = torch.float)
dataField = [('PhraseId', None),('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)]

# 创建dataset
print('Start Preprocessing...')
dataset = data.TabularDataset(path='./NLPcodes/sentiment-analysis-on-movie-reviews/train.csv',
                             format ='csv',
                             fields=dataField,
                             skip_header=True)
train_data, valid_data = dataset.split(split_ratio=0.8, random_state=random.seed(SEED))

# Load pre-trained GloVe Embedding
LABEL.build_vocab(train_data)
TEXT.build_vocab(train_data, vectors='glove.6B.300d')
embedding = TEXT.vocab.vectors.to(device)

# 创建Iterator
BATCH_SIZE = 16

train_iterator, valid_iterator = data.BucketIterator.splits((train_data, valid_data),
                                                            batch_size= BATCH_SIZE,
                                                            sort_key = lambda x: len(x.Phrase),
                                                            sort_within_batch = True,
                                                            device = device)
print('Preprocessing complete!')

# 定义模型
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, ouput_dim, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, ouput_dim)
        else:
            self.fc = nn.Linear(hidden_dim, ouput_dim)
    
    def forward(self, X):
        embedded = self.embedding(X)
        
        output, (hidden, cell) =  self.lstm(embedded)
        #output = [input dim, batch size, hid dim]
        #hidden = [directions, batch size, hid dim]
        #cell = [directions, batch size, hid dim]
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]),dim=1)
        out = self.fc(hidden)
        return out

# 初始化模型
from transformers import AdamW

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 374
OUTPUT_DIM = 5
BIDIRECTIONAL = True

model = LSTMClassifier(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, BIDIRECTIONAL)
model.to(device)
# 模型优化
optimizer = AdamW(model.parameters(),lr=5e-5, eps=1e-8)    

# loss函数和accuracy函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

def acc_fn(preds, y):
    preds = torch.argmax(preds, dim=1).flatten()
    accuracy = (preds==y).cpu().numpy().mean()
    return accuracy

# Train
def train(model, iterator, optimizer,loss_fn):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        
        optimizer.zero_grad()
        # forward pass
        preds = model(batch.Phrase.to(device)).squeeze(0)
        
        # compute loss and accuracy 
        loss = loss_fn(preds, batch.Sentiment.to(device).long())
        acc = acc_fn(preds, batch.Sentiment.to(device).long())
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        # backward pass
        loss.backward()
        # clip the norm of the gradients to 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
    # return average loss
    return epoch_loss/len(iterator), epoch_acc/len(iterator)

# Evalute
def evaluate(model, iterator, loss_fn):
    valid_loss = 0
    valid_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            preds = model(batch.Phrase.to(device)).squeeze(0)
            
            loss = loss_fn(preds, batch.Sentiment.to(device).long())
            acc = acc_fn(preds, batch.Sentiment.to(device))
            valid_loss += loss.item()
            valid_acc += acc.item()
    
    return valid_loss/len(iterator), valid_acc/len(iterator)

# time format
def time_form(time):
    minute = int(time/60)
    second = int(time - minute*60)
    return minute, second

# Train Model
import time

N_EPOCHS = 4
best_valid_loss = float('inf')

print("Start Training...")
for epoch in range(N_EPOCHS):
    # record running time
    start = time.time()
    # train and evaluate model
    train_loss, train_acc = train(model, train_iterator, optimizer, loss_fn)
    valid_loss, valid_acc = evaluate(model, valid_iterator, loss_fn)
    
    end = time.time()
    epoch_time = end-start
    # record the best_valid_loss model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './NLPcodes/best_LSTM_model.bin')
        
    minute, second = time_form(epoch_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {minute}m {second}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tValid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:.2f}%')
    
print('Training Complete!') 

# Predict
import torch.nn.functional as F

def predict(model, iterator):
    all_preds = []
    pred_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            preds = model(batch.Phrase.to(device)).squeeze(0)
            acc = acc_fn(preds, batch.Sentiment.to(device))

            pred_acc += acc.item()
            all_preds.append(preds)

    all_preds = torch.cat(all_preds, dim=0)
    # Apply Softmax to calculate probabilities
    probs = F.softmax(all_preds, dim=1).cpu().numpy()
    
    return probs, pred_acc/len(iterator)

# best_valid_loss model
#model.load_state_dict(torch.load('best_RNN_model.bin'))

#Create test dataloader
print('Starting Preprocessing...')
'''test_tsv = pd.read_csv("/home/zhangyuhan/NLPcodes/sentiment-analysis-on-movie-reviews/test.tsv", sep="\t")
test_label_csv = pd.read_csv("/home/zhangyuhan/NLPcodes/sentiment-analysis-on-movie-reviews/sampleSubmission.csv")
test_tsv.to_csv('/home/zhangyuhan/NLPcodes/sentiment-analysis-on-movie-reviews/test.csv', index=False)

test_csv = pd.read_csv('/home/zhangyuhan/NLPcodes/sentiment-analysis-on-movie-reviews/test.csv')
test = pd.merge(test_csv, test_label_csv, on='PhraseId')
test.to_csv('/home/zhangyuhan/NLPcodes/sentiment-analysis-on-movie-reviews/test.csv', index=False)
'''
dataset = data.TabularDataset(path='./NLPcodes/sentiment-analysis-on-movie-reviews/test.csv',
                             format ='csv',
                             fields=dataField,
                             skip_header=True)
test_iterator = data.BucketIterator(dataset, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.Phrase), 
                                 sort_within_batch=True,device=device)
print('Preprocessing Complete!')

# Predict
print('Start Predicting...')

start = time.time()
probs, pred_acc = predict(model, test_iterator)
end = time.time()

minute, second = time_form(end-start)
print(f'predicting time: {minute}m {second}s')
print(f'test accuracy: {pred_acc*100:.2f}%')
