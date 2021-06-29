import pandas as pd
import torch
import random

# random seed
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# deivice
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
# label: str->int
train_txt = pd.read_table('./NLPcodes/snli_1.0/snli_1.0_train.txt')
test_txt = pd.read_table('./NLPcodes/snli_1.0/snli_1.0_test.txt')

label_dicts = {'entailment':0, 'neutral':1, 'contradiction':2, '-':4}
train_txt['gold_label'] = train_txt['gold_label'].apply(lambda x: label_dicts[x])
test_txt['gold_label'] = test_txt['gold_label'].apply(lambda x: label_dicts[x])

# Create csv
train_txt.to_csv('./NLPcodes/snli_1.0/snli_1.0_train.csv', index=False, header=True)
test_txt.to_csv('./NLPcodes/snli_1.0/snli_1.0_test.csv', index=False, header=True)
'''
# Create field
from torchtext.legacy import data
import spacy

#MAX_SENTENCE_LEN = 64
TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', batch_first=True)
LABEL = data.LabelField(sequential= False, dtype=torch.float)
dataField = {'gold_label': ('gold_label', LABEL), 'sentence1': ('sentence1', TEXT), 'sentence2': ('sentence2', TEXT)}

# Create dataset
print('Start Preprocessing...')
dataset = data.TabularDataset(path='./NLPcodes/snli_1.0/snli_1.0_train.csv',
                             format ='csv',
                             fields=dataField,
                             skip_header=False)
train_data, valid_data = dataset.split(split_ratio=0.8, random_state=random.seed(SEED))

# Load pre-trained GloVe Embedding
LABEL.build_vocab(train_data)
TEXT.build_vocab(train_data, valid_data)
TEXT.vocab.load_vectors('glove.840B.300d')
embedding = TEXT.vocab.vectors.to(device)

# Create Iterator
BATCH_SIZE = 32

train_iterator, valid_iterator = data.BucketIterator.splits((train_data, valid_data),
                                                            batch_size= BATCH_SIZE,
                                                            device= device,
                                                            sort= False)
print('Preprocessing Complete!')

# Define LSTM model
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, drop_out, bidirectional=True):
        super(BiLSTM, self).__init__()

        self.directions = bidirectional+1
        self.layers_num = 2
        self.combin = 4     # premise, hypothesis, abs(premise-hypothesis), premise*hypothesis
        #self.h0 = self.c0 = torch.tensor([]).new_zeros((self.layers_num*self.directions, BATCH_SIZE, hidden_dim)).to(device)
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=self.layers_num, bidirectional= bidirectional, batch_first= True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)
        self.linear1 = nn.Linear(hidden_dim*self.directions*self.combin, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        for linear in [self.linear1, self.linear2, self.linear3]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

        self.output = nn.Sequential(
            self.linear1,
            self.relu,
            self.dropout,
            self.linear2,
            self.relu,
            self.dropout,
            self.linear3
        )
    
    def forward(self, sentence1, sentence2):
        premise_embedded = self.embedding(sentence1)
        hypothesis_embedded = self.embedding(sentence2)

        premise_proj = self.relu(self.projection(premise_embedded))
        hypothesis_proj = self.relu(self.projection(hypothesis_embedded))
        
        _, (premise_ht, _) = self.lstm(premise_proj)
        _, (hypothesis_ht, _) = self.lstm(hypothesis_proj)

        premise = premise_ht[-2:].transpose(0, 1).contiguous().view(len(sentence1), -1)
        hypothesis = hypothesis_ht[-2:].transpose(0, 1).contiguous().view(len(sentence2), -1)
                                                                                                                     
        combined = torch.cat((premise, hypothesis, torch.abs(premise-hypothesis), premise*hypothesis),dim=1)
        output = self.output(combined)
        return output

# Create LSTM model
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
HIDDEN_DIM = 100
OUTPUT_DIM = 4
DROP_OUT = 0.5
#BIDIRECTIONAL = True

model = BiLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, DROP_OUT)
model.to(device)

# Optimizer
from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=0.0004)

# Criterion
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

def acc_fn(prods, labels):
    preds = torch.argmax(prods, dim=1).flatten()
    accuracy = (preds==labels).cpu().numpy().mean()
    return accuracy

# Train
def train(model, iterator, optimizer, loss_fn):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        labels = batch.gold_label.to(device)
        premise = batch.sentence1.to(device)
        hypothesis = batch.sentence2.to(device)
        
        probs = model(premise, hypothesis)

        loss = loss_fn(probs, labels.long())
        acc = acc_fn(probs, labels.long())
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    return epoch_loss/len(iterator), epoch_acc/len(iterator)

# Evaluate
def evaluate(model, iterator,loss_fn):
    valid_loss = 0
    valid_acc = 0

    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            labels = batch.gold_label.to(device)
            premise = batch.sentence1.to(device)
            hypothesis = batch.sentence2.to(device)
            
            probs = model(premise, hypothesis)

            loss = loss_fn(probs, labels.long())
            acc = acc_fn(probs, labels.long())
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

N_EPOCHS = 10 
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
        torch.save(model.state_dict(), './NLPcodes/best_snli_BiLSTM_model.bin')
        
    minute, second = time_form(epoch_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {minute}m {second}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tValid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:.2f}%')
    
print('Training Complete!') 

# Predict
import torch.nn.functional as F

def predict(model, iterator):
    all_probs = []
    pred_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            labels = batch.gold_label.to(device)
            premise = batch.sentence1.to(device)
            hypothesis = batch.sentence2.to(device)
            
            probs = model(premise, hypothesis)
            acc = acc_fn(probs, labels.long())
            pred_acc += acc.item()
            all_probs.append(probs)

    all_probs = torch.cat(all_probs, dim=0)
    # Apply Softmax to calculate probabilities
    preds = F.softmax(all_probs, dim=1).cpu().numpy()
    
    return preds, pred_acc/len(iterator)

# Create test Iterator
print('Start Preprocessing...')
dataset = data.TabularDataset(path='./NLPcodes/snli_1.0/snli_1.0_test.csv',
                             format ='csv',
                             fields=dataField,
                             skip_header=False)
test_iterator = data.BucketIterator(dataset, batch_size= BATCH_SIZE, device= device, sort=False)
print('Preprocessing Complete!')

tart = time.time()
probs, pred_acc = predict(model, test_iterator)
end = time.time()

minute, second = time_form(end-start)
print(f'predicting time: {minute}m {second}s')
print(f'test accuracy: {pred_acc*100:.2f}%')