import torch
import random
import pandas as pd

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read in datafile
train_csv = pd.read_csv("./NLPcodes/snli_1.0/snli_1.0_train_3_labels.csv")

premise = train_csv.sentence1.values
hypothesis = train_csv.sentence2.values
labels = train_csv.gold_label.values

# split train/valid data
from sklearn.model_selection import train_test_split
train_premise, valid_premise, train_hypothesis, valid_hypothesis, train_labels, valid_labels = train_test_split(premise, hypothesis, labels, test_size = 0.2, random_state=SEED)
# data preprocess
from transformers import BartTokenizer
import lightseq

MAX_SENTENCE_LEN = 128
BERT_NAME = 'facebook/bart-base'

def preprocess(sentence1, sentence2): 
    tokenizer = BartTokenizer.from_pretrained(BERT_NAME, do_lower_case=True)

    input_ids =[]
    type_ids = []
    attention_masks = []

    for (x, y) in zip(sentence1, sentence2):
        x, y = str(x), str(y)
        encoded = tokenizer(x, y,
                            add_special_tokens = True,
                            max_length= MAX_SENTENCE_LEN,
                            padding= 'max_length')
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    return torch.tensor(input_ids), torch.tensor(attention_masks)  

print("Starting Preprocessing...")
train_X_inputs, train_X_masks = preprocess(train_premise, train_hypothesis)
train_Y = torch.tensor(train_labels)
valid_X_inputs, valid_X_masks = preprocess(valid_premise, valid_hypothesis)
valid_Y = torch.tensor(valid_labels)

# create dataloader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

BATCH_SIZE = 16

train_data = TensorDataset(train_X_inputs, train_X_masks, train_Y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler= train_sampler, batch_size= BATCH_SIZE)

valid_data = TensorDataset(valid_X_inputs, valid_X_masks, valid_Y)
valid_sampler = RandomSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler= valid_sampler, batch_size = BATCH_SIZE)

print("Preprocessing Complete!")

# define bert model
import torch.nn as nn
from transformers import BartModel
import lightseq

class BertClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim, drop_out):
        super(BertClassifier, self).__init__()

        # store dims for linear operation
        self.input_dim = 768
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        #self.bert = BartModel.from_pretrained(BERT_NAME)
        self.bert = lightseq.Transformer("lightseq_bart_base.bp", BATCH_SIZE)
        self.output = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(self.hidden_dim, self.output_dim)
        ) 

    def forward(self, input_ids, attention_masks):
        bert_outs = self.bert(input_ids= input_ids, attention_mask= attention_masks)
        last_hidden = bert_outs[0][:, 0, :]
        logits = self.output(last_hidden)
        return logits

# create bert model
HIDDEN_DIM = 300
OUTPUT_DIM = 3
DROP_OUT = 0.3
LEARNING_RATE = 0.001

model = BertClassifier(HIDDEN_DIM, OUTPUT_DIM, DROP_OUT)
model.to(device)

from torch.optim import Adam

optimizer = Adam(model.parameters(), LEARNING_RATE)

# criterion
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

def acc_fn(logits, labels):
    preds = torch.argmax(logits, dim=1).flatten()
    accuracy = (preds==labels).cpu().numpy().mean()
    return accuracy

# train
def train(model, dataloader, optimizer, loss_fn):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)

        logits = model(input_ids, attention_masks)

        loss = loss_fn(logits, labels)
        accuracy = acc_fn(logits, labels)

        epoch_loss += loss.item()
        epoch_acc += accuracy.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

# evaluate
def evaluate(model, dataloader, loss_fn):
    valid_loss = 0
    valid_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)

            logits = model(input_ids, attention_masks)

            loss = loss_fn(logits, labels)
            accuracy = acc_fn(logits, labels)

            valid_loss += loss.item()
            valid_acc += accuracy.item()
    
    return valid_loss/len(dataloader), valid_acc/len(dataloader)

# predict
import torch.nn.functional as F

def predict(model, dataloader):
    all_logits = []
    pred_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks, labels = tuple(t.to(device) for t in batch)

            logits = model(input_ids, attention_masks)
            all_logits.append(logits)

            accuracy = acc_fn(logits, labels)
            pred_acc += accuracy.item()

    all_logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs, pred_acc/len(dataloader)

# time format
import time

def time_form(time):
    minute = int(time/60)
    second = int(time - minute*60)
    return minute, second

# Train Model
N_EPOCHS = 4
best_valid_loss = float('inf')

print("Start Training...")
for epoch in range(N_EPOCHS):
    # record running time
    start = time.time()
    # train and evaluate model
    train_loss, train_acc = train(model, train_dataloader, optimizer, loss_fn)
    valid_loss, valid_acc = evaluate(model, valid_dataloader, loss_fn)
    
    end = time.time()
    epoch_time = end-start
    # record the best_valid_loss model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './NLPcodes/best_snli_bert_model.bin')
        
    minute, second = time_form(epoch_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {minute}m {second}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tValid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:.2f}%')
    
print('Training Complete!') 

# read in datafile
test_csv = pd.read_csv("./NLPcodes/snli_1.0/snli_1.0_test_3_labels.csv")

premise = test_csv.sentence1.values
hypothesis = test_csv.sentence2.values
labels = test_csv.gold_label.values

print("Starting Preprocessing...")
test_X_inputs, test_X_masks = preprocess(premise, hypothesis)
test_Y = torch.tensor(labels)

test_data = TensorDataset(test_X_inputs, test_X_masks, test_Y)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler= test_sampler, batch_size= BATCH_SIZE)
print("Preprocessing Complete!")

# predict
print("Start predicting...")
start = time.time()
probs, pred_acc = predict(model, test_dataloader)
end = time.time()

minute, second = time_form(end-start)
print(f'predicting time: {minute}m {second}s')
print(f'test accuracy: {pred_acc*100:.2f}%')
print("Predicting Complete!")