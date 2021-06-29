import pandas as pd
import torch
import random

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

train_path = "/home/zhangyuhan/NLPcodes/sentiment-analysis-on-movie-reviews/train.tsv"
test_path = "/home/zhangyuhan/NLPcodes/sentiment-analysis-on-movie-reviews/test.tsv"
test_labels_path = "/home/zhangyuhan/NLPcodes/sentiment-analysis-on-movie-reviews/sampleSubmission.csv"
MAX_LEN = 128 
batch_size = 32
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # define device

# read in files
train_csv = pd.read_csv(train_path, sep="\t")
test_csv = pd.read_csv(test_path, sep="\t")
test_label_csv = pd.read_csv(test_labels_path)

X = train_csv.Phrase.values
Y = train_csv.Sentiment.values
X_test = test_csv.Phrase.values
Y_test = test_label_csv.Sentiment.values

# split train and valid
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state= SEED)

# Pre-trained tokenier
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=True)

def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []
    
    for txt in data:
        encoded_txt = tokenizer.encode_plus(text = txt,
                                             add_special_tokens = True, # add [CLS], [SEP]
                                             max_length = MAX_LEN,
                                             padding = 'max_length',
                                             return_attention_mask = True)
        input_ids.append(encoded_txt.get('input_ids'))
        attention_masks.append(encoded_txt.get('attention_mask'))
        
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    
    return input_ids, attention_masks

# data-preprocessing
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

print('Start Preprocessing...')

train_inputs, train_masks = preprocessing_for_bert(X_train)
valid_inputs, valid_masks = preprocessing_for_bert(X_valid)
test_inputs, test_masks = preprocessing_for_bert(X_test)

# create DataLoader
train_labels = torch.tensor(Y_train)
valid_labels = torch.tensor(Y_valid)
test_labels = torch.tensor(Y_test)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

valid_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
valid_sampler =SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

print('Preprocessing Complete!')

# define model
import torch.nn as nn
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        
        in_D, hidden_D, out_D = 768, 50, 5
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.classifier = nn.Sequential(nn.Linear(in_D, hidden_D),
                                       nn.ReLU(),
                                       #nn.Dropout(0.3),
                                       nn.Linear(hidden_D, out_D))
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls =outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

def initialize_model(epochs):
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier()
    bert_classifier.to(device)
    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8)    # Default epsilon value
    # Set up the learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

# loss函数和accuracy函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

def acc_fn(logits, y):
    preds = torch.argmax(logits, dim=1).flatten()
    accuracy = (preds==y).cpu().numpy().mean()
    return accuracy

# Train
def train(model, dataloader, optimizer, scheduler, loss_fn):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in dataloader:
        # load batch to device
        in_ids, attn_mask, labels =tuple(t.to(device) for t in batch)
        
        optimizer.zero_grad()
        # forward pass
        logits = model(in_ids, attn_mask)
        
        # compute loss and accuracy 
        loss = loss_fn(logits, labels)
        acc = acc_fn(logits, labels)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        # backward pass
        loss.backward()
        # clip the norm of the gradients to 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters and learning rate
        optimizer.step()
        scheduler.step()
    # return average loss
    return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

# Evalute
def evaluate(model, dataloader, loss_fn):
    valid_loss = 0
    valid_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            in_ids, attn_mask, labels = tuple(t.to(device) for t in batch)
            logits = model(in_ids, attn_mask)
            
            loss = loss_fn(logits, labels)
            acc = acc_fn(logits, labels)
            valid_loss += loss.item()
            valid_acc += acc.item()
    
    return valid_loss/len(dataloader), valid_acc/len(dataloader)

# time format
def time_form(time):
    minute = int(time/60)
    second = int(time - minute*60)
    return minute, second

# Train Model
import time

N_EPOCHS = 10
model, optimizer, scheduler = initialize_model(N_EPOCHS)
best_valid_loss = float('inf')

print("Start Training...")
for epoch in range(N_EPOCHS):
    # record running time
    start = time.time()
    # train and evaluate model
    train_loss, train_acc = train(model, train_dataloader, optimizer, scheduler, loss_fn)
    valid_loss, valid_acc = evaluate(model, valid_dataloader, loss_fn)
    
    end = time.time()
    epoch_time = end-start
    # record the best_valid_loss model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.bin')
        
    minute, second = time_form(epoch_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {minute}m {second}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tValid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:.2f}%')
    
print('Training Complete!') 

# Predict
import torch.nn.functional as F

def bert_predict(model, dataloader):
    all_logits = []
    pred_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            in_ids, attn_mask, labels = tuple(t.to(device) for t in batch)
            logits = model(in_ids, attn_mask)
            acc = acc_fn(logits, labels)

            pred_acc += acc.item()
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)
    # Apply Softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    
    return probs, pred_acc/len(dataloader)

# best_valid_loss model
#model.load_state_dict(torch.load('best_model.bin'))

# Predict
print('Start Predicting...')

start = time.time()
probs, pred_acc = bert_predict(model, test_dataloader)
end = time.time()

minute, second = time_form(end-start)
print(f'predicting time: {minute}m {second}s')
print(f'test accuracy: {pred_acc*100:.2f}%')
