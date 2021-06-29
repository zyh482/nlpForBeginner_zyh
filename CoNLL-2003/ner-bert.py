import torch 
import pandas as pd
import os
import random

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read in datafile
file_names = ['train', 'valid', 'test']
extension = '.txt'
data_dir = '/home/zhangyuhan/NLPcodes/CoNLL-2003/'
files = {}
for f in file_names:
    files[f] = pd.read_csv(os.path.join(data_dir ,f+extension), 
                            sep=' ', 
                            header=None, 
                            names=['word', 'POS', 'syntactic chunk', 'tag'])
    # delete any rows with 'nan'
    files[f] = files[f].dropna(axis=0, how='any')

train_words, valid_words, test_words = files['train'].word.values, files['valid'].word.values, files['test'].word.values
train_tags, valid_tags, test_tags = files['train'].tag.values, files['valid'].tag.values, files['test'].tag.values

# tags <-> labels
TAGS = files['train']['tag'].value_counts().keys().to_list()
tag2label_dict = {}
label2tag_dict = {}
for i in range(len(TAGS)):
    tag2label_dict[TAGS[i]] = i
    label2tag_dict[i] = TAGS[i]

def tag_label_convert(keys, dict):
    out = []
    for key in keys:
        out.append(dict[key])
    return torch.tensor(out)

# data preprocess
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

BERTNAME = 'bert-base-uncased'
MAX_LEN = 8
BATCH_SIZE = 32

def preprocess(words, tags):
    input_ids = []
    attention_masks = []

    tokenizer = BertTokenizer.from_pretrained(BERTNAME, do_lower_case= True)

    for word in words:
        encoded = tokenizer.encode_plus(word, max_length= MAX_LEN, padding='max_length', truncation= True)
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    labels = tag_label_convert(tags, tag2label_dict)

    ids = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    dataset = TensorDataset(ids, masks, labels)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler= sampler, batch_size= BATCH_SIZE)

    return dataloader

print("Start preprocessing...")
train_dataloader = preprocess(train_words, train_tags)
valid_dataloader = preprocess(valid_words, valid_tags)
test_dataloader = preprocess(test_words, test_tags)
print('Preprocessing complete!')

# define model
from transformers import BertModel
import torch.nn as nn

class NerBertModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(NerBertModel, self).__init__()

        self.bert = BertModel.from_pretrained(BERTNAME)
        self.out = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input_ids, attention_masks):
        bert_outs = self.bert(input_ids, attention_masks)
        last_hidden = bert_outs[0][:, 0, :]
        logits = self.out(last_hidden)
        return logits
    
# Create bert model
INPUT_DIM = 768     # bert-base
HIDDEN_DIM = 100
OUTPUT_DIM = len(TAGS)
DROP_OUT = 0.5
LEARNING_RATE = 1e-5

model = NerBertModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROP_OUT)
model.to(DEVICE)

# optimizer
from torch.optim import Adam
optimizer = Adam(model.parameters(), LEARNING_RATE)

# criterion
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

loss_fn = CrossEntropyLoss()
loss_fn.to(DEVICE)

def criterion(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precise, recall,  fbeta_score, support = precision_recall_fscore_support(labels, preds, average='micro')
    return accuracy, precise, recall, fbeta_score

# train
def train(model, dataloader, optimizer):
    train_loss = 0
    all_labels = []
    all_preds = []

    model.train()

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids, attention_masks, labels = tuple(t.to(DEVICE) for t in batch)

        logits = model(input_ids, attention_masks)

        loss = loss_fn(logits, labels)
        train_loss += loss.item()

        preds = torch.argmax(logits, dim=1).flatten()
        all_preds.append(preds)
        all_labels.append(labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    train_acc, train_precise, train_recall, train_F = criterion(all_labels.cpu(), all_preds.cpu())

    return train_loss/len(dataloader), train_acc, train_precise, train_recall, train_F

# evaluate
def evaluate(model, dataloader):
    valid_loss = 0
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks, labels = tuple(t.to(DEVICE) for t in batch)

            logits = model(input_ids, attention_masks)

            loss = loss_fn(logits, labels)
            valid_loss += loss.item()

            preds = torch.argmax(logits, dim=1).flatten()
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    valid_acc, valid_precise, valid_recall, valid_F = criterion(all_labels.cpu(), all_preds.cpu())

    return valid_loss/len(dataloader), valid_acc, valid_precise, valid_recall, valid_F

# predict
def predict(model, dataloader):
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks, labels = tuple(t.to(DEVICE) for t in batch)

            logits = model(input_ids, attention_masks)

            preds = torch.argmax(logits, dim=1).flatten()
            all_labels.append(labels)
            all_preds.append(preds)
    
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    pred_acc, pred_precise, pred_recall, pred_F = criterion(all_labels.cpu(), all_preds.cpu())

    return preds, pred_acc, pred_precise, pred_recall, pred_F

# time format
import time

def time_form(time):
    minute = int(time/60)
    second = int(time - minute*60)
    return minute, second

# train model
N_EPOCHS = 10
#best_valid_loss = float('inf')

print("Start Training...")
for epoch in range(N_EPOCHS):
    # record running time
    start = time.time()
    # train and evaluate model
    train_loss, train_acc, train_precise, train_recall, train_F = train(model, train_dataloader, optimizer)
    valid_loss, valid_acc, valid_precise, valid_recall, valid_F = evaluate(model, valid_dataloader)
    
    end = time.time()
    epoch_time = end-start
    '''
    # record the best_valid_loss model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './best_ner_bert_model.bin')
    '''
    minute, second = time_form(epoch_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {minute}m {second}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | \
        Train Precise: {train_precise*100:.2f}% | Train Recall: {train_recall*100:.2f}% | \
        Train F: {train_F:.2f}')
    print(f'\tValid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:.2f}% | \
        Valid Precise: {valid_precise*100:.2f}% | Valid Recall: {valid_recall*100:.2f}% | \
        Valid F: {valid_F:.2f}')
    
print('Training Complete!') 

# predict
print("Start predicting...")
start = time.time()
preds, pred_acc, pred_precise, pred_recall, pred_F = predict(model, test_dataloader)
end = time.time()

minute, second = time_form(end-start)
print(f'predicting time: {minute}m {second}s')
print(f'Test accuracy: {pred_acc*100:.2f}% | Test Precise {pred_precise*100:.2f}% | \
    Test Recall: {pred_recall*100:.2f}% | Test F: {pred_F:.2f} ')
print("Predicting Complete!")

