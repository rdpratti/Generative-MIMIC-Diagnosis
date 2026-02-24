from lime.lime_text import LimeTextExplainer
import os
from tqdm.notebook import tqdm
import copy
import re as re
import random
import time
import torch
from torch import nn
from torch.nn.functional import softmax
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from ml_things import plot_dict, plot_confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score,roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
#from transformers.utils import (is_tf_available, is_torch_available)
from transformers import set_seed
from torchmetrics import AUROC
#from torcheval.metrics import MulticlassAUROC
import itertools
from datetime import datetime, timezone, timedelta
from pytz import timezone
#import optuna
#from optuna.integration import PyTorchLightningPruningCallback
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")


def get_dataframe_from_parquet(data_file):

    #path = 'E:/Education/CCSU-Thesis-2024/Data/'
    path = Path('/content/')
    file1 = path + data_file
    df = pd.read_parquet(file1,engine='auto',dtype_backend='numpy_nullable')
    df['label'] = df['label'].astype('int')
    diag_df = create_subsequences(df, 318)
    #print(df.sort_values(by='id'))
    #diag_df = diag_df.sample(n=500)
    print(diag_df.groupby(by=["label","icd10_code"]).count())
    possible_labels = diag_df[['label','icd10_code']].drop_duplicates()
    label_dict = possible_labels.set_index('icd10_code')['label'].to_dict()

    return(diag_df[['id', 'text','label', 'icd10_code']], label_dict)

def get_needed_dataset(run='distinct', dtype='train'):

    if run == 'distinct':
        if dtype == 'train':
            data_file = "intsl_train_group_full_no2c.snappy.parquet"
        elif dtype == 'val1':
            data_file = "intsl_val1_group_full_no2c.snappy.parquet"
        elif dtype == 'val2':
            data_file = "intsl_val2_group_full_no2c.snappy.parquet"
        elif dtype == 'test':    
            data_file = "intsl_test_group_full_no2c.snappy.parquet"
    elif run == 'similar':
        if dtype == 'train':
            data_file = "intsl_train_group_full_no3c.snappy.parquet"
        elif dtype == 'val1':
            data_file = "intsl_val1_group_full_no3c.snappy.parquet"
        elif dtype == 'val2':
            data_file = "intsl_val2_group_full_no3c.snappy.parquet"
        elif dtype == 'test':
            data_file = "intsl_test_group_full_no3c.snappy.parquet"
    
    data_df, labels = get_dataframe_from_parquet(data_file)
    
    return (data_df, labels)

def find_unique_dataset(diags, data_df, size_num):

    np.random.seed(100)
    classa = diags[0]
    classb = diags[1]
    classc = diags[2]
    classd = diags[3]
    
    classa_ids = data_df.loc[data_df.icd10_code == classa, 'id'].unique()
    classb_ids = data_df.loc[data_df.icd10_code == classb, 'id'].unique()
    classc_ids = data_df.loc[data_df.icd10_code == classc, 'id'].unique()
    classd_ids = data_df.loc[data_df.icd10_code == classd, 'id'].unique()
    
    
    classa_aug_ids = np.random.choice(classa_ids,size=size_num,replace=False)
    classb_aug_ids = np.random.choice(classb_ids,size=size_num,replace=False)
    classc_aug_ids = np.random.choice(classc_ids,size=size_num,replace=False)
    classd_aug_ids = np.random.choice(classd_ids,size=size_num,replace=False)
    Aug_list = np.concatenate([classa_aug_ids, classb_aug_ids, classc_aug_ids, classd_aug_ids])
    
    
    Aug_id_df = pd.DataFrame(Aug_list, columns=['id'])
    Aug_id_df['id'] = Aug_id_df['id'].astype(int)
    
    aug_data = pd.merge(Aug_id_df, data_df, left_on=['id'], right_on=['id'], how='left')
    
    return aug_data

def  get_diagnosis_data(diagnoses_list, run,breakdown=[111,19,19,15]):

    set_seed(100)
    
    print(diagnoses_list)
    print(run)
    #if run == 'distinct':
    #  data_file = "intsl_train_group_full_no2c.snappy.parquet"
    #  data_file2 = "intsl_val1_group_full_no2c.snappy.parquet"
    #  data_file4 = "intsl_val2_group_full_no2c.snappy.parquet"
    #  data_file3 = "intsl_test_group_full_no2c.snappy.parquet"
    #elif run == 'similar':
    #  data_file = "intsl_train_group_full_no3c.snappy.parquet"
    #  data_file2 = "intsl_val1_group_full_no3c.snappy.parquet"
    #  data_file4 = "intsl_val2_group_full_no3c.snappy.parquet"
    #  data_file3 = "intsl_test_group_full_no3c.snappy.parquet"

    train_data, labels1 = get_needed_dataset(run, dtype='train')
    val1_data, labels2 = get_needed_dataset(run, dtype='val1')
    val2_data, labels4 = get_needed_dataset(run, dtype='val2')
    test_data, labels3 = get_needed_dataset(run, dtype='test')
    
    #train_data, labels1 = get_dataframe_from_parquet(data_file)
    #val1_data, labels2 = get_dataframe_from_parquet(data_file2)
    #val2_data, labels4 = get_dataframe_from_parquet(data_file4)
    #test_data, labels3 = get_dataframe_from_parquet(data_file3)
    
    label_dict = labels1 | labels2 
    #classa = diagnoses_list[0]
    #classb = diagnoses_list[1]
    #classc = diagnoses_list[2]
    #classd = diagnoses_list[3]
    train_size = breakdown[0]
    val1_size = breakdown[1]
    val2_size = breakdown[2]
    test_size = breakdown[3]
    print(train_data.groupby('icd10_code')['id'].nunique())
    
    aug_train_data = find_unique_dataset(diagnoses_list, 
                                         train_data, train_size)
    aug_test_data = find_unique_dataset(diagnoses_list, 
                                         test_data, test_size)
    aug_val1_data = find_unique_dataset(diagnoses_list, 
                                         val1_data, val1_size)
    aug_val2_data = find_unique_dataset(diagnoses_list, 
                                         val2_data, val2_size)
    #classa_train_ids = train_data.loc[train_data.icd10_code == classa, 'id'].unique()
    #classb_train_ids = train_data.loc[train_data.icd10_code == classb, 'id'].unique()
    #classc_train_ids = train_data.loc[train_data.icd10_code == classc, 'id'].unique()
    #classd_train_ids = train_data.loc[train_data.icd10_code == classd, 'id'].unique()
    #classa_test_ids = test_data.loc[test_data.icd10_code == classa, 'id'].unique()
    #classb_test_ids = test_data.loc[test_data.icd10_code == classb, 'id'].unique()
    #classc_test_ids = test_data.loc[test_data.icd10_code == classc, 'id'].unique()
    #classd_test_ids = test_data.loc[test_data.icd10_code == classd, 'id'].unique()
    #classa_val_ids = val_data.loc[val_data.icd10_code == classa, 'id'].unique()
    #classb_val_ids = val_data.loc[val_data.icd10_code == classb, 'id'].unique()
    #classc_val_ids = val_data.loc[val_data.icd10_code == classc, 'id'].unique()
    #classd_val_ids = val_data.loc[val_data.icd10_code == classd, 'id'].unique()
    
    #classa_val2_ids = val2_data.loc[val2_data.icd10_code == classa, 'id'].unique()
    #classb_val2_ids = val2_data.loc[val2_data.icd10_code == classb, 'id'].unique()
    #classc_val2_ids = val2_data.loc[val2_data.icd10_code == classc, 'id'].unique()
    #classd_val2_ids = val2_data.loc[val2_data.icd10_code == classd, 'id'].unique()
    #classa_aug_train_ids = np.random.choice(classa_train_ids,size=train_size,replace=False)
    #classb_aug_train_ids = np.random.choice(classb_train_ids,size=train_size,replace=False)
    #classc_aug_train_ids = np.random.choice(classc_train_ids,size=train_size,replace=False)
    #classd_aug_train_ids = np.random.choice(classd_train_ids,size=train_size,replace=False)
    #classa_aug_test_ids = np.random.choice(classa_test_ids,size=test_size,replace=False)
    #classb_aug_test_ids = np.random.choice(classb_test_ids,size=test_size,replace=False)
    #classc_aug_test_ids = np.random.choice(classc_test_ids,size=test_size,replace=False)
    #classd_aug_test_ids = np.random.choice(classd_test_ids,size=test_size,replace=False)
    #classa_aug_val_ids = np.random.choice(classa_val_ids,size=val_size,replace=False)
    #classb_aug_val_ids = np.random.choice(classb_val_ids,size=val_size,replace=False)
    #classc_aug_val_ids = np.random.choice(classc_val_ids,size=val_size,replace=False)
    #classd_aug_val_ids = np.random.choice(classd_val_ids,size=val_size,replace=False)
    
    #Aug_train_list = np.concatenate([classa_aug_train_ids, classb_aug_train_ids, classc_aug_train_ids, classd_aug_train_ids])
    #Aug_test_list = np.concatenate([classa_aug_test_ids, classb_aug_test_ids, classc_aug_test_ids, classd_aug_test_ids])
    #Aug_val_list = np.concatenate([classa_aug_val_ids, classb_aug_val_ids, classc_aug_val_ids, classd_aug_val_ids])
    #Aug_train_id_df = pd.DataFrame(Aug_train_list, columns=['id'])
    #Aug_train_id_df['id'] = Aug_train_id_df['id'].astype(int)
    #Aug_test_id_df = pd.DataFrame(Aug_test_list, columns=['id'])
    #Aug_test_id_df['id'] = Aug_test_id_df['id'].astype(int)
    #Aug_val_id_df = pd.DataFrame(Aug_val_list, columns=['id'])
    #Aug_val_id_df['id'] = Aug_val_id_df['id'].astype(int)

    #aug_train_data = pd.merge(Aug_train_id_df, train_data, left_on=['id'], right_on=['id'], how='left')
    #aug_test_data = pd.merge(Aug_test_id_df, test_data, left_on=['id'], right_on=['id'], how='left')
    #aug_val_data = pd.merge(Aug_val_id_df, val_data, left_on=['id'], right_on=['id'], how='left')

    print('Train counts')
    print(len(aug_train_data))
    print('Test counts')
    print(len(aug_test_data))
    print('Val1 counts')
    print(len(aug_val1_data))
    print('Val2 counts')
    print(len(aug_val2_data))

    aug_train_data[['id']].to_csv('aug_train_ids.csv')	
    aug_test_data[['id']].to_csv('aug_test_ids.csv')
    aug_val1_data[['id']].to_csv('aug_val1_ids.csv')	
    aug_val2_data[['id']].to_csv('aug_val2_ids.csv')	
	

    return(aug_train_data, label_dict, aug_test_data, 
           aug_val1_data, aug_val2_data)


# Create a custom dataset class for text classification
class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels, adm_ids, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.adm_ids = adm_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        #print('labels', type(labels),'adm ids', type(adm_ids))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        adm_id = self.adm_ids[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].squeeze()  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze()  # Remove batch dimension

        return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': torch.tensor(label, dtype=torch.long),
        'adm_id': adm_id  # Leave as is if string or convert to tensor if needed
        }

    def get_raw_texts(self):
        return self.texts


# Build our customer BERT classifier
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        #self.bert = BertForSequenceClassification.from_pretrained(bert_model_name,num_labels=num_classes)
        self.bert = BertModel.from_pretrained(bert_model_name,num_labels=num_classes)
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.fold = 0
        self.batch_size = 0
        self.learning_rate = 0
        self.beta1 = 0
        self.beta2 = 0
        self.wdecay = 0
        self.epoch = 0
        self.best_weights = None
        self.test_probs = None
        self.best_train_probs = None
        self.best_optimizer = None
        self.best_accuracy = 0
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def get_classes(self):
        return self.num_classes

    def set_dropout(self, dropout_rate):
        """ Method to dynamically adjust dropout rate in the model """
        self.dropout.p = dropout_rate  # Adjust dropout rate in the classifier head

        # Optionally, adjust the BERT transformer layers' dropout (hidden layers)
        self.bert.config.hidden_dropout_prob = dropout_rate
        return

    def set_best_accuracy(self, accuracy):
        self.best_accuracy = accuracy
        return

    def get_best_accuracy(self):
        return self.best_accuracy

    def set_best_train_probs(self, train_probs):
        self.best_train_probs = train_probs
        return 

    def get_best_train_probs(self):
        return self.best_train_probs

    def set_test_probs(self, train_probs):
        self.test_probs = train_probs
        return 
        
    def get_test_probs(self):
        return self.test_probs

    def set_best_weights(self, weights):
        self.best_weights = weights
        return

    def get_best_weights(self):
        return self.best_weights

    def set_best_optimizer(self, opt):
        self.best_optimizer = opt
        return

    def get_best_optimizer(self):
        return self.best_optimizer

    def set_hyperparameters(self, best_fold, batch_size, learning_rate, beta1, beta2, wdecay, epoch):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.wdecay = wdecay
        self.epoch = epoch
        return

    def get_hyperparameters(self):

        return(self.fold, self.batch_size, self.learning_rate, self.beta1, self.beta2, self.wdecay, self.epoch)

    def get_best_fold(self):

        return(self.fold)

    def get_best_batch_size(self):
        return(self.batch_size)

    def get_best_lr(self):
        return(self.learning_rate)

    def get_best_beta1(self):
        return(self.beta1)

    def get_best_beta2(self):
        return(self.beta2)

    def get_best_wdecay(self):
        return(self.wdecay)

    def get_best_epoch(self):
        return(self.epoch)

    def forward(self, input_ids, attention_mask,labels=None, token_type_ids=None):
        # Pass the inputs through the BERT model

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)

        # Get the last hidden state
        last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

        # Get the representation of the [CLS] token (first token)
        cls_output = last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]

        # Pass through the classifier to get logits
        logits = self.classifier(cls_output)  # Shape: [batch_size, num_classes]
        if labels is not None:
            logits = logits.to(labels.device)
        

        loss = None
        if labels is not None:
            # Calculate the loss only if labels are provided
            loss = self.criterion(logits, labels)

        return {'logits':logits, 'loss':loss}

def calc_prediction(row):

    max_score = max(row.iloc[11], row.iloc[12], row.iloc[13], row.iloc[14])

    if row.iloc[11] == max_score:
        return 0
    elif row.iloc[12] == max_score:
        return 1
    elif row.iloc[13] == max_score:
        return 2
    elif row.iloc[14] == max_score:
        return 3
    else:
        return 99


def vote_score(adm_ids, probs, true_labels, num_classes):

    class_1_score = [item[0] for item in probs]
    class_2_score = [item[1] for item in probs]
    class_3_score = [item[2] for item in probs]
    class_4_score = [item[3] for item in probs]
    dict = {'adm_id': adm_ids, 'actuals':true_labels, 'class_1_score': class_1_score, 'class_2_score': class_2_score,'class_3_score': class_3_score,'class_4_score': class_4_score}
    adm_scores = pd.DataFrame(dict).sort_values(by='adm_id')

    for n in range(num_classes):
        ename = 'element-' + str(n)

    temp_scores = adm_scores.groupby(['adm_id', 'actuals']).agg(Adm_Count=('adm_id', 'size'),
                                                   Class_1_Mean=('class_1_score', 'mean'), Class_1_Max=('class_1_score', 'max'),
                                                   Class_2_Mean=('class_2_score', 'mean'), Class_2_Max=('class_2_score', 'max'),
                                                   Class_3_Mean=('class_3_score', 'mean'), Class_3_Max=('class_3_score', 'max'),
                                                   Class_4_Mean=('class_4_score', 'mean'), Class_4_Max=('class_4_score', 'max')).reset_index()

    temp_scores['Class_1_Prob'] = (temp_scores['Class_1_Max'] + (temp_scores['Class_1_Mean'] * temp_scores['Adm_Count']/2))/(1 + temp_scores['Adm_Count']/2)
    temp_scores['Class_2_Prob'] = (temp_scores['Class_2_Max'] + (temp_scores['Class_2_Mean'] * temp_scores['Adm_Count']/2))/(1 + temp_scores['Adm_Count']/2)
    temp_scores['Class_3_Prob'] = (temp_scores['Class_3_Max'] + (temp_scores['Class_3_Mean'] * temp_scores['Adm_Count']/2))/(1 + temp_scores['Adm_Count']/2)
    temp_scores['Class_4_Prob'] = (temp_scores['Class_4_Max'] + (temp_scores['Class_4_Mean'] * temp_scores['Adm_Count']/2))/(1 + temp_scores['Adm_Count']/2)
    temp_scores['Prediction'] = temp_scores.apply(calc_prediction, axis=1)

    pred_labels = temp_scores['Prediction']
    actuals = temp_scores['actuals']
    probs = temp_scores[['Class_1_Prob', 'Class_2_Prob', 'Class_3_Prob', 'Class_4_Prob']].to_numpy()

    return (actuals, pred_labels, probs)


def get_perf_data(cycle, y_true, y_pred):

    #print('get perf data input')
    #print('perf y_true', y_true[0:5])
    #print('perf y_pred', y_pred[0:5])
    #print('Generating Performance Data')
    cm = confusion_matrix(y_true, y_pred)
    print(cycle, 'Perf Routine Confusion Matrix:')
    #print('y_true', y_true[0:5])
    #print('y_pred',y_pred[0:5])
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # Precision
    precision = precision_score(y_true, y_pred, average=None)
    # Recall
    sensitivity = recall_score(y_true, y_pred, average=None)
    # F1-Score
    f1 = f1_score(y_true, y_pred, average=None)

    labels = [0, 1, 2, 3]
    # Binarize ytest with shape (n_samples, n_classes)
    y_true = label_binarize(y_true, classes=labels)
    # Binarize ypreds with shape (n_samples, n_classes)
    y_pred = label_binarize(y_pred, classes=labels)
        
    roc_auc = roc_auc_score(y_true, y_pred,average=None,multi_class='ovr')
    return(accuracy, precision, sensitivity, f1, roc_auc, cm)

# Define the train() function
def train(model, data_loader, optimizer, scheduler, device):

    print('beginning training', get_current_time())    
    model.train()
    # Tracking variables.
    predictions_labels = []
    prediction_probs = []
    adm_id_list = []
    true_labels = []
  # Total loss for this epoch.
    total_loss = 0
    for batch in tqdm(data_loader, total = len(data_loader)):
        
        model = model.to(device)
        true_labels += batch['label'].numpy().flatten().tolist()
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device,dtype = torch.long)
        attention_mask = batch['attention_mask'].to(device,dtype = torch.long)
        labels = batch['label'].to(device,dtype = torch.long)
        adm_ids = batch['adm_id'].to(device,dtype = torch.long)

        b_labels = labels  # Shape: [8]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        logits = outputs['logits']
        loss = outputs['loss']


        total_loss += loss
        loss.backward()
        # Next line is to prevent the "exploding gradients" problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        probabilities = nn.Softmax(dim=1)(logits)
        #probabilities = softmax(logits, dim=-1)
        np_probs = probabilities.detach().cpu().numpy().tolist()

        # Making predictions
        predictions = torch.argmax(probabilities, dim=1).flatten().tolist()
        predictions_labels += predictions
        adm_id_list += adm_ids.detach().cpu().numpy().tolist()
        #np_logits = logits.cpu().detach().numpy()
        prediction_probs.extend(np_probs)
        
    # Calculate the average loss over the training data.
    print('got to end of all batches', get_current_time())
    actuals, pred_labels, probs = vote_score(adm_id_list, prediction_probs, true_labels, 4)
    accuracy, precision, sensitivity, f1, roc_auc, cm = get_perf_data('Train', actuals, pred_labels)
    print('Epoch Train Performance ')
    print('\n', 'acc', accuracy, 'prec', precision, 'sens', sensitivity, 'f1', f1, 'roc', roc_auc)
    avg_epoch_loss = total_loss / len(data_loader)
    
    return(accuracy, precision, sensitivity, f1, roc_auc, cm, avg_epoch_loss, probs)


# Build our evaluation method
def evaluate(model, data_loader, device):
    model.eval()
    prediction_labels = []
    prediction_probs = []
    actual_labels = []
    adm_id_list = []
    total_loss = 0
    model = model.to(device)
    with torch.no_grad():
        for batch in tqdm(data_loader,total=len(data_loader)):
            actual_labels += batch['label'].numpy().flatten().tolist()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device,dtype = torch.long)
            adm_ids = batch['adm_id'].to(device,dtype = torch.long)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs['logits']
            loss = outputs['loss']

            total_loss += loss

            probabilities = softmax(logits, dim=-1)
            np_probs = probabilities.detach().cpu().numpy().tolist()
            # 2. Making predictions
            predictions = torch.argmax(probabilities, dim=1).flatten().tolist()
            prediction_labels += predictions
            adm_id_list += adm_ids.detach().cpu().numpy().tolist()
            #np_logits = logits.cpu().detach().numpy()
            prediction_probs.extend(np_probs)
            #prediction_logits += predictions

    actuals, pred_labels, probs = vote_score(adm_id_list, prediction_probs, actual_labels, 4)
    accuracy, precision, sensitivity, f1, roc_auc, cm = get_perf_data('Val', actuals, pred_labels)
    model.set_test_probs(probs)
    print('Epoch Evaluate Performance')
    print('\n', 'acc', accuracy, 'prec', precision, 'sens', sensitivity, 'f1', f1, 'roc', roc_auc)
    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(data_loader)

    #class_report = classification_report(actual_labels, prediction_labels)

    return(accuracy, precision, sensitivity, f1, roc_auc, cm, avg_epoch_loss, probs, actuals, pred_labels)

# Build our prediction method

def worker_init_fn(worker_id):
  temp_seed = 100 + worker_id
  np.random.seed(temp_seed)
  random_seed(temp_seed)
  return

def init_tokenizer(bert_model_name,train_texts, train_labels, train_adm_ids, max_length, val_texts, val_labels, val_adm_ids, batch_size):
#Initialize tokenizer, dataset, and data loader\
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
    train_dataset = TextClassificationDataset(train_texts, train_labels, train_adm_ids, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, val_adm_ids, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    return(tokenizer, train_dataloader, val_dataloader)

def setup_dataloader(tokenizer, in_texts, in_labels, in_adm_ids, max_length, batch_size, type):
#Create a dataloader
    
    print('setup dataloader input length', len(in_texts))
    temp_dataset = TextClassificationDataset(in_texts, in_labels, in_adm_ids, tokenizer, max_length)
    if type == 'train':
        temp_dataloader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    else:
        temp_dataloader = DataLoader(temp_dataset, batch_size=batch_size)    
    
    return(temp_dataloader)

def setup_device(bert_model_name, num_classes, learning_rate, beta1, beta2, wdecay, train_dataloader, num_epochs, dropout):
#Step 11: Set up the device and model
# Change this to take in a gamma to run thru a list pof them in cross validation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier(bert_model_name,num_classes).to(device)
    model.set_dropout(dropout)
    step_size = 1
    #Step 12: Set up optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2),  weight_decay=wdecay)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=total_steps)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
    # Code for step scheduler
    #step_size = 1
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    return(device, model, optimizer, scheduler)

def store_perf_data(fold_no, batch_size, learning_rate, beta1, beta2, wdecay, epoch, cycle, accuracy,
                    precision, sensitivity, f1, roc_auc, cm, perf_df):

    #print(cycle, 'accuracy', '\n', accuracy, 'precision', precision, 'sensitivity', sensitivity, 'f1', f1, 'roc_auc', roc_auc)
    ccols = ['Fold_no', 'Batch', 'LR', 'Beta1', 'Beta2', 'Wdecay', 'Epoch', 'Cycle',
              'T1', 'F1', 'F2', 'F3', 'F4', 'T2', 'F5', 'F6', 'F7', 'F8', 'T3', 'F9', 'F10', 'F11', 'F12', 'T4',
              'Accuracy','Prec1', 'Prec2', 'Prec3','Prec4','Sens1', 'Sens2', 'Sens3', 'Sens4',
             'FScore1', 'FScore2', 'FScore3', 'FScore4', 'Roc_Auc1', 'Roc_Auc2', 'Roc_Auc3', 'Roc_Auc4']
    temp_df = pd.DataFrame([[fold_no, batch_size, learning_rate, beta1, beta2, wdecay, epoch, cycle,
                             cm[0,0], cm[0,1], cm[0,2], cm[0,3], cm[1,0], cm[1,1], cm[1,2], cm[1,3],
                             cm[2,0], cm[2,1], cm[2,2], cm[2,3], cm[3,0], cm[3,1], cm[3,2], cm[3,3], accuracy,
                             precision[0], precision[1], precision[2], precision[3],
                             sensitivity[0], sensitivity[1], sensitivity[2], sensitivity[3],
                             f1[0], f1[1], f1[2], f1[3],
                             roc_auc[0], roc_auc[1], roc_auc[2], roc_auc[3]]], columns=ccols)

    perf_df = pd.concat([perf_df, temp_df])
    return(perf_df)

def run_epochs(fold_no, batch_size, learning_rate, beta1, beta2, wdecay, num_epochs, model, train_dataloader, val_dataloader, optimizer, scheduler,
               device, perf_df, label_dict):

    print('params', fold_no, batch_size, learning_rate, beta1, beta2, wdecay, num_epochs)
    print('train_length',len(train_dataloader), 'val_length',len(val_dataloader))
    best_acc = 0
    best_checkpoint = None
    best_model = None
    best_fold = 0
    best_epoch = 0
    best_wdecay = 0
    best_lr = 0
    best_beta1 = 0
    best_beta2 = 0
    best_batch_size = 0
    best_train_probs = None
    weights_list = []
    weights_ix = -1


    saved_val = {'val_actuals':[], 'val_preds':[]}
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}

    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("Training on batches....", get_current_time())
        train_acc, train_precision, train_sensitivity, train_f1, train_roc_auc, train_cm, train_loss, train_probs = train(model, train_dataloader, optimizer, scheduler, device)
        loss_train_avg = train_loss/len(train_dataloader)
        print('Training Loss:', loss_train_avg)

        print()
        print("Validation on batches....", get_current_time())
        val_acc, val_precision, val_sensitivity, val_f1, val_roc_auc, val_cm, val_loss, val_probs, actuals, pred_labels = evaluate(model, val_dataloader, device)

        current_weights = copy.deepcopy(model.state_dict())

        weights_list.append(current_weights)
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = current_weights
            best_optimizer = optimizer.state_dict()
            best_model = model
            best_fold = fold_no
            best_epoch = epoch
            best_wdecay = wdecay
            best_lr = learning_rate
            best_beta1 = beta1
            best_beta2 = beta2
            best_batch_size = batch_size
            best_train_probs = train_probs

        if (epoch == num_epochs):
            conf_matrix = confusion_matrix(saved_val.val_actuals, saved_val.val_predict)
            cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels=['Neg','Pos'])
            cm_display.plot()
            plt.show()

        loss_val_avg = val_loss/len(val_dataloader)
        print('Epoch ', (epoch + 1), ' Average Validation Loss:', loss_val_avg)
        #accuracy_per_class(val_predict, val_labels,label_dict)
        perf_df = store_perf_data(fold_no, batch_size, learning_rate, beta1, beta2, wdecay, epoch, 'Train', train_acc,
                                  train_precision, train_sensitivity, train_f1, train_roc_auc, train_cm, perf_df)
        perf_df = store_perf_data(fold_no, batch_size, learning_rate, beta1, beta2, wdecay,epoch, 'Val', val_acc,
                                  val_precision, val_sensitivity, val_f1, val_roc_auc, val_cm, perf_df)

        # write best performaing model
        print(type(best_weights))
    if best_acc > model.get_best_accuracy():
        model.set_best_accuracy(best_acc)
        model.set_hyperparameters(best_fold, best_batch_size, best_lr, best_beta1, best_beta2, best_wdecay, best_epoch)
        model.set_best_weights(best_weights)
        model.set_best_optimizer(best_optimizer)
        model.set_best_train_probs(best_train_probs)
    return(perf_df, best_model, val_acc)


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    return

def run_test(test_model, device, test_df):
    set_seed(100)
    print('************ Test Results ****************')
    print('Best Model Stats', 'acc', test_model.get_best_accuracy())
    print('*****************************************')
    bert_model_name = 'bert-base-uncased'
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    test_adm_ids = test_df.id.tolist()
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)

    test_dataset = TextClassificationDataset(test_texts, test_labels, test_adm_ids, tokenizer, max_length = 512)
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    test_model = test_model.to(device)

    test_acc, test_precision, test_sensitivity, test_f1, test_roc_auc, test_cm, test_loss, probs, actuals, pred_labels = evaluate(test_model, test_dataloader, device)
    test_model.set_test_probs(probs)
    probs_df = pd.DataFrame(probs)
    print('probs len', len(probs_df), probs_df.columns)
    probs_df.to_csv('BERT_probs.csv')
    test_actuals_df = pd.DataFrame(actuals)
    print('actuals len', len(test_actuals_df))
    test_actuals_df.to_csv('BERT_actuals.csv')

    print('test acc', test_acc, 'test loss', test_loss )
    print('Confusion Matrix:')
    print(test_cm)
    #cm_display = ConfusionMatrixDisplay(confusion_matrix = test_cm, display_labels=['0','1', '2', '3'])
    #plt.show()
    return test_model

# This flow runs one set of epochs
# This has been cleaned up on 12-10-24
#def one_epoch_set(run, diagnoses_list,hyp_list,breakdown=[111,15, 15, 15]):
def one_epoch_set(hyp_list,note_data, label_dict, test_data, val1_data, val2_data):

    print('starting one_epoch_set', get_current_time())
    
    set_seed(100)
    # Set up parameters
    # These are hyperparameters
    bert_model_name = 'bert-base-uncased'
    fold_no = 0
    num_classes = 4
    max_length = hyp_list[5]
    num_epochs = hyp_list[6]
    dropout = 0.3

    batch_size = hyp_list[0]
    #batch_size = 8
    learning_rate = hyp_list[1]
    #learning_rate = 2e-5
    beta1 =  hyp_list[2]
    #beta1 =  0.85
    beta2 =  hyp_list[3]
    #beta2 =  0.98
    wdecay =  hyp_list[4]
    #wdecay =  0

    # Get Data
    
    #note_data, label_dict, test_data, val1_data, val2_data = get_diagnosis_data(diagnoses_list, run, breakdown)

    print('Training admissions Sequences', len(note_data))
    print('Validation 1 admissions Sequences', len(val1_data))
    print('Validation 2 admissions Sequences', len(val2_data))
    print('Test admissions Sequences', len(test_data))

    # Split train into train and bert_val
    #df.loc[:, df.columns != 'b']
    
    #X = note_data[['id', 'text']]
    #y = note_data['label']
    
    #X_train,X_bval, y_train, y_bval = train_test_split(X, y, test_size=0.12, stratify=y, random_state=100) 
    
    # Define data elements from processing
    train_texts = note_data['text'].tolist()
    train_labels = note_data['label'].tolist()
    train_adms = note_data['id'].tolist()
    val_texts = val1_data['text'].tolist()
    val_labels = val1_data['label'].tolist()
    val_adms = val1_data['id'].tolist()
    
    print('train text', len(train_texts), 'train label', len(train_labels),
          'val text', len(val_texts), 'val label', len(val_labels))
          

    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
    
    train_dataloader = setup_dataloader(tokenizer,train_texts, train_labels, train_adms, max_length, batch_size, 'train')
    val_dataloader = setup_dataloader(tokenizer,val_texts, val_labels, val_adms, max_length, batch_size, 'val')
        
    device, model, optimizer, scheduler = setup_device(bert_model_name, num_classes, learning_rate, beta1, beta2, wdecay, train_dataloader,
                                                       num_epochs, dropout)

    # Define Performance dataframe
    ccols = ['Fold_no', 'Batch', 'LR', 'Beta1', 'Beta2', 'Wdecay', 'Epoch', 'Cycle',
              'T1', 'F1', 'F2', 'F3', 'F4', 'T2', 'F5', 'F6', 'F7', 'F8', 'T3', 'F9', 'F10', 'F11', 'F12', 'T4',
              'Accuracy','Prec1', 'Prec2', 'Prec3','Prec4','Sens1', 'Sens2', 'Sens3', 'Sens4',
             'FScore1', 'FScore2', 'FScore3', 'FScore4', 'Roc_Auc1', 'Roc_Auc2', 'Roc_Auc3', 'Roc_Auc4']
    perf_df = pd.DataFrame(columns = ccols)
    perf_df.drop(perf_df.index, inplace=True)


    #Run Epochs
    perf_df, best_model,val_acc = run_epochs(0, batch_size, learning_rate, beta1, beta2, wdecay, num_epochs, model, train_dataloader,
                         val_dataloader, optimizer, scheduler, device, perf_df, label_dict)
    perf_df[perf_df.Cycle == 'Val'].sort_values(by=['Accuracy'], ascending=[False])
    perf_df.to_csv('perf.csv')
    run_test(test_model=best_model, device=device, test_df=test_data)
    print('val2 data in one epoch')
    print(val2_data[0:5])
    return(best_model, device, tokenizer)

def write_hyperparameters(lr, beta1, beta2, wdecay, epoch):

  import csv

  parms = [[lr, beta1, beta2, wdecay, epoch]]

  fields = ['LR', 'Beta1', 'Beta2', 'Wdecay', 'Epoch']

  with open('/content/models/bert_hyperparameters.csv', 'w') as f:

     # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(fields)
    write.writerows(parms)
  return

def save_best_model(best_model, batch_size, fold_no, epoch, lr, beta1, beta2, wdecay):

    path = '/content/models/'
    tm = time.strftime('%m-%d-%Y')
    lr = str(lr).replace('.', '')
    beta1 = str(beta1).replace('.', '')
    beta2 = str(beta2).replace('.', '')
    wdecay = str(wdecay).replace('.', '')
    fname = 'best-model-' + tm + '.pth'
    print(fname)
    torch.save(best_model, path + fname)
    print("Model state_dict has been saved successfully.")
    return


def get_hyperparameter_list(epochs_list, batch_list, lr_list, wdecay_list, dropout_list, beta1_list, beta2_list, count):

    rand_epochs = random.choices(population=epochs_list, weights = [0.5, 0.25, 0.25],k=count)
    rand_batch = random.choices(population=batch_list, weights = [0.5, 0.4, 0.1], k=count)
    rand_lr = random.choices(population=lr_list, weights = [0.4, 0.3, 0.2, 0.1] ,k=count)
    rand_wdecay = random.choices(population=wdecay_list, weights = [0.1, 0.2, 0.4, 0.4] ,k=count)
    rand_dropout = random.choices(population=dropout_list, weights = [0.1, 0.3, 0.4, 0.3] ,k=count)
    rand_beta1 = random.choices(population=beta1_list, weights = [0.35, 0.22, 0.22, 0.21] ,k=count)
    rand_beta2 = random.choices(population=beta2_list, weights = [0.22, 0.35, 0.22, 0.21] ,k=count)
    hyperparameter_list = list(zip(rand_epochs, rand_batch, rand_lr, rand_wdecay, rand_dropout, rand_beta1, rand_beta2))

    return(hyperparameter_list)

def get_current_time():
    from datetime import datetime, timezone, timedelta
    time_change = timedelta(hours=5)
    current_time = datetime.now() - time_change
    return current_time

def hyperparameter_cv():
    # Set up parameters
    # This flow uses CV to test different sets of hyperparameters
    # It random selects 30 combination of hyperparameters to test
    set_seed(100)
    bert_model_name = 'bert-base-uncased'
    num_classes = 4
    max_length = 512
    best_acc = 0
    best_model = None
    best_optimizer = None
    # hyperparameter values to sample from and test
    epochs_list = [4,6,8]
    batch_list = [8, 16, 32]
    lr_list = [5e-5, 4e-5, 2e-5, 1e-5]
    gamma_list = [.4, .3, .2, .1]
    wdecay_list = [0, 0.01, 0.001, 0.1]
    dropout_list = [0, 0.1, 0.2, 0.3]
    beta1_list = [0.8, 0.85, 0.9, 0.99]
    beta2_list = [0.9, 0.98, 0.99, 0.999]
    # generate a list of parameters
    hparms_list = get_hyperparameter_list(epochs_list, batch_list, lr_list, wdecay_list, dropout_list, beta1_list, beta2_list, 30)
    #hparms_list = hparms_list[0:2]

    weights_list = []
    hparms_index = -1
    print('hparms_list', len(hparms_list))
    # Get train, val and test datasets
    #diagnosis_list2 = ['I21.4', 'I25.10', 'I35.2', 'I50.9']
    diagnosis_list1 = ['A41.9', 'I21.4', 'J96.00', 'N17.9']
    note_data, label_dict, test_data, val1_data, val2_data = get_diagnosis_data(diagnosis_list1, 'distinct')
    train_adm_ids = note_data.id.tolist()

    print(note_data[0:5])
    # Cross-Validation
    skf_train = StratifiedShuffleSplit(n_splits=len(hparms_list), test_size=0.1, random_state=100)
    skf_val = StratifiedShuffleSplit(n_splits=len(hparms_list), test_size=0.8, random_state=100)

    ccols = ['Fold_no', 'Batch', 'LR', 'Beta1', 'Beta2', 'Wdecay', 'Epoch', 'Cycle',
             'T1', 'F1', 'F2', 'F3', 'F4', 'T2', 'F5', 'F6', 'F7', 'F8', 'T3', 'F9', 'F10', 'F11', 'F12', 'T4',
             'Accuracy','Prec1', 'Prec2', 'Prec3','Prec4','Sens1', 'Sens2', 'Sens3', 'Sens4',
             'FScore1', 'FScore2', 'FScore3', 'FScore4', 'Roc_Auc1', 'Roc_Auc2', 'Roc_Auc3', 'Roc_Auc4']
    perf_df = pd.DataFrame(columns = ccols)
    perf_df.drop(perf_df.index, inplace=True)
    fold_no = 0

    time_change = timedelta(hours=5)

    train_texts_in = note_data['text'].tolist()
    train_labels_in = note_data['label'].tolist()
    train_adms_in = note_data['id'].tolist()
    val_texts_in = val1_data['text'].tolist()
    val_labels_in = val1_data['label'].tolist()
    val_adms_in = val1_data['id'].tolist()

    for (train_index_train, val_index_train), (train_index_val, val_index_val) in zip(skf_train.split(train_texts_in, train_labels_in),
                                                                                  skf_val.split(val_texts_in, val_labels_in)):
        hparms_index += 1
        print('hparms_index', hparms_index)
        num_epochs = hparms_list[hparms_index][0]
        batch_size = hparms_list[hparms_index][1]
        learning_rate = hparms_list[hparms_index][2]
        beta1 =  hparms_list[hparms_index][5]
        beta2 =  hparms_list[hparms_index][6]
        wdecay =  hparms_list[hparms_index][3]
        dropout = hparms_list[hparms_index][4]

        fold_no += 1
        start_time = datetime.now() - time_change
        print('Starting Fold', fold_no, 'startiing time:', start_time)
        print('hyperparameters', 'epochs', num_epochs, 'batch', batch_size,'Learning Rate', learning_rate, 'beta1', beta1, 'beta2', beta2, 'wdecay', wdecay)

        # original code which split on sequences
        train_texts = [train_texts_in[index] for index in train_index_train]
        train_labels = [train_labels_in[index] for index in train_index_train]
        train_adms = [train_adms_in[index] for index in train_index_train]
        val_texts = [val_texts_in[index] for index in val_index_val]
        val_labels = [val_labels_in[index] for index in val_index_val]
        val_adms = [val_adms_in[index] for index in val_index_val]

        common_adms = list(set(train_adms).intersection(set(val_adms)))
        print('common adms', len(common_adms), 'train adm counts', len(train_adms), 'val adm counts', len(val_adms))
        val_adms_list = list(zip(val_adms, val_index_val))
        dup_adms = [tup for tup in val_adms_list if tup[0] in train_adms]
        print('dups', len(dup_adms))
        print(dup_adms[0:5])

        tokenizer, train_dataloader, val_dataloader = init_tokenizer(bert_model_name,train_texts, train_labels, train_adms, max_length,
                                                             val_texts, val_labels, val_adms, batch_size)

        device, model, optimizer, scheduler = setup_device(bert_model_name, num_classes, learning_rate, beta1, beta2, wdecay, train_dataloader,
                                                   num_epochs,dropout)

        perf_df,best_model, val_acc = run_epochs(fold_no, batch_size, learning_rate, beta1, beta2, wdecay, num_epochs, model, train_dataloader, val_dataloader, optimizer, scheduler,
                         device, perf_df, label_dict)
        end_time = datetime.now() - time_change
        elapsed_time = end_time - start_time
        print('Fold: ', fold_no, ' Completion Time: ', end_time, ' elapsed time: ', elapsed_time)
        if model.get_best_accuracy() > best_acc:
            best_acc = model.get_best_accuracy()
            best_weights = model.best_weights
            best_model = model
            best_optimizer = model.best_optimizer

        print('after setup optimizer')
        param_groups = best_optimizer['param_groups']

        
    print("Cross-validation completed.")
    print('best acc:', model.get_best_accuracy())
    print('best hyperparameters')
    print(best_model.get_best_batch_size(), best_model.get_best_fold(),
          best_model.get_best_epoch(), best_model.get_best_lr(), best_model.get_best_beta1(), best_model.get_best_beta2(), best_model.get_best_wdecay())
    save_best_model(best_model, best_model.get_best_batch_size(), best_model.get_best_fold(),
                    best_model.get_best_epoch(), best_model.get_best_lr(), best_model.get_best_beta1(), best_model.get_best_beta2(), best_model.get_best_wdecay())
    write_hyperparameters(best_model.get_best_lr(), best_model.get_best_beta1(),
                          best_model.get_best_beta2(), best_model.get_best_wdecay(), best_model.get_best_epoch())
    perf_df.to_csv('perf.csv')

    bcols = ['Fold_no', 'Batch', 'LR', 'Beta1', 'Beta2', 'Wdecay', 'Epoch', 'Accuracy']
    perf_df.loc[perf_df.Cycle == 'Val', bcols].sort_values(by=['Accuracy'], ascending=[False])

    run_test(test_model=best_model, device=device, test_df=test_data)
    return

def create_subsequences(df_input, seq_len):
#chunk notes ointo subsequences based on needed BERT 
# window length

    from tqdm import tqdm
    df_len = len(df_input)
    want=pd.DataFrame({'id':[],'text':[],'label':[], 'icd10_code':[]})

    for i in tqdm(range(df_len)):
        x=df_input.text.iloc[i].split()
        n=int(len(x)/seq_len)
        for j in range(n):
            temp_dict = {'text':' '.join(x[j*seq_len:(j+1)*seq_len]),
                         'label':df_input.label.iloc[i],
                         'icd10_code':df_input.icd10_code.iloc[i],
                         'id':df_input.id.iloc[i]}
            want = pd.concat([want, pd.DataFrame(temp_dict, index=[0])], ignore_index=True)
            
        if len(x)%seq_len>10:
            
            temp_dict = {'text':' '.join(x[-(len(x)%seq_len):]),
                         'label':df_input.label.iloc[i], 
                         'icd10_code':df_input.icd10_code.iloc[i],
                         'id':df_input.id.iloc[i]}
            want = pd.concat([want, pd.DataFrame(temp_dict, index=[0])], ignore_index=True)

    return want

def check_dup_adms(test_adms, val_adms, train_adms):

    #val_adms_list = list(zip(val_adms, val_index_val))
    dup_val_adms = [tup for tup in set(val_adms) if tup in set(train_adms)]
    dup_test_adms = [tup for tup in set(test_adms) if tup in set(train_adms)]
    print('total_val', len(val_adms), 'dup val adms', len(dup_val_adms), 'total_test', len(test_adms), 'dup test adms', len(dup_test_adms))
    return

def generalizability_cv():
# This is cross validation for the measurement of generalizability
# This used two splits: one for the train and one for the val
    set_seed(100)
    bert_model_name = 'bert-base-uncased'
    num_classes = 4
    max_length = 512
    best_acc = 0
    best_model = None
    best_optimizer = None
    num_epochs = 7
    batch_size = 8
    learning_rate = 4e-5
    wdecay = 0
    dropout = 0.3
    beta1 = 0.8
    beta2 = 0.98

    accuracy_list = []

    diagnosis_list2 = ['I21.4', 'I25.10', 'I35.2', 'I50.9']
    note_data, label_dict, test_data, val1_data, val1_data = get_diagnosis_data(diagnosis_list2, 'similar')
    print('train data:', len(note_data), 'test data:', len(test_data))


    # Cross-Validation
    skf_train = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=100)
    skf_val = StratifiedShuffleSplit(n_splits=5, test_size=0.8, random_state=100)

    ccols = ['Fold_no', 'Batch', 'LR', 'Beta1', 'Beta2', 'Wdecay', 'Epoch', 'Cycle',
              'T1', 'F1', 'F2', 'F3', 'F4', 'T2', 'F5', 'F6', 'F7', 'F8', 'T3', 'F9', 'F10', 'F11', 'F12', 'T4',
              'Accuracy','Prec1', 'Prec2', 'Prec3','Prec4','Sens1', 'Sens2', 'Sens3', 'Sens4',
             'FScore1', 'FScore2', 'FScore3', 'FScore4', 'Roc_Auc1', 'Roc_Auc2', 'Roc_Auc3', 'Roc_Auc4']
    perf_df = pd.DataFrame(columns = ccols)
    perf_df.drop(perf_df.index, inplace=True)

    fold_no = 0

    train_texts_in = note_data['text'].tolist()
    train_labels_in = note_data['label'].tolist()
    train_adms_in = note_data['id'].tolist()
    val_texts_in = val1_data['text'].tolist()
    val_labels_in = val1_data['label'].tolist()
    val_adms_in = val1_data['id'].tolist()

    for (train_index_train, val_index_train), (train_index_val, val_index_val) in zip(skf_train.split(train_texts_in, train_labels_in),
                                                                                  skf_val.split(val_texts_in, val_labels_in)):

        fold_no += 1
        print('Starting Fold', fold_no, 'batch', batch_size,'Learning Rate', learning_rate, 'beta1', beta1, 'beta2', beta2, 'wdecay', wdecay)
        # original code which split on sequences
        train_texts = [train_texts_in[index] for index in train_index_train]
        train_labels = [train_labels_in[index] for index in train_index_train]
        train_adms = [train_adms_in[index] for index in train_index_train]
        val_texts = [val_texts_in[index] for index in val_index_val]
        val_labels = [val_labels_in[index] for index in val_index_val]
        val_adms = [val_adms_in[index] for index in val_index_val]

        common_adms = list(set(train_adms).intersection(set(val_adms)))
        print('common adms', len(common_adms), 'train adm counts', len(train_adms), 'val adm counts', len(val_adms))
        val_adms_list = list(zip(val_adms, val_index_val))
        dup_adms = [tup for tup in val_adms_list if tup[0] in train_adms]
        print('dups', len(dup_adms))
        print(dup_adms[0:5])

        #train_labels = [labels[index] for index in train_index]
        #val_labels = [labels[index] for index in val_index]
        #train_adms = [train_adm_ids[index] for index in train_index]
        #val_adms = [train_adm_ids[index] for index in val_index]

        tokenizer, train_dataloader, val_dataloader = init_tokenizer(bert_model_name,train_texts, train_labels, train_adms, max_length,
                                                                 val_texts, val_labels, val_adms, batch_size)

        device, model, optimizer, scheduler = setup_device(bert_model_name, num_classes, learning_rate, beta1, beta2, wdecay, train_dataloader,
                                                   num_epochs,dropout)

        perf_df, best_model, val_acc = run_epochs(fold_no, batch_size, learning_rate, beta1, beta2, wdecay, num_epochs, model, train_dataloader, val_dataloader,
                                             optimizer, scheduler, device, perf_df, label_dict)

        print('fold', fold_no, 'last validation accuracy:', val_acc)
        accuracy_list.append(val_acc)

    mean_accuracy = torch.tensor(accuracy_list).mean()
    print(f"Mean Accuracy across all folds: {mean_accuracy:.4f}")

    print("Generalizability Cross-validation completed.")
    perf_df[perf_df.Cycle == 'Val'].sort_values(by=['Accuracy'], ascending=[False])

    perf_df.to_csv('perf.csv')
    ccols = ['Fold_no', 'Batch', 'Epoch', 'Cycle',
             'Accuracy', 'Prec1', 'Prec2', 'Prec3',
             'Prec4', 'Sens1', 'Sens2', 'Sens3', 'Sens4',
             'Roc_Auc1', 'Roc_Auc2', 'Roc_Auc3', 'Roc_Auc4']
    ccols2 = ['Fold_no', 'Batch', 'Epoch', 'Cycle',
              'T1', 'F1', 'F2', 'F3', 'F4', 'T2', 'F5', 'F6', 'F7', 'F8', 'T3', 'F9',
              'F10', 'F11', 'F12', 'T4', 'Accuracy']

    perf_df.loc[perf_df.Cycle == 'Val', ccols].sort_values(by=['Accuracy'], ascending=[False])[0:10]
    return
