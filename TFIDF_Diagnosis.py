# ## TDIDF Diagnosis

# This module contains the functions required to generate a model 
# for predicting the diagnosis code of a patient using 3 days
# of MIMIC ICU patients' notes.  
# 
# This model uses TDIDF matrixes and a Random Forest algorithm.
# Steps:
# a) read data and merge notes so each row represents a 
#    patient admission
# b) apply the Porter stemmer to the note text
# c) convert notes to a vector matrix
# d) train the model on the training dataset
# e) test the model on the test dataset
# f) report performance data


import nltk
import random
import numpy as np
from nltk.stem import PorterStemmer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import re
import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score,roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm

def get_diagnosis_data(note_data, sample_num, diagnosis_list):

    set_seed(100)

    print('full notes size:', len(note_data))
    print(note_data.groupby('icd10_code')['id'].unique())
    
    classa = diagnosis_list[0]
    classb = diagnosis_list[1]
    classc = diagnosis_list[2]
    classd = diagnosis_list[3]
    classa_ids = note_data.loc[note_data.icd10_code == classa, 'id'].unique()
    classb_ids = note_data.loc[note_data.icd10_code == classb, 'id'].unique()
    classc_ids = note_data.loc[note_data.icd10_code == classc, 'id'].unique()
    classd_ids = note_data.loc[note_data.icd10_code == classd, 'id'].unique()
    classa_aug_ids = np.random.choice(classa_ids,size=sample_num,replace=False)
    classb_aug_ids = np.random.choice(classb_ids,size=sample_num,replace=False)
    classc_aug_ids = np.random.choice(classc_ids,size=sample_num,replace=False)
    classd_aug_ids = np.random.choice(classd_ids,size=sample_num,replace=False)
    

    Aug_list = np.concatenate([classa_aug_ids, classb_aug_ids, classc_aug_ids,classd_aug_ids])
    Aug_id_df = pd.DataFrame(Aug_list, columns=['id'])
    Aug_id_df['id'] = Aug_id_df['id'].astype(int)
    aug_data = pd.merge(Aug_id_df, note_data, left_on=['id'], right_on=['id'], how='left')
    
    print('Total Id counts')
    print(len(classa_ids), len(classb_ids), len(classc_ids), len(classd_ids))
    print('Note counts')
    print(len(classa_aug_ids), len(classb_aug_ids), len(classc_aug_ids), len(classd_aug_ids))
    print('note id length', len(aug_data.id))
    return(aug_data)

def combine_notes(notes):

    temp_note = ' '
    new_ct = 0
    for note in notes:
        new_ct += 1
 #      print('new_ct: ', new_ct, 'note: ', note[0:20])
        temp_note += note
 #   print('out_length: ', len(temp_note))
    return temp_note

def get_data_from_datasets(run='distinct'):
    #path = 'E:/Education/CCSU-Thesis-2024/Data/'
    path = '/content/'
    
    if run == 'distinct':
        data_file = "intsl_train_group_full_no2c.snappy.parquet"
        data_file2 = "intsl_val1_group_full_no2c.snappy.parquet"
        data_file3 = "intsl_test_group_full_no2c.snappy.parquet"
    elif run == 'similar':
        data_file = "intsl_train_group_full_no3c.snappy.parquet"
        data_file2 = "intsl_val1_group_full_no3c.snappy.parquet"
        data_file3 = "intsl_test_group_full_no3c.snappy.parquet"

    file1 = path + data_file
    file2 = path + data_file2
    file3 = path + data_file3

    df1 = pd.read_parquet(file1,engine='auto',dtype_backend='numpy_nullable')
    print('train file records', len(df1))
    df2 = pd.read_parquet(file2,engine='auto',dtype_backend='numpy_nullable')
    print('val file records',len(df2))
    test_data = pd.read_parquet(file3,engine='auto',dtype_backend='numpy_nullable')
    print('test file records', len(test_data))

    train_data = pd.concat([df1, df2], ignore_index=True)

    train_data = train_data[['id', 'icd10_code', 'text','label']]
    #train_data = train_data_small.groupby(['id','icd10_code'],as_index=False).agg({'text': combine_notes}).reset_index()
    #print('Train Note Length After Combining Notes', len(train_data))
    train_data['note_length'] = train_data['text'].apply(len)
    notes_length_stats = train_data.groupby(['icd10_code']).note_length.describe().round(1)
    print('Train Note Length Stats')
    print(notes_length_stats)
    #train_data.groupby(['icd10_code']).count()
    
    test_data_small = test_data[['id', 'icd10_code', 'text','label']]
    Test_full_note = test_data_small.groupby(['id','icd10_code','label'],as_index=False).agg({'text': combine_notes}).reset_index()
    Test_full_note['note_length'] = Test_full_note['text'].apply(len)
    print('Test Note Count After Combining Notes')
    tnotes_length_stats = Test_full_note.groupby(['icd10_code']).note_length.describe().round(1)
    print('Test Note Stats After Combining Notes')
    print(tnotes_length_stats)
    #Test_full_note.groupby(['icd10_code']).count()
    
    regex1 = r'[ ]{3,25}'
    train_data['text'] = train_data['text'].replace(to_replace=regex1, value = ' ', regex = True)
    train_data['note_length'] = train_data['text'].apply(len)
    notes_length_stats = train_data.groupby(['icd10_code']).note_length.describe().round(1)
    print('Train Note Stats After Shrinking Spaces')
    print(notes_length_stats)
    Test_full_note.groupby(['icd10_code']).count()
    Test_full_note['text'] = Test_full_note['text'].replace(to_replace=regex1, value = ' ', regex = True)
    Test_full_note['note_length'] = Test_full_note['text'].apply(len)
    test_notes_length_stats = Test_full_note.groupby(['icd10_code']).note_length.describe().round(1)
    print('Test Note Stats After Shrinking Spaces')
    print(test_notes_length_stats)
    #Test_full_note.groupby(['icd10_code']).count()
    return(train_data, Test_full_note)

def read_setup(data_df):
    data_small_df = data_df[['id', 'icd10_code', 'text','label']]
    full_note_df = data_small_df.groupby(['id','icd10_code','label'],as_index=False).agg({'text': combine_notes}).reset_index()
    regex1 = r'[ ]{3,25}'
    full_note_df['text'] = full_note_df['text'].replace(to_replace=regex1, value = ' ', regex = True)
    full_note_df['note_length'] = full_note_df['text'].apply(len)
    notes_length_stats = full_note_df.groupby(['icd10_code']).note_length.describe().round(1)
    print(notes_length_stats)
    print(full_note_df.groupby(['icd10_code']).count())
    return(full_note_df)

# apply stemmer
def apply_stemmer(note_df):

    stemmer = PorterStemmer()
    words = stopwords.words("english")
    note_df['cleaned'] = note_df['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    
    return(note_df)

# Train
def train(X, y):
    
    # We'll use a Random Forest for classification.
    # Split the dataset into training and testing sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.145, random_state=100,stratify=y)
    vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    # Initialize and train the classifier
    selector = SelectKBest(chi2, k=1200)  # Select top 1200 features (but will use available features)
    X_selected = selector.fit_transform(X_train, y_train)

    # Step 3: Train a Random Forest Classifier
    clf = RandomForestClassifier(random_state=100)
    clf.fit(X_selected, y_train)

    # Define the hyperparameter grid to search over
    param_grid = {
        'n_estimators': [300, 500, 1000],        # Number of trees
        'max_depth': [10, 20, None],             # Maximum depth of trees
        'min_samples_split': [2, 5, 10],         # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4],           # Minimum samples required in a leaf node
        'max_features': ['sqrt', 'log2']         # Number of features to consider for splitting
    }

    #    'bootstrap': [True, False]               # Whether bootstrap sampling is used

    # Set up GridSearchCV to tune hyperparameters using 5-fold cross-validation
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=8, n_jobs=-1, verbose=3, scoring='accuracy')

    # Fit the grid search to the training data
    grid_search.fit(X_selected, y_train)

    # Output the best parameters and the best cross-validation accuracy
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)

    # Evaluate the best model on the test set
    # Step 5: Apply feature selection to the test data using the same selector
    X_val_selected = selector.transform(X_val)  # Using transform on test data

    best_rf_model = grid_search.best_estimator_
    val_accuracy = best_rf_model.score(X_val_selected, y_val)
    print(f"Val Accuracy: {val_accuracy:.4f}")
    return(best_rf_model, vectorizer, selector, X_selected)
    
#  Evaluate 
def evaluate(model, X_test, y_test, vectorizer, selector):

    X_test = vectorizer.transform(X_test)
    X_test_selected = selector.transform(X_test)  # Using transform on test data
    #best_rf_model = grid_search.best_estimator_
    y_pred = model.predict(X_test_selected)
    print(y_test[0:5])

    cm = confusion_matrix(y_test, y_pred)
    print('tf y_pred', y_pred[0:5])
    print('TFIDF Confusion Matrix:')
    print(cm)
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Precision
    precision = precision_score(y_test, y_pred, average=None)
    # Recall
    sensitivity = recall_score(y_test, y_pred, average=None)
    # F1-Score
    f1 = f1_score(y_test, y_pred, average=None)
    probabilities = model.predict_proba(X_test_selected)
    print('probs dim', probabilities.ndim, 'probs size', probabilities.size,'probs shape', probabilities.shape,
    'probs len', len(probabilities))
    roc_auc = roc_auc_score(y_true=y_test, y_score=probabilities, multi_class='ovr', average=None)
    print('TFIDF Performance Measurements')
    print('accuracy', accuracy,'precision', precision,
          '\n','sensitivity', sensitivity, 'f1', f1, 
          '\n', 'ROC', roc_auc )
    return(probabilities, y_pred)

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    return

