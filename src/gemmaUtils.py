# The file defines utility functions to be used by gemma analysis work.

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
import requests
import json
from typing import List, Dict
import time
from tqdm import tqdm
import sys
import os
from datetime import datetime
from pathlib import Path
import shutil
import gc
import psutil
import BERT_Diagnosis as BD
from pathlib import Path
#sys.path.append('E:/VSCode-Projects/Thesis')
sys.path.append('~/projects/Thesis')
from llama_cpp import Llama
import llama_cpp
import logging

def setup(logger):
    """Complete setup environment tasks
    
    Args:
         logger: Logger instance
    
    Logs:
        - Python version
        - llama-cpp-python version
        - Key library versions
        - System information
    """

    import platform
    import sklearn
    
    logger.info("="*60)
    logger.info("ENVIRONMENT SETUP")
    logger.info("="*60)
    
    # Python version
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"Platform: {platform.platform()}")
    
    # Key library versions
    logger.info(f"llama-cpp-python: {llama_cpp.__version__}")
    logger.info(f"pandas: {pd.__version__}")
    logger.info(f"numpy: {np.__version__}")
    logger.info(f"scikit-learn: {sklearn.__version__}")
    
    # Optional: Check for RAG dependencies
    try:
        import sentence_transformers
        logger.info(f"sentence-transformers: {sentence_transformers.__version__}")
    except ImportError:
        logger.info("sentence-transformers: NOT INSTALLED (RAG mode unavailable)")
    
    try:
        import faiss
        logger.info(f"faiss: INSTALLED")
    except ImportError:
        logger.info("faiss: NOT INSTALLED (RAG mode unavailable)")
    
    logger.info("="*60)
    
    return()

def diagnose_gguf_file(model_path, logger):
    """Diagnose GGUF file issues"""
    import struct
    
    logger.info(f"Diagnosing file: {model_path}")
    
    # 1. Check file exists and permissions
    if not os.path.exists(model_path):
        logger.error(f"File does not exist!")
        return False
    
    # 2. Check file size
    size_gb = os.path.getsize(model_path) / (1024**3)
    logger.info(f"File size: {size_gb:.2f} GB")
    
    # 3. Check GGUF magic number
    try:
        with open(model_path, 'rb') as f:
            magic = f.read(4)
            logger.info(f"File magic bytes: {magic}")
            logger.info(f"File magic hex: {magic.hex()}")
            
            if magic != b'GGUF':
                logger.error(f"Invalid GGUF file! Expected b'GGUF', got {magic}")
                return False
            
            # Read version
            version = struct.unpack('<I', f.read(4))[0]
            logger.info(f"GGUF version: {version}")
            
            if version > 3:
                logger.warning(f"GGUF version {version} may not be supported by your llama-cpp-python")
                return False
                
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return False
    
    logger.info("File appears to be a valid GGUF file")
    return True

def discover_models(mpath, logger):
    """ Find all downloaeed gguf models """ 
    model_path = Path(mpath)
    print("Models found:")
    for gguf_file in model_path.rglob("*.gguf"):
        size_gb = gguf_file.stat().st_size / (1024**3)
        print(f"\n{gguf_file.name}")
        print(f"  Path: {gguf_file}")
        print(f"  Size: {size_gb:.2f} GB")
    return()
    
def setup_logging(log_dir="logs", log_level=logging.INFO):
    """
    Set up logging to both file and console.
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'Generative_MIMIC_Diagnosis_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger('MyApplication')
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simpler logging)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Only INFO and above to console
    console_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def copy_code_to_git(spath, gpath):
    """ Version Control
        a) copy this code to github directory
        b) Git commands need to be executed from command line:
        - git add "Gemma Diagnosis Prediction Example.ipynb"
        - git add "Gemma Diagnosis Prediction Example.py"
        - git commit -m "commit comment
    """
        
    shutil.copy2(spath, gpath)
    print(f"\n[OK] Total files copied: 1")
    return()


def check_avail_memory(logger):
    """ Check Available Memory
        - Model Requires at Least 3G</li> 
    """
    
    # Check available memory
    mem = psutil.virtual_memory()
    print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
    print(f"Free RAM: {mem.free / (1024**3):.2f} GB")

    # Force garbage collection
    gc.collect()

    # If less than 3 GB available, you'll have issues
    if mem.available < 3 * 1024**3:
        print("Error: Less than 3GB RAM available. Close other programs.")
        sys.exit(1)

    return


def _validate_and_clean(df, stage='unknown', text_col='text',logger=None):
    """
    Validate text column integrity. Called at key pipeline stages.
    Logs issues and removes bad records rather than silently passing them through.
    """
    
    OFFICE_MARKERS = [
     'microsoft office word',
    'worddocument',
    'documentsummaryinformation',
    'summaryinformation',
    'normalcj',
    'paragraph fontri',
    'root entry',
    'compobj',
    'dfollowedhyperlink',
    'rtable normal',
    'no listu',
    'header_title',
    'footer_title',
    ]
    
    issues_found = 0
    drop_indices = []
    
    for idx, row in df.iterrows():
        text = row[text_col]
        reason = None
        
        # 1. Not a string at all
        if not isinstance(text, str):
            reason = f"non_string: type={type(text)}"
        
        # 2. Empty or whitespace only
        elif not text.strip():
            reason = "empty_text"
        
        # 3. Too short to be a real clinical note
        elif len(text.split()) < 20:
            reason = f"too_short: {len(text.split())} words"
        
        # 4. Word document identifiers (common corruption issue)
        elif any(marker in text.lower() for marker in OFFICE_MARKERS):
            reason = "office_marker_found"
        
        # 5. note > 1000 words
        word_count = len(text.split())
        if word_count > 1000 * 1.1:  # allow 10% tolerance
            logger.warning(f"Test subsequence exceeds limit: {word_count} words for id={row.get('id', '?')}")
        
        # 6. Binary noise - low alpha ratio
        else:
            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            tokens = text.split()
            avg_word_len = sum(len(t) for t in tokens) / len(tokens)
            single_char_ratio = sum(1 for t in tokens if len(t) == 1) / len(tokens)
            
            if alpha_ratio < 0.4:
                reason = f"low_alpha_ratio={alpha_ratio:.2f}"
            elif avg_word_len < 2.5:
                reason = f"low_avg_word_len={avg_word_len:.2f}"
            elif single_char_ratio > 0.3:
                reason = f"high_single_char_ratio={single_char_ratio:.2f}"
        
        if reason:
            logger.warning(
                f"[{stage}] Dropping record idx={idx}, "
                f"id={row.get('id', '?')}, "
                f"icd10={row.get('icd10_code', '?')}: {reason} | "
                f"preview='{str(text)[:80]}'"
            )
            drop_indices.append(idx)
            issues_found += 1
    
    if issues_found > 0:
        df = df.drop(index=drop_indices).reset_index(drop=True)
        logger.warning(f"[{stage}] Removed {issues_found} records. Remaining: {len(df)}")
    else:
        logger.info(f"[{stage}] Validation passed. {len(df)} records OK.")
    
    return df


def load_data(trfile, tsfile, train_sample_size, 
              test_sample_size, train_seq_size, 
              test_seq_size,logger, seed_value=42):

    """ Load MIMIC data</font></b></p><br>
        a) Read Training Data 
        b) Random Select one Training Admission for each Diagnosis</li>
        c) Create 1000 Word Subsequences for each training admission copying diagnosis to each subsequence</li>   
        d) Read Test Data
        e) Random Select two Test Admissions for each Diagnosis</li>
        f) Create 1000 Word Subsequences for each test admission copying diagnosis to each subsequence</li>
        g) Print length of all admission subsequences</li>   
    """
    # Load MIMIC-III data
    logger.info("Loading MIMIC-III data from parquet file...")
    
    train_df_in = pd.read_parquet(trfile)  
    
    # Check for corrupt notes in training data
    train_df = _validate_and_clean(df=train_df_in, stage='train_load', logger=logger)

 
     # Display data info

    logger.info(f"\nDataset shape: {train_df_in.shape}")
    logger.info(f"Columns: {train_df_in.columns.tolist()}")
    logger.info("\nFirst few rows:")
    logger.info(train_df_in.head(2))
    logger.info(f"unique ids: {train_df_in['id'].nunique()}")
    logger.info(f"\n{train_df_in.groupby('icd10_code')['id'].nunique()}")
    


    # get one of each diagnosis for few shot training
    #train_sample_df = train_df_in.groupby('icd10_code', group_keys=False).apply(lambda x: x.sample(n=train_sample_size))
    train_sample_df = (train_df_in.groupby('icd10_code') 
                                  .apply(lambda x: x.sample(n=train_sample_size,random_state=seed_value))
                                  .reset_index(level=0)  
                                  .reset_index(drop=True) 
                      )
    logger.info(f"before rename Columns: {train_sample_df.columns.tolist()}")
    logger.info(f"before rename unique ids: {train_sample_df['id'].nunique()}")
    #train_sample_df = train_sample_df.rename(columns={'index': 'icd10_code'})
    logger.info(f"train sample df Columns: {train_sample_df.columns.tolist()}")
    logger.info(f"after rename unique ids: {train_sample_df['icd10_code'].nunique()}")
    # split admissions of notes subseqeunces of 1000 characters
    train_sample_df2 = BD.create_subsequences(train_sample_df, train_seq_size)
    train_sample_df2 = _validate_and_clean(df=train_sample_df2, stage='post_split', logger=logger)
    logger.info(f"Train shapes: {train_sample_df.shape}, {train_sample_df2.shape}")
    
    # get test cases
    test_df_in = pd.read_parquet(tsfile) 
    # Display data info of test data
    
    logger.info(f"\nDataset shape: {test_df_in.shape}") 
    logger.info(f"Columns: {test_df_in.columns.tolist()}")
    logger.info("\nFirst few rows:")
    logger.info(test_df_in.head(2))
    logger.info(f"unique ids: {test_df_in['id'].nunique()}")
    logger.info(test_df_in.groupby('icd10_code')['id'].nunique())

    # get same size of each diagnosis for prediction
    test_sample_df = (test_df_in.groupby('icd10_code') 
                        .apply(lambda x: x.sample(n=test_sample_size,random_state=seed_value))
                        .reset_index(level=0)  
                        .reset_index(drop=True) 
                      )
    #test_sample_df = test_sample_df.rename(columns={'index': 'icd10_code'})
    logger.info(f"test sample df Columns: {test_sample_df.columns.tolist()}")

    #test_sample_df = test_df_in.groupby('icd10_code', group_keys=False).apply(lambda x: x.sample(n=test_sample_size))
    
    logger.info("got test samples")
    
    # split admissions of notes subseqeunces of 1000 characters
    test_sample_df2 = BD.create_subsequences(test_sample_df, test_seq_size)
    test_sample_df2['word_count'] = test_sample_df2['text'].apply(lambda x: len(str(x).split()))
    logger.info("Before BD")
    logger.info(f"{test_sample_df.shape}, {test_sample_df2.shape}")
    logger.info(f"\n{test_df_in.groupby('icd10_code')['id'].nunique()}")
    logger.info(f"Test shapes: {test_sample_df.shape}, {test_sample_df2.shape}")
    logger.info("After BD")
    logger.info(f"BD Shape: {test_sample_df2.shape}")
    logger.info(test_sample_df2['word_count'].describe())
    logger.info(f"BD Over 1000 words: {(test_sample_df2['word_count'] > 1000).sum()}")
    logger.info(test_sample_df2.groupby('icd10_code')['word_count'].describe())
    test_sample_df2 = _validate_and_clean(df=test_sample_df2, stage='post_split', logger=logger)
    
    word_counts = test_sample_df2.iloc[10:20]['text'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    logger.info(f"Word counts for rows 10-20: {word_counts.tolist()}")
    
    return(train_sample_df2, test_sample_df2)
