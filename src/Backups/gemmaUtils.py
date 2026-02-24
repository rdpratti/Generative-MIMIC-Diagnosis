# The file defines utility functions to be used by gemma analysis work.

import pandas as pd
import numpy as np
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
sys.path.append('E:/VSCode-Projects/Thesis')
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

def load_data(trfile, tsfile, train_sample_size, test_sample_size, train_seq_size, test_seq_size, logger):

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
    
    train_df_in = pd.read_parquet(trfile)  # Replace with your file path

    # Display data info

    logger.info(f"\nDataset shape: {train_df_in.shape}")
    logger.info(f"Columns: {train_df_in.columns.tolist()}")
    logger.info("\nFirst few rows:")
    logger.info(train_df_in.head(2))
    logger.info(f"unique ids: {train_df_in['id'].nunique()}")
    logger.info(f"\n{train_df_in.groupby('icd10_code')['id'].nunique()}")
    
    # get one of each diagnosis for few shot training
    train_sample_df = train_df_in.groupby('icd10_code', group_keys=False).apply(lambda x: x.sample(n=train_sample_size))
    # split admissions of notes subseqeunces of 1000 characters
    train_sample_df2 = BD.create_subsequences(train_sample_df, train_seq_size)
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

    # get five of each diagnosis for prediction
    test_sample_df = test_df_in.groupby('icd10_code', group_keys=False).apply(lambda x: x.sample(n=test_sample_size))
    
    logger.info("got test samples")
    # split admissions of notes subseqeunces of 1000 characters
    test_sample_df2 = BD.create_subsequences(test_sample_df, test_seq_size)
    logger.info("got seqs")
    logger.info(f"{test_sample_df.shape}, {test_sample_df2.shape}")
    logger.info(f"\n{test_df_in.groupby('icd10_code')['id'].nunique()}")
    logger.info(f"Test shapes: {test_sample_df.shape}, {test_sample_df2.shape}")

    word_counts = test_sample_df2.iloc[10:20]['text'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    logger.info(f"Word counts for rows 10-20: {word_counts.tolist()}")
    
    return(train_sample_df2, test_sample_df2)
