"""
Gemma Diagnosis - MIMIC-III Diagnosis Code Prediction Driver
=============================================================
Main driver script for predicting ICD-10 diagnosis codes from MIMIC-III 
clinical notes using local Gemma GGUF models via llama.cpp. Supports both
standard few-shot and RAG-enhanced classification modes.

Functions
---------
save_results(classifier, metrics, logger)
    Saves prediction results to CSV and logs sample predictions and
    misclassifications.

build_needed_classes(mpath, model_name, train_sample_df2, test_sample_df2,
                     icd10_csv_path, example_size, example_ct, test_size,
                     use_rag, logger, balanced_retrieval, embedding_model,
                     temperature, seed, use_few_shot)
    Instantiates the LlamaCppWrapper and builds either a standard
    FewShotDiagnosisClassifier or a RAGFewShotDiagnosisClassifier depending
    on the use_rag flag. Fits the classifier on training data and returns
    the fitted classifier instance.

run_single_patient_test(test_sample_df2, classifier, logger)
    Runs a prediction on the first patient in the test set and logs the
    predicted vs true ICD-10 code, confidence score, and full class
    probability distribution.

run_full_patient_test(test_sample_df2, classifier, logger)
    Runs classifier.evaluate2() on the full test DataFrame and returns
    the metrics dictionary including results_df.

execute_classification(temp, train_ct, test_ct, train_seq_size, test_seq_size,
                       example_size, example_ct, test_size, use_rag,
                       balanced_rag, use_few_shot, logger, seed)
    Top-level orchestration function. Configures all paths and model settings,
    then sequentially calls setup, data loading, classifier building, and
    full patient evaluation. Logs timing for each major stage.

parse_args()
    Defines and parses a basic argument parser with temperature, seed,
    and output file arguments.

main()
    Entry point. Initializes logging, parses all command-line arguments,
    and calls execute_classification() with the provided parameters.

Command-Line Arguments
----------------------
--temperature       Sampling temperature for Gemma model (default: 0.1)
--train_ct          Number of training admissions per diagnosis
--test_ct           Number of test admissions per diagnosis
--train_seq_size    Word count for training subsequences
--test_seq_size     Word count for test subsequences
--example_size      Max word size of few-shot examples
--example_ct        Number of few-shot examples per diagnosis
--test_size         Max word size of test subsequences (default: 500)
--use_rag           Flag to enable RAG-enhanced classifier
--balanced_rag      Flag to retrieve equal examples per diagnosis in RAG
--cv                Flag to enable cross-validation mode
--seed              Random seed for reproducibility (default: 42)
--few_shot_type     Few-shot selection strategy type (default: 'C')
"""

from gemmaUtils import (
    setup, discover_models, check_avail_memory, 
    setup_logging, load_data)
from llama_wrapper import LlamaCppWrapper
from rag_retriever import MedicalRAGRetriever
from diagnosis_classifiers import (
    FewShotDiagnosisClassifier, 
    RAGFewShotDiagnosisClassifier)

import pandas as pd
import numpy as np
import time
import sys
import os
from pathlib import Path
import logging
import argparse

def save_results(self, results_df: pd.DataFrame, filepath: str, logger):
        """Save prediction results to CSV."""
        results_df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")

# ============================================================================
# MAIN WORKFLOW FUNCTIONS
# ============================================================================

def build_needed_classes(mpath, model_name, train_sample_df2, 
                         test_sample_df2,icd10_csv_path=None, 
                         example_size=500, example_ct=4, test_size=500,
                         use_rag=False, logger=None, balanced_retrieval=False,
                         embedding_model = None, temperature=0.0, seed=42, use_few_shot='C'):
    """Build Needed Classes
    
    Args:
        mpath: Path to models directory
        model_name: Name of GGUF model file
        train_sample_df2: Training data DataFrame
        test_sample_df2: Test data DataFrame
        icd10_csv_path: Optional path to ICD-10 descriptions CSV
        use_rag: Whether to use RAG-enhanced classifier
        logger: Logger instance
        balanced_retrieval: get equal number of examples frome each diagnosis 
    
    Returns:
        classifier: Fitted classifier instance (standard or RAG)
    """
    logger.info(f"Build Needed Classes")    
    logger.info(f"Temperature: {temperature} randowm seed: {seed}")    
    MODEL_PATH = mpath / model_name

    llama_model = LlamaCppWrapper(
        model_path=str(MODEL_PATH), 
        n_ctx=8192, 
        n_threads=8, 
        n_batch=512, 
        use_mmap=True,
        temp=temperature,
        seed=seed,
        logger=logger
    )
    
    # train_sample_df2['word_count'] = train_sample_df2['text'].str.split().str.len()
    # Only use subsequences with at least 1000 words for training examples
    # train_examples = train_sample_df2[train_sample_df2['word_count'] >= 500].copy()
    
    if use_rag:
        logger.info("\nCreating RAG-enhanced classifier...")
        logger.info(f"Balanced Retrieval: {balanced_retrieval}")    
        # Create RAG retriever
        retriever = MedicalRAGRetriever(
            embedding_model_name=embedding_model,
            logger=logger
        )
        
        # Create RAG classifier
        classifier = RAGFewShotDiagnosisClassifier(
            gemma_model=llama_model,
            rag_retriever=retriever,
            icd10_csv_path=icd10_csv_path,
            example_size=example_size, 
            test_size=test_size,
            logger=logger,
            k_examples = example_ct,
            balanced_retrieval = balanced_retrieval,
            embedding_model=embedding_model
        )
        
        logger.info("\nFitting RAG classifier (building index)...")
        logger.info(f"Fit Before Call: {train_sample_df2[0:3][['id', 'icd10_code', 'text']].to_string(index=False)}")  
        classifier.fit(train_sample_df2)
        logger.info("\nFinished Fitting RAG Classifier")
        
    else:
        logger.info("\nCreating standard few-shot classifier...")

        classifier = FewShotDiagnosisClassifier(
            gemma_model=llama_model,
            example_size=example_size, 
            test_size=test_size,
            k_examples=example_ct,
            icd10_csv_path=icd10_csv_path,
            logger=logger,
            embedding_model=embedding_model,
            use_few_shot=use_few_shot
        )
        
        logger.info("\nFitting classifier with training examples...")
        
        classifier.fit(train_sample_df2)
        logger.info("\nFinished Fitting Classifier")
    
    word_counts = test_sample_df2.iloc[10:20]['text'].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    logger.debug(f"Word counts - Min: {word_counts.min()}, Max: {word_counts.max()}, Mean: {word_counts.mean():.1f}")
    
    return classifier

def run_single_patient_test(test_sample_df2, classifier, logger = None):
    """ ============================================================================
        TEST ON A SINGLE PATIENT FIRST
        ============================================================================
    """
    logger.info("\n" + "="*80)
    logger.info("TESTING ON SINGLE PATIENT")
    logger.info("="*80)
    test_patient = test_sample_df2.iloc[0]
    logger.debug(f"\nText Length: {len(test_patient['text'].split())} words ({len(test_patient['text'])} chars)")
    logger.debug(f"Patient ID: {test_patient['id']}")
    logger.debug(f"True Diagnosis: {test_patient['icd10_code']}")
    logger.debug(f"\nClinical Note Preview:\n{test_patient['text'][:300]}...")

    # Get prediction with probabilities
    
    single_result = classifier.predict_single(
                            patient_text=test_patient['text'],
                            patient_id=test_patient['id']
                            )

    # Display prediction results
    logger.info("\n" + "="*80)
    logger.info("PREDICTION RESULTS")
    logger.info("="*80)

    correct = single_result['prediction'] == test_patient['icd10_code']
    status_icon = "[OK]" if correct else "[X]"  #Changed 12/3/25 5:27 PM

    logger.info(f"\n{'Predicted:':<15} {single_result['prediction']}")
    logger.info(f"{'True Label:':<15} {test_patient['icd10_code']}")
    logger.info(f"{'Status:':<15} {status_icon} {'CORRECT' if correct else 'INCORRECT'}")

    if single_result.get('confidence') is not None:
        logger.info(f"{'Confidence:':<15} {single_result['confidence']:.2%}")

    # Display probabilities table
    if single_result.get('probabilities') is not None:
        logger.info("\n" + "-"*80)
        logger.info("CLASS PROBABILITY DISTRIBUTION")
        logger.info("-"*80)
        logger.info(f"{'ICD-10 Code':<15} {'Probability':<12} {'Visual':<30} {'Notes'}")
        logger.info("-"*80)
        prob_pairs = list(zip(classifier.classes_, single_result['probabilities']))
        sorted_probs = sorted(prob_pairs, key=lambda x: x[1], reverse=True)
    
        for code, prob in sorted_probs:
            # Visual bar
            bar_length = int(prob * 40)
            bar = "=" * bar_length
        
            # Notes
            notes = []
            if code == test_patient['icd10_code']:
                notes.append("TRUE")
            if code == single_result['prediction']:
                notes.append("PRED")
            notes_str = ", ".join(notes) if notes else ""
        
            logger.info(f"{code:<15} {prob:>6.2%}       {bar:<30} {notes_str}")
    
            logger.info("-"*80)
    else:
        logger.warning(f"\n[WARNING] Probabilities not available")

    logger.info("\n" + "="*80)

    return()

def run_full_patient_test(test_sample_df2, classifier, logger = None):
    """ 
    Complete A Set of Admission Predictions on Test Data</font></b></p><br>
    - Get A Subset of Test Subsequences/li> 
    - Run classifier.evaluate2</li>
    - Return metrics</li>   
    """

    # For quick testing, use a small subset
    logger.info("\n" + "="*80)
    logger.info("EVALUATING ON TEST SET")
    logger.info("="*80)

    # Option 1: Test on small subset first (faster)
    test_subset = test_sample_df2  # Just 20 subsequences for quick test
    logger.info(f"\nTesting on {len(test_subset)} patients...")
    metrics = classifier.evaluate2(test_subset, verbose=False)
    
    return metrics

def save_results(classifier, metrics, logger = None):
    """ ============================================================================
        STEP 10: SAVE RESULTS
        ============================================================================
    """
    logger.info("\nSaving results...")
    classifier.save_results(metrics['results_df'], 'diagnosis_predictions.csv')

    # Show some example predictions
    logger.info("\nSample predictions:")
    logger.info(metrics['results_df'][['id', 'true_icd10_code', 'predicted_icd10_code', 'correct']].head(10))

    # Show misclassifications
    misclassified = metrics['results_df'][~metrics['results_df']['correct']]
    if len(misclassified) > 0:
        logger.debug(f"\nMisclassified cases ({len(misclassified)} total):")
        logger.debug(misclassified[['id', 'true_icd10_code', 'predicted_icd10_code']].head())
    else:
        logger.info("\nNo misclassifications!")
    return()

def execute_classification(temp, train_ct, test_ct, train_seq_size, test_seq_size, 
                           example_size=500, example_ct=4,
                           test_size=500,use_rag=False, balanced_rag=False, use_few_shot='C',
                           logger=None, seed=42):
    
    timings={}
    start_total = time.time()
    project_root = Path.cwd()
    #spath = Path("E:/VSCode-Projects/Thesis/Gemma_Diagnosis.py")
    spath = Path("~/projects/Thesis/Gemma_Diagnosis.py").expanduser()
    #gpath = Path("E:/Education/GitHub/Generative-MIMIC-Diagnosis/")
    #mpath = 'E:/Education/llama/models'
    mpath = Path("~/models").expanduser()
    #dpath = "E:/Education/CCSU-Thesis-2024/Data/"
    dpath = Path("~/thesis/data/").expanduser()
    # 2c = Distinct, 3c = Similar
    trfile = dpath / "intsl_train_group_full_no3c.snappy.parquet"
    tsfile = dpath / 'intsl_test_group_full_no3c.snappy.parquet'
        
    # Optional: Path to ICD-10 descriptions CSV
    #icd10_csv = "E:/VSCode-Projects/Thesis/data/raw/ICD10-Code-Descriptions-3.csv"
    icd10_csv = Path("~/projects/Thesis/data/raw/ICD10-Code-Descriptions-A.csv").expanduser()
        
    # Model configuration
    model_name = "gemma-3n-E4B-it-Q4_K_M.gguf"
    #model_name = "/gemma-2-27b-it-Q4_K_M.gguf"
    #model_name = "/gemma-2-2b-it-q4_k_m.gguf"
    #model_name = "/gemma-2-9b-it-q4_k_m.gguf"
    #embedding_model = 'medicalai/ClinicalBERT'
    embedding_model = 'emilyalsentzer/Bio_ClinicalBERT'

    
    # RAG Configuration
    use_rag = use_rag  # Set to True to use RAG-enhanced classifier
    balanced_retrieval = balanced_rag

    #Data Sampling
    train_sample = train_ct
    test_sample = test_ct
    
    
    #Cross-Validation Configuration
    temperature = temp

    # ========================================================================
    # EXECUTION
    # ========================================================================
            
    try:


        logger.info("="*80)
        logger.info("Gemma Diagnosis Starting")
        logger.info("="*80)
    
        logger.info(f"rag-or-no-rag: {use_rag}")
        logger.info(f"balanced rag: {balanced_rag}")
        logger.info(f"model name: {model_name}")
        logger.info(f"train sample: {train_sample}")
        logger.info(f"test sample: {test_sample}")
        logger.info(f"train seq size: {train_seq_size}")
        logger.info(f"test seq size: {test_seq_size}")
        logger.info(f"temperature: {temp}")
        logger.info(f"example size: {example_size}")
        logger.info(f"example ct: {example_ct}")
        logger.info(f"Use_few_shot: {use_few_shot}")
        #Environment Setup
        setup(logger)
        discover_models(mpath, logger)
        check_avail_memory(logger)
    
        start = time.time()
        #Load Data
        train_df, test_df = load_data(
            trfile, tsfile, train_sample, 
            test_sample, train_seq_size, test_seq_size,
            logger, seed)
        run_time = time.time() - start

        logger.info(f"Completed getting data: Time: {run_time:.2f}s")
        
        #Build Classifier
        
        start_time = time.time()
        classifier = build_needed_classes(
            mpath, model_name, train_df, test_df,  
            icd10_csv, example_size, example_ct,
            test_size,use_rag, logger,balanced_retrieval,
            embedding_model, temperature=temp, seed=999, use_few_shot=use_few_shot)
        
        timings['build_needed_classes'] = time.time() - start
        logger.info(f"Completed creating classifier Time: {timings['build_needed_classes']:.2f}s")
        
        #Test Single Patient
        #run_single_patient_test(test_df, classifier, logger)
        #print("Completed run single patient")
        
        start = time.time()
        run_full_patient_test(test_df,classifier, logger)
        timings['run_full_patient_test'] = time.time() - start
        logger.info(f"Completed run full patient Time: {timings['run_full_patient_test']:.2f}s")
        
        # Time Reporting
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        #Optional Copy code to git
        #copy_code_to_git(spath, gpath)
    except Exception as e:
        logger.exception(f"Unexpected error in main: {e}")
        
        # Force flush all handlers
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        raise
    finally:
        logger.info("Cleanup complete")
        
        # Flush and close all handlers
        for handler in logger.handlers:
            handler.flush()
            handler.close()
    return

def parse_args():
    parser = argparse.ArgumentParser(description='Test Gemma few-shot classifier')
    
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Sampling temperature (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='results.txt',
                       help='Output file for results')
    
    return parser.parse_args()

def main():

    # Initiate Logger
    #lpath = "E:/Education/CCSU-Thesis-2024/Data/logs"
    lpath = Path("~/thesis/logs").expanduser() 
    logger = setup_logging(log_dir=lpath, log_level=logging.DEBUG)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--train_ct', type=int, required=True)
    parser.add_argument('--test_ct', type=int, required=True)
    parser.add_argument('--train_seq_size', type=int, required=True)
    parser.add_argument('--test_seq_size', type=int, required=True)
    parser.add_argument('--example_size', type=int, required=True)
    parser.add_argument('--test_size', type=int, required=False, default=500)
    parser.add_argument('--example_ct', type=int, required=True)
    parser.add_argument('--use_rag', action='store_true')
    parser.add_argument('--balanced_rag', action='store_true')
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--few_shot_type', type=str, default='C')
    args = parser.parse_args()
    
    cross_validation = args.cv
    print(args.temperature, args.train_ct, args.test_ct, 
          args.train_seq_size, args.test_seq_size, args.example_size, args.example_ct,
          args.test_size, args.use_rag, args.balanced_rag,args.seed, args.few_shot_type)
    
    execute_classification(temp=args.temperature,
                           train_ct=args.train_ct, 
                           test_ct=args.test_ct, 
                           train_seq_size=args.train_seq_size, 
                           test_seq_size=args.test_seq_size, 
                           example_size=args.example_size,
                           example_ct=args.example_ct,
                           test_size=args.test_size,
                           use_rag=args.use_rag, 
                           balanced_rag=args.balanced_rag,  
                           use_few_shot=args.few_shot_type,
                           logger=logger,
                           seed=args.seed )
    
    return

if __name__ == '__main__':
    main()

