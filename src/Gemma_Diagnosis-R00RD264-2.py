# MIMIC-III Diagnosis Code Prediction using Local Gemma 
# This script uses llama.cpp library to predict diagnosis codes from patient notes

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

def save_results(self, results_df: pd.DataFrame, filepath: str, logger):
        """Save prediction results to CSV."""
        results_df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")

# ============================================================================
# MAIN WORKFLOW FUNCTIONS
# ============================================================================

def build_needed_classes(mpath, model_name, train_sample_df2, test_sample_df2, 
                        icd10_csv_path=None, use_rag=False, logger=None, balanced_rag=False):
    """Build Needed Classes
    
    Args:
        mpath: Path to models directory
        model_name: Name of GGUF model file
        train_sample_df2: Training data DataFrame
        test_sample_df2: Test data DataFrame
        icd10_csv_path: Optional path to ICD-10 descriptions CSV
        use_rag: Whether to use RAG-enhanced classifier
        logger: Logger instance
        balanced_rag:  get equal examples from each diagnoses
    
    Returns:
        classifier: Fitted classifier instance (standard or RAG)
    """
    if logger is None:
            print("Logger in build needed is None")
    else:
            print("Logger in build needed has a Value")
    
    MODEL_PATH = mpath + model_name
    llama_model = LlamaCppWrapper(
        model_path=MODEL_PATH, 
        n_ctx=4096, 
        n_threads=8, 
        n_batch=512, 
        use_mmap=True,
        logger=logger
    )
    
    if use_rag:
        logger.info("\nCreating RAG-enhanced classifier...")
        
        # Create RAG retriever
        retriever = MedicalRAGRetriever(
            embedding_model_name='medicalai/ClinicalBERT',
            logger=logger
        )
        
        # Create RAG classifier
        classifier = RAGFewShotDiagnosisClassifier(
            gemma_model=llama_model,
            rag_retriever=retriever,
            icd10_csv_path=icd10_csv_path,
            logger=logger,
            k_examples = 1,
            balanced_retrieval= balanced_rag
        )
        
        logger.info("\nFitting RAG classifier (building index)...")
        classifier.fit(train_sample_df2)
        logger.info("\nFinished Fitting RAG Classifier")
        
    else:
        logger.info("\nCreating standard few-shot classifier...")
        
        classifier = FewShotDiagnosisClassifier(
            gemma_model=llama_model,
            icd10_csv_path=icd10_csv_path,
            logger=logger
        )
        
        logger.info("\nFitting classifier with training examples...")
        classifier.fit(train_sample_df2, logger)
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
                            logger=logger, 
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
    metrics = classifier.evaluate2(test_subset, verbose=False, logger=logger)
    
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

def main():

    start_time = time.time()
    project_root = Path.cwd()
    spath = Path("E:/VSCode-Projects/Thesis/Gemma_Diagnosis.py")
    gpath = Path("E:/Education/GitHub/Generative-MIMIC-Diagnosis/")
    mpath = 'E:/Education/llama/models'
    dpath = "E:/Education/CCSU-Thesis-2024/Data/"
    trfile = dpath + 'intsl_train_group_full_no3c.snappy.parquet'
    tsfile = dpath + 'intsl_test_group_full_no3c.snappy.parquet'
    lpath = "E:/Education/CCSU-Thesis-2024/Data/logs"
    model_name = "/gemma-3n-E4B-it-Q4_K_M.gguf"
    #model_name = "/gemma-2-2b-it-q4_k_m.gguf"
    #model_name = "/gemma-2-9b-it-q4_k_m.gguf"

    # Optional: Path to ICD-10 descriptions CSV
    icd10_csv = "E:/VSCode-Projects/Thesis/data/raw/ICD10-Code-Descriptions.csv"
    
    # Model configuration
    #model_name = "/gemma-3n-E4B-it-Q4_K_M.gguf"
    
    # RAG Configuration
    use_rag = True  # Set to True to use RAG-enhanced classifier
    balanced_rag = False
    #Data Sampling
    train_sample = 100
    test_sample = 15
    seq_size = 1000
    
    # ========================================================================
    # EXECUTION
    # ========================================================================
    # Initiate Logger
    logger = setup_logging(log_dir=lpath, log_level=logging.INFO)
    
    try:
        logger.info("="*80)
        logger.info("Gemma Diagnosis Starting")
        logger.info("="*80)
    
        logger.info(f"rag-or-no-rag: {use_rag}")
        logger.info(f"model name: {model_name}")
        logger.info(f"train sample: {train_sample}")
        logger.info(f"test sample: {test_sample}")
        logger.info(f"seq size: {seq_size}")
        #Environment Setup
        setup(logger)
        discover_models(mpath, logger)
        check_avail_memory(logger)
    
        #Load Data
        train_df, test_df = load_data(
            trfile, tsfile, train_sample, 
            test_sample, seq_size, logger)

        logger.info("Completed getting data")
        #Build Classifier

        classifier = build_needed_classes(
            mpath, model_name, train_df, test_df,  
            icd10_csv, use_rag, logger, balanced_rag)
        logger.info("Completed creating classifier")
        
        #Test Single Patient
        #run_single_patient_test(test_df, classifier, logger)
        #print("Completed run single patient")
        
        run_full_patient_test(test_df,classifier, logger)
        logger.info("Completed run full patient")
        
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

if __name__ == '__main__':
    main()

