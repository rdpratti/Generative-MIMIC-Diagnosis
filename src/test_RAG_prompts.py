"""
Test script for RAG-enhanced prompt generation.
Tests the complete workflow: RAG retrieval -> prompt generation -> save to file.
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import BERT_Diagnosis as BD
from gemmaUtils import setup_logging, load_data
from llama_wrapper import LlamaCppWrapper
from rag_retriever import MedicalRAGRetriever
from diagnosis_classifiers import (
    FewShotDiagnosisClassifier,
    RAGFewShotDiagnosisClassifier
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
ICD10_CSV_PATH = '../data/raw/ICD10-Code-Descriptions.csv'
dpath = "E:/Education/CCSU-Thesis-2024/Data/"
TRAINING_DATA_PATH = dpath + 'intsl_train_group_full_no3c.snappy.parquet'
OUTPUT_DIR = '../data/processed/rag_prompts'

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# Mock Objects
# ============================================================================

class MockGemmaModel:
    """Mock Gemma model - not needed for prompt testing"""
    pass

# ============================================================================
# Helper Functions
# ============================================================================

def load_and_prepare_data(filepath, seq_size=1000):
    """Load training data and create subsequences"""
    logger.info(f"Loading training data from: {filepath}")
    train_df = pd.read_parquet(filepath)
    
    logger.info(f"Shape before subsequencing: {train_df.shape}")
    train_df = BD.create_subsequences(train_df, seq_size)
    logger.info(f"Shape after subsequencing: {train_df.shape}")
    
    classes = sorted(train_df['icd10_code'].unique())
    logger.info(f"Classes: {classes}")
    
    return train_df, classes

def select_test_cases(train_df, classes, n_per_class=2):
    """Select test cases - n examples from each diagnosis class"""
    test_cases = []
    
    logger.info(f"\nSelecting {n_per_class} test cases per class...")
    for diagnosis_code in classes:
        matching = train_df[train_df['icd10_code'] == diagnosis_code]
        if len(matching) >= n_per_class:
            samples = matching.sample(n=n_per_class, random_state=42)
        else:
            samples = matching
            logger.warning(f"Only {len(matching)} examples for {diagnosis_code}")
        
        for _, row in samples.iterrows():
            test_cases.append({
                'id': row['id'],
                'diagnosis': diagnosis_code,
                'text': row['text']
            })
            
    logger.info(f"Selected {len(test_cases)} test cases total")
    return test_cases

def save_prompt_to_file(prompt, test_case, k, output_dir, prompt_number):
    """Save a single prompt to file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"prompt_{prompt_number:03d}_k{k}_{test_case['diagnosis']}.txt"
    filepath = Path(output_dir) / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"PROMPT #{prompt_number} - RAG-ENHANCED FEW-SHOT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Test Case ID: {test_case['id']}\n")
        f.write(f"True Diagnosis: {test_case['diagnosis']}\n")
        f.write(f"K (examples retrieved): {k}\n")
        f.write("="*80 + "\n\n")
        f.write(prompt)
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("TEST CASE TEXT:\n")
        f.write("="*80 + "\n")
        f.write(test_case['text'])
        f.write("\n")
    
    return filepath

# ============================================================================
# Main Test Function
# ============================================================================

def test_rag_prompt_generation():
    """Test RAG-enhanced prompt generation and save results"""
    
    logger.info("="*80)
    logger.info("TESTING RAG-ENHANCED PROMPT GENERATION")
    logger.info("="*80)
    
    # Step 1: Load and prepare data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Loading and Preparing Data")
    logger.info("="*80)
    
    try:
        train_df, classes = load_and_prepare_data(TRAINING_DATA_PATH)
    except Exception as e:
        logger.exception(f"Failed to load data: {e}")
        return
    
    # Step 2: Import and initialize RAG components
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Initializing RAG Classifier")
    logger.info("="*80)
    
    try:
        from diagnosis_classifiers import RAGFewShotDiagnosisClassifier
        from rag_retriever import MedicalRAGRetriever  
        
        logger.info("✓ Imports successful")
        
    except ImportError as e:
        logger.exception(f"Import failed: {e}")
        logger.error("Update the MedicalRAGRetriever import path")
        return
    
    try:
        # Initialize retriever
        rag_retriever = MedicalRAGRetriever(
            embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2',
            logger=logger
        )
        
        # Initialize classifier
        mock_model = MockGemmaModel()
        classifier = RAGFewShotDiagnosisClassifier(
            gemma_model=mock_model,
            rag_retriever=rag_retriever,
            icd10_csv_path=ICD10_CSV_PATH,
            logger=logger
        )
        
        logger.info("✓ RAG classifier initialized")
        
    except Exception as e:
        logger.exception(f"Failed to initialize classifier: {e}")
        return
    
    # Step 3: Build RAG index
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Building RAG Index")
    logger.info("="*80)
    
    try:
        classifier.fit(train_df)
        logger.info(f"✓ RAG index built with {len(train_df)} examples")
        logger.info(f"✓ Classes: {classifier.classes_}")
        
    except Exception as e:
        logger.exception(f"Failed to build index: {e}")
        return
    
    # Step 4: Select test cases
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Selecting Test Cases")
    logger.info("="*80)
    
    test_cases = select_test_cases(train_df, classes, n_per_class=2)
    
    # Step 5: Generate prompts with different k values
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Generating Prompts with RAG Retrieval")
    logger.info("="*80)
    
    k_values = [1, 3, 5]  # Test with different numbers of retrieved examples
    results = []
    prompt_number = 1
    
    for k in k_values:
        logger.info(f"\n--- Testing with k={k} retrieved examples ---")
        
        for test_case in test_cases:
            logger.info(f"\nGenerating prompt #{prompt_number}")
            logger.info(f"  Test case ID: {test_case['id']}")
            logger.info(f"  True diagnosis: {test_case['diagnosis']}")
            logger.info(f"  Retrieving {k} similar examples...")
            
            try:
                # Generate prompt with RAG retrieval
                prompt = classifier._create_prompt_with_rag(
                    patient_text=test_case['text'],
                    patient_id=test_case['id'],
                    k=k
                )
                
                # Verify prompt structure
                has_examples = prompt.count('Example')
                has_diagnosis_options = 'Based on the patient\'s clinical presentation' in prompt
                has_instructions = 'CRITICAL INSTRUCTIONS' in prompt
                
                logger.info(f"  ✓ Prompt generated")
                logger.info(f"    - Contains {has_examples} examples")
                logger.info(f"    - Has diagnosis options: {has_diagnosis_options}")
                logger.info(f"    - Has instructions: {has_instructions}")
                
                # Save prompt to file
                filepath = save_prompt_to_file(
                    prompt=prompt,
                    test_case=test_case,
                    k=k,
                    output_dir=OUTPUT_DIR,
                    prompt_number=prompt_number
                )
                
                logger.info(f"  ✓ Saved to: {filepath.name}")
                
                # Store results
                results.append({
                    'prompt_number': prompt_number,
                    'test_case_id': test_case['id'],
                    'true_diagnosis': test_case['diagnosis'],
                    'k': k,
                    'num_examples_in_prompt': has_examples,
                    'has_diagnosis_options': has_diagnosis_options,
                    'has_instructions': has_instructions,
                    'filepath': str(filepath),
                    'prompt_length': len(prompt)
                })
                
                prompt_number += 1
                
            except Exception as e:
                logger.exception(f"Failed to generate prompt for case {test_case['id']}: {e}")
                continue
    
    # Step 6: Create summary report
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Creating Summary Report")
    logger.info("="*80)
    
    results_df = pd.DataFrame(results)
    
    # Save results summary
    summary_path = Path(OUTPUT_DIR) / 'prompt_generation_summary.csv'
    results_df.to_csv(summary_path, index=False)
    logger.info(f"✓ Summary saved to: {summary_path}")
    
    # Create detailed report
    report_path = Path(OUTPUT_DIR) / 'prompt_generation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAG-ENHANCED PROMPT GENERATION TEST REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training data: {TRAINING_DATA_PATH}\n")
        f.write(f"ICD-10 codes: {ICD10_CSV_PATH}\n")
        f.write(f"Output directory: {OUTPUT_DIR}\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-"*80 + "\n")
        f.write(f"Training examples: {len(train_df)}\n")
        f.write(f"Classes: {classes}\n")
        f.write(f"Test cases per class: 2\n")
        f.write(f"K values tested: {k_values}\n")
        f.write(f"Total prompts generated: {len(results_df)}\n")
        f.write("\n")
        
        f.write("RESULTS BY K VALUE:\n")
        f.write("-"*80 + "\n")
        for k in k_values:
            k_results = results_df[results_df['k'] == k]
            f.write(f"\nK = {k}:\n")
            f.write(f"  Prompts generated: {len(k_results)}\n")
            f.write(f"  Avg examples in prompt: {k_results['num_examples_in_prompt'].mean():.1f}\n")
            f.write(f"  Avg prompt length: {k_results['prompt_length'].mean():.0f} chars\n")
            f.write(f"  All have diagnosis options: {k_results['has_diagnosis_options'].all()}\n")
            f.write(f"  All have instructions: {k_results['has_instructions'].all()}\n")
        
        f.write("\n")
        f.write("RESULTS BY DIAGNOSIS:\n")
        f.write("-"*80 + "\n")
        for diagnosis in classes:
            diag_results = results_df[results_df['true_diagnosis'] == diagnosis]
            f.write(f"\n{diagnosis}:\n")
            f.write(f"  Prompts generated: {len(diag_results)}\n")
            f.write(f"  K values: {sorted(diag_results['k'].unique())}\n")
        
        f.write("\n\n")
        f.write("DETAILED RESULTS:\n")
        f.write("-"*80 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n")
        
        f.write("\n")
        f.write("FILES GENERATED:\n")
        f.write("-"*80 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"{Path(row['filepath']).name}\n")
    
    logger.info(f"✓ Report saved to: {report_path}")
    
    # Step 7: Validation checks
    logger.info("\n" + "="*80)
    logger.info("STEP 7: Validation Checks")
    logger.info("="*80)
    
    checks = {
        'All prompts generated': len(results_df) == len(test_cases) * len(k_values),
        'All have diagnosis options': results_df['has_diagnosis_options'].all(),
        'All have instructions': results_df['has_instructions'].all(),
        'Correct K=1 examples': (results_df[results_df['k']==1]['num_examples_in_prompt'] == 1).all(),
        'Correct K=3 examples': (results_df[results_df['k']==3]['num_examples_in_prompt'] == 3).all(),
        'Correct K=5 examples': (results_df[results_df['k']==5]['num_examples_in_prompt'] == 5).all(),
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {check_name}")
        if not passed:
            all_passed = False
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Total prompts generated: {len(results_df)}")
    logger.info(f"Test cases: {len(test_cases)}")
    logger.info(f"K values: {k_values}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Files generated: {len(results_df) + 2}")  # +2 for summary and report
    
    if all_passed:
        logger.info("\n🎉 ALL VALIDATION CHECKS PASSED!")
    else:
        logger.warning("\n⚠️  SOME VALIDATION CHECKS FAILED")
    
    logger.info("\nGenerated files:")
    logger.info(f"  - {len(results_df)} prompt files")
    logger.info(f"  - 1 summary CSV: {summary_path.name}")
    logger.info(f"  - 1 report: {report_path.name}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    try:
        test_rag_prompt_generation()
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        raise


