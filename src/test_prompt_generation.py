"""
Test script for RAGFewShotDiagnosisClassifier prompt generation.
Tests the format_diagnosis_options() and _create_prompt_with_rag() methods.
"""

import pandas as pd
import logging
from pathlib import Path
from diagnosis_classifiers import (FewShotDiagnosisClassifier, 
                         RAGFewShotDiagnosisClassifier)
import BERT_Diagnosis as BD


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
ICD10_CSV_PATH = '../data/raw/ICD10-Code-Descriptions.csv'
dpath = "E:/Education/CCSU-Thesis-2024/Data/"
TRAINING_DATA_PATH = dpath + 'intsl_train_group_full_no3c.snappy.parquet'

# ============================================================================
# Standalone Prompt Generation Function
# ============================================================================

def format_diagnosis_options(classes, icd10_df):
    """
    Standalone function to format diagnosis options.
    This mimics what your classifier method does.
    
    Args:
        classes: List of ICD-10 codes (e.g., ['A41.9', 'I21.4', 'J96.00', 'N17.9'])
        icd10_df: DataFrame with columns ['icd10_code', 'description']
    
    Returns:
        Formatted string for the prompt
    """
    lines = ["Based on the patient's clinical presentation, predict which diagnosis class applies:\n"]
    
    for idx, code in enumerate(classes):
        match = icd10_df[icd10_df['icd10_code'] == code]
        if len(match) > 0:
            desc = match['description'].values[0]
            lines.append(f"                    {idx} = {code} ({desc})")
        else:
            lines.append(f"                    {idx} = {code} (description not found)")
            logger.warning(f"Description not found for code: {code}")
    
    return "\n".join(lines)

def create_sample_prompt(diagnosis_section, patient_text, patient_id=None, examples=None):
    """
    Create a complete prompt with diagnosis section and optional examples.
    
    Args:
        diagnosis_section: Formatted diagnosis options string
        patient_text: Clinical note text
        patient_id: Optional patient ID
        examples: Optional list of (text, diagnosis_idx) tuples for few-shot examples
    
    Returns:
        Complete prompt string
    """
    prompt = f"""<start_of_turn>user

You are an experienced attending physician reviewing clinical notes to make a diagnostic assessment.

{diagnosis_section}

CRITICAL INSTRUCTIONS:
- Respond with ONLY a single digit: 0, 1, 2, or 3
- Do not include the diagnosis code (e.g., A41.9)
- Do not add any explanation, punctuation, spaces, or newlines
- Output format: Just the digit, nothing else

"""
    
    # Add few-shot examples if provided
    if examples:
        prompt += "Here are some similar example cases:\n\n"
        for idx, (example_text, diagnosis_idx) in enumerate(examples, 1):
            prompt += f"""Example {idx}:
Clinical Note: {example_text[:500]}...
Diagnosis: {diagnosis_idx}

"""
    
    # Add new patient case
    prompt += "Now, review the following new patient case:\n\n"
    if patient_id:
        prompt += f"Patient ID: {patient_id}\n"
    prompt += f"Clinical Note: {patient_text}\n\n"
    prompt += "Diagnosis (respond with only the number 0, 1, 2, or 3):<end_of_turn>"
    prompt += "\n<start_of_turn>model"
    
    return prompt

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_training_data(filepath):
    """Load training data from parquet file"""
    
    if not Path(filepath).exists():
        logger.error(f"Training data file not found: {filepath}")
        raise FileNotFoundError(f"Could not find {filepath}")
    
    logger.info(f"Loading training data from: {filepath}")
    train_df = pd.read_parquet(filepath)
    
    required_cols = ['id', 'text', 'icd10_code']
    missing_cols = [col for col in required_cols if col not in train_df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.info(f"Available columns: {train_df.columns.tolist()}")
        raise ValueError(f"Training data missing columns: {missing_cols}")
    
    # split admissions of notes subsequences of 1000 characters
    seq_size = 1000
    # Log BEFORE creating subsequences
    logger.info(f"Train shape BEFORE subsequencing: {train_df.shape}")

    # Create subsequences
    train_df2 = BD.create_subsequences(train_df, seq_size)
    # Log AFTER creating subsequences
    logger.info(f"Train shape AFTER subsequencing: {train_df2.shape}")  
    logger.info(f"Unique diagnoses: {sorted(train_df2['icd10_code'].unique())}")
    
    return train_df2

def create_sample_icd10_csv(filepath='sample_icd10_codes.csv'):
    """Create a sample ICD-10 reference CSV"""
    
    icd10_data = pd.DataFrame({
        'icd10_code': ['A41.9', 'I21.4', 'J96.00', 'N17.9'],
        'description': [
            'Sepsis, unspecified organism',
            'Acute subendocardial myocardial infarction',
            'Acute respiratory failure, unspecified',
            'Acute kidney failure, unspecified'
        ]
    })
    
    icd10_data.to_csv(filepath, index=False)
    logger.info(f"Created sample ICD-10 CSV at {filepath}")
    return filepath

def load_or_create_icd10_csv(filepath):
    """Load ICD-10 CSV or create sample if it doesn't exist"""
    
    if Path(filepath).exists():
        logger.info(f"Loading ICD-10 codes from: {filepath}")
        icd10_df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(icd10_df)} ICD-10 codes")
        return icd10_df
    else:
        logger.warning(f"ICD-10 file not found at {filepath}")
        logger.info("Creating sample ICD-10 CSV...")
        sample_path = create_sample_icd10_csv()
        return pd.read_csv(sample_path)

# ============================================================================
# Test Function
# ============================================================================

def test_prompt_generation(training_data_path, icd10_csv_path):
    """Test prompt generation without requiring full classifier"""
    
    logger.info("="*80)
    logger.info("TESTING PROMPT GENERATION (Standalone)")
    logger.info("="*80)
    
    # Load data
    icd10_df = load_or_create_icd10_csv(icd10_csv_path)
    
    try:
        train_df = load_training_data(training_data_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load training data: {e}")
        logger.info("\nPlease update TRAINING_DATA_PATH in the script.")
        return
    
    # Get classes from training data
    classes = sorted(train_df['icd10_code'].unique())
    logger.info(f"\nClasses found in training data: {classes}")
    
    # Test 1: Generate diagnosis section
    logger.info("\n" + "="*80)
    logger.info("TEST 1: format_diagnosis_options()")
    logger.info("="*80)
    
    diagnosis_section = format_diagnosis_options(classes, icd10_df)
    # print("\nGenerated Diagnosis Section:")
    # print(diagnosis_section)
    
    # Test 2: Create full prompt without examples (zero-shot)
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Zero-Shot Prompt (No Examples)")
    logger.info("="*80)
    
    test_example = train_df.iloc[-1]
    test_patient_text = test_example['text']
    test_patient_id = test_example['id']
    true_diagnosis = test_example['icd10_code']
    
    logger.info(f"Test Patient ID: {test_patient_id}")
    logger.info(f"True Diagnosis: {true_diagnosis}")
    
    zero_shot_prompt = create_sample_prompt(
        diagnosis_section=diagnosis_section,
        patient_text=test_patient_text,
        patient_id=test_patient_id,
        examples=None  # No examples
    )
    
    # print("\nZero-Shot Prompt:")
    # print("="*80)
    # print(zero_shot_prompt)
    # print("="*80)
    
    # Test 3: Create prompt with few-shot examples
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Few-Shot Prompt (With 3 Examples)")
    logger.info("="*80)
    
    # Get some example cases (first 3 from training data, excluding test case)
    example_cases = []
    for diagnosis_code in classes:  # classes is already sorted unique codes
        # Get one example with this diagnosis
        matching_rows = train_df[train_df['icd10_code'] == diagnosis_code]
        if len(matching_rows) > 0:
            sample_row = matching_rows.sample(n=1, random_state=42).iloc[0]
            diagnosis_idx = classes.index(sample_row['icd10_code'])
            example_cases.append((sample_row['text'], diagnosis_idx))

    logger.info(f"Selected {len(example_cases)} examples, one per class")

    for idx, (text, diag_idx) in enumerate(example_cases):
        logger.info(f"  Example {idx+1}: Class {diag_idx} ({classes[diag_idx]})")
    
    few_shot_prompt = create_sample_prompt(
        diagnosis_section=diagnosis_section,
        patient_text=test_patient_text,
        patient_id=test_patient_id,
        examples=example_cases
    )
    
    # print("\nFew-Shot Prompt:")
    # print("="*80)
    # print(few_shot_prompt)
    # print("="*80)
    
    # Test 4: Verify prompt structure
    logger.info("\n" + "="*80)
    logger.info("TEST 4: VERIFY PROMPT STRUCTURE")
    logger.info("="*80)
    
    checks = {
        'Has diagnosis options': 'Based on the patient\'s clinical presentation' in few_shot_prompt,
        'Has 4 classes (0-3)': all(f'{i} =' in few_shot_prompt for i in range(4)),
        'Has expected ICD codes': all(code in few_shot_prompt for code in classes),
        'Has descriptions': '(' in few_shot_prompt and ')' in few_shot_prompt,
        'Has examples': 'Example' in few_shot_prompt,
        'Has 4 examples': few_shot_prompt.count('Example') == 4,
        'Has test case': 'new patient case' in few_shot_prompt,
        'Has test patient ID': str(test_patient_id) in few_shot_prompt,
        'Has instruction format': '<start_of_turn>user' in few_shot_prompt,
        'Ends correctly': '<start_of_turn>model' in few_shot_prompt,
        'Has critical instructions': 'CRITICAL INSTRUCTIONS' in few_shot_prompt
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL CHECKS PASSED!")
    else:
        logger.warning("\n⚠️  SOME CHECKS FAILED")
    
    # Test 5: Test with different numbers of examples
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Testing Different Numbers of Examples")
    logger.info("="*80)
    
    for k in [0, 1, 3, 5]:
        example_subset = example_cases[:k] if k > 0 else None
        prompt_k = create_sample_prompt(
            diagnosis_section=diagnosis_section,
            patient_text=test_patient_text,
            patient_id=test_patient_id,
            examples=example_subset
        )
        example_count = prompt_k.count('Example')
        logger.info(f"k={k}: Generated {example_count} examples in prompt")
        
        if example_count == k:
            logger.info(f"  ✓ Correct")
        else:
            logger.warning(f"  ✗ Expected {k}, got {example_count}")
    
    # Save outputs
    # ========================================================================
    # SAVE PROMPTS TO FILE
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("SAVING PROMPTS TO FILE")
    logger.info("="*80)
    
    output_file = '../data/processed/prompt_test_output.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PROMPT GENERATION TEST RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Training data: {training_data_path}\n")
        f.write(f"ICD-10 CSV: {icd10_csv_path}\n")
        f.write("="*80 + "\n\n")
        
        # Test case info
        f.write("TEST CASE INFORMATION:\n")
        f.write("="*80 + "\n")
        f.write(f"Patient ID: {test_patient_id}\n")
        f.write(f"True Diagnosis: {true_diagnosis}\n")
        f.write(f"Patient Text (first 500 chars):\n{test_patient_text[:500]}...\n")
        f.write("\n\n")
        
        # Diagnosis section
        f.write("DIAGNOSIS OPTIONS SECTION:\n")
        f.write("="*80 + "\n")
        f.write(diagnosis_section)
        f.write("\n\n")
        
        # Zero-shot prompt
        f.write("ZERO-SHOT PROMPT (No Examples):\n")
        f.write("="*80 + "\n")
        f.write(zero_shot_prompt)
        f.write("\n\n")
        
        # Few-shot prompt with 3 examples
        f.write("FEW-SHOT PROMPT (3 Examples):\n")
        f.write("="*80 + "\n")
        f.write(few_shot_prompt)
        f.write("\n\n")
        
        # Additional prompts with different k values
        f.write("PROMPTS WITH DIFFERENT NUMBERS OF EXAMPLES:\n")
        f.write("="*80 + "\n\n")
        
        for k in [1, 5]:
            example_subset = example_cases[:k] if k <= len(example_cases) else example_cases
            prompt_k = create_sample_prompt(
                diagnosis_section=diagnosis_section,
                patient_text=test_patient_text,
                patient_id=test_patient_id,
                examples=example_subset
            )
            f.write(f"\n--- PROMPT WITH {k} EXAMPLE(S) ---\n")
            f.write("="*80 + "\n")
            f.write(prompt_k)
            f.write("\n\n")
        
        # Training data statistics
        f.write("\nTRAINING DATA STATISTICS:\n")
        f.write("="*80 + "\n")
        f.write(f"Total examples: {len(train_df)}\n")
        f.write(f"Unique patients: {train_df['id'].nunique()}\n")
        f.write("\nDiagnosis distribution:\n")
        for code in classes:
            count = (train_df['icd10_code'] == code).sum()
            pct = count / len(train_df) * 100
            f.write(f"  {code}: {count} ({pct:.1f}%)\n")
        
        # Example cases used
        f.write("\n\nEXAMPLE CASES USED IN PROMPTS:\n")
        f.write("="*80 + "\n")
        for idx, (example_text, diagnosis_idx) in enumerate(example_cases, 1):
            f.write(f"\nExample {idx}:\n")
            f.write(f"  Diagnosis Index: {diagnosis_idx}\n")
            f.write(f"  Diagnosis Code: {classes[diagnosis_idx]}\n")
            f.write(f"  Text (first 300 chars): {example_text[:300]}...\n")
    
    logger.info(f"✓ Prompts saved to: {output_file}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    
    try:
        test_prompt_generation(TRAINING_DATA_PATH, ICD10_CSV_PATH)
    except Exception as e:
        logger.exception(f"Test failed with error: {e}")
        raise
