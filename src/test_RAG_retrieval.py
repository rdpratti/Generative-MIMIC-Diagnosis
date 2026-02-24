"""
Test script for RAG retrieval functionality.
Verifies that similar examples are retrieved correctly.
"""
import sys

import pandas as pd
import logging
from pathlib import Path
from llama_wrapper import LlamaCppWrapper

print()
print("="*80)
print("ENVIRONMENT CHECK")
print("="*80)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Test sentence_transformers BEFORE importing your modules
try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence_transformers imported successfully")
except ImportError as e:
    print(f"✗ sentence_transformers import failed: {e}")
    sys.exit(1)


# NOW import your custom modules
print("Importing BERT_Diagnosis...")
try:
    import BERT_Diagnosis as BD
    print("✓ BERT_Diagnosis imported successfully")
except ImportError as e:
    print(f"✗ BERT_Diagnosis import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Add parent directory to path if needed
sys.path.append('E/VSCode-Projects/Thesis')

# Import from modular structure
print("Importing modular components...")
try:
    from rag_retriever import MedicalRAGRetriever
    print("✓ MedicalRAGRetriever imported")
    from diagnosis_classifiers import RAGFewShotDiagnosisClassifier
    print("✓ RAGFewShotDiagnosisClassifier imported")
    from gemmaUtils import (setup_logging,load_data)
    print("✓ setup_logging and load_data imported")
except ImportError as e:
    print(f"✗ Modular import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print("="*80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================
dpath = "E:/Education/CCSU-Thesis-2024/Data/"
TRAINING_DATA_PATH = dpath + 'intsl_train_group_full_no3c.snappy.parquet'
TESTING_DATA_PATH = dpath + 'intsl_test_group_full_no3c.snappy.parquet'
project_root = Path(__file__).parent.parent
ICD10_CSV_PATH = project_root / 'data' / 'raw' / 'ICD10-Code-Descriptions.csv'

# ============================================================================
# Test RAG Retrieval
# ============================================================================

def test_rag_retrieval():
    """Test that RAG retriever finds relevant similar examples"""

    mpath = 'E:/Education/llama/models'
    model_name = "/gemma-2-9b-it-q4_k_m.gguf"
    
    logger = setup_logging(
        log_dir ='E:/Education/CCSU-Thesis-2024/Data/logs', 
        log_level=logging.INFO)
    logger.info("="*80)
    logger.info("TESTING RAG RETRIEVAL")
    logger.info("="*80)
    
    # Load training data
    logger.info(f"\nLoading training data from: {TRAINING_DATA_PATH}")
       
    #Load Data
    train_df, test_df = load_data(trfile=TRAINING_DATA_PATH, 
                                  tsfile=TESTING_DATA_PATH, 
                                  train_sample_size=50, 
                                  test_sample_size=5, 
                                  seq_size=1000, logger=logger)

    # Test 1: Build RAG index
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Building RAG Index")
    logger.info("="*80)
    
    
        # Initialize RAG components
    logger.info("\nInitializing RAG components...")
    try:
        # Initialize the retriever
        rag_retriever = MedicalRAGRetriever(
            embedding_model_name='medicalai/ClinicalBERT',  
            logger=logger
        )
        logger.info("✓ RAG retriever initialized")

        #Initialize llama model
        MODEL_PATH = mpath + model_name
        
        llama_model = LlamaCppWrapper(
                model_path=MODEL_PATH, 
                n_ctx=4096, 
                n_threads=8, 
                n_batch=512, 
                use_mmap=True,
                logger=logger
)    
        # Initialize the classifier
        
        classifier = RAGFewShotDiagnosisClassifier(
                        gemma_model=llama_model,
                        rag_retriever=rag_retriever,
                        icd10_csv_path=ICD10_CSV_PATH,  # Make sure this is defined
                        logger=logger
                    )
        logger.info("✓ RAG classifier initialized")
    
    except Exception as e:
        logger.exception(f"Failed to initialize RAG components: {e}")
        return

    # Build the index
    logger.info("\nBuilding RAG index...")
    logger.info(f"Training data: {len(train_df)} examples")
    logger.info(f"Diagnoses: {sorted(train_df['icd10_code'].unique())}")

    try:
        # This calls fit() which builds the RAG index
        classifier.fit(train_df)
    
        logger.info("✓ RAG index built successfully!")
        logger.info(f"✓ Classes: {classifier.classes_}")
        logger.info(f"✓ Index size: {len(classifier.examples_df)} examples")
    
    except Exception as e:
        logger.exception(f"Failed to build RAG index: {e}")
        return

    # Test 1a: Verify index was created
    logger.info("\nVerifying index...")
    try:
        # Check if index files were created (adjust path as needed)
        index_path = 'rag_index'
        from pathlib import Path
    
        if Path(index_path).exists():
            logger.info(f"✓ Index files found at: {index_path}")
            # List files in index directory
            index_files = list(Path(index_path).glob('*'))
            logger.info(f"  Files: {[f.name for f in index_files]}")
        else:
            logger.warning(f"⚠️  Index directory not found at: {index_path}")
            logger.info("  (Index may be stored in memory only)")
        
    except Exception as e:
        logger.warning(f"Could not verify index files: {e}")

    # Test 1b: Test a simple retrieval
    logger.info("\nTesting retrieval...")
    try:
        # Get a test query
        test_case = train_df.iloc[0]
        test_query = test_case['text']
        test_id = test_case['id']
    
        logger.info(f"Query case ID: {test_id}")
        logger.info(f"Query diagnosis: {test_case['icd10_code']}")
        logger.info(f"Query text (first 150 chars): {test_query[:150]}...")
    
        # Retrieve similar examples
        retrieved = rag_retriever.retrieve(
            query_text=test_query,
            k=1,
            exclude_same_id=test_id
            )
    
        logger.info(f"\n✓ Retrieved {len(retrieved)} similar examples:")
        for idx, row in retrieved.iterrows():
            logger.info(f"  Example {idx + 1}:")
            logger.info(f"    ID: {row['id']}")
            logger.info(f"    Diagnosis: {row['icd10_code']}")
            logger.info(f"    Similarit distance: {row['similarity_distance']:,.4f}")
            logger.info(f"    Text preview: {row['text'][:100]}...")
        
    except Exception as e:
        logger.exception(f"Retrieval test failed: {e}")
        return

    logger.info("\n✓ TEST 1 COMPLETE: RAG index built and tested successfully!")
    # Test 2: Check data quality for RAG
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Data Quality Checks for RAG")
    logger.info("="*80)
    
    # Check for empty texts
    empty_texts = train_df['text'].isna().sum()
    logger.info(f"Empty texts: {empty_texts}")
    if empty_texts > 0:
        logger.warning(f"⚠️  {empty_texts} empty texts found - these should be filtered")
    else:
        logger.info("✓ No empty texts")
    
    # Check text lengths
    train_df['text_length'] = train_df['text'].str.len()
    logger.info(f"\nText length statistics:")
    logger.info(f"  Mean: {train_df['text_length'].mean():.0f} chars")
    logger.info(f"  Median: {train_df['text_length'].median():.0f} chars")
    logger.info(f"  Min: {train_df['text_length'].min():.0f} chars")
    logger.info(f"  Max: {train_df['text_length'].max():.0f} chars")
    
    # Check distribution by diagnosis
    logger.info(f"\nExamples per diagnosis:")
    classes = sorted(train_df['icd10_code'].unique())
    for code in classes:
        count = (train_df['icd10_code'] == code).sum()
        pct = count / len(train_df) * 100
        logger.info(f"  {code}: {count:4d} ({pct:5.1f}%)")
    
    # Test 3: Sample queries for each diagnosis
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Sample Queries (One Per Diagnosis)")
    logger.info("="*80)
    
    test_cases = []
    for diagnosis_code in classes:
        matching = train_df[train_df['icd10_code'] == diagnosis_code]
        if len(matching) > 0:
            sample = matching.sample(n=1, random_state=42).iloc[0]
            test_cases.append({
                'id': sample['id'],
                'diagnosis': diagnosis_code,
                'text': sample['text'],
                'text_length': len(sample['text'])
            })
            logger.info(f"\n{diagnosis_code}:")
            logger.info(f"  ID: {sample['id']}")
            logger.info(f"  Text length: {len(sample['text'])} chars")
            logger.info(f"  Text preview: {sample['text'][:150]}...")
    
    # Save test cases
    test_cases_df = pd.DataFrame(test_cases)
    output_file = 'E:/VScode-Projects/Thesis/data/processed/rag_test_cases.csv'
    test_cases_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Test cases saved to: {output_file}")
    
    # Test 4: Check for duplicate texts (can cause retrieval issues)
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Check for Duplicates")
    logger.info("="*80)
    
    duplicates = train_df.duplicated(subset=['text']).sum()
    logger.info(f"Duplicate texts: {duplicates}")
    if duplicates > 0:
        logger.warning(f"⚠️  {duplicates} duplicate texts found")
        logger.info("This is OK for subsequences from same admission")
    else:
        logger.info("✓ No duplicate texts")
    
    # Test 5: Test retrieval quality for each diagnosis
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Retrieval Quality Analysis")
    logger.info("="*80)

    for test_case in test_cases:
        logger.info(f"\nTesting retrieval for {test_case['diagnosis']}:")
        logger.info(f"  Query ID: {test_case['id']}")
    
        try:
            retrieved = rag_retriever.retrieve(
                query_text=test_case['text'],
                k=5,
                exclude_same_id=test_case['id']
            )
        
            # Check how many retrieved examples match the query diagnosis
            matches = sum(retrieved['icd10_code'] == test_case['diagnosis'])
            precision = matches / len(retrieved)
        
            logger.info(f"  Retrieved: {len(retrieved)} examples")
            logger.info(f"  Same diagnosis: {matches}/{len(retrieved)} ({precision:.1%})")
            logger.info(f"  Diagnoses retrieved: {list(retrieved['icd10_code'].value_counts().to_dict().items())}")
        
            # Show top 3 by similarity
            logger.info(f"  Top 3 most similar:")
            for idx, row in retrieved.head(3).iterrows():
                match_symbol = "✓" if row['icd10_code'] == test_case['diagnosis'] else "✗"
                logger.info(f"    {match_symbol} {row['icd10_code']} - Distance: {row['similarity_distance']:.4f}")
    
        except Exception as e:
            logger.error(f"  Retrieval failed: {e}")

    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info("✓ Data loaded successfully")
    logger.info("✓ Data ready for RAG indexing")
    logger.info(f"✓ {len(test_cases)} test cases prepared (one per diagnosis)")
    logger.info("\nNext steps:")
    logger.info("1. Initialize MedicalRAGRetriever")
    logger.info("2. Build index with this training data")
    logger.info("3. Test retrieval with saved test cases")

if __name__ == "__main__":
    
    test_rag_retrieval()
    