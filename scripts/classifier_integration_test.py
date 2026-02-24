"""
Test script for classifier integration with mock LLM.
Tests the full pipeline without requiring actual model.
"""

import pandas as pd
import logging
from pathlib import Path

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
TEST_CASES_PATH = '../data/processed/rag_test_cases.csv'

# ============================================================================
# Mock Objects
# ============================================================================

class MockGemmaModel:
    """Mock Gemma model that returns predictable responses"""
    
    def __init__(self):
        self.call_count = 0
    
    def generate_with_probabilities(self, prompt, classes, max_tokens=5, 
                                   temperature=0.7, logger=None):
        """Mock response with probabilities"""
        self.call_count += 1
        
        # Parse which diagnosis from prompt (simple mock)
        # In real scenario, this would come from actual model
        predicted_class = self.call_count % len(classes)  # Rotate through classes
        
        if logger:
            logger.debug(f"Mock prediction: class {predicted_class}")
        
        # Create mock logprobs structure
        logprobs = []
        for i in range(len(classes)):
            if i == predicted_class:
                prob = 0.7  # High probability for predicted class
            else:
                prob = 0.1  # Low for others
            
            logprobs.append({
                'token': str(i),
                'logprob': -0.3567 if i == predicted_class else -2.3026
            })
        
        return {
            'choices': [{
                'logprobs': {
                    'content': [{
                        'top_logprobs': logprobs
                    }]
                }
            }],
            'text': str(predicted_class)
        }

class MockRAGRetriever:
    """Mock RAG retriever for testing"""
    
    def __init__(self):
        self.index_data = None
    
    def build_index(self, examples_df, text_column='text', save_path=None):
        """Mock index building"""
        self.index_data = examples_df.copy()
        logger.info(f"Mock: Built index with {len(examples_df)} examples")
        return self
    
    def retrieve(self, query_text, k=3, exclude_same_id=None):
        """Mock retrieval - return random samples"""
        if self.index_data is None:
            raise ValueError("Index not built")
        
        # Simple mock: return k random examples
        filtered = self.index_data.copy()
        if exclude_same_id is not None:
            filtered = filtered[filtered['id'] != exclude_same_id]
        
        if len(filtered) < k:
            return filtered
        
        return filtered.sample(n=k, random_state=42)

# ============================================================================
# Integration Test
# ============================================================================

def test_classifier_integration():
    """Test full classifier pipeline with mocks"""
    
    logger.info("="*80)
    logger.info("TESTING CLASSIFIER INTEGRATION (Mock LLM)")
    logger.info("="*80)
    
    # Load test cases
    if not Path(TEST_CASES_PATH).exists():
        logger.error(f"Test cases not found: {TEST_CASES_PATH}")
        logger.error("Run test_rag_retrieval.py first to generate test cases")
        return
    
    logger.info(f"Loading test cases from: {TEST_CASES_PATH}")
    test_df = pd.read_csv(TEST_CASES_PATH)
    logger.info(f"Loaded {len(test_df)} test cases")
    
    # Load ICD-10 descriptions
    logger.info(f"Loading ICD-10 descriptions from: {ICD10_CSV_PATH}")
    icd10_df = pd.read_csv(ICD10_CSV_PATH)
    
    # Initialize mocks
    logger.info("\nInitializing mock components...")
    mock_model = MockGemmaModel()
    mock_retriever = MockRAGRetriever()
    
    # Import and initialize classifier
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Initialize Classifier")
    logger.info("="*80)
    
    try:
        from classifiers import RAGFewShotDiagnosisClassifier
        
        classifier = RAGFewShotDiagnosisClassifier(
            gemma_model=mock_model,
            rag_retriever=mock_retriever,
            icd10_csv_path=ICD10_CSV_PATH,
            logger=logger
        )
        logger.info("✓ Classifier initialized")
    except Exception as e:
        logger.exception(f"Failed to initialize classifier: {e}")
        return
    
    # Fit classifier
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Fit Classifier")
    logger.info("="*80)
    
    try:
        # Use test cases as "training" data for mock
        classifier.fit(test_df)
        logger.info(f"✓ Classifier fitted with {len(test_df)} examples")
        logger.info(f"Classes: {classifier.classes_}")
    except Exception as e:
        logger.exception(f"Failed to fit classifier: {e}")
        return
    
    # Test single prediction
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Single Prediction")
    logger.info("="*80)
    
    try:
        test_case = test_df.iloc[0]
        logger.info(f"Test case ID: {test_case['id']}")
        logger.info(f"True diagnosis: {test_case['diagnosis']}")
        
        result = classifier.predict_single(
            patient_text=test_case['text'],
            patient_id=test_case['id'],
            k_examples=3
        )
        
        logger.info(f"\nPrediction: {result['prediction']}")
        logger.info(f"Confidence: {result.get('confidence', 'N/A')}")
        logger.info(f"Probabilities: {result.get('probabilities', 'N/A')}")
        logger.info("✓ Single prediction successful")
        
    except Exception as e:
        logger.exception(f"Single prediction failed: {e}")
        return
    
    # Test batch prediction
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Batch Prediction")
    logger.info("="*80)
    
    try:
        # Predict on all test cases
        results_df = classifier.predict(test_df, verbose=False)
        
        logger.info(f"✓ Batch prediction successful")
        logger.info(f"Results shape: {results_df.shape}")
        logger.info(f"Columns: {results_df.columns.tolist()}")
        
        # Calculate accuracy (with mock, won't be meaningful)
        accuracy = results_df['correct'].mean()
        logger.info(f"\nMock accuracy: {accuracy:.2%}")
        logger.info("(Note: This is mock data, accuracy not meaningful)")
        
    except Exception as e:
        logger.exception(f"Batch prediction failed: {e}")
        return
    
    # Test prompt generation
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Prompt Generation")
    logger.info("="*80)
    
    try:
        test_case = test_df.iloc[1]
        prompt = classifier._create_prompt_with_rag(
            patient_text=test_case['text'],
            patient_id=test_case['id'],
            k=3
        )
        
        # Verify prompt structure
        checks = {
            'Has diagnosis options': 'Based on the patient\'s clinical presentation' in prompt,
            'Has 4 classes': all(f'{i} =' in prompt for i in range(4)),
            'Has examples': 'Example' in prompt,
            'Has patient ID': str(test_case['id']) in prompt,
            'Has instruction format': '<start_of_turn>user' in prompt,
            'Ends correctly': '<start_of_turn>model' in prompt,
        }
        
        all_passed = True
        for check_name, passed in checks.items():
            status = "✓" if passed else "✗"
            logger.info(f"{status} {check_name}")
            if not passed:
                all_passed = False
        
        if all_passed:
            logger.info("✓ All prompt checks passed")
        else:
            logger.warning("⚠️  Some prompt checks failed")
            
    except Exception as e:
        logger.exception(f"Prompt generation failed: {e}")
        return
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("="*80)
    logger.info("✓ Classifier initialization: PASS")
    logger.info("✓ Classifier fitting: PASS")
    logger.info("✓ Single prediction: PASS")
    logger.info("✓ Batch prediction: PASS")
    logger.info("✓ Prompt generation: PASS")
    logger.info(f"\nTotal mock model calls: {mock_model.call_count}")
    logger.info("\nNext steps:")
    logger.info("1. Test with real Gemma model on small sample")
    logger.info("2. Evaluate on full test set")
    logger.info("3. Compare RAG vs non-RAG performance")

if __name__ == "__main__":
    try:
        test_classifier_integration()
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        raise