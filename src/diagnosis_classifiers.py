import gc
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional
from unittest import result

# Third-party imports
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import torch

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score
)
from pathlib import Path
from sklearn.preprocessing import label_binarize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import math
from gemmaUtils import _validate_and_clean

class FewShotDiagnosisClassifier:
    """
    Few-shot learning classifier for ICD-10 diagnosis prediction using LLM.
    a) __init__(self, gemma_model, icd10_descriptions: Optional[Dict[str, str]] = None) 
    b) fit(self, examples_df: pd.DataFrame)
    c) _create_prompt(self, patient_text: str, patient_id: Optional[str] = None) -> str   
    d) _parse_response(self, response: str) -> str</li>
    e) _extract_class_probabilities(self, result) -> List[float]</li>
    f) _extract_class_probabilities2(self, logprobs_dict) -> List[float]</li>
    g) predict_single(self, patient_text: str, patient_id: Optional[str] = None) -> Dict</li>
    h) predict(self, test_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame</li>
    i) calc_prediction(self, row)</li>
    j) vote_score(self, details_df, num_classes)</li>
    k) get_perf_data(self, y_true, y_pred)</li>
    l) valuate(self, test_df: pd.DataFrame, verbose: bool = True) -> Dict</li>
    m) evaluate2(self, test_df: pd.DataFrame, verbose: bool = True) -> Dict</li>
    n) results_df.to_csv(filepath, index=False)</li>      
    """
    
    def __init__(self, gemma_model, example_size, test_size, k_examples, icd10_csv_path=None, 
                 logger=None, embedding_model=None, use_few_shot='C'):
        """
        Initialize the classifier.
        
        Parameters:
        - gemma_model: The Gemma model instance for generating predictions
        - icd10_descriptions: Optional dict mapping ICD10 codes to descriptions
        - k_examples: The number of examples to use per diagnosis
        - example_size: The number of words to include in the examples
        - test_size: The words in the test case to predict
        - logger: The logger to post messages to
        - embedding_model: model used to generate embedding
        - centroid_examples: whether to use centroid examples (original method) or diverse examples (new method)

        Generated Data:
        - examples_df: The full set of possible examples
        - classes_: The diagnositc classes identified in the dataset
        - few_shot_examples: The chosen few shot examples
        """
        self.gemma_model = gemma_model
        self.logger = logger
        self.k_examples = k_examples
        self.example_size = example_size
        self.test_size = test_size
        self.embedding_model = embedding_model
        #last 3 elements are set in fit()
        self.examples_df = None
        self.classes_ = None
        self.few_shot_examples = None
        self.use_few_shot = use_few_shot
        text_out1 = f"k_examples: {self.k_examples} example size {self.example_size}"
        text_out2 = f"test size {self.test_size} embedding model: {self.embedding_model}"
        self.logger.info(text_out1)
        self.logger.info(text_out2)
        
        # Load ICD-10 reference DataFrame
        if icd10_csv_path:
            self.icd10_df = pd.read_csv(icd10_csv_path)
            if self.logger:
                self.logger.info(f"Loaded {len(self.icd10_df)} ICD-10 codes from {icd10_csv_path}")
        else:
            self.icd10_df = None
            if self.logger:
                self.logger.warning("No ICD-10 CSV provided, will use hardcoded descriptions")
        
    def _get_prototypical_examples(self, train_df, k_per_class=1):
        """
        Get the most representative (prototypical) examples for each class.
        These are examples closest to each class's centroid in embedding space.
        """
        model = SentenceTransformer(self.embedding_model)
        prototypes = []

        train_df = _validate_and_clean(df=train_df, stage='examples', logger=self.logger)
        for diagnosis in sorted(train_df['icd10_code'].unique()):
            # Get all examples of this class
            class_df = train_df[train_df['icd10_code'] == diagnosis].copy()
        
            # Embed them
            embeddings = model.encode(class_df['text'].tolist())
        
            # Find centroid (average embedding)
            centroid = embeddings.mean(axis=0)
        
            # Get k closest to centroid
            
            similarities = cosine_similarity([centroid], embeddings)[0]
            top_k_indices = similarities.argsort()[-k_per_class:][::-1]
        
            prototypes.append(class_df.iloc[top_k_indices])
    
        return pd.concat(prototypes, ignore_index=True)
    
    def _get_diverse_examples(self, train_df, k_per_class=3):
        """
        Get diverse examples that cover different aspects of each class.
        Uses greedy selection to maximize diversity.
    
        NEW METHOD - Alternative to _get_prototypical_examples
        """
        MIN_WORDS = 50
        model = SentenceTransformer(self.embedding_model)  # Uses same embedding model
        prototypes = []
        train_df = _validate_and_clean(df=train_df, stage='examples', logger=self.logger)
        for diagnosis in sorted(train_df['icd10_code'].unique()):
            # Get all examples of this class
            class_df = train_df[train_df['icd10_code'] == diagnosis].copy()
            class_df['words'] = class_df['text'].apply(lambda x: len(x.split()))

            # Filter out short notes before selecting examples
            long_df = class_df[class_df['words'] >= MIN_WORDS].copy()
            if len(long_df) < k_per_class:
                self.logger.warning(f"Only {len(long_df)} long examples for {diagnosis} (needed {k_per_class}), using all available")
                long_df = long_df if len(long_df) > 0 else class_df  # fallback if none pass filter
        
            class_df = long_df  # use filtered set from here on
            self.logger.info(f"Average words per sequence {np.mean(class_df['words'])} words, minimum words {np.min(class_df['words'])} words")
            
            # Embed them
            embeddings = model.encode(class_df['text'].tolist())
        
            # --- DIVERSITY LOGIC STARTS HERE (different from centroid) ---
            selected_indices = []
        
            # Start with example closest to centroid
            centroid = embeddings.mean(axis=0)
            similarities = cosine_similarity([centroid], embeddings)[0]
            first_idx = similarities.argmax()
            selected_indices.append(first_idx)
        
            # Iteratively add most diverse examples
            for _ in range(k_per_class - 1):
                if len(selected_indices) >= len(embeddings):
                    break  # No more examples to select
            
                selected_embeddings = embeddings[selected_indices]
            
                # For each unselected example, find minimum similarity to selected set
                min_min_similarity = 1.1  # Start above any possible cosine similarity
                best_idx = -1 

                for i in range(len(embeddings)):
                    if i in selected_indices:
                        continue
                
                    # Similarity to all selected examples
                    similarities = cosine_similarity([embeddings[i]], selected_embeddings)[0]
                
                    # Find minimum similarity (most different)
                    min_sim = similarities.min()
                
                    # Select example furthest from nearest selected example
                    if min_sim < min_min_similarity:
                        min_min_similarity = min_sim
                        best_idx = i

            
                if best_idx != -1:
                    selected_indices.append(best_idx)
                # --- DIVERSITY LOGIC ENDS HERE ---
        
            prototypes.append(class_df.iloc[selected_indices])

        # Shuffle examples so they are not ordered by icd10 code to prevent bias
        result = pd.concat(prototypes, ignore_index=True)
        result = result.sample(frac=1, random_state=self.gemma_model.seed).reset_index(drop=True)
        return result

    def fit(self, examples_df: pd.DataFrame):

        """
        Fit the model with example cases.
         
        Parameters:
        - examples_df: DataFrame with columns ['id', 'text', 'icd10_code', 'label']
        """
        
        self.examples_df = examples_df.copy()
        self.classes_ = sorted(examples_df['icd10_code'].unique())
        
        # Validate examples
        if len(self.examples_df) < 1:
            raise ValueError("Need at least 1 example to fit the model")
        
        if self.use_few_shot == 'C':
            # Option 1: Centroid (your original - poor performance)
            self.logger.info("Creating prototypical examples...")
            self.few_shot_examples = self._get_prototypical_examples(examples_df, k_per_class=self.k_examples)
            self.logger.info(f"Created {len(self.few_shot_examples)} prototypical examples")
        elif self.use_few_shot == 'D': 
            # Option 2: Diversity (NEW - test this)
            self.logger.info("Creating diverse examples...")
            self.few_shot_examples = self._get_diverse_examples( self.examples_df, k_per_class=self.k_examples)
            self.logger.info(f"Created {len(self.few_shot_examples)} diverse examples")
        else:
            self.logger.info("Not including few-shot examples in prompt")
            self.few_shot_examples = pd.DataFrame()  # Empty DataFrame when not using examples
        
        
        self.logger.info(f"\nTotal examples: {len(self.few_shot_examples)}")
        if len(self.few_shot_examples) > 0:
            self.logger.info(f"\nPer-class distribution: {self.few_shot_examples['icd10_code'].value_counts().sort_index()}")
        else:
            self.logger.info("\nPer-class distribution: zero-shot mode, no examples")
        
        self.logger.info(f"Classes: {self.classes_}")
        
        self.logger.info(f"self.classes_ = {self.classes_}")
        
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.info("\n=== CHECKING EXAMPLE LABELS ===")
            for idx, row in self.few_shot_examples.iterrows():
                code = row['icd10_code']
                code_idx = self.classes_.index(code)
                self.logger.debug(f"Example {idx+1}: ICD {code} → Labeled as digit {code_idx}")
        
        return self
    
    def format_diagnosis_options(self):
        """
        Format diagnosis options for the prompt with class indices.
        Uses self.classes_ and self.icd10_df to generate the formatted text.
    
        Returns:
            Formatted string for the prompt
        """
        # The next few lines are used to test order bias
        import random
        #shuffled_classes = self.classes_.copy()
        #random.shuffle(shuffled_classes)

        lines = ["Based on the patient's clinical presentation, predict which diagnosis class applies:\n"]
    
        #for idx, code in enumerate(self.classes_):
        for idx, code in enumerate(self.classes_):
            if self.icd10_df is not None:
                # Look up description from DataFrame
                match = self.icd10_df[self.icd10_df['icd10_code'] == code]
                if len(match) > 0:
                    desc = match['description'].values[0]
                    #lines.append(f"                    {idx} = {desc}")
                    lines.append(f"                    {idx} = {code} ({desc})") 
                else:
                    lines.append(f"                    {idx} = Unknown Diagnosis")
                    if self.logger:
                        self.logger.warning(f"Description not found for code: {code}")
            else:
                lines.append(f"                    {idx} = {code}")
    
        return "\n".join(lines)

    def _generate_icd10_mapping(self) -> str:
        """Generate ICD-10 code mapping (SHARED by both RAG and non-RAG)."""
        if self.icd10_descriptions is None:
            return """0 = A41.9 (Sepsis, unspecified organism)
                      1 = I21.4 (Acute subendocardial myocardial infarction)
                      2 = J96.00 (Acute respiratory failure, unspecified)
                    3 = N17.9 (Acute kidney failure, unspecified)"""
        
        mapping_lines = []
        for idx, icd_code in enumerate(self.classes_):
            desc_row = self.icd10_descriptions[
                self.icd10_descriptions['icd10_code'] == icd_code
            ]
            description = desc_row.iloc[0]['description'] if not desc_row.empty else "Description not found"
            mapping_lines.append(f"{idx} = {icd_code} ({description})")
        
        return "\n".join(mapping_lines)
    
    def truncate_to_words(self, text, max_words=500):
        """Keep first 60% and last 40% of text"""
        words = text.split()
        if len(words) <= max_words:
            return text
    
        # Keep first 300 words and last 200 words
        keep_start = int(max_words * 0.6)
        keep_end = int(max_words * 0.4)

        return ' '.join(words[:max_words])
        #return ' '.join(words[:keep_start]) + ' [...] ' + ' '.join(words[-keep_end:])
    
    def _create_prompt(self, patient_text: str, patient_id: Optional[str] = None, example_size=500) -> str:
        """
        Create a few-shot prompt for a single patient.
        """
        if self.examples_df is None:
            raise ValueError("Model not fitted. Call fit() first.")
    
        # Generate diagnosis options dynamically
        diagnosis_section = self.format_diagnosis_options()
        #self.logger.info('diagnosis section')
        #self.logger.info(diagnosis_section)
        # Build prompt header
        prompt = f"""<start_of_turn>user

        You are an experienced attending physician reviewing clinical notes to make a diagnostic assessment.

        {diagnosis_section}

        CRITICAL INSTRUCTIONS:
        - Respond with ONLY a single digit: 0, 1, 2, or 3
        - Do not include the diagnosis code (e.g., A41.9)
        - Do not add any explanation, punctuation, spaces, or newlines
        - Output format: Just the digit, nothing else

        """
        if self.use_few_shot == 'C' or self.use_few_shot == 'D':
            self.logger.info('including examples')
            prompt += "Here are some example cases:\n\n"

            # Add prototypical examples (these have text!)
            for idx, row in self.few_shot_examples.iterrows():
                example_num = idx + 1
                code_idx = self.classes_.index(row['icd10_code'])
                truncated_text = self.truncate_to_words(row['text'], max_words=self.example_size)
                prompt += f"""Example {example_num}:
    
                Clinical Note: {truncated_text}...
                Diagnosis: {code_idx}
                """
        else:        
            self.logger.info('not including examples')
        # Add new patient case
        #truncated_test_text = self.truncate_to_words(patient_text, max_words=self.example_size)
        prompt += "Now, review the following new patient case:\n\n"
        if patient_id:
            prompt += f"Patient ID: {patient_id}\n"
        
        prompt += f"Clinical Note: {patient_text}\n\n"
        prompt += "Diagnosis (respond with only the number 0, 1, 2, or 3):<end_of_turn>"
        prompt += "\n<start_of_turn>model"

        return prompt
    
    def _parse_response(self, response: str) -> str:
        """
        Parse the model response to extract the ICD10 code.
        Model now outputs 0, 1, 2, or 3 which we map back to ICD codes.
        """
        response = response.strip()
    
        # Map single digit responses back to ICD codes
        if response in ['0', '1', '2', '3']:
            class_idx = int(response)
            if class_idx < len(self.classes_):
                return self.classes_[class_idx]
    
        # Try to find a digit at the start of response
        if response and response[0] in ['0', '1', '2', '3']:
            class_idx = int(response[0])
            if class_idx < len(self.classes_):
                return self.classes_[class_idx]
    
        # Fallback: check if any ICD code appears in response
        for code in self.classes_:
            if code in response:
                return code
    
        # Last resort: return "UNKNOWN"
        return "UNKNOWN"
    
   
    def _extract_class_probabilities2(self, output: Dict) -> List[float]:
        """
        Extract probability distribution for each class from model output.
        Searches through all token positions to find the one with digit classes.
        """
    
        try:
            if not output or 'choices' not in output:
                self.logger.warning(f"No valid output")
                return [0.25] * 4
        
            choice = output['choices'][0]
            logprobs_data = choice.get('logprobs', {})
            
            # ← NEW: Get from 'content' array first        # changed 12/3/2026 11:46 AM
            content_list = logprobs_data.get('content', [])
        
            if not content_list:  # ← NEW: Check if content exists      # changed 12/3/2026 11:46 AM
                self.logger.warning(f"No content in logprobs")
                return [0.25] * 4
        
            # ← CHANGED: Get top_logprobs from content[0] instead of directly from logprobs_data
            top_logprobs_list = content_list[0].get('top_logprobs', [])      # changed 12/3/2026 11:46 AM
            #top_logprobs_list = logprobs_data.get('top_logprobs', [])       # changed 12/3/2026 11:46 AM
        
            if not top_logprobs_list:
                self.logger.warning(f"No top_logprobs found")
                return [0.25] * 4
        
            #print(f"🔍 Searching through {len(top_logprobs_list)} token positions")    # changed 12/3/2026 11:59 AM
            self.logger.debug(f"Found {len(top_logprobs_list)} tokens in top_logprobs")          # changed 12/3/2026 11:59 AM
        
            # ← NEW CODE BLOCK: Build a dict for easy lookup       # changed 12/3/2026 11:59 AM
            token_dict = {}
            for item in top_logprobs_list:
                token_dict[item['token']] = item['logprob']
        
            self.logger.debug(f"Available tokens: {list(token_dict.keys())[:10]}")
            # ← END NEW CODE BLOCK                                 # changed 12/3/2026 11:59 AM   
        
            # Search through all token positions to find the one with digit classes   # Section removed 12/3/2026 11:59 AM    
            #for position_idx, top_logprobs in enumerate(top_logprobs_list):
            #    # Count how many digit tokens (0,1,2,3) are in this position
            #    digit_classes = [k for k in top_logprobs.keys() if k.strip() in ['0', '1', '2', '3']]
            #
            #    print(f"[DEBUG] Position {position_idx}: Found {len(digit_classes)} digit classes: {digit_classes}")
            #
            #    # If we found at least 2 digit classes, use this position
            #    if len(digit_classes) >= 2:
            #        print(f"[OK] Using position {position_idx} for class probabilities")
                
            # Extract probabilities for each class
            probs = []
            for class_idx in range(4):
                token = str(class_idx)
                if token in token_dict:
                    prob = math.exp(token_dict[token])
                    probs.append(prob)
                    print(f"  Class {class_idx}: logprob={token_dict[token]:.4f}, prob={prob:.4f}")
                else:
                    probs.append(1e-10)
                    print(f"  Class {class_idx}: not found (using 1e-10)")
                
            # Normalize to sum to 1
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
                self.logger.info(f"[OK] Normalized probabilities: {[f'{p:.4f}' for p in probs]}")
            else:
                probs = [0.25] * 4
                self.logger.warning(f"Total probability was 0, using uniform distribution")
                
            return probs
        
            # If we get here, we didn't find a position with digit classes       # Section removed 12/3/2026 11:59 AM
            # print(f"[WARNING] Could not find token position with digit classes")
            #print(f"[WARNING] Available tokens at each position:")
            #for idx, top_logprobs in enumerate(top_logprobs_list):
            #    print(f"   Position {idx}: {list(top_logprobs.keys())[:5]}")
            #        
            #return [0.25] * 4
        
        except Exception as e:
            self.logger.exception(f"Error during generation with probabilities: {e}")            
            #return [0.25] * 4
            raise

    def analyze_prompt_composition2(self, prompt: str, patient_id: Optional[str] = None):        
        
        delimiters = ["<start_of_turn>user", "assessment.", "CRITICAL INSTRUCTIONS", "Here are some example cases", 
                      "Now, review the following new patient case"]

        self.logger.info(f"Analyzing prompt for patient_id: {patient_id}")
        prompt_sections = {}
        for delimiter in delimiters:
            start_pos = prompt.find(delimiter)
            end_pos = start_pos + len(delimiter)
            prompt_sections[delimiter] = (start_pos, end_pos)
            self.logger.debug(f"Start: {start_pos}, End: {end_pos}, Delimiter text: {prompt[start_pos:end_pos]}")
            
        #for key, value in prompt_sections.items():
        #    print(f"{key}: {value}")

        start_length = prompt_sections["assessment."][0]
        assessment_length = prompt_sections["CRITICAL INSTRUCTIONS"][0] - prompt_sections["assessment."][0]
        if prompt_sections["Here are some example cases"][0] > 0:
            critical_ins = prompt_sections["Here are some example cases"][0] - prompt_sections["CRITICAL INSTRUCTIONS"][0]
            examples_length = prompt_sections["Now, review the following new patient case"][0] - prompt_sections["Here are some example cases"][0]
        else:
            critical_ins = prompt_sections["Now, review the following new patient case"][0] - prompt_sections["CRITICAL INSTRUCTIONS"][0]
            examples_length = 0
        
        note_length = len(prompt) - prompt_sections["Now, review the following new patient case"][0]
        log_msg = ", ".join(f"{k}={v}" for k, v in prompt_sections.items())
        self.logger.info(f"total length: {len(prompt)}, "
                         f"start_length: {start_length}, "
                         f"assessment_length: {assessment_length}")
        self.logger.info(f"critical_ins: {critical_ins}, "
                         f"examples_length: {examples_length}, "
                         f"note_length: {note_length}")

        self.logger.info(f"\nMetrics: {log_msg}")
        return
        
    def predict_single(self, patient_text: str, logger = None, patient_id: Optional[str] = None, **kwargs) -> Dict:
        """
        Predict diagnosis for a single patient with class probabilities.
    
        Returns:
        - Dictionary with 'prediction', 'probabilities', 'prompt', and 'raw_response'
        """
        self.logger.info(f"Processing ID: {round(patient_id)}")
        
        if self.examples_df is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        start = time.time()
        # Generate prompt - modify to encourage single-token response
        prompt = self._create_prompt(patient_text, patient_id)
        run_time = time.time() - start
        self.logger.info(f"Completed creating prompt Time: {run_time:.2f}s")
        #analysis = self.analyze_prompt_composition2(atient_text, patient_id)
        start = time.time()
        self.analyze_prompt_composition2(prompt, patient_id)
        run_time = time.time() - start
        self.logger.debug(f"Completed analyzing prompt Time: {run_time:.2f}s")
        # Get prediction WITH PROBABILITIES from Gemma
        start = time.time()
        result = self.gemma_model.generate_with_probabilities(prompt, self.classes_,
                                                              max_tokens=5)
        run_time = time.time() - start
        self.logger.info(f"Completed generating probabilities Time: {run_time:.2f}s")    
        if result is None:
            # Fallback to old method if logprobs not available
            raw_response = self.gemma_model.generate(prompt)
            prediction = self._parse_response(raw_response)
            return {
                'prediction': prediction,
                'probabilities': None,
                'prompt': prompt,
                'raw_response': raw_response
            }
    
        # Extract probabilities
        probabilities = self._extract_class_probabilities2(result)
        
        # ← ADD DEBUG HERE                                          #Added 12/3/2025 12:25 PM
        self.logger.debug(f"Probabilities = {probabilities}")
        self.logger.debug(f"Probabilities type = {type(probabilities)}")
        self.logger.debug(f"Probabilities is None? {probabilities is None}")
    
        
        # Get top prediction
        # if probabilities:              # Changed 12/3/2025 12:25 PM   
        if probabilities is not None:    # Changed 12/3/2025 12:25 PM   
            max_prob_idx = probabilities.index(max(probabilities))
            prediction = self.classes_[max_prob_idx]
            confidence = probabilities[max_prob_idx]  
        else:
            self.logger.warning(f"Probabilities is None, using fallback parsing")     # Changed 12/3/2025 12:25 PM   
            prediction = self._parse_response(result.get('text', ''))
            confidence = None  # ← Set here

        return {'prediction': prediction,
                'probabilities': probabilities,
                'confidence': confidence,  # ← Use the variable
                'prompt': prompt,
                'raw_response': result
                }

    
    def predict(self, test_df, verbose=False) -> pd.DataFrame:
        """
        Predict diagnoses for all patients in test_df.
        
        Parameters:
        - test_df: DataFrame with columns ['id', 'text', 'icd10_code', 'label']
        - verbose: Whether to print progress
        
        Returns:
        - DataFrame with predictions and metadata
        """
        if self.examples_df is None:
            raise ValueError("Model not fitted. Call fit() first.")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prompt_file = f'~/thesis/logs/prompts-output-{timestamp}.csv'
        prompt_outfile = Path(prompt_file).expanduser()
        f = open(prompt_outfile, 'a', encoding='utf-8')
        predictions = []


                
        for idx, row in tqdm(test_df.iterrows(), 
                             total=len(test_df), 
                             desc="Processing subsequences",
                             unit="subseq"):
            
            self.logger.info(f"Handling subsequence: {idx}")
            if verbose and (idx + 1) % 10 == 0:
                self.logger.info(f"Processing patient {idx + 1}/{len(test_df)}...")
            
            # Get prediction
            result = self.predict_single(
                            patient_text=row['text'],
                            patient_id=row['id']
                    )
            
            if result['prediction'] is None:
                self.logger.warning(f"Skipping subsequence {idx} for patient {row['id']}")
                continue  # move to next subsequence

            f.write(f"{result['prompt']}\n")
            probs = result['probabilities']  # Add this line to define probs
            
            # Predict with probabilities
            self.logger.info(f"Actual: {row['icd10_code']}")
            self.logger.info(f"Prediction: {result['prediction']}")
            self.logger.info(f"Confidence: {result['confidence']:.2%}")
            self.logger.info("\nClass Probabilities:")
            
            prob_pairs = list(zip(self.classes_, probs))
            sorted_probs = sorted(prob_pairs, key=lambda x: x[1], reverse=True)
        
            #for code, prob in sorted_probs:
                #print(f"  {code}: {prob:.4f} ({prob*100:.2f}%)")
            
            predictions.append({
                    'id': row['id'],
                    'text': row['text'],
                    'true_icd10_code': row['icd10_code'],
                    'true_label': row['label'],
                    'predicted_icd10_code': result['prediction'],
                    # Add individual class probabilities by index
                    'prob_class_0': probs[0] if len(probs) > 0 else 0.0,
                    'prob_class_1': probs[1] if len(probs) > 1 else 0.0,
                    'prob_class_2': probs[2] if len(probs) > 2 else 0.0,
                    'prob_class_3': probs[3] if len(probs) > 3 else 0.0,
                    'raw_response': result['raw_response'],
                    'correct': result['prediction'] == row['icd10_code']
            })
        
        results_df = pd.DataFrame(predictions)
        
        if verbose:
            self.logger.info(f"\nCompleted predictions for {len(test_df)} patients")
        
        #f.close()
        return results_df
    
    def calc_prediction(self, row, score_cols):
        """
        Find the class with maximum probability.
    
        Args:
            row: DataFrame row
            score_cols: List of column names for class scores
    
        Returns:
            Class index (0, 1, 2, 3) or 99 if error
        """
        try:
            scores = [row[col] for col in score_cols]
            max_score = max(scores)
            return scores.index(max_score)
        except Exception as e:
            self.logger.warning(f"Error in calc_prediction: {e}")
            raise
    
    def vote_score(self, details_df):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

         # Filter out failed predictions before scoring
        original_count = len(details_df)
        details_df = details_df[details_df['predictions'].notna()].copy()
        skipped = original_count - len(details_df)
        if skipped > 0:
            self.logger.warning(f"vote_score: excluded {skipped} subsequences with None predictions")

        adm_scores = details_df.sort_values(by='adm_id')

        #for n in range(num_classes):
        #    ename = 'element-' + str(n)

        details_df['type'] = 'D'
        details_df['adm_count'] = 1
        
        temp_scores = adm_scores.groupby(['adm_id', 'actuals']).agg(adm_count=('adm_id', 'size'),
                                        Class_1_Mean=('class_1_score', 'mean'), Class_1_Max=('class_1_score', 'max'),
                                        Class_2_Mean=('class_2_score', 'mean'), Class_2_Max=('class_2_score', 'max'),
                                        Class_3_Mean=('class_3_score', 'mean'), Class_3_Max=('class_3_score', 'max'),
                                        Class_4_Mean=('class_4_score', 'mean'), Class_4_Max=('class_4_score', 'max')).reset_index()

        temp_scores['class_1_score'] = (temp_scores['Class_1_Max'] + (temp_scores['Class_1_Mean'] 
                                                                     * temp_scores['adm_count']/2))/(1 + temp_scores['adm_count']/2)
        temp_scores['class_2_score'] = (temp_scores['Class_2_Max'] + (temp_scores['Class_2_Mean'] 
                                                                     * temp_scores['adm_count']/2))/(1 + temp_scores['adm_count']/2)
        temp_scores['class_3_score'] = (temp_scores['Class_3_Max'] + (temp_scores['Class_3_Mean'] 
                                                                     * temp_scores['adm_count']/2))/(1 + temp_scores['adm_count']/2)
        temp_scores['class_4_score'] = (temp_scores['Class_4_Max'] + (temp_scores['Class_4_Mean']
                                                                     * temp_scores['adm_count']/2))/(1 + temp_scores['adm_count']/2)
        
        score_cols = ['class_1_score', 'class_2_score', 'class_3_score', 'class_4_score']
    
        row_sums = temp_scores[score_cols].sum(axis=1)
        temp_scores[score_cols] = temp_scores[score_cols].div(row_sums, axis=0)
    
        # Optional: Log normalized values
        self.logger.info("Sample normalized scores:")
        self.logger.info(f"\n{temp_scores[score_cols].head()}")
        self.logger.info(f"Row sums after normalization: {temp_scores[score_cols].sum(axis=1).head()}")
        
        temp_scores['predictions'] = temp_scores.apply(lambda row: self.calc_prediction(row, score_cols), axis=1)
        
        temp_scores['predicted_icd'] = temp_scores['predictions']\
                                            .apply(lambda x: self.classes_[x] if x < len(self.classes_) else 'UNKNOWN')
        
        temp_scores['type'] = 'S'
        
        temp_scores2 = temp_scores[['adm_id','type', 'actuals','adm_count','predictions','predicted_icd', 'class_1_score','class_2_score',
                                    'class_3_score', 'class_4_score']]
        
        out_scores = pd.concat([details_df,temp_scores2], ignore_index=True)
        sorted_out_scores = out_scores.sort_values(['adm_id', 'type'])
        
        pred_labels = temp_scores['predicted_icd']
        actuals = temp_scores['actuals']
        probs = temp_scores[['class_1_score', 'class_2_score', 'class_3_score', 'class_4_score']].to_numpy()
        self.logger.info("=== VOTE_SCORE OUTPUT CHECK ===")
        self.logger.info(f"Unique actuals: {sorted(actuals.unique())}")
        self.logger.info(f"Unique predictions: {sorted(pred_labels.unique())}")
        self.logger.info(f"Type of predictions: {type(pred_labels.iloc[0])}")
        self.logger.info(f"Sample predictions: {pred_labels.head(10).tolist()}")

        self.logger.debug(f"Vote scores:\n{temp_scores}")
        out_file = Path('~/thesis/logs/sorted_out_scores_{timestamp}.csv').expanduser()
        #out_file = f'E:/Education/CCSU-Thesis-2024/Data/logs/output-scores/sorted_out_scores_{timestamp}.csv'
        sorted_out_scores.to_csv(out_file, index=False)

        return (actuals, pred_labels, probs)


    def get_perf_data(self, y_true, y_pred, probs):

        #print('get perf data input')
        self.logger.debug(f'perf y_true:\n{y_true[0:5]}')
        self.logger.debug(f'perf y_pred:\n{y_pred[0:5]}')
        self.logger.debug('Generating Performance Data')
        
        cm = confusion_matrix(y_true, y_pred)
        self.logger.info('Perf Routine Confusion Matrix:')
        self.logger.info(f'\n{cm}')
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        self.logger.info(f'Accuracy: {accuracy}')
        # Precision
        precision = precision_score(y_true, y_pred, average=None)
        self.logger.info(f'Precision: {precision}')
        # Recall
        sensitivity = recall_score(y_true, y_pred, average=None)
        self.logger.info(f'Sensitivity: {sensitivity}')
        # F1-Score
        f1 = f1_score(y_true, y_pred, average=None)

        #labels = ['A41.9', 'I21.4', 'J96.00', 'N17.9']  # replaced with classes_
        # Binarize ytest with shape (n_samples, n_classes)
        y_true_bin = label_binarize(y_true, classes=self.classes_)
        
                
        # ✅ Calculate ROC-AUC using probabilities (already normalized from vote_score)
        try:
            y_true_bin = label_binarize(y_true, classes=self.classes_)
            self.logger.debug(f'y_true_bin shape: {y_true_bin.shape}')
        
            # Verify shapes match
            if y_true_bin.shape[0] != probs.shape[0]:
                raise ValueError(f"Shape mismatch: y_true_bin {y_true_bin.shape} vs probs {probs.shape}")
        
            if y_true_bin.shape[1] != probs.shape[1]:
                raise ValueError(f"Number of classes mismatch: y_true_bin has {y_true_bin.shape[1]} classes, probs has {probs.shape[1]}")
            
            roc_auc = roc_auc_score(y_true_bin, probs, average=None, multi_class='ovr')
            self.logger.info(f'ROC-AUC per class: {roc_auc}')
            self.logger.info(f'ROC-AUC macro average: {roc_auc.mean():.4f}')
        except Exception as e:
            self.logger.warning(f'Could not calculate ROC-AUC: {e}')
            roc_auc = 0
            raise
    
        return(accuracy, precision, sensitivity, f1, roc_auc, cm)
        

    def evaluate2(self, test_df: pd.DataFrame, verbose: bool = False) -> Dict:
        """
        Predict and evaluate model performance.
        
        Returns:
        - Dictionary with evaluation metrics
        """
        # Get predictions
        self.logger.info('Beginning Evaluate')
        results_df = self.predict(test_df, verbose=verbose)
        
        #Skinny down results to just needed details for perf measurement
        details_df = results_df[['id', 'true_icd10_code', 'predicted_icd10_code', 'prob_class_0','prob_class_1',
                                    'prob_class_2','prob_class_3']].copy()
        new_cols = ['adm_id', 'actuals', 'predictions', 'class_1_score', 'class_2_score', 
                        'class_3_score', 'class_4_score']
        details_df.columns = new_cols
        
        actuals, pred_labels, probs = self.vote_score(details_df)
        
        self.logger.info(f'preds: {len(pred_labels)}, actuals: {len(actuals)}')
    
        accuracy, precision, sensitivity, f1, roc_auc, cm = self.get_perf_data(actuals, pred_labels, probs)
    
    
        return 
    
    def save_results(self, results_df: pd.DataFrame, filepath: str):
        """Save prediction results to CSV."""
        results_df.to_csv(filepath, index=False)
        self.logger.info(f"Results saved to {filepath}")

class RAGFewShotDiagnosisClassifier(FewShotDiagnosisClassifier):
    """
    RAG-enhanced diagnosis classifier.
    Retrieves relevant examples dynamically instead of using fixed examples.
    """
    
    import gc
    import torch

    def __init__(self, gemma_model, rag_retriever, icd10_csv_path,
                 example_size, test_size, logger=None, 
                 k_examples=1, balanced_retrieval=False, embedding_model=None):
        """
        Initialize RAG classifier.
        
        Args:
            gemma_model: Gemma model wrapper
            rag_retriever: MedicalRAGRetriever instance
            icd10_descriptions: Optional ICD code descriptions
            logger: Logger instance
            balanced_retrieval: equal number of examplkes from all diagnoses
        """
        super().__init__(
            gemma_model=gemma_model,
            example_size=example_size, 
            test_size=test_size, 
            k_examples=k_examples,
            icd10_csv_path=icd10_csv_path,
            logger=logger,
            embedding_model = embedding_model
            )
        
        self.rag_retriever = rag_retriever
        self.use_rag = True
        self.balanced_retrieval = balanced_retrieval
        self.logger.info(f"Use Rag: {self.use_rag}")
        self.logger.info(f"Balanced Rag: {self.balanced_retrieval}")
        return       
        
    def fit(self, examples_df: pd.DataFrame):
        """
        Fit by building RAG index instead of storing fixed examples.
        """

        self.logger.info(f"Fit received df with {len(examples_df)} rows")
        word_counts = examples_df['text'].apply(lambda x: len(x.split()))
        self.logger.info(f"Word counts: min={word_counts.min()}, max={word_counts.max()}, avg={word_counts.mean():.0f}")
        
        self.logger.info(f"At Begginning of Fit: {examples_df[0:3][['id', 'icd10_code', 'text']].to_string(index=False)}")
        self.examples_df = examples_df.copy()
        self.classes_ = sorted(examples_df['icd10_code'].unique())
        
         # DEBUG: Check what we're building index with
        self.logger.info(f"=== RAG FIT DEBUG ===")
        self.logger.info(f"examples_df shape: {examples_df.shape}")
        self.logger.info(f"examples_df columns: {examples_df.columns.tolist()}")
        self.logger.info(f"classes_: {self.classes_}")
        self.logger.info(f"Number of examples per class:")
        self.logger.info(examples_df['icd10_code'].value_counts())
    
        if len(examples_df) == 0:
            raise ValueError("Cannot fit RAG classifier with 0 examples!")
    
        if len(examples_df) < self.k_examples * len(self.classes_):
            self.logger.warning(
                f"Only {len(examples_df)} examples for {len(self.classes_)} classes "
                f"and k={self.k_examples}. May not have enough examples!"
        )
        # Build RAG index
        self.logger.info("Building RAG index from training examples...")
        self.rag_retriever.build_index(
            examples_df,
            text_column='text',
            save_path='rag_index'
        )

        # Warm up - force index and embedding model into RAM
        # Use first training record as dummy query
        warmup_text = self.examples_df.iloc[0]['text']
        self.logger.info("Warming up RAG index...")
        _ = self.rag_retriever.retrieve(warmup_text, k=1)
        self.logger.info("RAG warmup complete")
        
        self.logger.info(f"RAG classifier fitted with {len(self.examples_df)} examples")
        self.logger.info(f"Classes: {self.classes_}")
        
        return self
    
     
    def _create_prompt_with_rag(self, patient_text: str, patient_id=None) -> str:
        """
        Create prompt with dynamically retrieved examples.
    
        Args:
            patient_text: Patient note to classify
            patient_id: Optional patient ID to exclude from retrieval
        """
        # Retrieve similar examples
        # Retrieve similar examples for EACH diagnosis class
        
        start = time.time()
        if self.balanced_retrieval:
            all_retrieved_examples = []
            for diagnosis_code in self.classes_:
                # Retrieve k examples for this specific diagnosis
                examples_for_diagnosis = self.rag_retriever.retrieve(
                                            patient_text,
                                            k=self.k_examples,
                                            exclude_same_id=patient_id,
                                            filter_diagnosis=diagnosis_code  # You'll need to add this parameter
                                    )
        
                all_retrieved_examples.append(examples_for_diagnosis)
                #free up memory resources after each retrieval to prevent OOM 
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
            retrieved_examples = pd.concat(all_retrieved_examples, ignore_index=True)
    
            self.logger.debug(f"Retrieved {len(retrieved_examples)} examples ({self.k_examples} per diagnosis)")
        
        else:

            # Retrieve k examples for this specific diagnosis
            retrieved_examples = self.rag_retriever.retrieve(
                                            patient_text,
                                            k=self.k_examples * len(self.classes_),
                                            exclude_same_id=patient_id
                                    )

            self.logger.debug(f"Retrieved {len(retrieved_examples)} top 4 examples")
        
        run_time = time.time() - start
        self.logger.info(f"Completed rag retrieval Time: {run_time:.2f}s")
        # Generate diagnosis options dynamically
        diagnosis_section = self.format_diagnosis_options()
        #self.logger.info('Diagnostic Section')
        #self.logger.info(diagnosis_section)

        # Build prompt with retrieved examples
        prompt = f"""<start_of_turn>user

        You are an experienced attending physician reviewing clinical notes to make a diagnostic assessment.

        {diagnosis_section}

        CRITICAL INSTRUCTIONS:
        - Respond with ONLY a single digit: 0, 1, 2, or 3
        - Do not include the diagnosis code (e.g., A41.9)
        - Do not add any explanation, punctuation, spaces, or newlines
        - Output format: Just the digit, nothing else

        Here are some example cases:

    """
    
        # Add retrieved examples
        #for idx, row in retrieved_examples.iterrows():
            #example_num = idx + 1
        
        # Calculate available budget for examples
        MAX_TOTAL_CHARS = 28000
        overhead = 1200
        note_chars = len(patient_text)
        available_for_examples = MAX_TOTAL_CHARS - overhead - note_chars

        # Cap each example proportionally
        max_chars_per_example = available_for_examples // len(retrieved_examples)
        max_words_per_example = max_chars_per_example // 8  # conservative for medical text

        self.logger.info(f"Example budget: {max_chars_per_example} chars / {max_words_per_example} words each")

        for example_num, (_, row) in enumerate(retrieved_examples.iterrows(), start=1):
            code_idx = self.classes_.index(row['icd10_code'])
            example_text = ' '.join(row['text'].split()[:max_words_per_example])
            prompt += f"""Example {example_num}:
            Clinical Note: {example_text}...
            Diagnosis: {code_idx}

    """
          
        # Add new patient case
        prompt += "Now, review the following new patient case:\n\n"
        if patient_id:
            prompt += f"Patient ID: {patient_id}\n"
        prompt += f"Clinical Note: {patient_text}\n\n"
        prompt += "Diagnosis (respond with only the number 0, 1, 2, or 3):<end_of_turn>"
        prompt += "\n<start_of_turn>model"
        
        return prompt
    
    def predict_single(self, patient_text, patient_id=None) -> Dict:
        """
        Predict with RAG-retrieved examples.
        
        Args:
            patient_text: Clinical note
            patient_id: Optional patient ID
        """
        if self.logger:
            self.logger.info(f"Processing ID: {round(patient_id) if patient_id else 'unknown'}")
        
        if self.examples_df is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        start = time.time()
        # Generate prompt with RAG
        prompt = self._create_prompt_with_rag(patient_text, patient_id)
        self.analyze_prompt_composition2(prompt, patient_id)
        
        # Get prediction
        result = self.gemma_model.generate_with_probabilities(
            prompt,
            self.classes_,
            max_tokens=5
        )
        
        # Handle CUDA failure - skip this subsequence
        if result is None:
            self.logger.warning(f"Skipping patient {patient_id} due to CUDA failure")
            return {
            'prediction': None,
            'probabilities': None,
            'confidence': None,
            'raw_response': None,
            'prompt': prompt
            }
        
        # Extract probabilities
        probabilities = self._extract_class_probabilities2(result)
        
        if probabilities is not None:
            max_prob_idx = probabilities.index(max(probabilities))
            prediction = self.classes_[max_prob_idx]
            confidence = probabilities[max_prob_idx]
            self.logger.info(f"Predicted class {max_prob_idx}: {prediction} with confidence {confidence:.4f}")
        else:
            self.logger.warning("Probabilities is None, using fallback parsing")
            prediction = self._parse_response(result.get('text', ''))
            confidence = None
        
        run_time = time.time() - start
        self.logger.info(f"Completed prediction Time: {run_time:.2f}s")
        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'confidence': confidence,
            'raw_response': result,
            'prompt': prompt
        }