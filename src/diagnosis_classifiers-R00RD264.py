import math
from datetime import datetime
from typing import Dict, List, Optional

# Third-party imports
import pandas as pd
import numpy as np

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
from sklearn.preprocessing import label_binarize

class FewShotDiagnosisClassifier:
    """
    Few-shot learning classifier for ICD-10 diagnosis prediction using LLM.
    a) __init__(self, gemma_model, icd10_descriptions: Optional[Dict[str, str]] = None)</li> 
    b) fit(self, examples_df: pd.DataFrame)</li>
    c) _create_prompt(self, patient_text: str, patient_id: Optional[str] = None) -> str</li>   
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
    
    def __init__(self, gemma_model, icd10_csv_path=None, logger=None):
        """
        Initialize the classifier.
        
        Parameters:
        - gemma_model: The Gemma model instance for generating predictions
        - icd10_descriptions: Optional dict mapping ICD10 codes to descriptions
        """
        self.gemma_model = gemma_model
        self.logger = logger
        self.examples_df = None
        self.classes_ = None

        # Load ICD-10 reference DataFrame
        if icd10_csv_path:
            self.icd10_df = pd.read_csv(icd10_csv_path)
            if self.logger:
                self.logger.info(f"Loaded {len(self.icd10_df)} ICD-10 codes from {icd10_csv_path}")
        else:
            self.icd10_df = None
            if self.logger:
                self.logger.warning("No ICD-10 CSV provided, will use hardcoded descriptions")
        
    def fit(self, examples_df: pd.DataFrame, logger):
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
        
        logger.info(f"Model fitted with {len(self.examples_df)} examples")
        logger.info(f"Classes: {self.classes_}")
        
        return self
    
    def format_diagnosis_options(self):
        """
        Format diagnosis options for the prompt with class indices.
        Uses self.classes_ and self.icd10_df to generate the formatted text.
    
        Returns:
            Formatted string for the prompt
        """
        lines = ["Based on the patient's clinical presentation, predict which diagnosis class applies:\n"]
    
        for idx, code in enumerate(self.classes_):
            if self.icd10_df is not None:
                # Look up description from DataFrame
                match = self.icd10_df[self.icd10_df['icd10_code'] == code]
                if len(match) > 0:
                    desc = match['description'].values[0]
                    lines.append(f"                    {idx} = {code} ({desc})")
                else:
                    lines.append(f"                    {idx} = {code} (description not found)")
                    if self.logger:
                        self.logger.warning(f"Description not found for code: {code}")
            else:
                # Fallback to hardcoded descriptions
                hardcoded = {
                    'A41.9': 'Sepsis, unspecified organism',
                    'I21.4': 'Acute subendocardial myocardial infarction',
                    'J96.00': 'Acute respiratory failure, unspecified',
                    'N17.9': 'Acute kidney failure, unspecified'
                }
            
                desc = hardcoded.get(code, 'description not available')
                lines.append(f"                    {idx} = {code} ({desc})")
    
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
    
    def _create_prompt(self, patient_text: str, patient_id: Optional[str] = None) -> str:
        """
        Create a few-shot prompt for a single patient.
        """
        if self.examples_df is None:
            raise ValueError("Model not fitted. Call fit() first.")
    
        # Generate diagnosis options dynamically
        diagnosis_section = self.format_diagnosis_options()
    
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
    
        # Add new patient case
        prompt += "Now, review the following new patient case:\n\n"
        if patient_id:
            prompt += f"Patient ID: {patient_id}\n"
            prompt += f"Clinical Note: {patient_text}\n\n"
            prompt += "Diagnosis (respond with only the number 0, 1, 2, or 3):<end_of_turn>"
            prompt += "\n<start_of_turn>model"

        return prompt
    
    def _parse_response(self, response: str, logger) -> str:
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
    
   
    def _extract_class_probabilities2(self, output: Dict, logger) -> List[float]:
        """
        Extract probability distribution for each class from model output.
        Searches through all token positions to find the one with digit classes.
        """
        import math
    
        try:
            if not output or 'choices' not in output:
                logger.warning(f"No valid output")
                return [0.25] * 4
        
            choice = output['choices'][0]
            logprobs_data = choice.get('logprobs', {})
            
            # ← NEW: Get from 'content' array first        # changed 12/3/2026 11:46 AM
            content_list = logprobs_data.get('content', [])
        
            if not content_list:  # ← NEW: Check if content exists      # changed 12/3/2026 11:46 AM
                logger.warning(f"No content in logprobs")
                return [0.25] * 4
        
            # ← CHANGED: Get top_logprobs from content[0] instead of directly from logprobs_data
            top_logprobs_list = content_list[0].get('top_logprobs', [])      # changed 12/3/2026 11:46 AM
            #top_logprobs_list = logprobs_data.get('top_logprobs', [])       # changed 12/3/2026 11:46 AM
        
            if not top_logprobs_list:
                logger.warning(f"No top_logprobs found")
                return [0.25] * 4
        
            #print(f"🔍 Searching through {len(top_logprobs_list)} token positions")    # changed 12/3/2026 11:59 AM
            logger.debug(f"Found {len(top_logprobs_list)} tokens in top_logprobs")          # changed 12/3/2026 11:59 AM
        
            # ← NEW CODE BLOCK: Build a dict for easy lookup       # changed 12/3/2026 11:59 AM
            token_dict = {}
            for item in top_logprobs_list:
                token_dict[item['token']] = item['logprob']
        
            logger.debug(f"Available tokens: {list(token_dict.keys())[:10]}")
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
                logger.info(f"[OK] Normalized probabilities: {[f'{p:.4f}' for p in probs]}")
            else:
                probs = [0.25] * 4
                logger.warning(f"Total probability was 0, using uniform distribution")
                
            return probs
        
            # If we get here, we didn't find a position with digit classes       # Section removed 12/3/2026 11:59 AM
            # print(f"[WARNING] Could not find token position with digit classes")
            #print(f"[WARNING] Available tokens at each position:")
            #for idx, top_logprobs in enumerate(top_logprobs_list):
            #    print(f"   Position {idx}: {list(top_logprobs.keys())[:5]}")
            #        
            #return [0.25] * 4
        
        except Exception as e:
            logger.exception(f"Error during generation with probabilities: {e}")            
            #return [0.25] * 4
            raise

    
    def predict_single(self, patient_text: str, logger = None, patient_id: Optional[str] = None, **kwargs) -> Dict:
        """
        Predict diagnosis for a single patient with class probabilities.
    
        Returns:
        - Dictionary with 'prediction', 'probabilities', 'prompt', and 'raw_response'
        """
        logger.info(f"Processing ID: {round(patient_id)}")
        
        if self.examples_df is None:
            raise ValueError("Model not fitted. Call fit() first.")
    
        # Generate prompt - modify to encourage single-token response
        prompt = self._create_prompt(patient_text, patient_id)
    
        # Get prediction WITH PROBABILITIES from Gemma
        result = self.gemma_model.generate_with_probabilities(prompt,
                                                              self.classes_,
                                                              max_tokens=5,      # ← Explicitly set this
                                                              temperature=0.7,    # ← And this
                                                              logger=logger  
                                                             )
    
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
        probabilities = self._extract_class_probabilities2(result, logger)
        
        # ← ADD DEBUG HERE                                          #Added 12/3/2025 12:25 PM
        logger.debug(f"Probabilities = {probabilities}")
        logger.debug(f"Probabilities type = {type(probabilities)}")
        logger.debug(f"Probabilities is None? {probabilities is None}")
    
        
        # Get top prediction
        # if probabilities:              # Changed 12/3/2025 12:25 PM   
        if probabilities is not None:    # Changed 12/3/2025 12:25 PM   
            max_prob_idx = probabilities.index(max(probabilities))
            prediction = self.classes_[max_prob_idx]
            confidence = probabilities[max_prob_idx]  
            logger.info(f"[OK] Predicted class {max_prob_idx}: {prediction} with confidence {confidence:.4f}") # Changed 12/3/2025 12:25 PM   
        else:
            logger.warning(f"Probabilities is None, using fallback parsing")     # Changed 12/3/2025 12:25 PM   
            prediction = self._parse_response(result.get('text', ''))
            confidence = None  # ← Set here

        return {'prediction': prediction,
                'probabilities': probabilities,
                'confidence': confidence,  # ← Use the variable
                'prompt': prompt,
                'raw_response': result
                }

    
    def predict(self, test_df, verbose=False, logger=None) -> pd.DataFrame:
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
        
        #f = open('E:/Education/CCSU-Thesis-2024/Data/logs/prompts_output.txt', 'a', encoding='utf-8')
        predictions = []
        
        for idx, row in test_df.iterrows():
            logger.info(f"Handling subsequence: {idx}")
            if verbose and (idx + 1) % 10 == 0:
                logger.info(f"Processing patient {idx + 1}/{len(test_df)}...")
            
            # Get prediction
            result = self.predict_single(
                            patient_text=row['text'],
                            logger=logger,
                            patient_id=row['id']
                    )
            #f.write(f"{result['prompt']}\n")
            probs = result['probabilities']  # Add this line to define probs
            
            # Predict with probabilities
            logger.info(f"Prediction: {result['prediction']}")
            logger.info(f"Confidence: {result['confidence']:.2%}")
            logger.info("\nClass Probabilities:")
            
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
            logger.info(f"\nCompleted predictions for {len(test_df)} patients")
        
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
    
    def vote_score(self, details_df, logger = None):

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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
        logger.info("Sample normalized scores:")
        logger.info(f"\n{temp_scores[score_cols].head()}")
        logger.info(f"Row sums after normalization: {temp_scores[score_cols].sum(axis=1).head()}")
        
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
        
        logger.debug(f"Vote scores:\n{temp_scores}")
        out_file = f'sorted_out_scores_{timestamp}.csv'
        sorted_out_scores.to_csv(out_file, index=False)

        return (actuals, pred_labels, probs)


    def get_perf_data(self, y_true, y_pred, probs, logger = None):

        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
        from sklearn.metrics import precision_score, recall_score, f1_score,roc_curve, auc, roc_auc_score
        from sklearn.preprocessing import label_binarize

        #print('get perf data input')
        logger.debug(f'perf y_true:\n{y_true[0:5]}')
        logger.debug(f'perf y_pred:\n{y_pred[0:5]}')
        logger.debug('Generating Performance Data')
        
        cm = confusion_matrix(y_true, y_pred)
        logger.info('Perf Routine Confusion Matrix:')
        logger.info(f'\n{cm}')
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        logger.info(f'Accuracy: {accuracy}')
        # Precision
        precision = precision_score(y_true, y_pred, average=None)
        logger.info(f'Precision: {precision}')
        # Recall
        sensitivity = recall_score(y_true, y_pred, average=None)
        logger.info(f'Sensitivity: {sensitivity}')
        # F1-Score
        f1 = f1_score(y_true, y_pred, average=None)

        #labels = ['A41.9', 'I21.4', 'J96.00', 'N17.9']  # replaced with classes_
        # Binarize ytest with shape (n_samples, n_classes)
        y_true_bin = label_binarize(y_true, classes=self.classes_)
        
                
        # ✅ Calculate ROC-AUC using probabilities (already normalized from vote_score)
        try:
            y_true_bin = label_binarize(y_true, classes=self.classes_)
            logger.debug(f'y_true_bin shape: {y_true_bin.shape}')
        
            # Verify shapes match
            if y_true_bin.shape[0] != probs.shape[0]:
                raise ValueError(f"Shape mismatch: y_true_bin {y_true_bin.shape} vs probs {probs.shape}")
        
            if y_true_bin.shape[1] != probs.shape[1]:
                raise ValueError(f"Number of classes mismatch: y_true_bin has {y_true_bin.shape[1]} classes, probs has {probs.shape[1]}")
            
            roc_auc = roc_auc_score(y_true_bin, probs, average=None, multi_class='ovr')
            logger.info(f'ROC-AUC per class: {roc_auc}')
            logger.info(f'ROC-AUC macro average: {roc_auc.mean():.4f}')
        except Exception as e:
            logger.warning(f'Could not calculate ROC-AUC: {e}')
            roc_auc = 0
            raise
    
        return(accuracy, precision, sensitivity, f1, roc_auc, cm)
        

    def evaluate2(self, test_df: pd.DataFrame, verbose: bool = False, logger = None) -> Dict:
        """
        Predict and evaluate model performance.
        
        Returns:
        - Dictionary with evaluation metrics
        """
        # Get predictions
        logger.info('Beginning Evaluate')
        results_df = self.predict(test_df, verbose=verbose, logger=logger)
        
        #Skinny down results to just needed details for perf measurement
        details_df = results_df[['id', 'true_icd10_code', 'predicted_icd10_code', 'prob_class_0','prob_class_1',
                                    'prob_class_2','prob_class_3']].copy()
        new_cols = ['adm_id', 'actuals', 'predictions', 'class_1_score', 'class_2_score', 
                        'class_3_score', 'class_4_score']
        details_df.columns = new_cols
        
        actuals, pred_labels, probs = self.vote_score(details_df, logger)
        
        logger.info(f'preds: {len(pred_labels)}, actuals: {len(actuals)}')
    
        accuracy, precision, sensitivity, f1, roc_auc, cm = self.get_perf_data(actuals, pred_labels, probs, logger)
    
    
        return 
    
    def save_results(self, results_df: pd.DataFrame, filepath: str, logger):
        """Save prediction results to CSV."""
        results_df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")

class RAGFewShotDiagnosisClassifier(FewShotDiagnosisClassifier):
    """
    RAG-enhanced diagnosis classifier.
    Retrieves relevant examples dynamically instead of using fixed examples.
    """
    
    def __init__(self, gemma_model, rag_retriever, icd10_csv_path, logger=None, k_examples=1, balanced_retrieval=False):
        """
        Initialize RAG classifier.
        
        Args:
            gemma_model: Gemma model wrapper
            rag_retriever: MedicalRAGRetriever instance
            icd10_descriptions: Optional ICD code descriptions
            logger: Logger instance
            balanced_retrieval: do I want examples for all classes
        """
        super().__init__(
            gemma_model=gemma_model,
            icd10_csv_path=icd10_csv_path,  
            logger=logger
            )
        
        self.rag_retriever = rag_retriever
        self.use_rag = True
        self.k_examples=k_examples
        self.balanced_retrieval = balanced_retrieval 
        return       
        
    def fit(self, examples_df: pd.DataFrame):
        """
        Fit by building RAG index instead of storing fixed examples.
        """
        self.examples_df = examples_df.copy()
        self.classes_ = sorted(examples_df['icd10_code'].unique())
        
        # Build RAG index
        self.logger.info("Building RAG index from training examples...")
        self.rag_retriever.build_index(
            examples_df,
            text_column='text',
            save_path='rag_index'
        )
        
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
    
            retrieved_examples = pd.concat(all_retrieved_examples, ignore_index=True)
    
            self.logger.debug(f"Retrieved {len(retrieved_examples)} examples ({self.k_examples} per diagnosis)")
        
        else:
            # Retrieve 4 most similar no matter the diagnosis (unbalanced)    
            retrieved_examples = self.rag_retriever.retrieve(
                                        patient_text,
                                        k=self.k_examples * len(self.classes_),
                                        exclude_same_id=patient_id
                                    )

            example_count = self.k_examples * len(self.classes_),
            self.logger.debug(f"Retrieved {len(retrieved_examples)} unbalanced examples ({example_count} )")
        
        # Generate diagnosis options dynamically
        diagnosis_section = self.format_diagnosis_options()
    
        # Build prompt with retrieved examples
        prompt = f"""<start_of_turn>user

        You are an experienced attending physician reviewing clinical notes to make a diagnostic assessment.

        {diagnosis_section}

    CRITICAL INSTRUCTIONS:
    - Respond with ONLY a single digit: 0, 1, 2, or 3
    - Do not include the diagnosis code (e.g., A41.9)
    - Do not add any explanation, punctuation, spaces, or newlines
    - Output format: Just the digit, nothing else

    Here are some similar example cases:

"""
    
        # Add retrieved examples
        for idx, row in retrieved_examples.iterrows():
            example_num = idx + 1
            code_idx = self.classes_.index(row['icd10_code'])
        
            prompt += f"""Example {example_num}:
    Clinical Note: {row['text'][:500]}...
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
    
    def predict_single(self, patient_text, logger=None, patient_id=None) -> Dict:
        """
        Predict with RAG-retrieved examples.
        
        Args:
            patient_text: Clinical note
            logger: log object
            patient_id: Optional patient ID
        """
        if logger:
            logger.info(f"Processing ID: {round(patient_id) if patient_id else 'unknown'}")
        
        if self.examples_df is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Generate prompt with RAG
        prompt = self._create_prompt_with_rag(patient_text, patient_id)
        
        # Get prediction
        result = self.gemma_model.generate_with_probabilities(
            prompt,
            self.classes_,
            max_tokens=5,
            temperature=0.7,
            logger=logger
        )
        
        # Extract probabilities
        probabilities = self._extract_class_probabilities2(result, logger)
        
        if probabilities is not None:
            max_prob_idx = probabilities.index(max(probabilities))
            prediction = self.classes_[max_prob_idx]
            confidence = probabilities[max_prob_idx]
            self.logger.info(f"Predicted class {max_prob_idx}: {prediction} with confidence {confidence:.4f}")
        else:
            self.logger.warning("Probabilities is None, using fallback parsing")
            prediction = self._parse_response(result.get('text', ''))
            confidence = None

        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'confidence': confidence,
            'raw_response': result
        }