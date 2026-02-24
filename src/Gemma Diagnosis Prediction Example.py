# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python [conda env:research]
#     language: python
#     name: conda-env-research-py
# ---

# %% [markdown]
# MIMIC-III Diagnosis Code Prediction using Local Gemma via LM Studio
# This notebook uses LM Studio's local API to predict diagnosis codes from patient notes

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <p style="margin-left: 25px;">
#     <b><font size="5">Setup</font></b></p><br>
#     <font size="3">
#          <ol>
#           <li>Load Libraries</li> 
#           <li>Find GGUF Gemma Models</li>
#           <li>Check llama_cpp version</li>
#          </ol> </font>
# </div> 

# %%
# MIMIC-III Diagnosis Code Prediction using Local Gemma via LM Studio
# This notebook uses llama's local API to predict diagnosis codes from patient notes

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import requests
import json
from typing import List, Dict
import time
from tqdm import tqdm
import sys
from datetime import datetime

sys.path.append('E:/VSCode-Projects/Thesis')
import BERT_Diagnosis as BD
from pathlib import Path

import sys
log_file = open('E:\output.log', 'w')
sys.stdout = log_file


# Find all models downloaded by LM Studio
lm_studio_path = Path("E:\Education\llama\models")

print("Models found in LM Studio:")
for gguf_file in lm_studio_path.rglob("*.gguf"):
    size_gb = gguf_file.stat().st_size / (1024**3)
    print(f"\n{gguf_file.name}")
    print(f"  Path: {gguf_file}")
    print(f"  Size: {size_gb:.2f} GB")

import llama_cpp
print(f"llama-cpp-python version: {llama_cpp.__version__}")


# %%
# Create print log filename with date and time
def start_logging(type): 
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f'{type}_output_{timestamp}.log'
    print(f"Starting logging to: {log_filename}")
    log_file = open(log_filename, 'w', encoding='utf-8')
    sys.stdout = log_file
    return log_file, log_filename



# %% [markdown]
# <div class="alert alert-block alert-info">
#     <p style="margin-left: 25px;">
#     <b><font size="5">Version Control</font></b></p><br>
#     <font size="3">
#          <ol>
#           <li>Move Code from Jupyter Directory to Github Directory</li> 
#           <li>Need to run git commands from command line</li>
#          </ol> </font>
# </div>

# %%
# Version Control
# Copy jupyter notebook to github directory
# Git commands need to be executed from cpmmand line:
#   git add "Gemma Diagnosis Prediction Example.ipynb"
#   git add "Gemma Diagnosis Prediction Example.py"
#   git commit -m "commit comment"
from pathlib import Path
import shutil
from IPython.display import Javascript, display
display(Javascript('Jupyter.notebook.save_checkpoint();'))


# !jupytext --set-formats ipynb,py:percent "Gemma Diagnosis Prediction Example.ipynb"
# !jupytext --sync "Gemma Diagnosis Prediction Example.ipynb" 
source_path = Path("E:/Jupyter/notebooks/Generative AI Diagnosis Prediction/")
destination_path = Path("E:/Education/GitHub/Generative-MIMIC-Diagnosis/")

base_name = "Gemma Diagnosis Prediction Example"

# Copy all matching files (handles spaces in filename)
copied_count = 0
for file in source_path.iterdir():
    # Check if filename (without extension) matches
    if file.stem == base_name:
        dest_file = destination_path / file.name
        shutil.copy2(file, dest_file)
        print(f"[OK] Copied: {file.name} ({file.suffix})")
        copied_count += 1

print(f"\n[OK] Total files copied: {copied_count}")

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <p style="margin-left: 25px;">
#     <b><font size="5">Check Available Memory</font></b></p><br>
#     <font size="3">
#          <ol>
#           <li>Model Requires at Leat 3G</li> 
#          </ol> </font>
# </div>

# %%
import gc
import psutil

# Check available memory
mem = psutil.virtual_memory()
print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
print(f"Free RAM: {mem.free / (1024**3):.2f} GB")

# If less than 3 GB available, you'll have issues
if mem.available < 3 * 1024**3:
    print("WARNING: Less than 3GB RAM available. Close other programs.")

# Force garbage collection
gc.collect()


# %%
from llama_cpp import Llama
import traceback

MODEL_PATH = r"E:/Education/llama/models/gemma-3n-E4B-it-Q4_K_M.gguf"

try:
    print("Attempting to load model...")
    model = Llama(
        model_path=MODEL_PATH,
        n_ctx=2046,  # Start with smaller context
        n_threads=4,  # Use fewer threads
        logits_all=True,
        verbose=False,  # Do not see detailed loading info
        n_gpu_layers=0  # Force CPU to rule out GPU issues
    )
    print(f"[OK] Model loaded successfully!")
    
    # Test generation
    response = model("Hello", max_tokens=1)
    print(f"[OK] Test generation works")
    
except Exception as e:
    print(f"[ERROR] Error loading model:")
    print(traceback.format_exc())

# %%
##########################################
# LM_STUDIO Configuration
LM_STUDIO_URL = "http://localhost:1234/v1"  # Default LM Studio API endpoint
#MODEL_NAME = "gemma-2-2b-it"  # Adjust to your loaded model name
MODEL_NAME = "google/gemma-3n-e4b"  # Adjust to your loaded model name
MAX_PROMPT_CHARS = 2000  # Safety limit
MAX_PATIENT_TEXT_CHARS = 2000
##########################################


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <p style="margin-left: 25px;">
#     <b><font size="5">Load MIMIC data</font></b></p><br>
#     <font size="3">
#          <ol>
#           <li>Read Training Data</li> 
#           <li>Random Select one Training Admission for each Diagnosis</li>
#           <li>Create 1000 Word Subsequences for each training admission copying diagnosis to each subsequence</li>   
#           <li>Read Test Data</li>
#           <li>Random Select two Test Admissions for each Diagnosis</li>
#           <li>Create 1000 Word Subsequences for each test admission copying diagnosis to each subsequence</li>
#           <li>Print length of all admission subsequences</li>   
#          </ol> </font>
# </div>

# %%
# Load MIMIC-III data
print("Loading MIMIC-III data from parquet file...")
path = 'E:/Education/CCSU-Thesis-2024/Data/'
train_df_in = pd.read_parquet(path + 'intsl_train_group_full_no2c.snappy.parquet')  # Replace with your file path

# Display data info
print(f"\nDataset shape: {train_df_in.shape}")
print(f"Columns: {train_df_in.columns.tolist()}")
print("\nFirst few rows:")
print(train_df_in.head(2))
print('unique ids:', train_df_in['id'].nunique())
print(train_df_in.groupby('icd10_code')['id'].nunique())
# get one of each diagnosis for for shot training
train_sample_df = train_df_in.groupby('icd10_code', group_keys=False).apply(lambda x: x.sample(n=1))
# split admissions of notes subseqeunces of 1000 characters
train_sample_df2 = BD.create_subsequences(train_sample_df, 1000)
print(train_sample_df.shape, train_sample_df2.shape)


test_df_in = pd.read_parquet(path + 'intsl_test_group_full_no2c.snappy.parquet')  # Replace with your file path
# Display data info of test data
print(f"\nDataset shape: {test_df_in.shape}") 
print(f"Columns: {test_df_in.columns.tolist()}")
print("\nFirst few rows:")
print(test_df_in.head(2))
print('unique ids:', test_df_in['id'].nunique())
print(test_df_in.groupby('icd10_code')['id'].nunique())
# get five of each diagnosis for prediction
test_sample_df = test_df_in.groupby('icd10_code', group_keys=False).apply(lambda x: x.sample(n=12))
# split admissions of notes subseqeunces of 1000 characters
test_sample_df2 = BD.create_subsequences(test_sample_df, 1000)
print(test_sample_df.shape,test_sample_df2.shape)


# print sizes of each note in train
# print(test_sample_df2['text'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0))
# print sizes of each note in test
test_sample_df2['text'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <p style="margin-left: 25px;">
#     <b><font size="5">LlamaCppWrapper Class</font></b></p><br>
#     <font size="3">
#          <ol>
#           <li>__init__</li> 
#           <li>generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.1) -> str</li>
#           <li>generate_with_probabilities(self, prompt: str, classes: list, max_tokens: int = 10, 
#                                           temperature: float = 0.7)</li>   
#           <li>_convert_to_openai_format(self, llama_output: Dict) -> Dict</li>
#          </ol> </font>
# </div>

# %%
class LlamaCppWrapper:
    """Wrapper for llama-cpp-python with logprobs support - replaces LM Studio."""
    
    def __init__(self, 
                 model_path: str, 
                 n_ctx: int = 4096, 
                 n_threads: int = None, 
                 n_batch: int = 512,          
                 use_mmap: bool = True):
        """
        Initialize llama.cpp model.
        
        Args:
            model_path: Path to GGUF model file (e.g., "C:/models/gemma-2-9b-it-Q4_K_M.gguf")
            n_ctx: Context window size
            n_threads: Number of CPU threads (None = auto-detect)
        """
        # Auto-detect optimal thread count if not specified
        if n_threads is None:
            cpu_count = os.cpu_count() or 4
            n_threads = max(1, cpu_count // 2)  # Use half of available cores
        
        print(f"Loading model from: {model_path}")
        print(f"Context size: {n_ctx}")
        print(f"Using {n_threads} CPU threads")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            use_mmap=use_mmap,
            logits_all=True,
            n_gpu_layers=0,  # CPU only (change if you add GPU later)
            verbose=False
        )
        
        self.model_name = os.path.basename(model_path)
        print(f"[OK] Model '{self.model_name}' loaded successfully")
        #print(f"✓ Model '{self.model_name}' loaded successfully")
    
    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.1) -> str:
        """
        Generate response without probabilities.
        Compatible with your existing code.
        """
        print(f"[INFO] Prompt size: {len(prompt)} chars (~{len(prompt)//4} tokens)")
        
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                #stop=["\n\n", "###", "Example", "Patient Note:"],
                echo=False
            )
            
            if 'choices' in output and len(output['choices']) > 0:
                return output['choices'][0]['text'].strip()
            
            return "ERROR"
            
        except Exception as e:
            print(f"[WARNING] Error during generation: {e}")
            return "ERROR"
    
    def generate_with_probabilities(self, prompt: str, classes: list, 
                                    max_tokens: int = 10, temperature: float = 0.7):
        """
        Generate response WITH probabilities (logprobs).
        This is what enables true probability distributions for classification!
        """
        print(f"[DEBUG] Requesting logprobs for {len(classes)} classes")
        
        try:
            
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=10,  # Get top 20 token probabilities
                #stop=["\n\n", "###", "Example"],
                echo=False
            )
            
            # ADD THESE DEBUG LINES RIGHT HERE:
            print(f"[DEBUG]: output keys = {output.keys()}")
            print(f"[DEBUG]: choices = {output.get('choices', [])}")
            if output.get('choices'):
                choice = output['choices'][0]
                print(f"[DEBUG]: choice keys = {choice.keys()}")
                print(f"[DEBUG]: text = '{choice.get('text', '')}'")
                print(f"[DEBUG]: text repr = {repr(choice.get('text', ''))}")
                #print(f"[DEBUG]: logprobs = {choice.get('logprobs', {})}")
                print(f"[DEBUG]: logprobs = {repr(choice.get('logprobs', {}))}")
            # Check if we got logprobs
            if 'choices' in output and len(output['choices']) > 0:
                choice = output['choices'][0]
                if 'logprobs' in choice and choice['logprobs']:
                    print("[OK] Successfully retrieved logprobs")
                    # Convert to OpenAI-compatible format
                    result =  self._convert_to_openai_format(output)
                    if result:
                        return result
                    else:
                        print("[WARNING] Conversion returned None, using fallback")
                else:
                    print("[WARNING] No logprobs in response")
            else:
                print("[WARNING] No choices in output")
            return {'choices': [{'text': '0',  # Default to class 0
                                 'logprobs': {'top_logprobs': [{'0': -1.386,  # log(0.25)
                                                                '1': -1.386,
                                                                '2': -1.386,
                                                                '3': -1.386
                                                                }]
                                              }
                                }]
                    }
        except Exception as e:
            print(f"[ERROR] Error during generation with probabilities: {e}")
            import traceback
            traceback.print_exc()
    
            # [OK Return safe fallback instead of None
            return {'choices': [{'text': '0',  # Default to class 0
                    'logprobs': {'top_logprobs': [{'0': -1.386,  # log(0.25)
                                                   '1': -1.386,
                                                   '2': -1.386,
                                                   '3': -1.386
                                                 }]
                                 }
                               }]
                    }
    

    
    def _convert_to_openai_format(self, llama_output: Dict) -> Dict:
        """
        Convert llama-cpp-python output format to OpenAI-compatible format.
        This makes it work with your existing _extract_class_probabilities code.
        Searches through all token positions to find the one with digit classes.   # changed 12/3/2026 11:20 AM
        """
        choice = llama_output['choices'][0]
        logprobs_data = choice.get('logprobs', {})
        top_logprobs_all_positions = logprobs_data.get('top_logprobs', [])   # changed 12/3/2026 11:20 AM
        
        print(f"[DEBUG] Raw model output text: '{choice['text']}'")
        print(f"[DEBUG] Raw model output repr: {repr(choice['text'])}")
        # print(f"[DEBUG] Raw top_logprobs keys: {list(logprobs_data.get('top_logprobs', [{}])[0].keys())}")  # changed 12/3/2026 11:20 AM
        print(f"[DEBUG] Total token positions: {len(top_logprobs_all_positions)}") # changed 12/3/2026 11:20 AM
        
        # ← NEW CODE BLOCK START: Search through all positions    # changed 12/3/2026 11:20 AM
        best_position = None
        best_digit_count = 0
        
        for position_idx, top_logprobs_dict in enumerate(top_logprobs_all_positions):
            digit_tokens = [k for k in top_logprobs_dict.keys() if k.strip() in ['0', '1', '2', '3']]
            #print(f"🔍 Position {position_idx}: {len(digit_tokens)} digit classes found: {digit_tokens}")
            print(f"[DEBUG] Position {position_idx}: {len(digit_tokens)} digit classes found: {digit_tokens}")
            
        
            if len(digit_tokens) > best_digit_count:
                best_digit_count = len(digit_tokens)
                best_position = position_idx
    
        if best_position is None:
            print("[WARNING] No position with digit classes found, using position 0")
            best_position = 0
        else:
            print(f"[OK] Using position {best_position} (has {best_digit_count} digit classes)")
        # ← NEW CODE BLOCK END        # changed 12/3/2026 11:20 AM
        
        
        # Get the first token's top logprobs
        # llama-cpp-python format: {'token': logprob, 'token2': logprob2, ...}
        #top_logprobs_dict = logprobs_data.get('top_logprobs', [{}])[0]  # changed 12/3/2026 11:20 AM
        
        top_logprobs_dict = top_logprobs_all_positions[best_position]   # changed 12/3/2026 11:20 AM 
        
        # Convert to list of dicts (OpenAI format)
        top_logprobs_list = []
        for token, logprob in top_logprobs_dict.items():
            top_logprobs_list.append({
                'token': token.strip(),
                'logprob': logprob
            })
        
        # Sort by probability (highest first)
        top_logprobs_list.sort(key=lambda x: x['logprob'], reverse=True)
        
        #print(f"✓ Converted {len(top_logprobs_list)} tokens to OpenAI format")  # changed 12/3/2026 11:20 AM
        print(f"[OK] Converted {len(top_logprobs_list)} tokens to OpenAI format") # changed 12/3/2026 4:50 AM
        
        # Return in OpenAI-compatible format
        return {
            'choices': [{
                'message': {
                    'content': choice['text']
                },
                'logprobs': {
                    'content': [{
                        'token': logprobs_data.get('tokens', [''])[0],
                        'logprob': logprobs_data.get('token_logprobs', [0])[0],
                        'top_logprobs': top_logprobs_list
                    }]
                }
            }]
        }


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <p style="margin-left: 25px;">
#     <b><font size="5">Few Shot Diagnosis Class</font></b></p><br>
#     <font size="3">
#          <ol>
#           <li>__init__(self, gemma_model, icd10_descriptions: Optional[Dict[str, str]] = None)</li> 
#           <li>fit(self, examples_df: pd.DataFrame)</li>
#           <li>_create_prompt(self, patient_text: str, patient_id: Optional[str] = None) -> str</li>   
#           <li>_parse_response(self, response: str) -> str</li>
#           <li>_extract_class_probabilities(self, result) -> List[float]</li>
#           <li>_extract_class_probabilities2(self, logprobs_dict) -> List[float]</li>
#           <li>predict_single(self, patient_text: str, patient_id: Optional[str] = None) -> Dict</li>
#           <li>predict(self, test_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame</li>
#           <li>calc_prediction(self, row)</li>
#           <li>vote_score(self, details_df, num_classes)</li>
#           <li>get_perf_data(self, y_true, y_pred)</li>
#           <li>evaluate(self, test_df: pd.DataFrame, verbose: bool = True) -> Dict</li>
#           <li>evaluate2(self, test_df: pd.DataFrame, verbose: bool = True) -> Dict</li>
#           <li>results_df.to_csv(filepath, index=False)</li>      
#          </ol> </font>
# </div>

# %%
class FewShotDiagnosisClassifier:
    """
    Few-shot learning classifier for ICD-10 diagnosis prediction using LLM.
    """
    
    def __init__(self, gemma_model, icd10_descriptions: Optional[Dict[str, str]] = None):
        """
        Initialize the classifier.
        
        Parameters:
        - gemma_model: The Gemma model instance for generating predictions
        - icd10_descriptions: Optional dict mapping ICD10 codes to descriptions
        """
        self.gemma_model = gemma_model
        self.icd10_descriptions = icd10_descriptions or {}
        self.examples_df = None
        self.classes_ = None
        
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
        
        print(f"Model fitted with {len(self.examples_df)} examples")
        print(f"Classes: {self.classes_}")
        
        return self
    
    def _create_prompt(self, patient_text: str, patient_id: Optional[str] = None) -> str:
        """
        Create a few-shot prompt for a single patient.
        """
        if self.examples_df is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Map ICD codes to simple indices
        code_mapping = {i: code for i, code in enumerate(self.classes_)}
        
        # Build prompt header
        prompt = f"""<start_of_turn>user
        
                    You are an experienced attending physician reviewing clinical notes to make a diagnostic assessment.

                    Based on the patient's clinical presentation, predict which diagnosis class applies:

                    0 = A41.9 (Sepsis, unspecified organism)
                    1 = I21.4 (Acute subendocardial myocardial infarction)
                    2 = J96.00 (Acute respiratory failure, unspecified)
                    3 = N17.9 (Acute kidney failure, unspecified)

        CRITICAL INSTRUCTIONS:
        - Respond with ONLY a single digit: 0, 1, 2, or 3
        - Do not include the diagnosis code (e.g., A41.9)
        - Do not add any explanation, punctuation, spaces, or newlines
        - Output format: Just the digit, nothing else
        
      
    """
        
        
        #prompt += "\nHere are some example cases:\n\n"
        
        # # Add few-shot examples with the new format
#    for idx, row in self.examples_df.iterrows():
#        example_num = idx + 1
#        # Map the ICD code to its index (0, 1, 2, or 3)
#        code_idx = self.classes_.index(row['icd10_code'])
#        
#        prompt += f"""Example {example_num}:
#                    Clinical Note: {row['text']}
#                    Diagnosis: {code_idx}

#        """

        
        # Add new patient case
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
    
    def _extract_class_probabilities(self, result) -> List[float]:
        """Extract probabilities for all 4 classes, filtering out non-digit tokens."""
        import math
    
        try:
            print("🔍 Extracting class probabilities")
        
            # Debug: Show full result structure
            # print("\n" + "="*80)
            # print("DEBUG: Full result structure:")
            # print(json.dumps(result, indent=2, default=str))
            # print("="*80 + "\n")
        
            # if not result or 'choices' not in result:
                # print("[WARNING] Invalid result structure")
                # return [0.25] * 4
        
            choice = result['choices'][0]
            # print(f"Choice keys: {choice.keys()}")
        
            logprobs_data = choice.get('logprobs', {})
            # print(f"Logprobs data: {logprobs_data}")
        
            content_list = logprobs_data.get('content', [])
        
            if not content_list:
                print(f"[WARNING] No logprobs content")
                return [0.25] * 4
        
            # Get the first token position's logprobs
            first_token = content_list[0]
            top_logprobs_list = first_token.get('top_logprobs', [])
            print(f"[DEBUG]: top_logprobs_list = {top_logprobs_list}")
            print(f"[DEBUG]: Length = {len(top_logprobs_list)}")
            if top_logprobs_list:
                print(f"[DEBUG]: First item = {top_logprobs_list[0]}")
        
            if not top_logprobs_list:
                print(f"[WARNING] No top_logprobs")
                return [0.25] * 4
        
            # Use position 0 (where the digit is generated)
            top_logprobs = top_logprobs_list[0]
        
            #print(f"✓ Successfully retrieved logprobs")
            #print(f"[INFO] Available tokens (all): {list(top_logprobs.keys())[:10]}")
        
            # Convert log probs to regular probabilities
            probs_dict = {}
            for item in top_logprobs_list:
                token = item['token']
                logprob = item['logprob']
                if token:  # Skip empty string tokens
                    probs_dict[token] = math.exp(logprob)
        
            #print(f"[INFO] Available tokens: {list(probs_dict.keys())}")
        
            # Extract probabilities ONLY for valid class tokens (0, 1, 2, 3)
            class_probs = []
            found_classes = []
        
            for class_idx in range(4):
                token = str(class_idx)
                if token in probs_dict:
                    class_probs.append(probs_dict[token])
                    found_classes.append(class_idx)
                else:
                    class_probs.append(1e-10)  # Very small for missing classes
        
            print(f"[OK] Found {len(found_classes)} digit classes: {found_classes}")
        
            # IMPORTANT: Renormalize to sum to 1 (only among the 4 classes)
            # This redistributes probability from \n and other tokens to our classes
            total = sum(class_probs)
            if total > 0:
                probs = [p / total for p in class_probs]
            #    print(f"[OK] Renormalized probabilities (redistributed from non-digit tokens)")
            else:
                print("[WARNING] No valid class tokens found, using uniform")
                probs = [0.25] * 4
        
            # Display probabilities
            #print(f"\nClass Probabilities:")
            #for i, prob in enumerate(probs):
            #    code = self.classes_[i] if hasattr(self, 'classes_') else str(i)
            #    print(f"  {code}: {prob:.4f} ({prob*100:.2f}%)")
        
            return probs
        
        except Exception as e:
            print(f"[ERROR] Error extracting probabilities: {e}")
            import traceback
            traceback.print_exc()
            return [0.25] * 4
    
    def _extract_class_probabilities2(self, output: Dict) -> List[float]:
        """
        Extract probability distribution for each class from model output.
        Searches through all token positions to find the one with digit classes.
        """
        import math
    
        try:
            if not output or 'choices' not in output:
                print(f"[WARNING] No valid output")
                return [0.25] * 4
        
            choice = output['choices'][0]
            logprobs_data = choice.get('logprobs', {})
            
            # ← NEW: Get from 'content' array first        # changed 12/3/2026 11:46 AM
            content_list = logprobs_data.get('content', [])
        
            if not content_list:  # ← NEW: Check if content exists      # changed 12/3/2026 11:46 AM
                print(f"[WARNING] No content in logprobs")
                return [0.25] * 4
        
            # ← CHANGED: Get top_logprobs from content[0] instead of directly from logprobs_data
            top_logprobs_list = content_list[0].get('top_logprobs', [])      # changed 12/3/2026 11:46 AM
            #top_logprobs_list = logprobs_data.get('top_logprobs', [])       # changed 12/3/2026 11:46 AM
        
            if not top_logprobs_list:
                print(f"[WARNING] No top_logprobs found")
                return [0.25] * 4
        
            #print(f"🔍 Searching through {len(top_logprobs_list)} token positions")    # changed 12/3/2026 11:59 AM
            print(f"[DEBUG] Found {len(top_logprobs_list)} tokens in top_logprobs")          # changed 12/3/2026 11:59 AM
        
            # ← NEW CODE BLOCK: Build a dict for easy lookup       # changed 12/3/2026 11:59 AM
            token_dict = {}
            for item in top_logprobs_list:
                token_dict[item['token']] = item['logprob']
        
            print(f"[DEBUG] Available tokens: {list(token_dict.keys())[:10]}")
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
                print(f"[OK] Normalized probabilities: {[f'{p:.4f}' for p in probs]}")
            else:
                probs = [0.25] * 4
                print(f"[WARNING] Total probability was 0, using uniform distribution")
                
            return probs
        
            # If we get here, we didn't find a position with digit classes       # Section removed 12/3/2026 11:59 AM
            # print(f"[WARNING] Could not find token position with digit classes")
            #print(f"[WARNING] Available tokens at each position:")
            #for idx, top_logprobs in enumerate(top_logprobs_list):
            #    print(f"   Position {idx}: {list(top_logprobs.keys())[:5]}")
            #        
            #return [0.25] * 4
        
        except Exception as e:
            print(f"[ERROR] Error in _extract_class_probabilities: {e}")
            import traceback
            traceback.print_exc()
            return [0.25] * 4

    
    def predict_single(self, patient_text: str, patient_id: Optional[str] = None) -> Dict:
        """
        Predict diagnosis for a single patient with class probabilities.
    
        Returns:
        - Dictionary with 'prediction', 'probabilities', 'prompt', and 'raw_response'
        """
        
        print("Processing ID:", round(patient_id))
        if self.examples_df is None:
            raise ValueError("Model not fitted. Call fit() first.")
    
        # Generate prompt - modify to encourage single-token response
        prompt = self._create_prompt(patient_text, patient_id)
    
        # Get prediction WITH PROBABILITIES from Gemma
        result = self.gemma_model.generate_with_probabilities(prompt,
                                                              self.classes_,
                                                              max_tokens=5,      # ← Explicitly set this
                                                              temperature=0.7    # ← And this
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
        probabilities = self._extract_class_probabilities2(result)
        
        # ← ADD DEBUG HERE                                          #Added 12/3/2025 12:25 PM
        print(f"[DEBUG]: probabilities = {probabilities}")
        print(f"[DEBUG]: probabilities type = {type(probabilities)}")
        print(f"[DEBUG]: probabilities is None? {probabilities is None}")
    
        
        # Get top prediction
        # if probabilities:              # Changed 12/3/2025 12:25 PM   
        if probabilities is not None:    # Changed 12/3/2025 12:25 PM   
            max_prob_idx = probabilities.index(max(probabilities))
            prediction = self.classes_[max_prob_idx]
            confidence = probabilities[max_prob_idx]  
            print(f"[OK] Predicted class {max_prob_idx}: {prediction} with confidence {confidence:.4f}") # Changed 12/3/2025 12:25 PM   
        else:
            print(f"[WARNING] Probabilities is None, using fallback parsing")     # Changed 12/3/2025 12:25 PM   
            prediction = self._parse_response(result.get('text', ''))
            confidence = None  # ← Set here

        return {'prediction': prediction,
                'probabilities': probabilities,
                'confidence': confidence,  # ← Use the variable
                'prompt': prompt,
                'raw_response': result
                }

    
    def predict(self, test_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
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
        
        predictions = []
        
        for idx, row in test_df.iterrows():
            print("handling susequence:", idx)
            if verbose and (idx + 1) % 10 == 0:
                print(f"Processing patient {idx + 1}/{len(test_df)}...")
            
            # Get prediction
            result = self.predict_single(row['text'], row['id'])
            probs = result['probabilities']  # Add this line to define probs
            
            # Predict with probabilities
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("\nClass Probabilities:")
            
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
            print(f"\nCompleted predictions for {len(test_df)} patients")
        
        return results_df
    
    def calc_prediction(self, row):

        max_score = max(row.iloc[11], row.iloc[12], row.iloc[13], row.iloc[14])

        if row.iloc[11] == max_score:
            return 0
        elif row.iloc[12] == max_score:
            return 1
        elif row.iloc[13] == max_score:
            return 2
        elif row.iloc[14] == max_score:
            return 3
        else:
            return 99

    
    def vote_score(self, details_df, num_classes):

    
        adm_scores = details_df.sort_values(by='adm_id')

        for n in range(num_classes):
            ename = 'element-' + str(n)

        temp_scores = adm_scores.groupby(['adm_id', 'actuals']).agg(Adm_Count=('adm_id', 'size'),
                                        Class_1_Mean=('class_1_score', 'mean'), Class_1_Max=('class_1_score', 'max'),
                                        Class_2_Mean=('class_2_score', 'mean'), Class_2_Max=('class_2_score', 'max'),
                                        Class_3_Mean=('class_3_score', 'mean'), Class_3_Max=('class_3_score', 'max'),
                                        Class_4_Mean=('class_4_score', 'mean'), Class_4_Max=('class_4_score', 'max')).reset_index()

        temp_scores['Class_1_Prob'] = (temp_scores['Class_1_Max'] + (temp_scores['Class_1_Mean'] 
                                                                     * temp_scores['Adm_Count']/2))/(1 + temp_scores['Adm_Count']/2)
        temp_scores['Class_2_Prob'] = (temp_scores['Class_2_Max'] + (temp_scores['Class_2_Mean'] 
                                                                     * temp_scores['Adm_Count']/2))/(1 + temp_scores['Adm_Count']/2)
        temp_scores['Class_3_Prob'] = (temp_scores['Class_3_Max'] + (temp_scores['Class_3_Mean'] 
                                                                     * temp_scores['Adm_Count']/2))/(1 + temp_scores['Adm_Count']/2)
        temp_scores['Class_4_Prob'] = (temp_scores['Class_4_Max'] + (temp_scores['Class_4_Mean'] 
                                                                     * temp_scores['Adm_Count']/2))/(1 + temp_scores['Adm_Count']/2)
        temp_scores['Prediction'] = temp_scores.apply(self.calc_prediction, axis=1)

        pred_labels = temp_scores['Prediction']
        actuals = temp_scores['actuals']
        probs = temp_scores[['Class_1_Prob', 'Class_2_Prob', 'Class_3_Prob', 'Class_4_Prob']].to_numpy()
        
        print(temp_scores)

        return (actuals, pred_labels, probs)


    def get_perf_data(self, y_true, y_pred):

        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
        from sklearn.metrics import precision_score, recall_score, f1_score,roc_curve, auc, roc_auc_score
        from sklearn.preprocessing import label_binarize

        #print('get perf data input')
        print('perf y_true', '\n', y_true[0:5])
        print('perf y_pred', '\n', y_pred[0:5])
        print('Generating Performance Data')
        cm = confusion_matrix(y_true, y_pred)
        print('Perf Routine Confusion Matrix:')
        print(cm)
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(accuracy)
        # Precision
        precision = precision_score(y_true, y_pred, average=None)
        print(precision)
        # Recall
        sensitivity = recall_score(y_true, y_pred, average=None)
        print(sensitivity)
        # F1-Score
        f1 = f1_score(y_true, y_pred, average=None)

        labels = ['A41.9', 'I21.4', 'J96.00', 'N17.9']
        # Binarize ytest with shape (n_samples, n_classes)
        y_true = label_binarize(y_true, classes=labels)
        # Binarize ypreds with shape (n_samples, n_classes)
        y_pred = label_binarize(y_pred, classes=labels)
        
        roc_auc = 0
        #roc_auc = roc_auc_score(y_true, y_pred,average=None,multi_class='ovr')
        return(accuracy, precision, sensitivity, f1, roc_auc, cm)

    def evaluate(self, test_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Predict and evaluate model performance.
        
        Returns:
        - Dictionary with evaluation metrics
        """
        # Get predictions
        results_df = self.predict(test_df, verbose=verbose)
        
        
        # Calculate metrics
        accuracy = results_df['correct'].mean()
        
        # Per-class accuracy
        class_accuracy = {}
        for code in self.classes_:
            mask = results_df['true_icd10_code'] == code
            if mask.sum() > 0:
                class_accuracy[code] = results_df[mask]['correct'].mean()
        
        # Confusion matrix
        confusion_matrix = pd.crosstab(
            results_df['true_icd10_code'],
            results_df['predicted_icd10_code'],
            rownames=['True'],
            colnames=['Predicted']
        )
        
        metrics = {
            'accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'confusion_matrix': confusion_matrix,
            'results_df': results_df,
            'n_samples': len(test_df),
            'n_correct': results_df['correct'].sum()
        }
        
        if verbose:
            print("\n" + "="*80)
            print("EVALUATION RESULTS")
            print("="*80)
            print(f"\nOverall Accuracy: {accuracy:.2%} ({metrics['n_correct']}/{metrics['n_samples']})")
            print("\nPer-Class Accuracy:")
            for code, acc in class_accuracy.items():
                desc = self.icd10_descriptions.get(code, "")
                print(f"  {code} ({desc}): {acc:.2%}")
            print("\nConfusion Matrix:")
            print(confusion_matrix)
            print("\nMisclassified cases:")
            misclassified = results_df[~results_df['correct']]
            if len(misclassified) > 0:
                for _, row in misclassified.iterrows():
                    print(f"\n  ID: {row['id']}")
                    print(f"  True: {row['true_icd10_code']}, Predicted: {row['predicted_icd10_code']}")
                    print(f"  Text: {row['text'][:100]}...")
            else:
                print("  No misclassifications!")
                
            details_df = results_df[['id', 'true_icd10_code', 'prob_class_0','prob_class_1',
                                    'prob_class_2','prob_class_3']].copy()
            new_cols = ['adm_id', 'actuals', 'class_1_score', 'class_2_score', 
                        'class_3_score', 'class_4_score']
            details_df.columns = new_cols
        
        return metrics, details_df
    
    def evaluate2(self, test_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Predict and evaluate model performance.
        
        Returns:
        - Dictionary with evaluation metrics
        """
        # Get predictions
        results_df = self.predict(test_df, verbose=verbose)
        
        #Skinny down results to just needed details for perf measurement
        details_df = results_df[['id', 'true_icd10_code', 'predicted_icd10_code', 'prob_class_0','prob_class_1',
                                    'prob_class_2','prob_class_3']].copy()
        new_cols = ['adm_id', 'actuals', 'predictions', 'class_1_score', 'class_2_score', 
                        'class_3_score', 'class_4_score']
        details_df.columns = new_cols
        
        actuals, pred_labels, probs = self.vote_score(details_df, 4)
        
        # Convert numeric predictions back to ICD codes
        #temp_labels = ['A41.9', 'I21.4', 'J96.00', 'N17.9']
              
        pred_labels_icd = [self.classes_[label] for label in pred_labels]
        print('preds:', len(pred_labels), 'pred-converted:', len(pred_labels_icd), 'actuals:', len(actuals))
        accuracy, precision, sensitivity, f1, roc_auc, cm = self.get_perf_data(actuals, pred_labels_icd)
        print('Evaluate Performance')
        print('\n', 'acc', accuracy, 'prec', precision, 'sens', sensitivity, 'f1', f1, 'roc', roc_auc)

        return 
    
    def save_results(self, results_df: pd.DataFrame, filepath: str):
        """Save prediction results to CSV."""
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <p style="margin-left: 25px;">
#     <b><font size="5">Build Needed Classes</font></b></p><br>
#     <font size="3">
#          <ol>
#           <li>Create LlamaCppWrappr object</li> 
#           <li>Create FewShotDiagnosisClassifier Object</li>
#           <li>Call Classifier Fit method to store training data in classifier</li>   
#          </ol> </font>
# </div>

# %%
import os
from llama_cpp import Llama

##########################################
# LLama Configuration
#MODEL_PATH = r"C:/Users/rolan/.lmstudio/models/lmstudio-community/gemma-3n-E4B-it-text-GGUF/gemma-3n-E4B-it-Q4_K_M.gguf"
MODEL_PATH = r"E:/Education/llama/models/gemma-3n-E4B-it-Q4_K_M.gguf"
##########################################

#lm_studio_model = LMStudioModelWrapper(base_url=LM_STUDIO_URL,model_name=MODEL_NAME)
llama_model = LlamaCppWrapper(model_path=MODEL_PATH, n_ctx=4096, n_threads=8)

# Rest stays the same:
classifier = FewShotDiagnosisClassifier(gemma_model=llama_model)  
print("\nFitting classifier with training examples...")
classifier.fit(train_sample_df2)

# %%
print(test_sample_df2.iloc[10:20]['text'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0))

# %%
# ============================================================================
# STEP 8: TEST ON A SINGLE PATIENT FIRST
# ============================================================================
import re
fhandle, fname = start_logging("Single_Patient")

print("\n" + "="*80)
print("TESTING ON SINGLE PATIENT")
print("="*80)
test_patient = test_sample_df2.iloc[0]
print(f"\nText Length: {len(test_patient['text'].split())} words ({len(test_patient['text'])} chars)")
print(f"Patient ID: {test_patient['id']}")
print(f"True Diagnosis: {test_patient['icd10_code']}")
print(f"\nClinical Note Preview:\n{test_patient['text'][:300]}...")

# Get prediction with probabilities
single_result = classifier.predict_single(
    patient_text=test_patient['text'],
    patient_id=test_patient['id']
)

# Display prediction results
print("\n" + "="*80)
print("PREDICTION RESULTS")
print("="*80)

correct = single_result['prediction'] == test_patient['icd10_code']
#status_icon = "✓" if correct else "✗"     #Changed 12/3/25 5:27 PM
status_icon = "[OK]" if correct else "[X]"  #Changed 12/3/25 5:27 PM

print(f"\n{'Predicted:':<15} {single_result['prediction']}")
print(f"{'True Label:':<15} {test_patient['icd10_code']}")
print(f"{'Status:':<15} {status_icon} {'CORRECT' if correct else 'INCORRECT'}")

if single_result.get('confidence') is not None:
    print(f"{'Confidence:':<15} {single_result['confidence']:.2%}")

# Display probabilities table
if single_result.get('probabilities') is not None:
    print("\n" + "-"*80)
    print("CLASS PROBABILITY DISTRIBUTION")
    print("-"*80)
    print(f"{'ICD-10 Code':<15} {'Probability':<12} {'Visual':<30} {'Notes'}")
    print("-"*80)
    
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
        
        print(f"{code:<15} {prob:>6.2%}       {bar:<30} {notes_str}")
    
    print("-"*80)
else:
    print(f"\n[WARNING] Probabilities not available")

print("\n" + "="*80)

sys.stdout = sys.__stdout__
fhandle.close()
print(f"[OK] Logging stopped. Output saved to: {fname}")

# %%
print(single_result.keys())
print(single_result['probabilities'])

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <p style="margin-left: 25px;">
#     <b><font size="5">Complete A Set of Admission Predictions on Test Data</font></b></p><br>
#     <font size="3">
#          <ol>
#           <li>Get A Subset of Test Subsequences/li> 
#           <li>Run classifier.evaluate2</li>
#           <li>Return metrics</li>   
#          </ol> </font>
# </div>

# %%


# ============================================================================
# STEP 9: EVALUATE ON FULL TEST SET (or subset for speed)
# ============================================================================
fhandle, fname = start_logging("Set_of_Patients")
# For quick testing, use a small subset
print("\n" + "="*80)
print("EVALUATING ON TEST SET")
print("="*80)

# Option 1: Test on small subset first (faster)
test_subset = test_sample_df2[:59]  # Just 20 subsequences for quick test
print(f"\nTesting on {len(test_subset)} patients...")
#metrics = classifier.evaluate(test_subset, verbose=True)
metrics = classifier.evaluate2(test_subset)
sys.stdout = sys.__stdout__
fhandle.close()
print(f"[OK] Logging stopped. Output saved to: {fname}")
#print(details_df[0:20])

# Option 2: Run on full test set (uncomment when ready)
# print(f"\nTesting on all {len(test_df)} patients...")
# metrics = classifier.evaluate(test_df, verbose=True)


# %%
print(metrics)

# %%

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================

print("\nSaving results...")
classifier.save_results(metrics['results_df'], 'diagnosis_predictions.csv')

# Show some example predictions
print("\nSample predictions:")
print(metrics['results_df'][['id', 'true_icd10_code', 'predicted_icd10_code', 'correct']].head(10))

# Show misclassifications
misclassified = metrics['results_df'][~metrics['results_df']['correct']]
if len(misclassified) > 0:
    print(f"\nMisclassified cases ({len(misclassified)} total):")
    print(misclassified[['id', 'true_icd10_code', 'predicted_icd10_code']].head())
else:
    print("\nNo misclassifications!")

# %%
# Simple diagnostic test - run this in a new cell
test_prompt = "Choose a number 0, 1, 2, or 3: "

for temp in [0.1, 0.5, 0.7, 1.0]:
    print(f"\n{'='*60}")
    print(f"Testing with temperature = {temp}")
    print(f"{'='*60}")
    
    output = classifier.gemma_model.llm(
        test_prompt,
        max_tokens=1,
        temperature=temp,
        logprobs=True,
        echo=False
    )
    
    top_logprobs = output['choices'][0]['logprobs']['top_logprobs'][0]
    print(f"Number of tokens: {len(top_logprobs)}")
    print(f"Tokens: {list(top_logprobs.keys())}")


# %%
# Test with top_k to force more options
test_text = test_sample_df2.iloc[0]['text']
actual_prompt = classifier._create_prompt(test_text, "test123")

print("Testing with top_k parameter:")
print("="*80)

output = classifier.gemma_model.llm(
    actual_prompt,
    max_tokens=2,
    temperature=1.0,
    top_k=50,  # Force consideration of top 50 tokens
    top_p=0.95,  # Nucleus sampling
    logprobs=True,
    echo=False
)

generated = output['choices'][0]['text']
logprobs_list = output['choices'][0]['logprobs']['top_logprobs']

print(f"Generated: {repr(generated)}")
print(f"\nPosition 0 tokens: {list(logprobs_list[0].keys())[:10]}")
if len(logprobs_list) > 1:
    print(f"Position 1 tokens: {list(logprobs_list[1].keys())[:10]}")

# Check how many tokens are in each position
print(f"\nNumber of tokens at position 0: {len(logprobs_list[0])}")
if len(logprobs_list) > 1:
    print(f"Number of tokens at position 1: {len(logprobs_list[1])}")

# %%
# Test if this works in your version
output = classifier.gemma_model.llm(
    actual_prompt,
    max_tokens=2,
    temperature=1.0,
    logprobs=10,  # ← Request 10 top logprobs (not just True)
    echo=False
)

print("Testing with logprobs=10:")
print("="*80)
generated = output['choices'][0]['text']
logprobs_list = output['choices'][0]['logprobs']['top_logprobs']

print(f"Generated: {repr(generated)}")
print(f"Number of positions: {len(logprobs_list)}")

for i, top_logprobs in enumerate(logprobs_list):
    print(f"Position {i}: {len(top_logprobs)} tokens - {list(top_logprobs.keys())[:10]}")

# %%

# %%

# %%

# %%

# %%
