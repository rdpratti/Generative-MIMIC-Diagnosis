# llama_wrapper.py
# Wrapper class for llama-cpp-python to work with Gemma models

import os
import sys
import io
from contextlib import redirect_stderr
import psutil
from typing import Dict, List
from llama_cpp import Llama
import gemmaUtils as gu

class LlamaCppWrapper:
    """LlamaCppWrapper Class</font></b></p><br>
        a) __init__
        b) generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.1) -> str</li>
        c) generate_with_probabilities(self, prompt: str, classes: list, max_tokens: int = 10, 
                                       temperature: float = 0.7)</li>   
        d) _convert_to_openai_format(self, llama_output: Dict) -> Dict</li>
    """
    
    def __init__(self, 
                 model_path: str, 
                 n_ctx: int = 4096, 
                 n_threads: int = None, 
                 n_batch: int = 512,          
                 use_mmap: bool = True,
                 logger = None):
        """
        Initialize llama.cpp model.
        
        Args:
            model_path: Path to GGUF model file (e.g., "C:/models/gemma-2-9b-it-Q4_K_M.gguf")
            n_ctx: Context window size
            n_threads: Number of CPU threads (None = auto-detect)
        """

        if logger is None:
            print("Logger is None")
        else:
            print("Logger has a Value")

        # Auto-detect optimal thread count if not specified
        if n_threads is None:
            cpu_count = os.cpu_count() or 4
            n_threads = max(1, cpu_count // 2)  # Use half of available cores
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Context size: {n_ctx}")
        logger.info(f"Using {n_threads} CPU threads")
        
        print('Model:', model_path)
    
        # Verify model file exists and is readable
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
        file_size_gb = os.path.getsize(model_path) / (1024**3)
        logger.info(f"Model file size: {file_size_gb:.2f} GB")
        
        if not gu.diagnose_gguf_file(model_path, logger):
            raise ValueError(f"GGUF file validation failed: {model_path}")
        
        # ← ADD: Capture stderr to see C++ errors
        import sys
        import io
        from contextlib import redirect_stderr
    
        stderr_capture = io.StringIO()
        
        # ← ADD: Check memory before loading
        import psutil
        mem = psutil.virtual_memory()
        logger.info(f"Available RAM before loading: {mem.available / (1024**3):.2f} GB")
        logger.info(f"Free RAM before loading: {mem.free / (1024**3):.2f} GB")
        logger.info(f"Memory usage: {mem.percent}%")
        
        try:
            logger.info("Attempting to load model...")
            with redirect_stderr(stderr_capture):
                self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=n_batch,
                use_mmap=use_mmap,
                logits_all=True,
                n_gpu_layers=0,
                verbose=True,  # ← CHANGE: Enable verbose to see C++ output
                use_mlock=False,  # Don't lock memory
                numa=False,       # Disable NUMA
                )
            logger.info("Model loaded successfully")
        
            # ← ADD: Check memory after loading
            mem_after = psutil.virtual_memory()
            logger.info(f"Available RAM after loading: {mem_after.available / (1024**3):.2f} GB")
            logger.info(f"Memory used by model: {(mem.available - mem_after.available) / (1024**3):.2f} GB")
        
        except Exception as e:
            cpp_errors = stderr_capture.getvalue()
            logger.error(f"Failed to load model: {str(e)}")
            logger.error(f"C++ error output:\n{cpp_errors}")
        
            # Try to parse the specific error
            if "unsupported" in cpp_errors.lower():
                logger.error("Model architecture may be unsupported")
            elif "allocat" in cpp_errors.lower() or "memory" in cpp_errors.lower():
                logger.error("Memory allocation failed - try reducing n_ctx or closing other programs")
            elif "mmap" in cpp_errors.lower():
                logger.error("Memory mapping failed - try use_mmap=False")
        
            raise  
    
        print('got past create')
        self.model_name = os.path.basename(model_path)
        logger.info(f"[OK] Model '{self.model_name}' loaded successfully")
    
    def find_token_size(self, prompt, max_tokens, logger):

        if logger:
            # CHECK PROMPT LENGTH
            prompt_tokens = self.llm.tokenize(prompt.encode('utf-8'))
            prompt_length = len(prompt_tokens)
            context_limit = self.llm.n_ctx()
            remaining = context_limit - prompt_length
            logger.info(f"Prompt tokens: {prompt_length}/{context_limit} (remaining: {remaining})")
            logger.info(f"Requested max_tokens: {max_tokens}")
        
        # WARNING if close to limit
        if remaining < max_tokens:
            logger.info(f"⚠️ DANGER: Prompt uses {prompt_length} tokens, only {remaining} remaining!")
            logger.info(f"Reducing max_tokens from {max_tokens} to {remaining - 10}")
            max_tokens = max(10, remaining - 10)  # Leave buffer
        
        if remaining < 100:
            logger.info(f"❌ CRITICAL: Only {remaining} tokens remaining - prompt too long!")
            # Log the prompt size breakdown
            logger.info(f"Prompt character length: {len(prompt)}")
            return None  # Fail gracefully instead of crashing
        return

    def check_memory(self, logger):

        import psutil
        vm = psutil.virtual_memory()
        available_gb = vm.available / 1024**3
        logger.info(f"Available RAM before generation: {available_gb:.2f} GB")
        
        if available_gb < 1.0:
            logger.info(f"❌ LOW MEMORY: Only {available_gb:.2f} GB available!")
        return

    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.1, logger = None) -> str:
        """
        Generate response without probabilities.
        Compatible with your existing code.
        """
        logger.info(f"Prompt size: {len(prompt)} chars (~{len(prompt)//4} tokens)")
        
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
            
            #return "ERROR"
            
        except Exception as e:
            logger.info(f"Error during generation: {e}")
            #return "ERROR"
            raise
    
    def generate_with_probabilities(self, prompt: str, classes: list, 
                                    max_tokens: int = 10, temperature: float = 0.7,
                                    logger = None):
        """
        Generate response WITH probabilities (logprobs).
        This is what enables true probability distributions for classification!
        """
        logger.debug(f"Requesting logprobs for {len(classes)} classes")
        

        try:
            self.check_memory(logger)
            self.find_token_size(prompt, max_tokens, logger)  
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=10,  # Get top 20 token probabilities
                #stop=["\n\n", "###", "Example"],
                echo=False
            )
            
            # ADD THESE DEBUG LINES RIGHT HERE:
            logger.debug(f"output keys = {output.keys()}")
            logger.debug(f"choices = {output.get('choices', [])}")
            if output.get('choices'):
                choice = output['choices'][0]
                logger.debug(f"choice keys = {choice.keys()}")
                logger.debug(f"text = '{choice.get('text', '')}'")
                logger.debug(f"text repr = {repr(choice.get('text', ''))}")
                logger.debug(f"logprobs = {repr(choice.get('logprobs', {}))}")
            # Check if we got logprobs
            if 'choices' in output and len(output['choices']) > 0:
                choice = output['choices'][0]
                if 'logprobs' in choice and choice['logprobs']:
                    logger.info("[OK] Successfully retrieved logprobs")
                    # Convert to OpenAI-compatible format
                    result =  self._convert_to_openai_format(output, logger=logger)
                    if result:
                        return result
                    else:
                        logger.warning("Conversion returned None, using fallback")
                else:
                    logger.warning("No logprobs in response")
            else:
                logger.warning("No choices in output")
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
            logger.exception(f"Error during generation with probabilities: {e}")
    
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
    

    
    def _convert_to_openai_format(self, llama_output: Dict, logger) -> Dict:
        """
        Convert llama-cpp-python output format to OpenAI-compatible format.
        This makes it work with your existing _extract_class_probabilities code.
        Searches through all token positions to find the one with digit classes.   # changed 12/3/2026 11:20 AM
        """
        choice = llama_output['choices'][0]
        logprobs_data = choice.get('logprobs', {})
        top_logprobs_all_positions = logprobs_data.get('top_logprobs', [])   # changed 12/3/2026 11:20 AM
        
        logger.debug(f"Raw model output text: '{choice['text']}'")
        logger.debug(f"Raw model output repr: {repr(choice['text'])}")
        logger.debug(f"Total token positions: {len(top_logprobs_all_positions)}") # changed 12/3/2026 11:20 AM
        
        # ← NEW CODE BLOCK START: Search through all positions    # changed 12/3/2026 11:20 AM
        best_position = None
        best_digit_count = 0
        
        for position_idx, top_logprobs_dict in enumerate(top_logprobs_all_positions):
            digit_tokens = [k for k in top_logprobs_dict.keys() if k.strip() in ['0', '1', '2', '3']]
            #print(f"🔍 Position {position_idx}: {len(digit_tokens)} digit classes found: {digit_tokens}")
            logger.debug(f"Position {position_idx}: {len(digit_tokens)} digit classes found: {digit_tokens}")
            
        
            if len(digit_tokens) > best_digit_count:
                best_digit_count = len(digit_tokens)
                best_position = position_idx
    
        if best_position is None:
            logger.warning("No position with digit classes found, using position 0")
            best_position = 0
        else:
            logger.info(f"[OK] Using position {best_position} (has {best_digit_count} digit classes)")
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
        logger.info(f"[OK] Converted {len(top_logprobs_list)} tokens to OpenAI format") # changed 12/3/2026 4:50 AM
        
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