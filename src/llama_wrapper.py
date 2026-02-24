# llama_wrapper.py
# Wrapper class for llama-cpp-python to work with Gemma models

import os
import io
from contextlib import redirect_stderr
import psutil
from llama_cpp import Llama
import gemmaUtils as gu
import time
from typing import Dict

class LlamaCppWrapper:
    """LlamaCppWrapper Class
        a) __init__
        b) tokenize(self, text)
        c) find_token_size(self, prompt, max_tokens)
        d) check_memory(self)  
        e) generate_with_probabilities(self, prompt: str, classes: list, max_tokens: int = 10, 
                                       temperature: float = 0.7)   
        f) _convert_to_openai_format(self, llama_output: Dict) -> Dict
    """
    
    def __init__(self, 
                 model_path: str, 
                 n_ctx: int = 4096, 
                 n_threads: int = None, 
                 n_batch: int = 512,          
                 use_mmap: bool = True,
                 temp = 0.0,
                 seed = 42,
                 logger = None):
        """
        Initialize llama.cpp model.
        
        Args:
            model_path: Path to GGUF model file (e.g., "C:/models/gemma-2-9b-it-Q4_K_M.gguf")
            n_ctx: Context window size
            n_threads: Number of CPU threads (None = auto-detect)
        """

        self.model_path = model_path 
        self.n_ctx = n_ctx 
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.use_mmap = use_mmap
        self.temperature = temp
        self.seed = seed
        self.logger = logger
        self.n_gpu_layers= -1

        # Auto-detect optimal thread count if not specified
        if n_threads is None:
            cpu_count = os.cpu_count() or 4
            self.n_threads = max(1, cpu_count // 2)  # Use half of available cores
        
        self.logger.info(f"Loading model from: {self.model_path}")
        self.logger.info(f"Context size: {self.n_ctx}")
        self.logger.info(f"Using {self.n_threads} CPU threads")
        self.logger.debug(f"Temperature: {self.temperature} random seed: {self.seed}")            
        
        # Verify model file exists and is readable
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
        file_size_gb = os.path.getsize(model_path) / (1024**3)
        self.logger.info(f"Model file size: {file_size_gb:.2f} GB")
        
        if not gu.diagnose_gguf_file(self.model_path, logger):
            raise ValueError(f"GGUF file validation failed: {self.model_path}")
        
        # Capture stderr to see C++ errors
        
    
        stderr_capture = io.StringIO()
        
        # Check memory before loading
        mem = psutil.virtual_memory()
        self.logger.info(f"Available RAM before loading: {mem.available / (1024**3):.2f} GB")
        self.logger.info(f"Free RAM before loading: {mem.free / (1024**3):.2f} GB")
        self.logger.info(f"Memory usage: {mem.percent}%")
        
        try:
            self.logger.info("Attempting to load model...")
            with redirect_stderr(stderr_capture):
                self.llm = self._load_model()
            #    self.llm = Llama(
            #    model_path=self.model_path,
            #    n_ctx=self.n_ctx,
            #    n_threads=self.n_threads,
            #    n_batch=self.n_batch,
            #    use_mmap=self.use_mmap,
            #    logits_all=True,
            #    n_gpu_layers=self.n_gpu_layers,
            #    verbose=False,  
            #    use_mlock=False,  # Don't lock memory
            #    numa=False,       # Disable NUMA
            #    )
            self.logger.info("Model loaded successfully")
        
            # ADD: Check memory after loading
            mem_after = psutil.virtual_memory()
            self.logger.info(f"Available RAM after loading: {mem_after.available / (1024**3):.2f} GB")
            self.logger.info(f"Memory used by model: {(mem.available - mem_after.available) / (1024**3):.2f} GB")
        
        except Exception as e:
            cpp_errors = stderr_capture.getvalue()
            self.logger.error(f"Failed to load model: {str(e)}")
            self.logger.error(f"C++ error output:\n{cpp_errors}")
        
            # Try to parse the specific error
            if "unsupported" in cpp_errors.lower():
                self.logger.error("Model architecture may be unsupported")
            elif "allocat" in cpp_errors.lower() or "memory" in cpp_errors.lower():
                self.logger.error("Memory allocation failed - try reducing n_ctx or closing other programs")
            elif "mmap" in cpp_errors.lower():
                self.logger.error("Memory mapping failed - try use_mmap=False")
        
            raise  
    
        self.model_name = os.path.basename(self.model_path)
        self.logger.info(f"[OK] Model '{self.model_name}' loaded successfully")

    def tokenize(self, text):
        """Tokenize text using the underlying llama.cpp model"""
        if isinstance(text, str):
            text_bytes = text.encode('utf-8')
        else:
            text_bytes = text  # Already bytes

        return self.llm.tokenize(text_bytes)
    
    def find_token_size(self, prompt, max_tokens):

        prompt_tokens = self.llm.tokenize(prompt.encode('utf-8'))
        prompt_length = len(prompt_tokens)
        context_limit = self.llm.n_ctx()
        remaining = context_limit - prompt_length
        self.logger.debug(f"Prompt tokens: {prompt_length}/{context_limit} (remaining: {remaining})")
        self.logger.debug(f"Requested max_tokens: {max_tokens}")
        
        # WARNING if close to limit
        if remaining < max_tokens:
            self.logger.debug(f"DANGER: Prompt uses {prompt_length} tokens, only {remaining} remaining!")
            self.logger.debug(f"Reducing max_tokens from {max_tokens} to {remaining - 10}")
            max_tokens = max(10, remaining - 10)  # Leave buffer
        
        if remaining < 100:
            self.logger.debug(f"CRITICAL: Only {remaining} tokens remaining - prompt too long!")
            # Log the prompt size breakdown
            self.logger.debug(f"Prompt character length: {len(prompt)}")
            return None  # Fail gracefully instead of crashing
        return

    def check_memory(self):

        vm = psutil.virtual_memory()
        available_gb = vm.available / 1024**3
        self.logger.debug(f"Available RAM before generation: {available_gb:.2f} GB")
        
        if available_gb < 1.0:
            self.logger.info(f"LOW MEMORY: Only {available_gb:.2f} GB available!")
        return

    def _load_model(self):
        
        
        llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                use_mmap=self.use_mmap,
                logits_all=True,
                seed = self.seed,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,  
                use_mlock=False,  # Don't lock memory
                numa=False,       # Disable NUMA
                )
               
        return llm
    

    def _reload_model(self):
        self.logger.info("Reloading model to clear VRAM...")
        try:
            del self.llm
            import gc
            gc.collect()
            self.llm = self._load_model()
            self.logger.info("Model reloaded successfully")
        except Exception as e:
            self.logger.error(f"Model reload failed: {e}")
            raise  # don't swallow this one


    def _cleanup_gpu_memory(self, inference_count: int = None):
        """
        Clear KV cache and release GPU memory between inference calls.
        Call this after each llm() invocation.
        """
        # 1. Reset KV cache
        try:
            self.llm.reset()
            self.logger.debug("KV cache reset via llm.reset()")
        except AttributeError:
            self.logger.warning("Could not reset KV cache - no reset method found")
        
        try:
            self.llm._ctx.kv_cache_clear()
            self.logger.debug("KV cache cleared via kv_cache_clear()")
        except AttributeError:
            self.logger.warning("Could not reset KV cache - no reset method found")

        # 2. Python garbage collection
        import gc
        gc.collect()

        # 3. PyTorch CUDA cache (if available)
        try:
            import torch
            torch.cuda.empty_cache()
            self.logger.debug("PyTorch CUDA cache cleared")
        except ImportError:
            pass

        # 4. Log VRAM state
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.free',
                '--format=csv,noheader,nounits'],
                capture_output=True, text=True
                )
            used, free = result.stdout.strip().split(', ')
            self.logger.info(f"VRAM after cleanup: {used} MB used, {free} MB free")
        except Exception:
            self.logger.warning("Failed to query VRAM state")
        
    def generate_with_probabilities(self, prompt: str, 
                                    classes: list, 
                                    max_tokens: int = 10):
        """
        Generate response WITH probabilities (logprobs).
        This is what enables true probability distributions for classification!
        """
        self.logger.debug(f"Requesting logprobs for {len(classes)} classes")
        self.logger.debug(f"Temperature: {self.temperature} random seed: {self.seed}")    

        start = time.time()

        # 1. CUDA health check first
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError as e:
                self.logger.warning(f"CUDA sync failed on inference {self.inference_count}, reloading model: {e}")
                self._reload_model()
                return None  # skip this inference, caller handles None
        
        try:
            self.llm.reset()  # clears KV cache
            self.check_memory()
            run_time = time.time() - start
            self.logger.debug(f"Completed check memory Time: {run_time:.2f}s")
            start = time.time()
            self.find_token_size(prompt, max_tokens)  
            run_time = time.time() - start
            self.logger.debug(f"Completed find_token_size Time: {run_time:.2f}s")
            start = time.time()
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                seed=self.seed,        
                logprobs=10,    
                echo=False
            )

            self._inference_count = getattr(self, '_inference_count', 0) + 1
            self._cleanup_gpu_memory(self._inference_count)
            
            run_time = time.time() - start
            self.logger.debug(f"Completed llm call Time: {run_time:.2f}s")
            
            self.logger.debug(f"output keys = {output.keys()}")
            self.logger.debug(f"choices = {output.get('choices', [])}")
            if output.get('choices'):
                choice = output['choices'][0]
                self.logger.debug(f"choice keys = {choice.keys()}")
                self.logger.debug(f"text = '{choice.get('text', '')}'")
                self.logger.debug(f"text repr = {repr(choice.get('text', ''))}")
                self.logger.debug(f"logprobs = {repr(choice.get('logprobs', {}))}")
                raw_output = choice['text']
                self.logger.debug(f"Raw output: '{raw_output}'")
                self.logger.debug(f"Output repr: {repr(raw_output)}")
                self.logger.debug(f"Stripped: '{raw_output.strip()}'")
            # Check if we got logprobs
            if 'choices' in output and len(output['choices']) > 0:
                choice = output['choices'][0]
                if 'logprobs' in choice and choice['logprobs']:
                    self.logger.info("[OK] Successfully retrieved logprobs")
                    # Convert to OpenAI-compatible format
                    result = self._convert_to_openai_format(output)
                    if result:
                        return result
                    else:
                        self.logger.warning("Conversion returned None, using fallback")
                else:
                    self.logger.warning("No logprobs in response")
            else:
                self.logger.warning("No choices in output")
            return {'choices': [{'text': '0',  
                                 'logprobs': {'top_logprobs': [{'0': -1.386,  # log(0.25)
                                                                '1': -1.386,
                                                                '2': -1.386,
                                                                '3': -1.386
                                                                }]
                                              }
                                }]
                    }
        except Exception as e:
            self.logger.exception(f"Error during generation with probabilities: {e}")
    
            # Return safe fallback instead of None
            return {'choices': [{'text': '0',  
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
        Searches through all token positions to find the one with digit classes.   
        """
        choice = llama_output['choices'][0]
        logprobs_data = choice.get('logprobs', {})
        top_logprobs_all_positions = logprobs_data.get('top_logprobs', [])   
        
        self.logger.debug(f"Raw model output text: '{choice['text']}'")
        self.logger.debug(f"Raw model output repr: {repr(choice['text'])}")
        self.logger.debug(f"Total token positions: {len(top_logprobs_all_positions)}") 
        
        # Search through all positions    
        best_position = None
        best_digit_count = 0
        
        for position_idx, top_logprobs_dict in enumerate(top_logprobs_all_positions):
            digit_tokens = [k for k in top_logprobs_dict.keys() if k.strip() in ['0', '1', '2', '3']]
            self.logger.debug(f"Position {position_idx}: {len(digit_tokens)} digit classes found: {digit_tokens}")
            
        
            if len(digit_tokens) > best_digit_count:
                best_digit_count = len(digit_tokens)
                best_position = position_idx
    
        if best_position is None:
            self.logger.warning("No position with digit classes found, using position 0")
            best_position = 0
        else:
            self.logger.debug(f"[OK] Using position {best_position} (has {best_digit_count} digit classes)")
                
        
        top_logprobs_dict = top_logprobs_all_positions[best_position]    
        
        # Convert to list of dicts (OpenAI format)
        top_logprobs_list = []
        for token, logprob in top_logprobs_dict.items():
            top_logprobs_list.append({
                'token': token.strip(),
                'logprob': logprob
            })
        
        # Sort by probability (highest first)
        top_logprobs_list.sort(key=lambda x: x['logprob'], reverse=True)
        
        self.logger.debug(f"[OK] Converted {len(top_logprobs_list)} tokens to OpenAI format") 
        
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