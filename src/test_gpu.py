from llama_cpp import Llama
import sys

print("="*60)
print("GPU VERIFICATION TEST")
print("="*60)

# Test 1: CPU Only (baseline)
print("\n[Test 1] CPU-only (baseline speed)...")
try:
    llm_cpu = Llama(
        model_path="E:/Education/llama/models/gemma-3n-E4B-it-Q4_K_M.gguf",
        n_gpu_layers=0,  # CPU only
        n_ctx=512,
        verbose=False
    )
    response = llm_cpu("Test", max_tokens=20)
    print("✅ CPU test completed")
    del llm_cpu
except Exception as e:
    print(f"❌ CPU test failed: {e}")

# Test 2: GPU with verbose output
print("\n[Test 2] GPU with n_gpu_layers=20 (watch for CUDA messages)...")
try:
    llm_gpu = Llama(
        model_path="E:/Education/llama/models/gemma-3n-E4B-it-Q4_K_M.gguf",
        n_gpu_layers=20,  # Try offloading 20 layers
        n_ctx=512,
        verbose=True  # Should show GPU info
    )
    print("\n✅ GPU model loaded!")
    print("\nGenerating 50 tokens to measure speed...")
    response = llm_gpu("Write a short story", max_tokens=50)
    print("✅ GPU generation completed!")
    
except Exception as e:
    print(f"❌ GPU test failed: {e}")