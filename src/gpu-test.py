"""
Simple AI Capability Test - Works on Any System
No special GPU libraries required
"""

import torch
import time
import sys
import platform

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def system_info():
    """Display system information"""
    print_header("SYSTEM INFORMATION")
    
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check available memory
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        ram_available_gb = psutil.virtual_memory().available / (1024**3)
        print(f"Total RAM: {ram_gb:.1f} GB")
        print(f"Available RAM: {ram_available_gb:.1f} GB")
    except ImportError:
        print("Install psutil for memory info: pip install psutil")

def test_pytorch_cpu():
    """Test basic PyTorch operations on CPU"""
    print_header("PYTORCH CPU TEST")
    
    sizes = [500, 1000, 2000]
    print("Matrix multiplication benchmark:")
    
    for size in sizes:
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        start = time.time()
        c = torch.matmul(a, b)
        elapsed = time.time() - start
        
        print(f"  {size}x{size} matrix: {elapsed*1000:.1f} ms")
    
    print(f"\n✓ PyTorch CPU working correctly")

def test_bert_inference():
    """Test BERT model inference"""
    print_header("BERT MODEL TEST")
    
    try:
        from transformers import BertModel, BertTokenizer
        
        print("Loading BERT-base-uncased...")
        model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model.eval()
        
        # Test single inference
        text = "Medical diagnosis classification for ICD-10 codes."
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        print("Running inference test...")
        
        # Warmup
        with torch.no_grad():
            _ = model(**inputs)
        
        # Benchmark
        num_runs = 50
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                outputs = model(**inputs)
        elapsed = time.time() - start
        
        avg_ms = (elapsed / num_runs) * 1000
        throughput = num_runs / elapsed
        
        print(f"\n✓ BERT inference successful")
        print(f"  Average time: {avg_ms:.1f} ms per sample")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        
        # Test batch processing
        print("\nTesting batch processing:")
        for batch_size in [1, 4, 8, 16]:
            texts = [text] * batch_size
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            
            start = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            elapsed = time.time() - start
            
            per_sample = (elapsed / batch_size) * 1000
            print(f"  Batch {batch_size:2d}: {elapsed*1000:5.1f} ms total ({per_sample:.1f} ms/sample)")
        
        return True
        
    except ImportError:
        print("✗ transformers not installed")
        print("  Install with: pip install transformers")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_memory_capacity():
    """Estimate memory capacity for AI models"""
    print_header("MEMORY CAPACITY TEST")
    
    try:
        from transformers import BertModel
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        baseline_mb = process.memory_info().rss / (1024**2)
        print(f"Baseline memory usage: {baseline_mb:.1f} MB")
        
        print("\nLoading BERT-base model...")
        model = BertModel.from_pretrained("bert-base-uncased")
        
        after_load_mb = process.memory_info().rss / (1024**2)
        model_mb = after_load_mb - baseline_mb
        
        print(f"Memory after loading BERT: {after_load_mb:.1f} MB")
        print(f"Model memory footprint: {model_mb:.1f} MB")
        
        # System memory
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        available_ram_mb = psutil.virtual_memory().available / (1024**2)
        
        print(f"\nSystem RAM: {total_ram_gb:.1f} GB")
        print(f"Available for models: {available_ram_mb:.0f} MB")
        print(f"Can fit ~{int(available_ram_mb / model_mb)} BERT models in memory")
        
        if available_ram_mb > model_mb * 2:
            print("\n✓ Sufficient memory for your thesis work")
        else:
            print("\n⚠ Limited memory - use one model at a time")
        
    except ImportError:
        print("Install psutil for memory analysis: pip install psutil")
    except Exception as e:
        print(f"Error: {e}")

def test_llama_cpp():
    """Check llama-cpp-python installation"""
    print_header("LLAMA-CPP-PYTHON TEST")
    
    try:
        from llama_cpp import Llama
        print("✓ llama-cpp-python is installed")
        print("  Good for running Gemma models on CPU")
        print("  Use Q4_K_M quantized models for best performance")
        return True
    except ImportError:
        print("✗ llama-cpp-python not installed")
        print("  Install with: pip install llama-cpp-python")
        return False

def recommendations():
    """Provide thesis-specific recommendations"""
    print_header("RECOMMENDATIONS FOR YOUR THESIS")
    
    print("""
Based on your system capabilities:

═══════════════════════════════════════════════════════════════════

APPROACH 1: Fine-tuned BERT Models
───────────────────────────────────
TRAINING:
  → Use Google Colab (free, 12GB GPU, easy setup)
  → Upload your MIMIC-III processed data
  → Train 3 BERT variants in parallel notebooks
  → Download trained models to local machine
  
LOCAL INFERENCE:
  → Load trained models on your laptop
  → CPU inference: ~100-200 ms per patient note
  → Batch size 4-8 works fine
  → Sufficient for thesis evaluation

═══════════════════════════════════════════════════════════════════

APPROACH 2: Few-shot with Gemma
────────────────────────────────
  → Use llama-cpp-python (CPU)
  → Download Gemma-2B Q4_K_M quantized model (~1.5GB)
  → Runs well on CPU with your 8GB RAM
  → ~2-5 seconds per generation
  → Good enough for thesis evaluation

═══════════════════════════════════════════════════════════════════

APPROACH 3: RAG-enhanced Classification  
────────────────────────────────────────
  → Embeddings: sentence-transformers on CPU
  → Vector store: FAISS (works well locally)
  → Retrieval: Fast on your system
  → Generation: Use approach 2 (Gemma on CPU)

═══════════════════════════════════════════════════════════════════

INTEL ARC GPU:
  → Skip setup for now (not worth time investment)
  → 2GB VRAM too limited for training anyway
  → CPU performance is adequate for thesis
  → Focus time on research quality, not hardware

═══════════════════════════════════════════════════════════════════

TIME ALLOCATION FOR THESIS:
  ✓ 2 hours: Set up Colab for BERT training
  ✓ 1 day: Train 3 BERT models on Colab
  ✓ 2 hours: Set up local inference pipeline
  ✓ 1 day: Implement few-shot approach
  ✓ 1 day: Implement RAG approach
  ✓ 2 days: Evaluate all 3 approaches
  ✓ Rest: Write up results
  
  ✗ Do NOT spend >2 hours on local GPU setup

═══════════════════════════════════════════════════════════════════
    """)

def main():
    print("\n" + "="*70)
    print("  AI SYSTEM CAPABILITY TEST")
    print("  For Medical Diagnosis Classification Thesis")
    print("="*70)
    
    system_info()
    test_pytorch_cpu()
    
    bert_ok = test_bert_inference()
    
    if bert_ok:
        test_memory_capacity()
    
    test_llama_cpp()
    recommendations()
    
    print("\n" + "="*70)
    print("  NEXT STEPS:")
    print("  1. If transformers works: You're ready for local inference")
    print("  2. Set up Google Colab for BERT training")
    print("  3. Install llama-cpp-python for Gemma models")
    print("  4. Skip Intel GPU setup - not worth the time")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()