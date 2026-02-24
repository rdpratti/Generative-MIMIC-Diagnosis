from llama_cpp import Llama

llm = Llama(
    model_path="E:/Education/llama/models/gemma-3n-E4B-it-Q4_K_M.gguf",
    n_gpu_layers=35,
    verbose=True
)

prompt = "Hello"
output = llm(prompt, max_tokens=20)
print(output['choices'][0]['text'])