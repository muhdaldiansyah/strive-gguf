from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os

# 1. Download the model
model_path = hf_hub_download(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",  # or another mirror
    filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)  # :contentReference[oaicite:1]{index=1}

# 2. Set environment variables
os.environ["LLAMA_LOG_LEVEL"] = "ERROR"
os.environ["GGML_LOG_LEVEL"] = "ERROR"

# 3. Initialize the model
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False
)

# 4. Use it for chat
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello, who are you?"}],
    temperature=0.2,
    max_tokens=128
)
print(response["choices"][0]["message"]["content"])
