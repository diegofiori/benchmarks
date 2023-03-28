from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to("cuda")


def compute_latency(hf_model, input_dict, n_warmup=5, n_repeat=10):
    model.eval()

    for _ in range(n_warmup):
        with torch.no_grad():
            hf_model(**input_dict)

    latencies = []
    for _ in range(n_repeat):
        with torch.no_grad():
            start = time.time()
            hf_model.generate(**input_dict, max_length=1024, num_beams=5)
            end = time.time()
        latencies.append(end - start)
    return np.mean(latencies)


input_text = "\n".join(["Hello, my dog is cute."] * 300)
input_hf = tokenizer(input_text, return_tensors="pt", truncation=True).to("cuda")
long_latency = compute_latency(model, input_hf)
print(f"tokens: {input_hf['input_ids'].shape[1]}, latency: {long_latency}")
input_text = "\n".join(["Hello, my dog is cute."] * 150)
input_hf = tokenizer(input_text, return_tensors="pt", truncation=True).to("cuda")
short_latency = compute_latency(model, input_hf)
print(f"tokens: {input_hf['input_ids'].shape[1]}, latency: {short_latency}")
print(f"speedup: {long_latency / short_latency}")
