from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b").to("cuda").half()


def compute_latency(hf_model, input_dict, max_length,  n_warmup=5, n_repeat=5):
    model.eval()

    for _ in range(n_warmup):
        with torch.no_grad():
            hf_model(**input_dict)

    latencies = []
    for i in range(n_repeat):
        print(i)
        with torch.no_grad():
            start = time.time()
            hf_model.generate(**input_dict, max_length=max_length, num_beams=5)
            end = time.time()
        latencies.append(end - start)
    return np.mean(latencies)


input_text = "\n".join(["Hello, my dog is cute."] * 300)
input_hf = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
print(input_hf["input_ids"].shape)
long_latency = compute_latency(model, input_hf, 2048)
print(f"tokens: {input_hf['input_ids'].shape[1]}, latency: {long_latency}")
input_text = "\n".join(["Hello, my dog is cute."] * 150)
input_hf = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
short_latency = compute_latency(model, input_hf, 1536)
print(f"tokens: {input_hf['input_ids'].shape[1]}, latency: {short_latency}")
print(f"speedup: {long_latency / short_latency}")
