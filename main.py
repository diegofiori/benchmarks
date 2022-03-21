import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from nebullvm.api.frontend.huggingface import optimize_huggingface_model
import torch


def run_hugginface_model(model, encoded_input, steps=100):
    times = []
    for _ in range(steps):
        st = time.time()
        with torch.no_grad():
            _ = model(**encoded_input)
        times.append(time.time() - st)
    return sum(times) / len(times) * 1000


def optimized_and_run(text, model, tokenizer, save_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoded_input = tokenizer(text, return_tensors='pt')
    vanilla_time = run_hugginface_model(model.to(device), encoded_input.to(device))
    long_text = " ".join([text]*100)
    long_encoded_input = tokenizer(
        long_text, return_tensors='pt', truncation=True
    )
    vanilla_time_long = run_hugginface_model(model.to(device), long_encoded_input.to(device))

    extra_input_info = [{}] + [{"max_value": 1, "min_value": 0}] * (len(long_encoded_input) - 1)
    model.to("cpu")
    with TemporaryDirectory() as tmp_dir:
        optimized_model = optimize_huggingface_model(
            model=model,
            tokenizer=tokenizer,
            target_text=text,
            batch_size=1,
            max_input_sizes=[tuple(value.size()[1:]) for value in
                             long_encoded_input.values()],
            save_dir=tmp_dir,
            extra_input_info=extra_input_info,
            use_torch_api=False
        )
        optimized_time_long = run_hugginface_model(optimized_model, long_encoded_input)
        optimized_time = run_hugginface_model(optimized_model, encoded_input)
    time_dict = {
        "vanilla_time": vanilla_time,
        "vanilla_time_long": vanilla_time_long,
        "optimized_time": optimized_time,
        "optimized_time_long": optimized_time_long
    }
    Path(save_dir).mkdir(exist_ok=True)
    with open(os.path.join(save_dir, "time_info.json"), "w") as f_out:
        json.dump(time_dict, f_out)


def optimized_and_run_static(text, model, tokenizer, save_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
    extra_input_info = [{}] + [{"max_value": 1, "min_value": 0}] * (len(encoded_input) - 1)
    vanilla_time = run_hugginface_model(model.to(device), encoded_input.to(device))
    model.to("cpu")
    with TemporaryDirectory() as tmp_dir:
        optimized_model = optimize_huggingface_model(
            model=model,
            tokenizer=tokenizer,
            target_text=text,
            batch_size=1,
            max_input_sizes=[tuple(value.size()[1:]) for value in
                             encoded_input.values()],
            save_dir=tmp_dir,
            extra_input_info=extra_input_info,
            use_torch_api=False,
            use_static_shape=True,
        )
        optimized_time = run_hugginface_model(optimized_model, encoded_input)
    time_dict = {
        "vanilla_time": vanilla_time, "optimized_time": optimized_time
    }
    Path(save_dir).mkdir(exist_ok=True)
    with open(os.path.join(save_dir, "time_info.json"), "w") as f_out:
        json.dump(time_dict, f_out)


if __name__ == "__main__":
    from transformers import GPT2Tokenizer, GPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    text = "Replace me by any text you'd like."
    optimized_and_run(text, model, tokenizer, save_dir="gpt2")
    optimized_and_run_static(text, model, tokenizer, "gpt2-static-short")
    long_text = " ".join([text]*200)
    optimized_and_run_static(long_text, model, tokenizer, "gpt2-static-long")

    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    text = "Hello, my dog is cute"
    optimized_and_run(text, model, tokenizer, save_dir="bert")
    optimized_and_run_static(text, model, tokenizer, "bert-static-short")
    long_text = " ".join([text] * 200)
    optimized_and_run_static(long_text, model, tokenizer, "bert-static-long")
