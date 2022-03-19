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
    encoded_input = tokenizer(text, return_tensors='pt')
    vanilla_time = run_hugginface_model(model, encoded_input)
    long_text = " ".join([text]*100)
    long_encoded_input = tokenizer(
        long_text, return_tensors='pt', truncation=True
    )
    vanilla_time_long = run_hugginface_model(model, long_encoded_input)

    extra_input_info = [{}] + [{"max_value": 1, "min_value": 0}] * (len(long_encoded_input) - 1)
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


if __name__ == "__main__":
    from transformers import GPT2Tokenizer, GPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    text = "Replace me by any text you'd like."
    optimized_and_run(text, model, tokenizer, save_dir="gpt2")

    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    text = "Hello, my dog is cute"
    optimized_and_run(text, model, tokenizer, save_dir="bert")

