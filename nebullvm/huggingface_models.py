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


def optimized_and_run(texts, model, tokenizer, save_dir, quantization_ths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _text = texts[0]
    encoded_input = tokenizer(_text, return_tensors='pt')
    vanilla_time = run_hugginface_model(model.to(device), encoded_input.to(device))
    _long_text = " ".join([_text]*100)
    long_encoded_input = tokenizer(
        _long_text, return_tensors='pt', truncation=True
    )
    vanilla_time_long = run_hugginface_model(model.to(device), long_encoded_input.to(device))

    extra_input_info = [{}] + [{"max_value": 1, "min_value": 0}] * (len(long_encoded_input) - 1)
    model.to("cpu")
    with TemporaryDirectory() as tmp_dir:
        optimized_model = optimize_huggingface_model(
            model=model,
            tokenizer=tokenizer,
            input_texts=texts,
            batch_size=1,
            max_input_sizes=[tuple(value.size()[1:]) for value in
                             long_encoded_input.values()],
            save_dir=tmp_dir,
            extra_input_info=extra_input_info,
            use_torch_api=False,
            perf_loss_ths=quantization_ths,
            ignore_compilers=["tvm"],
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


def optimized_and_run_static(texts, model, tokenizer, save_dir, quantization_ths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoded_input = tokenizer(texts[0], return_tensors='pt', truncation=True)
    extra_input_info = [{}] + [{"max_value": 1, "min_value": 0}] * (len(encoded_input) - 1)
    vanilla_time = run_hugginface_model(model.to(device), encoded_input.to(device))
    model.to("cpu")
    with TemporaryDirectory() as tmp_dir:
        optimized_model = optimize_huggingface_model(
            model=model,
            tokenizer=tokenizer,
            input_texts=texts,
            batch_size=1,
            max_input_sizes=[tuple(value.size()[1:]) for value in
                             encoded_input.values()],
            save_dir=tmp_dir,
            extra_input_info=extra_input_info,
            use_torch_api=False,
            use_static_shape=True,
            tokenizer_args={"truncation": True},
            perf_loss_ths=quantization_ths,
            ignore_compilers=["tvm"],
        )
        optimized_time = run_hugginface_model(optimized_model, encoded_input)
    time_dict = {
        "vanilla_time": vanilla_time, "optimized_time": optimized_time
    }
    Path(save_dir).mkdir(exist_ok=True)
    with open(os.path.join(save_dir, "time_info.json"), "w") as f_out:
        json.dump(time_dict, f_out)

TEXTS = [
    "Replace me by any text you'd like.",
    "The prime minister of UK selected the wrong piece of cake!",
    "I really like pies.",
    "No one really thinks that I can solve the problem, but I will.",
    "Nebuly is the most amazing company ever built! Do you agree?!?",
    "Op, op the unicorn is coming!",
    "A major challenge we hear from our enterprise machine learning customers is managing the ever-growing hardware inference targets modern use cases demand.",
    "As more organizations embrace a multi-cloud approach, the difficulty of hardware identification and benchmarking becomes even more challenging.",
    "The OctoML Machine Learning Deployment Platform now supports inferencing targets.",
    "I personally don't like octoML as a company"
]


if __name__ == "__main__":
    import argparse
    from transformers import GPT2Tokenizer, GPT2Model
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantization_ths", "-q", type=float, help="Quantization ths"
    )
    parser.add_argument(
        "--static", "-s", action="store_true", help="Use static inference"
    )
    args = parser.parse_args()
    quantization_ths = args.quantization_ths
    static_inference = args.static
    if quantization_ths is None:
        base_path = "base"
    else:
        base_path = "quantization"
    Path(base_path).mkdir(exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    text = TEXTS[0]
    if static_inference:
        optimized_and_run_static([text], model, tokenizer, os.path.join(base_path, "gpt2-static-short"), quantization_ths=quantization_ths)
        long_text = " ".join([text] * 200)
        optimized_and_run_static([long_text], model, tokenizer, os.path.join(base_path, "gpt2-static-long"), quantization_ths=quantization_ths)
    else:
        optimized_and_run(TEXTS, model, tokenizer, save_dir=os.path.join(base_path, "gpt2-dynamic"), quantization_ths=quantization_ths)

    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    text = "Hello, my dog is cute"
    if static_inference:
        optimized_and_run_static([text], model, tokenizer,
                                 os.path.join(base_path, "bert-static-short"),
                                 quantization_ths=quantization_ths)
        long_text = " ".join([text] * 200)
        optimized_and_run_static([long_text], model, tokenizer,
                                 os.path.join(base_path, "bert-static-long"),
                                 quantization_ths=quantization_ths)
    else:
        optimized_and_run(TEXTS, model, tokenizer, save_dir=os.path.join(base_path, "bert-dynamic"), quantization_ths=quantization_ths)

