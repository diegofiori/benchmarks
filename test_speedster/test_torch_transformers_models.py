import argparse
import time

import numpy as np
import torch
from transformers import AutoTokenizer, TFAutoModel, AutoModel

from speedster import optimize_model


def create_list_of_strings():
    texts = [
        "This is a test",
        "This is another test",
        "This is a third test",
        "This is a fourth test",
        "This is a fifth test",
    ]
    long_texts = [
        "This is a test" * 100,
        "This is another test" * 100,
    ]
    data = (texts + long_texts) * 10
    return data


def get_most_downloaded_models():
    model_names = [
        "bert-base-uncased",
        "xml-roberta-base",
        "distilbert-base-uncased",
        "gpt2",
        "distilbert-base-uncased-finetuned-sst-2-english",
    ]
    return model_names


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def tokenize_data(tokenizer, data):
    return tokenizer(data, return_tensors="pt", padding=True)


def optimize_hf_model(batch_size):
    model_names = get_most_downloaded_models()
    data = create_list_of_strings()
    speed_up_dict = {}
    for model_name in model_names:
        tokenizer, model = load_model_and_tokenizer(model_name)
        tokenized_data = [tokenize_data(tokenizer, data[i : i + batch_size]) for i in range(0, len(data), batch_size)]
        optimized_model = optimize_model(
            model,
            input_data=tokenized_data,
            metric_drop_ths=0.1,
            metric="numeric_precision",
            optimization_time="unconstrained",
            ignore_compilers=["tvm"],
        )
        original_times = []
        optimized_times = []
        with torch.no_grad():
            for i, batch in enumerate(tokenized_data):
                st = time.time()
                _ = model(**batch)
                if i > 10:
                    original_times.append(time.time() - st)

                for i, batch in enumerate(tokenized_data):
                    st = time.time()
                    _ = optimized_model(**batch)
                    if i > 10:
                        optimized_times.append(time.time() - st)
        speed_up_dict[
            model_name.__class__.__name__
        ] = float(np.median(original_times) / np.median(optimized_times))
    return speed_up_dict


def save_json(filename: str, dictionary: dict):
    import json
    with open(filename, "w") as f:
        json.dump(dictionary, f)


def print_average_speed_up(speed_up_dict: dict):
    average_speed_up = np.mean(list(speed_up_dict.values()))
    print(f"Average speed up: {average_speed_up}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--output_file", "-o", type=str, default="speed_up.json",
        help="Output file"
    )
    args = parser.parse_args()
    speedup_dict = optimize_hf_model(args.batch_size)
    save_json(args.output_file, speedup_dict)
    print_average_speed_up(speedup_dict)
