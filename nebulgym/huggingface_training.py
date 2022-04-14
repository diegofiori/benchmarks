from tempfile import TemporaryDirectory
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

try:
    from torch_ort import ORTModule
except ImportError:
    import warnings

    warnings.warn("No torch-ort installation found")
    from torch.nn import Sequential as ORTModule


def _load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2)
    return model, tokenizer


def _get_dataset(tokenizer):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    imdb = load_dataset("imdb")
    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenized_imdb, data_collator


def _get_trainer(
        model,
        tokenizer,
        tokenized_imdb,
        data_collator,
        epochs: int,
        temp_dir: str
):
    training_args = TrainingArguments(
        output_dir=temp_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer


def run_huggingface_train(epochs: int):
    model, tokenizer = _load_model()
    tokenized_imdb, data_collator = _get_dataset(tokenizer)
    with TemporaryDirectory() as tmp_dir:
        trainer = _get_trainer(
            model, tokenizer, tokenized_imdb, data_collator, epochs, tmp_dir
        )
        st = time.time()
        trainer.train()
        return time.time() - st


def run_onnx_train(epochs: int):
    model, tokenizer = _load_model()
    tokenized_imdb, data_collator = _get_dataset(tokenizer)
    model = ORTModule(model)
    with TemporaryDirectory() as tmp_dir:
        trainer = _get_trainer(
            model, tokenizer, tokenized_imdb, data_collator, epochs, tmp_dir
        )
        st = time.time()
        trainer.train()
        return time.time() - st


if __name__ == "__train__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="The number of epochs")
    args = parser.parse_args()
    epochs = int(args.epochs or 5)
    hf_time = run_huggingface_train(epochs)
    onnx_time = run_onnx_train(epochs)
    print(f"HF vanilla time {hf_time}. ONNX time: {onnx_time}")
