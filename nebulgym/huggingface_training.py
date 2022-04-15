import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch.utils.data.dataset
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

try:
    rammer_path = f"{Path.home()}/nnfusion/build/src/tools/nnfusion"
    if not Path(rammer_path).exists():
        raise ImportError
    os.environ["PATH"] = os.path.abspath(rammer_path) + ":" + os.environ["PATH"]
    sys.path.insert(1, os.path.abspath(f"{Path.home()}/nnfusion/src/python"))

    from nnfusion.trainer import PTTrainer as RummerTrainer
except ImportError:
    import warnings
    warnings.warn("No Rummer installation detected")


def _load_model():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)
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


def _get_dataloaders(data, batch_size):
    class _TextDataset(torch.utils.data.dataset.Dataset):
        def __init__(self, dataset):
            self.internal_dataset = dataset
            self.keys = ['input_ids', 'attention_mask', 'label']

        def __getitem__(self, item):
            return {
                key: torch.tensor(self.internal_dataset[item][key])
                for key in self.keys
            }

    train_ds = _TextDataset(data["train"])
    val_ds = _TextDataset(data["val"])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(val_ds, batch_size)
    return train_dl, valid_dl


class _WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super(_WrapperModel, self).__init__()
        self._model = model

    def forward(self, input_ids, attention_mask, labels):
        out = self._model(input_ids,
                          attention_mask=attention_mask,
                          labels=labels)
        return out.loss


def run_rammer_train(epochs: int):
    model, tokenizer = _load_model()
    data, _ = _get_dataset(tokenizer)
    train_loader, val_loader = _get_dataloaders(data, 16)

    device = "cuda:0"
    model.to(device)
    # TODO: should switch to train() once nnf dropout kernel ready
    model.eval()
    wrapper = _WrapperModel(model)

    codegen_flags = {
        "autodiff":
            True,  # add backward graph
        "training_mode":
            True,  # move weight external
        "extern_result_memory":
            True,  # move result external
        "training_optimizer":
            '\'' + json.dumps({
                "optimizer": "SGD",
                "learning_rate": 0.0001
            }) + '\'',  # training optimizer configs
        "blockfusion_level":
            0,  # TODO: fix blockfusion problem in bert training
        "enable_all_bert_fusion":
            True,  # enable all bert fusion optimizations
    }
    trainer = RummerTrainer(wrapper, device=device, codegen_flags=codegen_flags)
    st = time.time()
    sum_nnf_loss = 0
    sum_iter = 0
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if sum_iter == 100:
                print("Epoch {}, batch {}, nnf_loss {}".format(
                    epoch, i, sum_nnf_loss / sum_iter))
                sum_nnf_loss = 0
                sum_iter = 0

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(0).to(device)

            nnf_loss = trainer(input_ids, attention_mask, labels)

            sum_nnf_loss += nnf_loss
            sum_iter += 1
        with torch.no_grad():
            sum_nnf_loss_val = 0
            for val_batch in val_loader:
                val_loss = model(**val_batch)[0]
                sum_nnf_loss_val += val_loss
            print("Validation -- Epoch {}, nnf_loss {}".format(
                epoch, sum_nnf_loss_val/len(val_loader)
            ))

    return time.time() - st


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="The number of epochs")
    args = parser.parse_args()
    epochs = int(args.epochs or 5)
    try:
        onnx_time = run_onnx_train(epochs)
    except Exception as ex:
        onnx_time = np.inf
        print(ex)

    try:
        rammer_time = run_rammer_train(epochs)
    except Exception as ex:
        rammer_time = np.inf
        print(ex)

    hf_time = run_huggingface_train(epochs)
    print(
        f"HF vanilla time {hf_time}. \n"
        f"ONNX time: {onnx_time}. \n"
        f"Rammer time {rammer_time}.\n"
    )
