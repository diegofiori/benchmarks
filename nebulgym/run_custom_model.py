import json
import time

from fastai.data.external import untar_data, URLs
from fastai.data.transforms import get_image_files
from fastai.vision.augment import Resize
from fastai.vision.data import ImageDataLoaders
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
import torch

from nebulgym.decorators.torch_decorators import accelerate_model, accelerate_dataset

BACKEND = "PYTORCH"


def get_fastai_dataloaders():
    path = untar_data(URLs.PETS)
    files = get_image_files(path / "images")

    def label_func(f): return f[0].isupper()

    dls = ImageDataLoaders.from_name_func(path, files, label_func,
                                          item_tfms=Resize(224), bs=1, num_workers=0)
    return dls


class CustomDataset(Dataset):
    def __init__(self, dl):
        self._dl = dl
        self._pointer = 0
        self._iter_dl = iter(self._dl)

    def __getitem__(self, item):
        if self._pointer >= len(self._dl):
            self._pointer = 0
            self._iter_dl = iter(self._dl)
        x, y = next(self._iter_dl)
        self._pointer += 1
        return x[0], int(y[0])

    def __len__(self):
        return len(self._dl)


class CustomModel(Module):
    def __init__(self):
        super().__init__()
        self._avg_pool = torch.nn.AvgPool2d(4)
        self._linear = torch.nn.Linear(3136, 1024)
        self._relu = torch.nn.ReLU()
        self._linears = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self._avg_pool(x).mean(dim=-3).view(-1, 3136)
        x = self._relu(self._linear(x))
        return self._linears(x)


def compute_accuracy(pred, target):
    with torch.no_grad():
        pred = pred.argmax(dim=-1)
        return torch.mean(1.*(pred == target))


def initialize_model_and_data(dls, bs):
    ds = CustomDataset(dls.train)
    dl = DataLoader(ds, batch_size=bs)
    model = CustomModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cost_fun = torch.nn.CrossEntropyLoss()
    return dl, model, optimizer, cost_fun


def train_model(dl, model, optimizer, cost_fun, max_epochs):
    model.train()
    st = time.time()
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(max_epochs):
        avg_loss = 0
        avg_accuracy = 0
        for x, y in tqdm(dl):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = cost_fun(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_accuracy += compute_accuracy(pred, y)
            avg_loss += loss.detach().mean()
        print(
            f"Epoch {epoch}: Loss={avg_loss / len(dl)}, accuracy: {avg_accuracy / len(dl)}")
    total_time = time.time() - st
    print(
        f"Total time: {total_time} s. Time per epoch: {total_time / max_epochs} s")

    return total_time


def train_and_save(max_epochs, bs, json_dict):
    dls = get_fastai_dataloaders()
    dl, model, optimizer, cost_fun = initialize_model_and_data(dls, bs)
    total_time = train_model(dl, model, optimizer, cost_fun, max_epochs)
    name = f"standard_{max_epochs}"
    json_dict[name] = total_time


@accelerate_dataset(preloaded_data=20, max_memory_size=None)
class CustomPatchedDataset(Dataset):
    def __init__(self, dl):
        self._dl = dl
        self._pointer = 0
        self._iter_dl = iter(self._dl)

    def __getitem__(self, item):
        if self._pointer >= len(self._dl):
            self._pointer = 0
            self._iter_dl = iter(self._dl)
        x, y = next(self._iter_dl)
        self._pointer += 1
        return x[0], int(y[0])

    def __len__(self):
        return len(self._dl)


@accelerate_model(patch_backprop=True, backends=[BACKEND])
class CustomPatchedModel(Module):
    def __init__(self):
        super().__init__()
        self._avg_pool = torch.nn.AvgPool2d(4)
        self._linear = torch.nn.Linear(3136, 1024)
        self._relu = torch.nn.ReLU()
        self._linears = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self._avg_pool(x).mean(dim=-3).view(-1, 3136)
        x = self._relu(self._linear(x))
        return self._linears(x)


def initialize_fast_model_and_data(dls, bs):
    ds = CustomPatchedDataset(dls.train)
    dl = DataLoader(ds, batch_size=bs)
    model = CustomPatchedModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cost_fun = torch.nn.CrossEntropyLoss()
    return dl, model, optimizer, cost_fun


def train_and_save_with_nebulgym(max_epochs, bs, json_dict):
    dls = get_fastai_dataloaders()
    dl, model, optimizer, cost_fun = initialize_fast_model_and_data(dls, bs)
    total_time = train_model(dl, model, optimizer, cost_fun, max_epochs)
    name = f"nebulgym_{max_epochs}"
    json_dict[name] = total_time


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", "-e", type=int)
    parser.add_argument("--batch_size", "-bs", type=int)
    args = parser.parse_args()
    bs = args.batch_size or 8
    max_epochs = args.epochs or 10
    json_dict = {}
    train_and_save(max_epochs, bs, json_dict)
    train_and_save_with_nebulgym(max_epochs, bs, json_dict)
    with open("results.json", "w") as f:
        json.dump(json_dict, f)