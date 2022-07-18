import copy
import json
from typing import Type, Dict

import torch
from torchvision.models import resnet152
from nebulgym.decorators.torch_decorators import accelerate_model
from torch.utils.data import DataLoader

from run_custom_model import CustomPatchedDataset, CustomDataset, \
    get_fastai_dataloaders, train_model


BACKEND = "PYTORCH"
LR = 1e-3


def patch_model_class(model_class):
    new_model_class = copy.deepcopy(model_class)
    new_model_class = accelerate_model(
        patch_backprop=False, reduce_memory=True, backends=[BACKEND]
    )(new_model_class)
    return new_model_class


def run_standard_training(model_class: Type[torch.nn.Module], batch_size: int, max_epochs: int, json_dict: Dict):
    dls = get_fastai_dataloaders()
    ds = CustomDataset(dls.train)
    dl = DataLoader(ds, batch_size=batch_size)
    model = model_class()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fun = torch.nn.CrossEntropyLoss()
    runtime = train_model(dl, model, optimizer, loss_fun, max_epochs)
    json_dict["standard"] = runtime


def run_nebulgym_training(
        model_class: Type[torch.nn.Module],
        batch_size: int,
        max_epochs: int,
        json_dict: Dict,
):
    dls = get_fastai_dataloaders()
    ds = CustomPatchedDataset(dls.train)
    dl = DataLoader(ds, batch_size=batch_size)
    model_class = patch_model_class(model_class)
    model = model_class()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fun = torch.nn.CrossEntropyLoss()
    runtime = train_model(dl, model, optimizer, loss_fun, max_epochs)
    json_dict["nebulgym"] = runtime


class ResnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._inner_model = resnet152()

    def forward(self, input):
        return self._inner_model(input)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int)
    parser.add_argument("--batch_size", "-bs", type=int)
    args = parser.parse_args()
    bs = args.batch_size or 8
    epochs = args.epochs or 10
    result_dict = {}
    m_class = ResnetModel
    run_standard_training(m_class, bs, epochs, result_dict)
    run_nebulgym_training(m_class, 2*bs, epochs, result_dict)

    with open("result.json", "w") as f:
        json.dump(result_dict, f)