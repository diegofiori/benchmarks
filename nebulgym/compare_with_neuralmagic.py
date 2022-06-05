import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sparseml.pytorch.optim import ScheduledModifierManager

from run_custom_model import CustomPatchedModel, CustomModel, \
    CustomPatchedDataset, CustomDataset, get_fastai_dataloaders, \
    compute_accuracy


def get_training_objects(use_nebulgym: bool, bs):
    dls = get_fastai_dataloaders()
    if use_nebulgym:
        model = CustomPatchedModel()
        ds = CustomPatchedDataset(dls.train)
    else:
        model = CustomModel()
        ds = CustomDataset(dls.train)
    dl = DataLoader(ds, batch_size=bs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cost_fun = torch.nn.CrossEntropyLoss()
    return model, dl, optimizer, cost_fun


def train_model_neuralmagic(use_nebulgym, bs, max_epochs):
    model, train_dl, optimizer, cost_fun = get_training_objects(use_nebulgym, bs)
    steps_per_epoch = len(train_dl)
    manager = ScheduledModifierManager.from_yaml("neuralmagic_recipe.yaml")
    optimizer = manager.modify(model, optimizer, steps_per_epoch)
    model.train()
    st = time.time()
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(max_epochs):
        avg_loss = 0
        avg_accuracy = 0
        for x, y in tqdm(train_dl):
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
            f"Epoch {epoch}: Loss={avg_loss / len(train_dl)}, "
            f"accuracy: {avg_accuracy / len(train_dl)}"
        )
    manager.finalize(model)
    total_time = time.time() - st
    print(
        f"Total time: {total_time} s. Time per epoch: {total_time / max_epochs} s")

    return total_time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", "-e", type=int)
    parser.add_argument("--batch_size", "-bs", type=int)
    args = parser.parse_args()
    bs = args.batch_size or 8
    max_epochs = args.epochs or 10
    json_dict = {}
    neural_time = train_model_neuralmagic(False, bs, max_epochs)
    json_dict["neuralmagic"] = neural_time
    nebulgym_time = train_model_neuralmagic(True, bs, max_epochs)
    json_dict["nebulgym"] = nebulgym_time
    import json

    with open("result_neuralmagic.json", "w") as f:
        json.dump(json_dict, f)

