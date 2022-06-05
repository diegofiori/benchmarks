import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from composer import Trainer, ComposerModel
from torchmetrics import Accuracy

from run_custom_model import CustomPatchedModel, CustomModel, \
    CustomPatchedDataset, CustomDataset, get_fastai_dataloaders, \
    compute_accuracy


class ComposerIdiot(ComposerModel):
    def __init__(self, optimized):
        super().__init__()
        self.model = CustomPatchedModel() if optimized else CustomModel()

    def forward(self, batch):
        inputs, _ = batch
        return self.model(inputs)

    def loss(self, outputs, batch):
        _, targets = batch
        return F.cross_entropy(outputs, targets)

    def validate(self, batch):
        inputs, targets = batch
        with torch.no_grad():
            preds = self.model(inputs)
        return preds, targets

    def metrics(self, train: bool = False):
        return Accuracy()


def initialize_mosaicml(dls, bs, max_epochs):
    train_dataloader = DataLoader(CustomDataset(dls.train), batch_size=bs)
    eval_dataloader = DataLoader(CustomDataset(dls.valid), batch_size=bs)
    model = ComposerIdiot(False)
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration=f'{max_epochs}ep',
        device='cpu',
    )
    return trainer


def initialize_mosaic_with_nebulgym(dls, bs, max_epochs):
    train_dataloader = DataLoader(CustomPatchedDataset(dls.train), batch_size=bs)
    eval_dataloader = DataLoader(CustomPatchedDataset(dls.valid), batch_size=bs)
    model = ComposerIdiot(True)
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration=f'{max_epochs}ep',
        device='cpu',
    )
    return trainer


def train_model(bs, max_epochs, with_nebulgym):
    dls = get_fastai_dataloaders()
    if with_nebulgym:
        trainer = initialize_mosaic_with_nebulgym(dls, bs, max_epochs)
    else:
        trainer = initialize_mosaicml(dls, bs, max_epochs)
    st = time.time()
    trainer.fit()
    return time.time() - st


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int)
    parser.add_argument("--batch_size", "-bs", type=int)
    args = parser.parse_args()
    bs = args.batch_size or 8
    max_epochs = args.epochs or 10
    json_dict = {}
    mosaic_time = train_model(bs, max_epochs, False)
    print(f"Mosaic time: {mosaic_time}")
    json_dict["mosaic"] = mosaic_time
    nebulgym_time = train_model(bs, max_epochs, True)
    print(f"Nebulgym time: {nebulgym_time}")
    json_dict["nebulgym"] = nebulgym_time
    import json
    with open("result_mosaic.json", "w") as f:
        json.dump(json_dict, f)
