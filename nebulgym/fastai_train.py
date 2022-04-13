import copy
import time

from fastai.vision.all import (
    untar_data, URLs, ImageDataLoaders, RandomResizedCrop, Normalize,
    imagenet_stats, cnn_learner, resnet34, accuracy
)
try:
    from torch_ort import ORTModule

    class NebORTModule(ORTModule):
        def __getitem__(self, item):
            return self._torch_module._flattened_module[item]

except ImportError:
    import warnings
    warnings.warn("No torch-ort installation found")
    from torch.nn import Sequential as NebORTModule


def _get_dls():
    path = untar_data(URLs.IMAGENETTE_160)
    dls = ImageDataLoaders.from_folder(
        path,
        valid='val',
        item_tfms=RandomResizedCrop(128, min_scale=0.35),
        batch_tfms=Normalize.from_stats(*imagenet_stats),
        num_workers=0
    )
    return dls


def run_fastai_train(num_epochs: int = 5):
    dls = _get_dls()
    learn = cnn_learner(dls, resnet34, metrics=accuracy, pretrained=False)
    st = time.time()
    learn.fit_one_cycle(num_epochs, 5e-3)
    training_time = time.time() - st
    return training_time


def run_model_with_onnxruntime(num_epochs: int = 5):
    dls = _get_dls()
    learn = cnn_learner(dls, resnet34, metrics=accuracy, pretrained=False)
    model_copy = copy.deepcopy(learn.model)
    onnx_model = NebORTModule(model_copy)
    learn.model = onnx_model
    st = time.time()
    learn.fit_one_cycle(num_epochs, 5e-3)
    training_time = time.time() - st
    return training_time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="The number of epochs")
    args = parser.parse_args()
    num_epochs = int(args.epochs or 5)
    fastai_time = run_fastai_train(num_epochs)
    onnx_time = run_model_with_onnxruntime(num_epochs)
    print(f"FastAI vanilla time {fastai_time}. ONNX time: {onnx_time}")
