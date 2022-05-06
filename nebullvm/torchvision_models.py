import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from nebullvm import optimize_torch_model
from torch.utils.data import DataLoader



def run_torch_model(model, input_tensor, steps=100):
    times = []
    for _ in range(steps):
        st = time.time()
        with torch.no_grad():
            _ = model(input_tensor)
        times.append(time.time() - st)
    return sum(times) / len(times) * 1000


def optimize_and_run(model, input_shape, save_dir, quantization_ths, from_dataloader: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(input_shape)
    vanilla_time = run_torch_model(model.to(device), input_tensor.to(device))
    with TemporaryDirectory() as tmp_dir:
        if from_dataloader:
            data = [((torch.randn(input_shape[1:]), ), 0) for _ in range(500)]
            dataloader = DataLoader(data, batch_size=input_shape[0])
            optimized_model = optimize_torch_model(
                model,
                save_dir=tmp_dir,
                use_torch_api=False,
                quantization_ths=quantization_ths,
                dataloader=dataloader,
                ignore_compilers=["tvm"],
            )
        else:
            optimized_model = optimize_torch_model(
                model,
                batch_size=input_shape[0],
                input_sizes=[input_shape[1:]],
                save_dir=tmp_dir,
                use_torch_api=False,
                quantization_ths=quantization_ths,
                ignore_compilers=["tvm"],
            )
        optimized_time = run_torch_model(optimized_model, input_tensor)
    time_dict = {
        "vanilla_time": vanilla_time,
        "optimized_time": optimized_time,
    }
    Path(save_dir).mkdir(exist_ok=True)
    with open(os.path.join(save_dir, "time_info.json"), "w") as f_out:
        json.dump(time_dict, f_out)


if __name__ == "__main__":
    import argparse
    import torchvision.models as models
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantization_ths",
        "-q",
        type=float,
        help="The drop in precision accepted after quantization"
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        help="The batch size"
    )
    parser.add_argument(
        "--from_data",
        "-d",
        action="store_true",
        help="Flag for optimizing the model from dataset or from metadata."
    )
    args = parser.parse_args()
    quantization_ths = args.quantization_ths
    bs = args.batch_size or 1
    from_data = args.from_data or False
    print(f"Quantization: {quantization_ths}")
    input_shape = (bs, 3, 256, 256)
    model_tuples = [
        (models.resnet18(), "resnet18"),
        (models.squeezenet1_0(), "squeezenet"),
        (models.efficientnet_b0(), "efficientnet_b0"),
        (models.efficientnet_b1(), "efficientnet_b1"),
        (models.efficientnet_b2(), "efficientnet_b2"),
        (models.efficientnet_b3(), "efficientnet_b3"),
        (models.efficientnet_b4(), "efficientnet_b4"),
        (models.efficientnet_b5(), "efficientnet_b5"),
        (models.efficientnet_b6(), "efficientnet_b6"),
        (models.efficientnet_b7(), "efficientnet_b7"),
        (models.convnext_tiny(), "convnext_tiny"),
        (models.convnext_small(), "convnext_small"),
        (models.convnext_base(), "convnext_base"),
        (models.convnext_large(), "convnext_large"),
        (models.resnet34(), "resnet34"),
        (models.resnet50(), "resnet50"),
        (models.resnet101(), "resnet101"),
        (models.resnet152(), "resnet152"),
    ]
    if quantization_ths is None:
        base_path = "base"
    else:
        base_path = "quantization"
    Path(base_path).mkdir(exist_ok=True)

    for model, model_name in model_tuples:
        model_dir = os.path.join(base_path, model_name)
        if Path(model_dir).exists():
            continue
        optimize_and_run(model, input_shape, model_dir, quantization_ths, from_data)
