import argparse
import time

import numpy as np
import torch.utils.data
from speedster import optimize_model
from torchvision import datasets, transforms


def build_validation_dataloader_imagenet(batch_size: int):
    val_dataset = datasets.ImageNet(
        root="data",
        split="val",
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    return val_dataloader


def get_torchvision_models():
    """The function returns a list containing an example for each torchvision
    architecture. Model should be used with no pretrained weights.
    """
    import torchvision.models as models

    models_list = [
        models.alexnet(pretrained=False),
        models.densenet121(pretrained=False),
        models.googlenet(pretrained=False),
        models.inception_v3(pretrained=False),
        models.mnasnet0_5(pretrained=False),
        models.mobilenet_v2(pretrained=False),
        models.resnet18(pretrained=False),
        models.resnext50_32x4d(pretrained=False),
        models.shufflenet_v2_x0_5(pretrained=False),
        models.squeezenet1_0(pretrained=False),
        models.vgg19(pretrained=False),
        models.vgg19_bn(pretrained=False),
        models.wide_resnet50_2(pretrained=False),
        models.vit_b_16(pretrained=False),
    ]
    return models_list


def optimizer_with_speedster(batch_size: int):
    models = get_torchvision_models()
    val_dataloader = build_validation_dataloader_imagenet(batch_size)

    speed_up_dict = {}

    for model in models:
        try:
            model = model.eval()
            optimized_model = optimize_model(
                model,
                input_data=val_dataloader,
                metric_drop_ths=0.1,
                metric="numeric_precision",
                optimization_time="unconstrained",
                ignore_compilers=["tvm"],
            )
        except Exception as e:
            print(f"Error optimizing {model}: {e}")
            speed_up_dict[model.__class__.__name__] = 1
            continue
        original_times = []
        optimized_times = []
        with torch.no_grad():
            for i, (data, target) in enumerate(val_dataloader):
                st = time.time()
                _ = model(data)
                if i > 10:
                    original_times.append(time.time() - st)

            for i, (data, target) in enumerate(val_dataloader):
                st = time.time()
                _ = optimized_model(data)
                if i > 10:
                    optimized_times.append(time.time() - st)
        speed_up_dict[
            model.__class__.__name__
        ] = np.median(original_times) / np.median(optimized_times)
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
    parser.add_argument(
        "--batch_size", "-b", type=int, default=32, help="Batch size for validation"
    )
    parser.add_argument(
        "--output_file", "-o", type=str, default="speed_up.json", help="Output file"
    )
    args = parser.parse_args()
    speedup_dict = optimizer_with_speedster(args.batch_size)
    print_average_speed_up(speedup_dict)
    save_json(args.output_file, speedup_dict)
