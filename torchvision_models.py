import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from nebullvm import optimize_torch_model


def run_torch_model(model, input_tensor, steps=100):
    times = []
    for _ in range(steps):
        st = time.time()
        with torch.no_grad():
            _ = model(input_tensor)
        times.append(time.time() - st)
    return sum(times) / len(times) * 1000


def optimize_and_run(model, input_shape, save_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(input_shape)
    vanilla_time = run_torch_model(model.to(device), input_tensor.to(device))

    with TemporaryDirectory() as tmp_dir:
        optimized_model = optimize_torch_model(
            model,
            batch_size=input_shape[0],
            input_sizes=[input_shape[1:]],
            save_dir=tmp_dir,
            use_torch_api=True,

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
    import torchvision.models as models

    resnet18 = models.resnet18()
    alexnet = models.alexnet()
    vgg16 = models.vgg16()
    squeezenet = models.squeezenet1_0()
    efficientnet_b0 = models.efficientnet_b0()
    efficientnet_b1 = models.efficientnet_b1()
    efficientnet_b2 = models.efficientnet_b2()
    efficientnet_b3 = models.efficientnet_b3()
    efficientnet_b4 = models.efficientnet_b4()
    efficientnet_b5 = models.efficientnet_b5()
    efficientnet_b6 = models.efficientnet_b6()
    efficientnet_b7 = models.efficientnet_b7()
    convnext_tiny = models.convnext_tiny()
    convnext_small = models.convnext_small()
    convnext_base = models.convnext_base()
    convnext_large = models.convnext_large()

    input_shape = (1, 3, 256, 256)
    optimize_and_run(resnet18, input_shape, "resnet18")
    optimize_and_run(alexnet, input_shape, "alexnet")
    optimize_and_run(vgg16, input_shape, "vgg16")
    optimize_and_run(squeezenet, input_shape, "squeezenet")
    optimize_and_run(efficientnet_b0, input_shape, "efficientnet_b0")
    optimize_and_run(efficientnet_b1, input_shape, "efficientnet_b1")
    optimize_and_run(efficientnet_b3, input_shape, "efficientnet_b3")
    optimize_and_run(efficientnet_b4, input_shape, "efficientnet_b4")
    optimize_and_run(efficientnet_b5, input_shape, "efficientnet_b5")
    optimize_and_run(efficientnet_b6, input_shape, "efficientnet_b6")
    optimize_and_run(efficientnet_b7, input_shape, "efficientnet_b7")
    optimize_and_run(convnext_tiny, input_shape, "convnext_tiny")
    optimize_and_run(convnext_small, input_shape, "convnext_small")
    optimize_and_run(convnext_base, input_shape, "convnext_base")
    optimize_and_run(convnext_large, input_shape, "convnext_large")
