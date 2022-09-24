import json
import os
import time
import types
from pathlib import Path

import torch
from nebullvm.api.functions import optimize_model
from nebullvm.utils.feedback_collector import FEEDBACK_COLLECTOR


def run_torch_model(model, input_tensors):
    times = []
    model.eval()
    for input_tensor in input_tensors:
        st = time.time()
        with torch.no_grad():
            _ = model(input_tensor)
        times.append(time.time() - st)
    return sum(times) / len(times) * 1000


def optimize_and_run(model, input_shape, optimization_type, quantization_ths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_tensors = [torch.randn(input_shape).to(device) for _ in range(100)]
    vanilla_time = run_torch_model(model.eval().to(device), test_tensors)
    data = [((torch.randn(input_shape), ), 0) for _ in range(500)]
    optimized_model = optimize_model(
        model.eval(),
        input_data=data,
        metric_drop_ths=quantization_ths,
        metric="numeric_precision",
        optimization_time=optimization_type,
        ignore_compilers=["tvm"],
    )
    time_dict = {
        "vanilla_time": vanilla_time,
        "optimized_time": run_torch_model(optimized_model, test_tensors),
    }
    return time_dict


def save_ouput_in_dict(dictionary):
    def wrap(function):
        def new_function(self, *args, **kwargs):
            dictionary.update(self._latency_dict)
            return function(self, *args, **kwargs)
        return new_function
    return wrap


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
        "--use_compression",
        "-c",
        action="store_true",
        help="Flag for optimizing the model using compression techniques."
    )
    args = parser.parse_args()
    quantization_ths = args.quantization_ths
    bs = args.batch_size or 1
    optimization_time = "unconstrained" if args.use_compression else "constrained"
    input_shape = (bs, 3, 224, 224)
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

    dictionary = {}
    FEEDBACK_COLLECTOR.send_feedback = types.MethodType(
        save_ouput_in_dict(dictionary)(
            FEEDBACK_COLLECTOR.send_feedback.__func__
        ), FEEDBACK_COLLECTOR
    )
    for model, model_name in model_tuples:
        model_dir = os.path.join(base_path, model_name)
        if Path(model_dir).exists():
            continue
        time_dict = optimize_and_run(
            model, input_shape, optimization_time, quantization_ths
        )
        time_dict.update(dictionary)
        Path(model_dir).mkdir(exist_ok=True)
        with open(Path(model_dir) / "time_info.json", "w") as f_out:
            json.dump(time_dict, f_out)
        dictionary.clear()
