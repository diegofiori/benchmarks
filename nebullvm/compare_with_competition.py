import platform
import time

import torch
import torchvision.models
from nebullvm.api.frontend.torch import _extract_info_from_data, \
    optimize_torch_model
from nebullvm.base import ModelParams, InputInfo
from nebullvm.measure import compute_optimized_running_time
from nebullvm.optimizers import ApacheTVMOptimizer
from nebullvm.utils.torch import get_outputs_sizes_torch, \
    create_model_inputs_torch
from torch.utils.data import DataLoader


def optimize_with_tvm(model, dataloader, batch_size, input_sizes, input_types, dynamic_axis, extra_input_info):
    if dataloader is not None:
        (
            batch_size,
            input_sizes,
            input_types,
            dynamic_axis,
        ) = _extract_info_from_data(
            model,
            dataloader,
            batch_size,
            input_sizes,
            input_types,
            dynamic_axis,
        )
    if input_types is None:
        input_types = ["float"] * len(input_sizes)
    if extra_input_info is None:
        extra_input_info = [{}] * len(input_sizes)

    input_infos = [
        InputInfo(size=input_size, dtype=input_type, **extra_info)
        for input_size, input_type, extra_info in zip(
            input_sizes, input_types, extra_input_info
        )
    ]
    model_params = ModelParams(
        batch_size=batch_size,
        input_infos=input_infos,
        output_sizes=get_outputs_sizes_torch(
            model,
            input_tensors=create_model_inputs_torch(batch_size, input_infos),
        ),
        dynamic_info=dynamic_axis,
    )
    st = time.time()
    tvm_opt_model = ApacheTVMOptimizer().optimize_from_torch(
        torch_model=model,
        model_params=model_params,
        perf_loss_ths=None,
        quantization_type=None,
    )
    tvm_optimization_time = time.time() - st
    tvm_model_latency = compute_optimized_running_time(tvm_opt_model)
    return tvm_model_latency, tvm_optimization_time


def optimize_with_deci(model, dataloader, batch_size, input_sizes, input_types, dynamic_axis, extra_input_info):
    st = time.time()
    optimized_model = optimize_torch_model(
        model, ".", dataloader, batch_size, input_sizes, input_types,
        extra_input_info, dynamic_axis=dynamic_axis,
        ignore_compilers=["tvm", "onnxruntime"]
    )
    optimization_time = time.time() - st
    latency = compute_optimized_running_time(optimized_model)
    return latency, optimization_time


def optimize_with_nebullvm(model, dataloader, quantization_ths, batch_size, input_sizes, input_types, dynamic_axis, extra_input_info):
    st = time.time()
    optimized_model = optimize_torch_model(
        model, ".", dataloader, batch_size, input_sizes, input_types,
        extra_input_info, dynamic_axis=dynamic_axis,
        perf_loss_ths=quantization_ths
    )
    optimization_time = time.time() - st
    latency = compute_optimized_running_time(optimized_model)
    return latency, optimization_time


if __name__ == "__main__":
    import argparse

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
    quantization_ths = args.quantization_ths or 3
    bs = args.batch_size or 1
    from_data = args.from_data or False
    input_shapes = [(3, 224, 224)]
    input_types, dynamic_axis, extra_input_info = None, None, None
    model = torchvision.models.efficientnet_b0().eval()

    if from_data:
        data = [((torch.randn(input_shapes[0]),), 0) for _ in range(500)]
        dataloader = DataLoader(data, batch_size=input_shapes[0][0])
    else:
        dataloader = None

    json_dict = {}
    tvm_latency, tvm_runtime = optimize_with_tvm(
        model, dataloader, bs, input_shapes, input_types, dynamic_axis, extra_input_info
    )
    json_dict["tvm_latency"] = tvm_latency
    json_dict["tvm_runtime"] = tvm_runtime
    if "darwin" not in platform.system().lower():
        deci_latency, deci_runtime = optimize_with_deci(
            model, dataloader, bs, input_shapes, input_types, dynamic_axis, extra_input_info
        )
        json_dict["deci_latency"] = deci_latency
        json_dict["deci_runtime"] = deci_runtime
    nebullvm_latency, nebullvm_runtime = optimize_with_nebullvm(
        model, dataloader, quantization_ths, bs, input_shapes, input_types, dynamic_axis, extra_input_info
    )
    json_dict["nebullvm_latency"] = nebullvm_latency
    json_dict["nebullvm_runtime"] = nebullvm_runtime
    import json
    with open("result_nebullvm_with_competition.json", "w") as f:
        json.dump(json_dict, f)

