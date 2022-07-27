import time

import onnxruntime as ort
import torch
import torchvision.models as models
from nebullvm.base import DeepLearningFramework, ModelParams

from nebullvm.utils.onnx import get_output_names, get_output_sizes_onnx
from nebullvm.optimizers.tensor_rt import TensorRTOptimizer

if __name__ == "__main__":
    model = models.resnet50()
    input_tensors = [torch.randn(1, 3, 224, 224).cuda() for _ in range(100)]
    with torch.no_grad():
        st = time.time()
        for tensor in input_tensors:
            res = model(tensor)
        torch_time = time.time()-st
        print("Torch: ", torch_time)
    onnx_model = "model.onnx"
    torch.onnx.export(
        model,
        input_tensors[0],
        onnx_model,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input_0"],
        output_names=["output_0"],
        dynamic_axes=None,
    )
    ort_model = ort.InferenceSession(
        onnx_model, providers=["TensorRTExecutionProvider"]
    )
    arrays = [tensor.cpu().detach().numpy() for tensor in input_tensors]
    st = time.time()
    for array in arrays:
        inputs = {
            "input_0": array,
        }
        res = model.run(
            output_names=get_output_names(onnx_model), input_feed=inputs
        )
    ort_time = time.time() - st
    print(f"ONNXRuntime time {ort_time}")
    model_params = ModelParams(
        batch_size=1,
        input_infos=[{"size": (3, 224, 224), "dtype": "float"}],
        output_sizes=get_output_sizes_onnx(onnx_model, [arrays[0]])
    )
    optimizer = TensorRTOptimizer()
    optimized_model = optimizer.optimize(
        onnx_model,
        DeepLearningFramework.NUMPY,
        model_params,
        input_tfms=None,
        quantization_type=None,
        metric_drop_ths=None,
    )
    st = time.time()
    for array in arrays:
        _ = optimized_model(array)
    tensor_rt_time = time.time()-st
    print("Pure TensorRT time: ", tensor_rt_time)
