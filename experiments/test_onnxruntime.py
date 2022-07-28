import json
import time

import onnxruntime as ort
import torch
import torchvision.models as models
from nebullvm.base import DeepLearningFramework, ModelParams

from nebullvm.utils.onnx import get_output_names, get_output_sizes_onnx
from nebullvm.optimizers.tensor_rt import TensorRTOptimizer
from nebullvm.optimizers import OpenVinoOptimizer, ONNXOptimizer

MODELS = [
    models.resnet50().eval(),
    models.vgg19().eval(),
    models.efficientnet_b3().eval(),
]


def run_cuda():
    result_dict = {}
    for model in MODELS:
        model = model.cuda()
        input_tensors = [torch.randn(1, 3, 224, 224).cuda() for _ in range(100)]
        with torch.no_grad():
            st = time.time()
            for tensor in input_tensors:
                _ = model(tensor)
            torch_time = time.time() - st
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
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 0
        # sess_options.log_verbosity_level = 1
        sess_options.enable_profiling = True

        providers = [('TensorrtExecutionProvider',
                      {'trt_max_workspace_size': 22147483648,
                       'trt_fp16_enable': False, 'trt_engine_cache_enable': True,
                       'trt_engine_cache_path': './trt_cache'})]
        # providers = ["CUDAExecutionProvider"]
        ort_model = ort.InferenceSession(
            onnx_model, sess_options, providers=providers
        )
        # warmup
        for _ in range(10):
            array = torch.randn(1, 3, 224, 224).cpu().numpy()
            inputs = {
                "input_0": array,
            }
            _ = ort_model.run(
                output_names=["output_0"], input_feed=inputs
            )
        output_names = ["output_0"]
        arrays = [tensor.cpu().detach().numpy() for tensor in input_tensors]
        st = time.time()
        for array in arrays:
            inputs = {
                "input_0": array,
            }
            _ = ort_model.run(
                output_names=output_names, input_feed=inputs
            )
        ort_time = time.time() - st
        print(f"ONNXRuntime + TRT time {ort_time}")
        model_params = ModelParams(
            batch_size=1,
            input_infos=[{"size": (3, 224, 224), "dtype": "float"}],
            output_sizes=get_output_sizes_onnx(onnx_model, [arrays[0]])
        )
        optimizer = ONNXOptimizer()
        optimized_model = optimizer.optimize(
            onnx_model,
            DeepLearningFramework.NUMPY,
            model_params,
            input_tfms=None,
            quantization_type=None,
            metric_drop_ths=None,
        )
        for array in arrays:
            _ = optimized_model(array)
        onnx_time = time.time() - st
        print("Pure ONNX time: ", onnx_time)
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
        tensor_rt_time = time.time() - st
        print("Pure TensorRT time: ", tensor_rt_time)
        result_dict[model.__class__.__name__] = {
            "torch": torch_time,
            "ort_rt": ort_time,
            "base_onnx": onnx_time,
            "tensor_rt": tensor_rt_time,
        }
    with open("result_cuda.json", "w") as f:
        json.dump(result_dict, f)


def run_openvino():
    result_dict = {}
    for model in MODELS:
        input_tensors = [torch.randn(1, 3, 224, 224) for _ in range(100)]
        with torch.no_grad():
            st = time.time()
            for tensor in input_tensors:
                _ = model(tensor)
            torch_time = time.time() - st
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
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_options.log_severity_level = 0
        # sess_options.log_verbosity_level = 1
        sess_options.enable_profiling = True

        providers = ["OpenVINOExecutionProvider"]
        ort_model = ort.InferenceSession(
            onnx_model, sess_options, providers=providers, provider_options=[
                {
                    "device_type" : "CPU_FP32",
                    "device_id": "",
                    "enable_vpu_fast_compile": False,
                    "num_of_threads": 4,
                }
            ]
        )
        # warmup
        for _ in range(10):
            array = torch.randn(1, 3, 224, 224).cpu().numpy()
            inputs = {
                "input_0": array,
            }
            _ = ort_model.run(
                output_names=["output_0"], input_feed=inputs
            )
        output_names = ["output_0"]
        arrays = [tensor.cpu().detach().numpy() for tensor in input_tensors]
        st = time.time()
        for array in arrays:
            inputs = {
                "input_0": array,
            }
            _ = ort_model.run(
                output_names=output_names, input_feed=inputs
            )
        ort_time = time.time() - st
        print(f"ONNXRuntime + OV time {ort_time}")
        model_params = ModelParams(
            batch_size=1,
            input_infos=[{"size": (3, 224, 224), "dtype": "float"}],
            output_sizes=get_output_sizes_onnx(onnx_model, [arrays[0]])
        )
        # optimizer = ONNXOptimizer()
        # optimized_model = optimizer.optimize(
        #     onnx_model,
        #     DeepLearningFramework.NUMPY,
        #     model_params,
        #     input_tfms=None,
        #     quantization_type=None,
        #     metric_drop_ths=None,
        # )
        # for array in arrays:
        #     _ = optimized_model(array)
        # onnx_time = time.time() - st
        # print("Pure ONNX time: ", onnx_time)
        optimizer = OpenVinoOptimizer()
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
        openvino_time = time.time() - st
        print("Pure OpenVino time: ", tensor_rt_time)
        result_dict[model.__class__.__name__] = {
            "torch": torch_time,
            "ort_rt": ort_time,
            # "base_onnx": onnx_time,
            "openvino": openvino_time,
        }
    with open("result_openvino.json", "w") as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
    if torch.cuda.is_available():
        run_cuda()
    run_openvino()
