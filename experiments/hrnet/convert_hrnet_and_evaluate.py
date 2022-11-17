import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime
import torch

from utils import get_hrnet


def convert_model_to_onnx(model: torch.nn.Module, save_path: str):
    input_tensor = torch.randn(1, 3, 384, 288)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        model = model.cuda()
    torch.onnx.export(
        model,
        input_tensor,
        save_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


def load_numpy_data(data_path: Path):
    img_list = list(data_path.glob("*.npy"))
    img_list.sort()
    return [np.load(str(img_path)) for img_path in img_list]


def compute_latency_onnx(model_path: str, input_data: List[np.ndarray]):
    sess = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    latencies = []
    for input_tensor in input_data:
        st = time.time()
        sess.run([output_name], {input_name: input_tensor})
        latencies.append(time.time() - st)
    return np.mean(latencies[5:])  # warmup


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="Path to model")
    parser.add_argument("--save", "-s", type=str, help="Path to save")
    parser.add_argument("--data", "-d", type=Path, help="Path to data")
    args = parser.parse_args()
    model = get_hrnet(args.model)
    convert_model_to_onnx(model, args.save)
    latency = compute_latency_onnx(args.save, load_numpy_data(args.data))
    print("####### Latency #######")
    print(f"Latency: {latency}")


if __name__ == "__main__":
    main()
