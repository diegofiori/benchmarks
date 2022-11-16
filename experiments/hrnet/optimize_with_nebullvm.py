from pathlib import Path

import torch.utils.data
from nebullvm.api.functions import optimize_model

from utils import get_hrnet, read_label, PoseEstimationDataset


def _load_test_data(data_path: Path):
    img_list = list(data_path.glob("*.npy"))
    img_list.sort()
    label_list = [read_label(img_path) for img_path in img_list]
    ds = PoseEstimationDataset(img_list, label_list, False)
    dl = torch.utils.data.DataLoader(ds, batch_size=1)
    return dl


def load_optimize(path_to_model: Path, path_to_data: Path, save_dir: Path):
    pose_model = get_hrnet(str(path_to_model))
    data_loader = _load_test_data(path_to_data)
    if torch.cuda.is_available():
        input_data = [((x.cuda(),), y.cuda()) for x, y in data_loader]
    else:
        input_data = [((x, ), y) for x, y in data_loader]
    optimized_model = optimize_model(
        pose_model,
        input_data=input_data,
        optimization_time="unconstrained",
        metric_drop_ths=10.,
        store_latencies=True,
    )
    optimized_model.save(save_dir)


def load_optimize_from_onnx(path_to_onnx: Path, path_to_data: Path, save_dir: Path):
    data_loader = _load_test_data(path_to_data)
    input_data = [
        ((x.detach().cpu().numpy(), ), y.detach().cpu().numpy())
        for x, y in data_loader
    ]
    optimized_model = optimize_model(
        str(path_to_onnx),
        input_data=input_data,
        optimization_time="unconstrained",
        metric_drop_ths=10.,
        store_latencies=True,
    )
    optimized_model.save(save_dir)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=Path, help="Path to model")
    parser.add_argument("--data", "-d", type=Path, help="Path to data")
    parser.add_argument("--save", "-s", type=Path, help="Path to save")
    parser.add_argument("--from_onnx", action="store_true", help="Load model from ONNX")
    args = parser.parse_args()
    if not args.from_onnx:
        load_optimize(args.model, args.data, args.save)
    else:
        load_optimize_from_onnx(args.model, args.data, args.save)
