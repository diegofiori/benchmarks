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


def load_optimize(path_to_model: Path, path_to_data: Path):
    pose_model = get_hrnet(str(path_to_model))
    data_loader = _load_test_data(path_to_data)
    input_data = [((x, ), y) for x, y in data_loader]
    _ = optimize_model(
        pose_model,
        input_data=input_data,
        optimization_time="unconstrained",
        metric_drop_ths=10.,
        store_latencies=True,
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=Path, help="Path to model")
    parser.add_argument("--data", "-d", type=Path, help="Path to data")
    args = parser.parse_args()
    load_optimize(args.model, args.data)
