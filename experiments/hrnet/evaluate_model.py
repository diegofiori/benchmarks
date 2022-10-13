import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn
import torch.utils.data
from nebullvm.inference_learners.base import BaseInferenceLearner, LearnerMetadata

from utils import read_label, PoseEstimationDataset, compute_pck_metric, \
    compute_oks_metric, get_hrnet


def _load_test_data(data_path: Path):
    img_list = list(data_path.glob("*.npy"))
    label_list = [read_label(img_path) for img_path in img_list]
    ds = PoseEstimationDataset(img_list, label_list, True)
    dl = torch.utils.data.DataLoader(ds, batch_size=1)
    return dl

@torch.no_grad()
def evaluate_model_performance(
        original_model: torch.nn.Module,
        optimized_model: BaseInferenceLearner,
        data_path: Path,
        save_path: Path = None,
        original_in_half: bool = True,
):
    original_latencies = []
    original_losses = []
    original_pck_list = []
    original_oks_list = []
    optimized_latencies = []
    optimized_losses = []
    optimized_pck_list = []
    optimized_oks_list = []
    loss_fn = torch.nn.MSELoss()
    test_dl = _load_test_data(data_path)
    if original_in_half:
        original_model.half()
    max_loss = 0
    img_max_loss = None
    heatmap_max_loss = None
    orig_pred_max_loss = None
    opt_pred_max_loss = None
    for input_tensor, heatmap, keypoint in test_dl:
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            heatmap = heatmap.cuda()
            keypoint = keypoint.cuda()
        # Original model
        st = time.time()
        original_pred = original_model(input_tensor.half() if original_in_half else input_tensor)
        original_latencies.append(time.time()-st)
        if original_in_half:
            original_pred = original_pred.float()
        original_loss = float(loss_fn(original_pred, heatmap).cpu())
        original_pck = float(compute_pck_metric(original_pred, keypoint, 0.2).cpu())
        original_oks = float(compute_oks_metric(original_pred, keypoint).cpu())
        original_losses.append(original_loss)
        original_pck_list.append(original_pck)
        original_oks_list.append(original_oks)

        # Optimized model
        # model built using the ONNX interface
        input_tensor = input_tensor.cpu().numpy()
        st = time.time()
        optimized_pred = optimized_model(input_tensor)[0]
        optimized_latencies.append(time.time() - st)
        optimized_pred = torch.from_numpy(optimized_pred)
        if torch.cuda.is_available():
            optimized_pred = optimized_pred.cuda()
        optimized_loss = float(loss_fn(optimized_pred, heatmap).cpu())
        optimized_pck = float(compute_pck_metric(optimized_pred, keypoint, 0.2).cpu())
        optimized_oks = float(compute_oks_metric(optimized_pred, keypoint).cpu())
        optimized_losses.append(optimized_loss)
        optimized_pck_list.append(optimized_pck)
        optimized_oks_list.append(optimized_oks)
        if optimized_loss > max_loss:
            img_max_loss = torch.from_numpy(input_tensor).cpu().permute(0, 2, 3, 1).numpy()[0]
            heatmap_max_loss = heatmap.cpu().numpy()[0]
            orig_pred_max_loss = original_pred.cpu()
            opt_pred_max_loss = optimized_pred.cpu()

    print("########### Evaluation results ###############")
    print(f"Latency\norig: {np.mean(original_latencies)}\nopt: {np.mean(optimized_latencies)}")
    print(f"Loss\norig: {np.mean(original_losses)}\nopt: {np.mean(optimized_losses)}")
    print(f"PCK\norig: {np.mean(original_pck_list)}\nopt: {np.mean(optimized_pck_list)}")
    print(f"OKS\norig: {np.mean(original_oks_list)}\nopt: {np.mean(optimized_oks_list)}")

    result_dict = {
        "original_latency": np.mean(original_latencies),
        "optimized_latency": np.mean(optimized_latencies),
        "original_loss": np.mean(original_losses),
        "optimized_loss": np.mean(optimized_losses),
        "original_pck": np.mean(original_pck_list),
        "optimized_pck": np.mean(optimized_pck_list),
        "original_oks": np.mean(original_oks_list),
        "optimized_oks": np.mean(optimized_oks_list),
        "max_losses": {
            "img": img_max_loss,
            "heatmap": heatmap_max_loss,
            "original_pred": orig_pred_max_loss,
            "optimized_pred": opt_pred_max_loss,
            "max_loss": max_loss,
        },
    }
    if save_path is not None:
        with open(save_path / "result_evaluation.json", "r") as f:
            json.dump(result_dict, f)
    return result_dict


def _plot_heatmaps(result_dict, save_file):
    max_losses_dict = result_dict["max_losses"]
    img = max_losses_dict["img"]
    heatmap = max_losses_dict["heatmap"]
    original_preds = max_losses_dict["original_pred"].numpy()[0]
    optimized_preds = max_losses_dict["optimized_pred"].numpy()[0]
    num_rows = len(heatmap) + 1
    fig = plt.figure(figsize=(15,5*num_rows))
    plt.subplot(num_rows, 3, 1)
    plt.imshow(img)
    for i in range(num_rows):
        plt.subplot(num_rows, 3, 3*i+3)
        plt.imshow(heatmap[i])
        plt.subplot(num_rows, 3, 3*i + 4)
        plt.imshow(original_preds[i])
        plt.subplot(num_rows, 3, 3*i + 5)
        plt.imshow(optimized_preds[i])
    fig.save(save_file)


def save_plots(result_dict, save_path):
    _plot_heatmaps(result_dict, save_path / "heatmaps.png")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--save_path", "-s", type=Path, default=None, help="Path to directory where storing the results.")
    parser.add_argument("--data_path", "-d", type=Path, help="Path to data directory.")
    parser.add_argument("--original_model_path", "-o", help="Path to the repo where the model is defined.")
    parser.add_argument("--learner_path", "-l", help="Path to the repo where the inference learner is stored.")
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    original_model = get_hrnet(args.original_model_path)
    inference_learner = LearnerMetadata.read(args.learner_path).load_model(args.learner_path)
    results = evaluate_model_performance(original_model, inference_learner, args.data_path, save_path=args.save_path, original_in_half=args.half)

    if args.save_path is not None:
        save_plots(results, args.save_path)