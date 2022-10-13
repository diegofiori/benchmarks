import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy
import torch.utils
import torch.utils.data


def get_hrnet(path_to_hrnet: str):
    path_to_lib = os.path.join(path_to_hrnet, "lib")
    sys.path.append(path_to_lib)
    try:
        from models.pose_hrnet import get_pose_net
        from config import cfg
        from config import update_config
        import torch

        class Args:
            cfg = os.path.join(path_to_hrnet, "experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml")
            opts = []
            modelDir = None
            logDir = None
            dataDir = None

        args = Args()

        update_config(cfg, args)

        pose_model = get_pose_net(cfg, is_train=False)
        pose_model.load_state_dict(torch.load(
            os.path.join(path_to_lib, "models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth")
        ), strict=False)
        return pose_model.cuda().eval()
    except Exception as ex:
        raise RuntimeError(
            f"No valid HRNet repo installation found. Got error {ex}"
        )


def read_label(image_path: Path):
    label_str = image_path.name.replace(".npy", "_label.json")
    with open(image_path.parent / label_str, "r") as f:
        label_dict = json.load(f)
    return label_dict


def _build_heatmap_per_layer(coord_y, coord_x, shape, scaling_factor=1):
    heatmap = np.zeros(shape)
    heatmap[coord_x//scaling_factor, coord_y//scaling_factor] = 1
    heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=2)
    heatmap = heatmap / heatmap.max()
    return torch.from_numpy(heatmap).float()


class PoseEstimationDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, return_keypoint_position=False):
        super().__init__()
        self.images = images
        self.labels = labels
        assert len(self.labels) == len(self.images), "Labels and images must be in the same number"
        self.keys = list(labels[0].keys())
        self._output_shape = [x // 4 for x in np.load(self.images[0]).shape[:-1]]
        self.return_keypoint_position = return_keypoint_position

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = torch.from_numpy(np.load(self.images[item])).permute(2, 0, 1) / 255
        label_dict = self.labels[0]
        label = torch.stack([
            _build_heatmap_per_layer(
                *label_dict[key][:-1],
                shape=self._output_shape,
                scaling_factor=4
            )
            for key in self.keys
        ])
        if self.return_keypoint_position:
            keypoint_pos = torch.stack([torch.tensor(label_dict[key][:-1]) for key in self.keys])
            return image, label, keypoint_pos
        return image, label


def get_pose_point(prediction_logits: torch.Tensor):
    # prediction_logits has shape N, K, H, W
    #  this means that we need to have a final tensor with shapes N, K, 2
    vals, h_idx = torch.max(prediction_logits, dim=2)
    w_idx = torch.argmax(vals, dim=-1)
    print(w_idx.shape, h_idx.shape)
    h_idx = h_idx.take(w_idx)
    print(h_idx.shape)
    return torch.cat([h_idx.unsqueeze(-1), w_idx.unsqueeze(-1)], dim=-1)


@torch.no_grad()
def compute_pck_metric(prediction, label, tau=0.5):
    pose_point_pred = get_pose_point(prediction)
    torso_dims = torch.norm(label[:, 1] - label[:, 11], dim=-1)
    distance = torch.norm(pose_point_pred-label, dim=-1)
    return torch.mean((torch.less_equal(distance, torso_dims*tau)) * 1.)


K_VEC = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0


@torch.no_grad()
def compute_oks_metric(prediction, label):
    pose_point_pred = get_pose_point(prediction)
    distance = torch.norm(pose_point_pred - label, dim=-1)
    k_vec = K_VEC.to(prediction.device)
    body_area = torch.prod(label.max(dim=1)[0] - label.min(dim=1)[0], dim=-1).unsqueeze(-1)
    exponent = torch.exp(-1 * distance**2 / (2*body_area*k_vec**2))
    return torch.mean(exponent)


