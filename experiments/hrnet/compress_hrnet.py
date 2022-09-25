import copy
import json
import os
import sys
from pathlib import Path

import deepspeed
import numpy as np
import scipy
import sklearn.model_selection
import torch.distributed
import torch.utils.data
from deepspeed.compression.compress import init_compression, redundancy_clean
from tqdm import tqdm


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


def _read_label(image_path: Path):
    label_str = image_path.name.replace(".npy", "_label.json")
    with open(image_path.parent / label_str, "r") as f:
        label_dict = json.load(f)
    return label_dict


def _build_singular_heatmap(coord_x, coord_y, shape, scaling_factor=1):
    heatmap = torch.zeros(*shape)
    heatmap[coord_x//scaling_factor, coord_y//scaling_factor] = 1
    return heatmap


class GaussianLayer(torch.nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(10),
            torch.nn.Conv2d(17, 17, 21, stride=1, padding=0, bias=None, groups=17)
        )
        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((21, 21))
        n[10, 10] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=3)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))


class HeatmapGenerator:
    def __init__(self):
        self._gaussian_filter = GaussianLayer()

    def apply_gaussian_filter(self, input_tensor):
        with torch.no_grad():
            input_tensor = input_tensor.float().unsqueeze(0)
            input_tensor = self._gaussian_filter(input_tensor)
        return input_tensor[0]


class PoseEstimationDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels
        assert len(self.labels) == len(self.images), "Labels and images must be in the same number"
        self.keys = list(labels[0].keys())
        self._output_shape = [x // 4 for x in np.load(self.images[0]).shape[:-1]]
        self._heatmap_generator = HeatmapGenerator()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = torch.from_numpy(np.load(self.images[item])).permute(2, 0, 1)
        label_dict = self.labels[0]
        label = torch.stack([
            _build_singular_heatmap(
                *label_dict[key][:-1],
                shape=self._output_shape,
                scaling_factor=4
            )
            for key in self.keys
        ])
        label = self._heatmap_generator.apply_gaussian_filter(label)
        return image, label


def get_data(path_to_data: str):
    image_paths = list(Path(path_to_data).glob("*.npy"))
    image_paths.sort()
    label_paths = [_read_label(img_p) for img_p in image_paths]
    train_imgs, test_imgs, train_labels, test_labels = sklearn.model_selection.train_test_split(image_paths, label_paths, test_size=0.2,
                                             train_size=None, random_state=52,
                                             shuffle=False, stratify=None)
    train_ds = PoseEstimationDataset(train_imgs, train_labels)
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.train_batch_size)
    test_ds = PoseEstimationDataset(test_imgs, test_labels)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.train_batch_size)
    return train_ds, test_dl


def train_model(model_engine, original_model, train_dls):
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        print('Current Epoch: ', epoch)
        train_loss = 0.
        total_num = 0
        with tqdm(total=len(train_dls.dataset)) as progressbar:
            for batch_idx, (data, target) in enumerate(train_dls):
                model_engine.train()
                original_model.cuda().train()
                if torch.cuda.is_available():
                    half_data, data, target = data.cuda().half(), data.cuda().float(), target.cuda()
                with torch.no_grad():
                    orig_pred = original_model(data)
                output = model_engine(half_data)
                loss = criterion(output, orig_pred.half())
                model_engine.backward(loss)
                train_loss += loss.item() * target.size()[0]
                total_num += target.size()[0]
                model_engine.step()

                progressbar.set_postfix(loss=train_loss / total_num)

                progressbar.update(target.size(0))
    return model_engine


def get_test_loss(model, dl_test):
    criterion = torch.nn.MSELoss()
    test_loss = 0
    total_num = 0
    with torch.no_grad():
        model.eval()
        half_precision = False
        if next(model.parameters()).dtype is torch.float16:
            half_precision = True
        for data, target in dl_test:
            if torch.cuda.is_available():
                data, target = data.cuda().float(), target.cuda()
                if half_precision:
                    data = data.half()
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * target.size()[0]
            total_num += target.size()[0]
        test_loss /= total_num
    return test_loss


def export_to_onnx(model, input_tensor, output_file_path):
    # input_names = [f"input_{i}" for i in range(len(input_tensors))]
    if torch.cuda.is_available():
        model.cuda().float().eval()
        input_tensor = input_tensor.cuda().float()
    input_names = ["input"]
    with torch.no_grad():
        outputs = model(input_tensor)
        if isinstance(outputs, torch.Tensor):
            output_names = ["output"]
        else:
            output_names = [f"output_{i}" for i in range(len(outputs))]
    torch.onnx.export(
        model,  # model being run
        input_tensor,  # model input (or a tuple for multiple inputs)
        str(output_file_path),
        # where to save the model (can be a file or file-like object)
        export_params=True,
        # store the trained parameter weights inside the model file
        opset_version=13,
        # the ONNX version to export the model to
        do_constant_folding=True,
        # whether to execute constant folding for optimization
        input_names=input_names,
        # the model's input names
        output_names=output_names,
    )


def main(path_to_hrnet: str, path_to_data: str, save_path: str):
    print("################################")
    print("Pre init distribution")
    deepspeed.init_distributed()
    model = get_hrnet(path_to_hrnet)
    train_ds, dl_test = get_data(path_to_data)
    print("Model and data ready.")
    if args.local_rank == 0:
        test_loss_pre_compression = get_test_loss(model, dl_test)
        print(f"Loss computed before compression: {test_loss_pre_compression}.")
    torch.distributed.barrier()
    original_model = copy.deepcopy(model).eval()
    model = init_compression(model, args.deepspeed_config)
    print("Initialize distribution")
    model_engine, optimizer, train_dls, __ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters(),
        training_data=train_ds
    )
    print("Model copied and prepared for compression")
    model = train_model(model_engine, original_model, train_dls)
    print("Knowledge distillation run on using the original model as teacher.")
    model = redundancy_clean(model, args.deepspeed_config)
    test_loss_post_compression = get_test_loss(model, dl_test)
    print(f"Loss computed post compression: {test_loss_post_compression}.")
    if args.local_rank == 0:
        Path(save_path).mkdir(exist_ok=True, parents=True)
        model_exported_path = os.path.join(save_path, "compressed_hrnet.onnx")
        export_to_onnx(model, dl_test.dataset[0][0], model_exported_path)
        loss_dict = {
            "pre_compression": test_loss_pre_compression,
            "post_compression": test_loss_post_compression
        }
        dict_path = os.path.join(save_path, "loss_dict.json")
        with open(dict_path, "w") as f:
            json.dump(loss_dict, f)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--path_to_model_dir", "-m", help="Path to model lib")
    parser.add_argument("--path_to_data", "-d", help="Path to the data.")
    parser.add_argument("--save_path", "-s", help="Save path")
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument("--train_batch_size", "-bs", type=int, default=32, help="Batch Size")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    main(
        path_to_hrnet=args.path_to_model_dir,
        path_to_data=args.path_to_data,
        save_path=args.save_path,
    )
