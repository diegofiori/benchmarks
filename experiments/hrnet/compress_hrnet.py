import copy
import json
import os
from pathlib import Path

import deepspeed
import numpy as np
import sklearn.model_selection
import torch.distributed
import torch.fx
import torch.utils
import torch.utils.data
from deepspeed.compression.compress import init_compression, redundancy_clean
from tqdm import tqdm

from utils import get_hrnet, compute_pck_metric, compute_oks_metric, read_label, PoseEstimationDataset


def get_data(path_to_data: str):
    image_paths = list(Path(path_to_data).glob("*.npy"))
    image_paths.sort()
    label_paths = [read_label(img_p) for img_p in image_paths]
    train_imgs, test_imgs, train_labels, test_labels = sklearn.model_selection.train_test_split(image_paths, label_paths, test_size=0.2,
                                             train_size=None, random_state=52,
                                             shuffle=False, stratify=None)
    train_ds = PoseEstimationDataset(train_imgs, train_labels)
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.train_batch_size)
    test_ds = PoseEstimationDataset(test_imgs, test_labels, True)
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


class FakeQuantizationModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input_tensor):
        x = self.quant(input_tensor)
        x = self.model(x)
        return self.dequant(x)


def fake_quantize(model: torch.nn.Module):
    # model = copy.deepcopy(model)
    fake_quantized_model = FakeQuantizationModel(model)
    fake_quantized_model.train()
    fake_quantized_model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    fake_quantized_model = torch.quantization.prepare_qat(fake_quantized_model, inplace=True)
    return fake_quantized_model


def fake_dequantize(quantized_model, model):
    q_state_dict = quantized_model.model.state_dict() # get state dict of the model inside the wrapper
    state_dict = model.state_dict()
    new_state_dict = {}
    for key in state_dict.keys():
        new_state_dict[key] = q_state_dict[key]
    print(len(new_state_dict))
    # print(max(np.abs(value1-value2) for value1, value2 in zip(state_dict.values(), new_state_dict.values())))
    model.load_state_dict(new_state_dict)
    return model


def fine_tune_with_quantization(model, train_dl):
    model = model.float()
    traced_model = torch.jit.trace(model, torch.randn(1, 3, 384, 288).cuda())
    q_model = fake_quantize(model)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_ft)
    for epoch in range(1, args.ft_epochs + 1):
        print('Current FT Epoch: ', epoch)
        train_loss = 0.
        total_num = 0
        with tqdm(total=len(train_dl.dataset)) as progressbar:
            for batch_idx, (data, target) in enumerate(train_dl):
                q_model.train()
                if torch.cuda.is_available():
                    data, target = data.cuda().float(), target.cuda()
                output = q_model(data)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(output, target)
                loss.backward()
                train_loss += loss.item() * target.size()[0]
                total_num += target.size()[0]
                optimizer.step()

                progressbar.set_postfix(loss=train_loss / total_num)

                progressbar.update(target.size(0))
    return fake_dequantize(q_model, traced_model)


def get_test_loss(model, dl_test):
    criterion = torch.nn.MSELoss()
    test_loss = 0
    total_num = 0
    oks_val = 0
    pck_val = 0
    with torch.no_grad():
        model.eval()
        half_precision = False
        if next(model.parameters()).dtype is torch.float16:
            half_precision = True
        for data, target, keypoint_pos in dl_test:
            if torch.cuda.is_available():
                data, target = data.cuda().float(), target.cuda()
                keypoint_pos = keypoint_pos.cuda()
                if half_precision:
                    data = data.half()
            output = model(data)
            loss = criterion(output, target)
            oks_val += compute_oks_metric(output, keypoint_pos).item() * target.size()[0]
            pck_val += compute_pck_metric(output, keypoint_pos).item() * target.size()[0]
            test_loss += loss.item() * target.size()[0]
            total_num += target.size()[0]
        test_loss /= total_num
        oks_val /= total_num
        pck_val /= total_num
    return test_loss, oks_val, pck_val


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


def compute_params(model: torch.nn.Module):
    total = 0
    for param in model.parameters():
        total += int(np.prod(param.shape))
    return total


def main(path_to_hrnet: str, path_to_data: str, save_path: str):
    print("################################")
    print("Pre init distribution")
    deepspeed.init_distributed()
    model = get_hrnet(path_to_hrnet)
    train_ds, dl_test = get_data(path_to_data)
    print("Model and data ready.")
    if args.local_rank == 0:
        test_loss_bc, oks_bc, pck_bc = get_test_loss(model, dl_test)
        params_pre_compression = compute_params(model)
        print(f"Loss computed before compression: {test_loss_bc}.")
        print(f"OKS metric value before compression: {oks_bc}")
        print(f"PCK0.2 metric value before compression: {pck_bc}")
        print(f"Parameters before compression {params_pre_compression}")
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
    test_loss_ac, oks_ac, pck_ac = get_test_loss(model, dl_test)
    params_post_compression = compute_params(model)
    print(f"Loss computed post compression: {test_loss_ac}.")
    print(f"OKS metric value after compression: {oks_ac}")
    print(f"PCK0.2 metric value after compression: {pck_ac}")
    print(f"Parameters after compression {params_post_compression}")
    if args.local_rank == 0:
        print("Starting fine-tuning with QAT")
        ft_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.train_batch_size)
        model = fine_tune_with_quantization(model, ft_dl)
        print("Model fine-tuned with QAT")
        test_loss_pqat, oks_pqat, pck_pqat = get_test_loss(model, dl_test)
        print(f"Loss computed post QAT: {test_loss_pqat}.")
        print(f"OKS metric value post QAT: {oks_pqat}")
        print(f"PCK0.2 metric value post QAT: {pck_pqat}")
        Path(save_path).mkdir(exist_ok=True, parents=True)
        model_exported_path = os.path.join(save_path, "compressed_hrnet.onnx")
        export_to_onnx(model, dl_test.dataset[0][0].unsqueeze(0), model_exported_path)
        loss_dict = {
            "loss_pre_compression": test_loss_bc,
            "loss_post_compression": test_loss_ac,
            "loss_post_qat": test_loss_pqat,
            "okc_pre_compression": oks_bc,
            "okc_post_compression": oks_ac,
            "okc_post_qat": oks_pqat,
            "pck_pre_compression": pck_bc,
            "pck_post_compression": pck_ac,
            "pck_post_qat": pck_pqat,
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
    parser.add_argument("--ft_epochs", type=int, default=5, help="Number of epochs for fine tuning")
    parser.add_argument("--lr_ft", type=float, default=1e-3, help="LR for fine tuning")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    main(
        path_to_hrnet=args.path_to_model_dir,
        path_to_data=args.path_to_data,
        save_path=args.save_path,
    )