import numpy as np


def _get_resnet_onnx_model():
    from torchvision.models import resnet50
    from torch import Tensor
    from torch.onnx import export

    model = resnet50(pretrained=True)
    model.eval()
    dummy_input = Tensor(1, 3, 224, 224)
    export(model, dummy_input, "resnet18.onnx", verbose=True)
    return "resnet18.onnx"


def test_resnet_onnx_model():
    from speedster import optimize_model
    model = _get_resnet_onnx_model()
    input_data = [
        ((np.random.normal(size=(1, 3, 224, 224)),), 0) for i in range(100)
    ]
    optimized_model = optimize_model(model, input_data, metric_drop_ths=0.1)


if __name__ == "__main__":
    test_resnet_onnx_model()