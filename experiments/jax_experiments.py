import copy
import time
from typing import List

import torch
import torch.nn
from jax import vjp
from torch.autograd import Function
from torch import Tensor
from torch.nn import Module
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class AvgAndFlatten(Module):
    def __init__(self, avg_dim):
        super().__init__()
        self.avg_dim = avg_dim

    def forward(self, x):
        bs = x.shape[0]
        x = x.mean(dim=self.avg_dim)
        return x.view(bs, -1)


class CustomModel(Module):
    def __init__(self):
        super().__init__()
        self._avg_pool = torch.nn.AvgPool2d(4)
        self._flatten = AvgAndFlatten(-3)
        self._linear = torch.nn.Linear(3136, 1024)
        self._relu = torch.nn.ReLU()
        self._linears = torch.nn.Sequential(
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self._flatten(self._avg_pool(x))
        x = self._relu(self._linear(x))
        return self._linears(x)


def extract_params(names, *args, correction_fn=None, **kwargs):
    def extraction_fn(layer):
        nw_args = [getattr(layer, name) for name in names]
        if correction_fn is not None:
            nw_args = [correction_fn(x) for x in nw_args]
        nw_args += list(args)
        return nw_args, kwargs
    return extraction_fn


needs_training = [hk.BatchNorm]


class JaxSequential(hk.Module):
    def __init__(self, layers, name=None):
        super().__init__(name=name)
        self.layers = layers
        self.needs_training = [type(l) in needs_training for l in layers]

    def __call__(self, inputs, *args, is_training: bool = True, **kwargs):
        out = inputs
        for i, layer in enumerate(self.layers):
            if i == 0:
                if self.needs_training[0]:
                    out = layer(out, *args, is_training=is_training, **kwargs)
                else:
                    out = layer(out, *args, **kwargs)
            elif self.needs_training[i]:
                out = layer(out, is_training=is_training)
            else:
                out = layer(out)
        return out


class JaxAvgAndFlatten(hk.Module):
    def __init__(self, avg_dim, name=None):
        super().__init__(name=name)
        self.avg_dim = avg_dim

    def __call__(self, inputs):
        bs = inputs.shape[0]
        out = jnp.mean(inputs, axis=self.avg_dim).reshape(bs, -1)
        return out


torch2haiku = {
    torch.nn.AvgPool2d: (hk.AvgPool, extract_params(["kernel_size", "stride"], "VALID")),
    torch.nn.BatchNorm1d: (hk.BatchNorm, extract_params([], create_scale=True, create_offset=True, decay_rate=0.9)),
    torch.nn.ReLU: (jax.nn.relu, None),
    torch.nn.Linear: (hk.Linear, extract_params(["out_features"], True)),
    AvgAndFlatten: (JaxAvgAndFlatten, extract_params(["avg_dim"], correction_fn=lambda x: x+2)),
    # torch.nn.Sequential: (hk.Sequential, None)
}

def build_haiku_model(model: Module):
    haiku_layers = []
    for layer in model.children():
        if isinstance(layer, torch.nn.Sequential):
            haiku_layers.append(build_haiku_model(layer))
            continue
        hk_layer, params_fn = torch2haiku[type(layer)]
        if params_fn is None:
            haiku_layers.append(hk_layer)
        else:
            args, kwargs = params_fn(layer)
            haiku_layers.append(hk_layer(*args, **kwargs))
    return JaxSequential(haiku_layers)


def transform_into_array(tensor):
    return jnp.array(tensor.detach().cpu().numpy())


def get_params(module, example_dict):
    example_dict = copy.deepcopy(example_dict)
    parameters = [param for param in module.parameters()]
    return get_params_from_torch(parameters, example_dict)


def get_params_from_torch(parameters, example_dict):
    parameters = [param for param in parameters]
    new_params = {}
    for key, value in example_dict.items():
        param_dict = {}
        if "linear" in key:
            param_dict["w"] = transform_into_array(parameters.pop(0).T)
            param_dict["b"] = transform_into_array(parameters.pop(0)).reshape(value["b"].shape)
        elif "batch_norm" in key:
            param_dict["scale"] = transform_into_array(parameters.pop(0)).reshape(value["scale"].shape)
            param_dict["offset"] = transform_into_array(parameters.pop(0)).reshape(value["offset"].shape)
        else:
            raise NotImplementedError()
        new_params[key] = param_dict
    assert len(parameters) == 0, "Not all the torch parameters were converted in JAX"
    return new_params


def recompose_gradients(grads, example_dict, torch_parameters):
    torch_gradients = []
    for key in example_dict.keys():
        grad_dict = grads[key]
        if "linear" in key:
            torch_gradients.append(torch.from_numpy(np.array(grad_dict["w"])).view(next(torch_parameters).shape))
            torch_gradients.append(torch.from_numpy(np.array(grad_dict["b"])).view(next(torch_parameters).shape))
        elif "batch_norm" in key:
            torch_gradients.append(torch.from_numpy(np.array(grad_dict["scale"])).view(next(torch_parameters).shape))
            torch_gradients.append(torch.from_numpy(np.array(grad_dict["offset"])).view(next(torch_parameters).shape))
        else:
            raise NotImplementedError()
    return torch_gradients


class MyJAXWrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        class MyJAXWrapperFunction(Function):
            @staticmethod
            def jvp(ctx, *grad_inputs):
                return Function.jvp(self, *grad_inputs)

            @staticmethod
            def forward(
                    ctx, inputs, *parameters,
            ) -> Tensor:
                # print("forward time")
                ctx.save_for_backward(inputs, *parameters)
                # transform in jnp
                # st = time.time()
                input_array = transform_into_array(inputs)
                params = get_params_from_torch(parameters, self.params_template)
                # print(f"params tfms: {time.time()-st}")
                # st = time.time()
                _predict = lambda x: self.jit_predict(x, state, input_array)
                pred, vjp_fun = vjp(_predict, params)
                ctx.vjp_fun = vjp_fun
                # print(f"forward: {time.time()-st}")
                # st = time.time()
                # transorm into tensor
                output = torch.from_numpy(np.array(pred)).requires_grad_(True)
                # print(f"output tfms: {time.time()-st}")
                return output

            @staticmethod
            def backward(ctx, grad_output):
                # print("backward")
                # st = time.time()
                dl_dy = transform_into_array(grad_output)
                # print(f"input_grad tfms: {time.time()-st}")
                # st = time.time()
                grads = ctx.vjp_fun(dl_dy)[0]
                # print(f"grad computation: {time.time()-st}")
                # st = time.time()
                inputs, *parameters = ctx.saved_tensors
                grads = recompose_gradients(grads, params, iter(parameters))
                # print(f"grad recomposition: {time.time()-st}")
                return None, *grads

        def forward(input_data):
            hk_model = build_haiku_model(model)
            pred = hk_model(input_data)
            return pred

        forward_t = hk.transform_with_state(forward)
        forward_t = hk.without_apply_rng(forward_t)
        rng = jax.random.PRNGKey(42)
        input_array = np.random.randn(256, 224, 224, 3)
        params, state = forward_t.init(rng, input_array)
        self.params_template = params
        self.state = state
        jit_forward = jax.jit(forward_t.apply)

        def predict(*args, **kwargs):
            r, self.state = jit_forward(*args, **kwargs)
            return r

        self.jit_predict = predict

        self._forward = MyJAXWrapperFunction.apply

    def forward(self, inputs):
        return self._forward(inputs, *self.model.parameters())


def get_data() -> List:
    return [torch.randn(256, 3, 224, 224) for _ in range(100)]


def train_model(model: torch.nn.Module, dataset: List, reshape: bool):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    st = time.time()
    for data in dataset:
        if reshape:
            data = data.transpose(1, 3)
        pred = model(data)
        loss = pred.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(time.time() - st)


if __name__ == "__main__":
    model = CustomModel()
    jax_model = MyJAXWrapperModel(copy.deepcopy(model))
    dataset = get_data()
    train_model(model, dataset, False)
    train_model(jax_model, dataset, True)
