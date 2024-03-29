import copy
import time
from typing import Tuple, List, Any

import torch
import numpy as np

try:
    import pycutlass
    from pycutlass import *
    from pycutlass.conv2d_operation import *
except ImportError:
    import warnings
    warnings.warn(
        "No valid CUTLASS installation found. Attempt using Cutlass kernels "
        "will cause an error at runtime."
    )


class CutlassConv2dFunc(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        raise NotImplementedError(
            "The Cutlass kernels must be used just for inference. "
            "The training ones has not been implemented yet."
        )

    @staticmethod
    def forward(
            ctx: Any,
            tensor_A: torch.Tensor,
            tensor_B: torch.Tensor,
            tensor_C: torch.Tensor,
            tensor_D_size,
            type_c,
            operation,
            problem_size,
            alpha,
            beta,
            activation_args,
            split_k_mode,
            split_k_slices,
            conv_kind,
            reduction_operation,
    ) -> Any:
        tensor_D = torch.ones(size=(tensor_D_size,),
                              dtype=getattr(torch, type_c),
                              device="cuda")

        arguments = Conv2dArguments(
            operation=operation, problem_size=problem_size,
            A=tensor_A,
            B=tensor_B, C=tensor_C, D=tensor_D,
            output_op=operation.epilogue_type(
                *([alpha, beta] + activation_args)),
            split_k_mode=getattr(cutlass.conv.SplitKMode, split_k_mode),
            split_k_slices=problem_size.split_k_slices
        )

        if split_k_mode == "Parallel" and split_k_slices > 1:
            implicit_gemm_size = cutlass.conv.implicit_gemm_problem_size(
                conv_kind, arguments.problem_size)
            reduction_arguments = ReductionArguments(
                reduction_operation,
                problem_size=[implicit_gemm_size.m(), implicit_gemm_size.n()],
                partitions=problem_size.split_k_slices,
                workspace=arguments.ptr_D,
                destination=tensor_D,
                source=tensor_C,
                output_op=reduction_operation.epilogue_type(
                    *([alpha, beta] + activation_args)),
                bias=arguments.bias
            )

        operation.run(arguments)

        if split_k_mode == "Parallel" and split_k_slices > 1:
            reduction_operation.run(reduction_arguments)
            reduction_arguments.sync()
        else:
            arguments.sync()
        return tensor_D


cutlass_conv2d = CutlassConv2dFunc.apply


def compute_dims(H, W, kernel, pad, dilation, stride):
    pad_h = pad[0] + pad[1]
    pad_w = pad[2] + pad[3]
    out_h = (H + pad_h - dilation[0]*(kernel[0]-1) - 1) // stride[0] + 1
    out_w = (W + pad_w - dilation[1] * (kernel[1] - 1) - 1) // stride[1] + 1
    return out_h, out_w


class CutlassConv2d(torch.nn.Module):
    def __init__(
            self,
            instruction_shape: Tuple[int, int, int],  # instruction_shape
            type_a: str,  # element_a, choices=['float64', 'float32', 'float16', 'bfloat16', 'int32', 'int8']
            type_b: str,  # element_b
            type_c: str,  # element_c
            type_acc: str,  # element_acc
            math_instruction: str, #  choices=["multiply_add", "multiply_add_fast_bf16", "multiply_add_fast_f32"]
            opcode: str,  # choices=["Simt", 'TensorOp']
            threadblock_shape: Tuple[int, int, int],  # default [128, 128, 8], describes the tile size a thread block with compute
            stages: int,  # default=4, "Number of pipelines you want to use"
            warp_count: Tuple[int, int, int],  # default=[4, 2, 1], This option describes the number of warps along M, N, and K of the threadblock
            compute_capability: int,  # default=80, option describes CUDA SM architecture number
            layout_a: str,  # default="TensorNHWC", choices=["TensorNHWC", "TensorNC32HW32"]
            alignment_a: int,  # default=1, "Memory alignement of input tensor A
            layout_b: str,  # default="TensorNHWC", choices=["TensorNHWC", "TensorC32RSK32"]
            alignment_b: int,  # default=1, "Memory alignement of input tensor B
            layout_c: str,  # default="TensorNHWC", choices=["TensorNHWC", "TensorNC32HW32"]
            alignment_c: int,  # default=1, Memory alignment of input tensor C and output tensor D
            type_epilogue: str,  # Data type of computation in the epilogue
            epilogue_functor: str,  # choices=['LinearCombination', 'FastLinearCombinationClamp', 'LinearCombinationClamp'],
            # "This option describes the epilogue part of the kernel"
            swizzling_functor: str,  #  default="IdentitySwizzle1", choices=["IdentitySwizzle1", "IdentitySwizzle2", "IdentitySwizzle4", "IdentitySwizzle8", "HorizontalSwizzle", "StridedDgradIdentitySwizzle1", "StridedDgradIdentitySwizzle4", "StridedDgradHorizontalSwizzle"]
            # This option describes how thread blocks are scheduled on GPU
            conv_kind: str,  # choices=['fprop', 'dgrad', 'wgrad']
            stride_support: str,  # default="Strided", choices=["Strided", "Unity"]
            iterator_algorithm: str,  # default="analytic", choices=["analytic", "optimized", "fixed_channels", "few_channels"]
            split_k_mode: str,  # default="Serial", choices=["Serial", "Parallel"]
            # "Split K Mode. Serial is used for non-splitK or serial-splitK.\
            #                         Parallel is used for parallel splitK."
            split_k_slices: int,  # default=1 help="Number of split-k partitions. (default 1)"
            nhwc: Tuple[int, int, int, int],  # input size (NHWC)
            krsc: Tuple[int, int, int, int],  # filter size (KRSC)
            pad: Tuple[int, int, int, int],  # "padding (pad_h, _, pad_w, _)"
            stride: Tuple[int, int],  # stride (stride_h, stride_w)
            dilation: Tuple[int, int],  # dilation (dilation_h, dilation_w)
            alpha: float,  # default=1.0, help="alpha"
            beta: float,  # default=0.0, help="beta"
            bias: bool,  # default=False, help="C is bias vector"
            activation_function: str,  # default="identity", choices=["identity", "relu", "leaky_relu", "tanh", "sigmoid", "silu", "hardswish", "gelu"]
            activation_args: List[float],  # default=[], help="addition arguments for activation"
    ):
        super().__init__()
        # set the memory occupation, # TODO: use a more precise method
        pycutlass.get_memory_pool(init_pool_size=2 ** 30, max_pool_size=2 ** 32)

        # convert input types in cutlass formats
        np.random.seed(0)
        reduction_operation = None
        type_a_str = type_a
        type_b_str = type_b
        type_c_str = type_c
        type_a = getattr(cutlass, type_a)
        type_b = getattr(cutlass, type_b)
        type_c = getattr(cutlass, type_c)
        element_acc = getattr(cutlass, type_acc)
        math_operation = getattr(MathOperation, math_instruction)
        opclass = getattr(cutlass.OpClass, opcode)
        type_epilogue = getattr(cutlass, type_epilogue)
        layout_a = getattr(cutlass, layout_a)
        layout_b = getattr(cutlass, layout_b)
        layout_c = getattr(cutlass, layout_c)
        iterator_algorithm = getattr(cutlass.conv.IteratorAlgorithm, iterator_algorithm)
        swizzling_functor = getattr(cutlass, swizzling_functor)
        stride_support = getattr(StrideSupport, stride_support)
        conv_kind = getattr(cutlass.conv.Operator, conv_kind)
        epilogue_functor_str = epilogue_functor

        math_inst = MathInstruction(
            instruction_shape, type_a, type_b,
            element_acc, opclass, math_operation
        )

        # Build a Tile Description based on the input information.
        # Note that informations as threadblock_shape, stages and warp_count
        #  are specific to each GPU device.
        tile_description = TileDescription(
            threadblock_shape, stages, warp_count, math_inst,  # math_inst should be a parameter imported by CuTLAS
        )
        A = TensorDescription(
            type_a, layout_a, alignment_a
        )

        B = TensorDescription(
            type_b, layout_b, alignment_b
        )

        C = TensorDescription(
            type_c, layout_c, alignment_c
        )
        if (activation_function == "identity"
                or (
                        split_k_mode == "Parallel" and split_k_slices > 1)):
            #
            epilogue_functor = getattr(pycutlass, epilogue_functor)(
                C.element, C.alignment, math_inst.element_accumulator,
                type_epilogue)
        else:
            epilogue_functor = getattr(pycutlass, "LinearCombinationGeneric")(
                getattr(pycutlass, activation_function)(type_epilogue),
                C.element, C.alignment, math_inst.element_accumulator,
                type_epilogue)

        operation = Conv2dOperation(
            conv_kind=conv_kind, iterator_algorithm=iterator_algorithm,
            arch=compute_capability, tile_description=tile_description,
            A=A, B=B, C=C, stride_support=stride_support,
            epilogue_functor=epilogue_functor,
            swizzling_functor=swizzling_functor
        )
        self._internal_cuda_kernel = operation.rt_module.emit()
        operations = [operation, ]

        if split_k_mode == "Parallel" and split_k_slices > 1:
            if activation_function == "identity":
                epilogue_functor_reduction = getattr(pycutlass, epilogue_functor_str)(
                    C.element, C.alignment, math_inst.element_accumulator,
                    type_epilogue)
            else:
                epilogue_functor_reduction = getattr(pycutlass, "LinearCombinationGeneric")(
                    getattr(pycutlass, activation_function)(
                        type_epilogue),
                    C.element, C.alignment, math_inst.element_accumulator,
                    type_epilogue)
            reduction_operation = ReductionOperation(
                shape=cutlass.MatrixCoord(4, 32 * C.alignment),
                C=C, element_accumulator=element_acc,
                element_compute=type_epilogue,
                epilogue_functor=epilogue_functor_reduction,
                count=C.alignment
            )
            operations.append(reduction_operation)

        pycutlass.compiler.add_module(operations)
        problem_size = cutlass.conv.Conv2dProblemSize(
            cutlass.Tensor4DCoord(*nhwc),
            cutlass.Tensor4DCoord(*krsc),
            cutlass.Tensor4DCoord(*pad),
            cutlass.MatrixCoord(*stride),
            cutlass.MatrixCoord(*dilation),
            cutlass.conv.Mode.cross_correlation,
            split_k_slices,
            1,
        )

        # User-provide inputs
        tensor_A_size = cutlass.conv.implicit_gemm_tensor_a_size(
            conv_kind, problem_size
        )
        tensor_B_size = cutlass.conv.implicit_gemm_tensor_b_size(
            conv_kind, problem_size
        )
        if bias:
            tensor_C_size = cutlass.conv.implicit_gemm_tensor_c_extent(
                conv_kind, problem_size
            ).at(3)
        else:
            tensor_C_size = cutlass.conv.implicit_gemm_tensor_c_size(
                conv_kind, problem_size
            )

        tensor_D_size = cutlass.conv.implicit_gemm_tensor_c_size(
            conv_kind, problem_size
        )

        if type_b_str != "int8":
            tensor_B = torch.ceil(torch.empty(size=(tensor_B_size,),
                                              dtype=getattr(torch,
                                                            type_b_str),
                                              device="cuda").uniform_(-8.5,
                                                                      7.5))
        else:
            tensor_B = torch.empty(size=(tensor_B_size,),
                                   dtype=getattr(torch, type_b_str),
                                   device="cuda").uniform_(-2, 2)

        if type_c_str != "int8":
            tensor_C = torch.ceil(torch.empty(size=(tensor_C_size,),
                                              dtype=getattr(torch,
                                                            type_c_str),
                                              device="cuda").uniform_(-8.5,
                                                                      7.5))
        else:
            tensor_C = torch.empty(size=(tensor_C_size,),
                                   dtype=getattr(torch, type_c_str),
                                   device="cuda").uniform_(-2, 2)

        self.tensor_D_size = tensor_D_size
        self.tensor_B = tensor_B  # weight
        self.tensor_C = tensor_C  # Bias
        self.type_c_str = type_c_str

        self.operation = operation
        self.problem_size = problem_size
        self.alpha = alpha
        self.beta = beta
        self.activation_args = activation_args
        self.split_k_mode = split_k_mode
        self.split_k_slices = split_k_slices
        self.conv_kind = conv_kind
        self.reduction_operation = reduction_operation
        self.out_H, self.out_W = compute_dims(nhwc[1], nhwc[2], krsc[1:3], pad, dilation, stride)
        self.out_C = krsc[0]

    def forward(self, tensor_A):
        N = tensor_A.shape[0]
        tensor_A = tensor_A.permute(0, 2, 3, 1)  # .reshape(-1).contiguous()
        return cutlass_conv2d(
            tensor_A,
            self.tensor_B,
            self.tensor_C,
            self.tensor_D_size,
            self.type_c_str,
            self.operation,
            self.problem_size,
            self.alpha,
            self.beta,
            self.activation_args,
            self.split_k_mode,
            self.split_k_slices,
            self.conv_kind,
            self.reduction_operation,
        ).contiguous().view(N, self.out_H, self.out_W, self.out_C).permute(0, 3, 1, 2)

    @classmethod
    def from_conv2d(cls, input_shape: Tuple[int, int, int], conv: torch.nn.Conv2d):
        # pytorch Conv2d has instruction_shape NCHW, while CUTLASS uses NHWC
        input_shape = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]
        weight = conv.weight
        kernel_size = weight.shape
        krsc = [kernel_size[0], kernel_size[2], kernel_size[3], kernel_size[1]]
        pad = conv._reversed_padding_repeated_twice
        stride = conv.stride
        dilation = conv.dilation
        if isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(dilation, int):
            dilation = [dilation, dilation]
        data_type = "float32" if weight.dtype is torch.float32 else "float16"
        bias = conv.bias
        self = cls(
            instruction_shape=[16, 8, 8],  # alternative [1,1,1] I'm not sure about it
            type_a=data_type,
            type_b=data_type,
            type_c=data_type,
            type_acc=data_type,
            math_instruction="multiply_add", # "multiply_add_fast_f32" if data_type=="float32" else "multiply_add",
            opcode="TensorOp",
            ############## Set the quantities below depending on the HW  ###################
            threadblock_shape=[128, 128, 16],
            stages=4,
            warp_count=[2, 2, 1],
            compute_capability=80,
            ##################  Finished  #######################
            layout_a="TensorNHWC",
            alignment_a=4 if data_type == "float32" else 2,
            layout_b="TensorNHWC",
            alignment_b=4 if data_type == "float32" else 2,
            layout_c="TensorNHWC",
            alignment_c=4 if data_type == "float32" else 2,
            type_epilogue=data_type,
            epilogue_functor="LinearCombination",  # TODO: understand what it does
            swizzling_functor="IdentitySwizzle1",  # TODO: understand what it does
            conv_kind="fprop",
            stride_support="Strided" if np.prod(conv.stride)>1 else "Unity",
            iterator_algorithm="analytic",  # Try with other configurations
            split_k_mode="Serial",
            split_k_slices=1,  # to be changed accordingly to dims that impact the k dimension in the ImplicitGemm
            nhwc=input_shape,
            krsc=krsc,
            pad=pad,
            stride=stride,
            dilation=dilation,
            alpha=1.0,
            # beta=0.0,
            beta=1.0,
            bias=bias is not None,
            activation_function="identity",  # change with other for further performance
            activation_args=[],
        )
        self.tensor_B = copy.deepcopy(weight).permute(0, 2, 3, 1).cuda().contiguous()
        if bias is not None:
            self.tensor_C = copy.deepcopy(bias).cuda().contiguous()
        # self.tensor_B = torch.zeros(self.tensor_B.shape).contiguous().cuda()
        # if bias is not None:
        #     self.tensor_C = torch.zeros(self.tensor_C.shape).contiguous().cuda()
        return self


def add_shape_hook(module: torch.nn.Module, shape_list):
    def append_input_shape(module, input_tensor, output_tensor):
        assert isinstance(input_tensor, tuple)
        input_tensor = tuple(t.shape for t in input_tensor)
        shape_list.append(input_tensor)

    def recursively_register_hook(module):
        if isinstance(module, torch.nn.Conv2d):
            module.register_forward_hook(append_input_shape)
        else:
            named_children = list(module.named_children())
            if len(named_children) > 0:
                for name, submodule in named_children:
                    recursively_register_hook(submodule)

    recursively_register_hook(module)


def trace_and_replace(module: torch.nn.Module, input_sample: torch.Tensor):
    new_module = copy.deepcopy(module)
    shape_list = []
    add_shape_hook(new_module, shape_list)
    with torch.no_grad():
        _ = new_module(input_sample)
    print(f"Number of conv2d layers: {len(shape_list)}")
    new_module = copy.deepcopy(module)
    new_module = replace_conv2d_module(new_module, shape_list)
    return new_module


def run_timed_experiment(layer, activations):
    running_time_list = []
    with torch.no_grad():
        for input_tensor in activations:
            starting_time = time.time()
            _ = layer(input_tensor)
            running_time_list.append(time.time() - starting_time)
    return sum(running_time_list) / len(activations)


def replace_conv2d_module(module: torch.nn.Module, input_shapes: List[Tuple]):
    if isinstance(module, torch.nn.Conv2d):
        shape = input_shapes.pop(0)[0]  # Just one tensor as input
        if args.verbose:
            print("####### Conv2d layer #######")
            print(module.in_channels, module.out_channels)

        if module.in_channels % 2 == 0 and module.out_channels % 2 == 0:
            new_module = CutlassConv2d.from_conv2d(shape, module)
            device = module.weight.device
            activations = [torch.randn(*shape).to(device) for _ in range(args.experiment_replicas)]
            new_module_runtime = run_timed_experiment(new_module, activations)
            base_module_runtime = run_timed_experiment(module, activations)
            if new_module_runtime > base_module_runtime:
                new_module = module
            if args.verbose:
                print("######## Runtimes ##########")
                print(f"Base: {base_module_runtime}\nCutlass: {new_module_runtime}")
        else:
            new_module = module
    else:
        named_children = list(module.named_children())
        if len(named_children) > 0:
            for name, submodule in named_children:
                new_submodule = replace_conv2d_module(submodule, input_shapes)
                if new_submodule is not module:
                    module._modules[name] = new_submodule

        new_module = module
    return new_module


if __name__ == "__main__":
    from argparse import ArgumentParser
    from torchvision import models
    parser = ArgumentParser()
    parser.add_argument("--input_shape", default=[1, 3, 224, 224], nargs=4, type=int, help="Input Shape")
    parser.add_argument("--half", action="store_true", help="Activate half precision")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--experiment_replicas", "-er", type=int, default=100, help="Number of times the experiments must be run for getting statistics.")
    args = parser.parse_args()
    input_shape = args.input_shape
    half = args.half
    base_model = models.resnet50().eval().cuda()
    input_sample = torch.randn(*input_shape).cuda()
    if half:
        base_model = base_model.half()
        input_sample = input_sample.half()
    cutlass_model = trace_and_replace(base_model, input_sample)
    input_data = [torch.randn(*input_shape).cuda() for _ in range(100)]
    if half:
        input_data = [x.half() for x in input_data]
    with torch.no_grad():
        times = []
        cutlass_times = []
        preds = []
        cutlass_preds = []
        for tensor in input_data:
            st = time.time()
            pred = base_model(tensor)
            times.append(time.time()-st)
            preds.append(pred)
        for tensor in input_data:
            st = time.time()
            pred = cutlass_model(tensor)
            cutlass_times.append(time.time()-st)
            cutlass_preds.append(pred)
    print("##################### Final Results ####################")
    print(f"Torch: {float(np.mean(times))}\nCutlass: {float(np.mean(cutlass_times))}")
    print(f"Difference: {np.mean(torch.stack([torch.abs((pred1-pred2)/(pred1+1e-7)) for pred1, pred2 in zip(preds, cutlass_preds)]).cpu().numpy())}")
    # for i, (pred1, pred2) in enumerate(zip(preds, cutlass_preds)):
    #     try:
    #         assert torch.equal(pred1, pred2)
    #     except:
    #         print(i)
    #         err = torch.abs((pred1-pred2)/(pred1+1e-7))
    #         arg_max = err.reshape(-1).argmax()
    #         print(err.max(), err.mean(), pred1.reshape(-1)[arg_max], pred2.reshape(-1)[arg_max])
