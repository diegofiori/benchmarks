from typing import Tuple

import torch


class CutlassConv2d(torch.nn.Module):
    def __init__(
            self,
            shape: Tuple[int, int, int],  # instruction_shape
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
            allignment_a: int,  # default=1, "Memory alignement of input tensor A
            layout_b: str,  # default="TensorNHWC", choices=["TensorNHWC", "TensorC32RSK32"]
            allignment_b: int,  # default=1, "Memory alignement of input tensor B
            layout_c: str,  # default="TensorNHWC", choices=["TensorNHWC", "TensorNC32HW32"]
            allignment_c: int,  # default=1, Memory alignment of input tensor C and output tensor D
            type_epilogue: str,  # Data type of computation in the epilogue
            epilogue_functor: str,  # choices=['LinearCombination', 'FastLinearCombinationClamp', 'LinearCombinationClamp'],
            # "This option describes the epilogue part of the kernel"
            swizzling_functor: str,  #  default="IdentitySwizzle1", choices=["IdentitySwizzle1", "IdentitySwizzle2", "IdentitySwizzle4", "IdentitySwizzle8", "HorizontalSwizzle", "StridedDgradIdentitySwizzle1", "StridedDgradIdentitySwizzle4", "StridedDgradHorizontalSwizzle"]
            # This option describes how thread blocks are scheduled on GPU
            conv_kind: str,  # choices=['fprop', 'dgrad', 'wgrad']

    ):
        super().__init__()

