import torch
import triton
import triton.language as tl
import pytest

@triton.jit
def kernel_matmul(x_ptr,
                  y_ptr,
                  BLOCK_SIZE: tl.constexpr,
                  M: tl.constexpr,
                  K: tl.constexpr,
                  N: tl.constexpr
                  ):
    m = tl.program_id(0)
    k = tl.program_id(1)

    # assume x is contiguous
    x_offsets = m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load()
    pass

def triton_matmul(x: torch.Tensor, y: torch.Tensor):
    pass

@pytest.mark.parametrize("M", [1, 4, 8, 16])
@pytest.mark.parametrize("K", [1, 2, 4, 8, 16, 64, 256, 1028])
@pytest.mark.parametrize("N", [1, 4, 8, 16])
@pytest.mark.parametrize("cast_to", [torch.float16, torch.float32]) # dtype to downcast to
# @pytest.mark.parametrize("fp_downcast_rounding", ['rtne','rtz'])    # round to nearest (rtne) / round to zero (rtz)
def test_dot_precision(M, K, N, cast_to, fp_downcast_rounding):
    torch.manual_seed(15)

    # make two matrices which we can 
    x = torch.rand(M, K, dtype=torch.float32)
    y = torch.rand(K, N, dtype=torch.float32)

    torch_ref = torch.matmul(x, y)
    triton_ref = triton_matmul(x, y)

    x, y = x.type(cast_to), y.type(cast_to)
    triton_downcast = triton_matmul(x, y)

    print((triton_ref-triton_downcast).abs())

    pass