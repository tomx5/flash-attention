import torch
import triton
import triton.language as tl
import pytest
import pdb

@triton.jit
def print_dtype(x_ptr, M: tl.constexpr, N: tl.constexpr):
    block_range = tl.arange(0, M)[:, None]*N + tl.arange(0, N)[:, None]
    x = tl.load(x_ptr + block_range)
    dtype = x_ptr.dtype.element_ty

    x = x
    dtype = dtype
    
    print(dtype)

def test_basic():

    print(torch.cuda.is_available())
    x = torch.zeros(2, 2, dtype=torch.float16)

    grid = (1,)
    print_dtype[(grid)](x, 2, 2)