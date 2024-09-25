import torch
import triton
import triton.language as tl
import pytest
import pdb

@triton.jit
def dot_default(x_ptr, y_ptr, o_ptr, M: tl.constexpr):
    block_range = tl.arange(0, M)[:, None]*M + tl.arange(0, M)[None, :]
    x = tl.load(x_ptr + block_range)
    y = tl.load(y_ptr + block_range)

    o = tl.dot(x, y)
    tl.store(o_ptr + block_range, o)

@triton.jit
def dot_cast_inside(x_ptr, y_ptr, o_ptr, M: tl.constexpr):
    block_range = tl.arange(0, M)[:, None]*M + tl.arange(0, M)[None, :]
    x = tl.load(x_ptr + block_range)
    y = tl.load(y_ptr + block_range)

    o = tl.dot(x, y, out_dtype=tl.float16)
    tl.store(o_ptr + block_range, o)

@triton.jit
def dot_cast_outside(x_ptr, y_ptr, o_ptr, M: tl.constexpr):
    block_range = tl.arange(0, M)[:, None]*M + tl.arange(0, M)[None, :]
    x = tl.load(x_ptr + block_range)
    y = tl.load(y_ptr + block_range)

    o = tl.dot(x, y)
    o = tl.cast(o, tl.float16, "rtne")
    tl.store(o_ptr + block_range, o)


@pytest.mark.parametrize("seed", [i for i in range(1000)])
def test_basic(seed):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    M = 32

    torch.manual_seed(43)

    x = torch.rand(M, M, dtype=torch.float16, device=device)
    y = torch.rand(M, M, dtype=torch.float16, device=device)

    print("x", x)
    print("y", y)

    grid = (1,)
    out_dot_default = torch.zeros(M, M, dtype=torch.float16, device=device)
    out_dot_cast_inside = torch.zeros(M, M, dtype=torch.float16, device=device)
    out_dot_cast_outside = torch.zeros(M, M, dtype=torch.float16, device=device)

    out_torch = torch.matmul(x, y)
    with torch.cuda.device(x.device):
        dot_default[grid](x, y, out_dot_default, M)
        dot_cast_inside[grid](x, y, out_dot_cast_inside, M)
        dot_cast_outside[grid](x, y, out_dot_cast_outside, M)


    print("dot - dot_cast", (out_dot_cast_inside-out_dot_cast_outside).abs().max().item())
    print("torch - dot", (out_torch-out_dot_cast_inside).abs().max().item())
    print("torch - dot_cast", (out_torch-out_dot_cast_outside).abs().max().item())
    print("torch - dot_default", (out_torch-out_dot_default).abs().max().item())

    assert torch.allclose(out_torch, out_dot_default, atol=0)
    assert torch.allclose(out_torch, out_dot_cast_inside, atol=0)
    assert torch.allclose(out_torch, out_dot_cast_outside, atol=0)




# New exp2 kernels
@triton.jit
def exp2_fp16(x_ptr, y_ptr, o_ptr, M: tl.constexpr):
    block_range = tl.arange(0, M)[:, None]*M + tl.arange(0, M)[None, :]
    x = tl.load(x_ptr + block_range)
    y = tl.load(y_ptr + block_range)

    log2_e = 1.4426950408889634  # log2(e)

    x = tl.exp2(log2_e * x)
    y = tl.exp2(log2_e * y)

    o = tl.dot(x, y, out_dtype=tl.float16)
    tl.store(o_ptr + block_range, o)

@triton.jit
def exp2_fp32_cast(x_ptr, y_ptr, o_ptr, M: tl.constexpr):
    block_range = tl.arange(0, M)[:, None]*M + tl.arange(0, M)[None, :]
    x = tl.load(x_ptr + block_range)
    y = tl.load(y_ptr + block_range)

    # cast to fp32
    x = x.to(tl.float32)
    y = y.to(tl.float32)
    
    log2_e = 1.4426950408889634  # log2(e)

    x = tl.exp2(log2_e * x)
    y = tl.exp2(log2_e * y)

    o = tl.dot(x, y, out_dtype=tl.float16)
    tl.store(o_ptr + block_range, o)

@pytest.mark.parametrize("seed", [i for i in range(100)])  # Reduced number of seeds for brevity
def test_exp2(seed):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    M = 16

    torch.manual_seed(43)

    x = torch.rand(M, M, dtype=torch.float16, device=device)
    y = torch.rand(M, M, dtype=torch.float16, device=device)

    print("x", x)
    print("y", y)

    grid = (1,)
    out_exp2_fp16 = torch.zeros(M, M, dtype=torch.float16, device=device)
    out_exp2_fp32_cast = torch.zeros(M, M, dtype=torch.float16, device=device)

    out_torch = torch.matmul(torch.exp(x), torch.exp(y))

    with torch.cuda.device(x.device):
        exp2_fp16[grid](x, y, out_exp2_fp16, M)
        exp2_fp32_cast[grid](x, y, out_exp2_fp32_cast, M)

    print("torch", out_torch)
    print("out_exp2_fp16", out_exp2_fp16)
    print("out_exp2_fp32_cast", out_exp2_fp32_cast)

    print("torch - out_exp2_fp16", (out_torch-out_exp2_fp16))
    print("torch - out_exp2_fp32_cast", (out_torch-out_exp2_fp32_cast))

    assert torch.allclose(out_torch, out_exp2_fp16, atol=1e-5)
    assert torch.allclose(out_torch, out_exp2_fp32_cast, atol=1e-5)