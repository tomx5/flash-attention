import pytest
import torch
import math

import pdb

import triton
import triton.language as tl

@triton.jit
def rotary_kernel_splitk(
    # Dimensions of X
    X,              # tensor being rotated. Has shape (batch (z), seqlen (s), group (g), head (h), head_dim (d))
    seqlen_x,       # seqlen of the x dim. shape is (batch (z), )
    rotary_dim: tl.constexpr,     # size of embedding space we end up rotating

    # COS/SIN and Offsetting Into It
    COS,            # tensor of shape (seqlen (m), ro_dim // 2)
    SIN,            # tensor of shape (seqlen (m), ro_dim // 2)
    SEQLEN_OFFSET,  # we use this as an offset into COS and SIN to apply the correct rotation
    SEQLEN_OFFSET_IS_TENSOR: tl.constexpr, # if seqlen_offset is a tensor it has shape (num_batch, )
    
    # PID Offsets
    batch_pid: tl.constexpr,      # pid for batch
    start_m: tl.constexpr,        # the token idx the current M_BLOCK starts at.
    group_pid: tl.constexpr,      # pid for group
    head_pid: tl.constexpr,       # pid to access head

    # Strides
    stride_batch: tl.constexpr,
    stride_m: tl.constexpr,
    stride_group: tl.constexpr,
    stride_head: tl.constexpr,
    stride_headdim: tl.constexpr,

    # Misc
    INTERLEAVED: tl.constexpr,

    # Meta-parameters
    BLOCK_M: tl.constexpr,     # block size to access chunks of tokens (# of tokens simultaneously)
    BLOCK_K: tl.constexpr,     # block size to access chunks of headdim (# of dimensions processed)
):
    """
    Note: 
    - for K in splitk let BLOCK_M = BLOCK_N, and start_m=start_n
    """
    # pdb.set_trace()
    range_m = start_m + tl.arange(0, BLOCK_M)
    range_d = tl.arange(0, BLOCK_K)

    x_ptr = X + (batch_pid * stride_batch) + (group_pid * stride_group) + (head_pid * stride_head)   # pointer to x block
    x_mask = (range_m < seqlen_x)[:, None] & (range_d < rotary_dim)[None, :]

    ro_dim_half = rotary_dim // 2       # length of cos/sin
    range_d_half = tl.arange(0, BLOCK_K // 2)

    # COS/SIN Range
    # cs_range = (SEQLEN_OFFSET + tl.arange(0, BLOCK_M))[:, None]*(ro_dim_half) + tl.arange(0, ro_dim_half)

    if SEQLEN_OFFSET_IS_TENSOR:
        seqlen_offset = tl.load(SEQLEN_OFFSET + batch_pid) # a tensor
    else:
        seqlen_offset = SEQLEN_OFFSET # an int

    if not INTERLEAVED:
        range_d_half_duplicate = range_d % (rotary_dim // 2)

        x0_range = range_m[:, None]*stride_m + range_d_half_duplicate[None, :]*stride_headdim                # BLOCK_M x 1st half of headdim (fast to load)
        x1_range = range_m[:, None]*stride_m + range_d_half_duplicate[None, :]*stride_headdim + ro_dim_half  # BLOCK_M x 2nd half of headdim (fast to load)

        
        print('range_m', range_m)
        print('x0_range', x0_range)
        print('x1_range', x1_range)

        print('range_d_half_duplicate', range_d_half_duplicate)

        x0_mask = (range_m < seqlen_x)[:, None] & (range_d_half_duplicate < rotary_dim)[None, :]                  # Mask for the first half
        x1_mask = (range_m < seqlen_x)[:, None] & (range_d_half_duplicate + ro_dim_half < rotary_dim)[None, :]    # Mask for the second half

        range_m_cos_sin = range_m + seqlen_offset # offsets cos and sin based on current m position range and seqlen offset
        COS = COS + (range_m_cos_sin[:, None] * rotary_dim + range_d_half_duplicate[None, :])
        SIN = SIN + (range_m_cos_sin[:, None] * rotary_dim + range_d_half_duplicate[None, :])
        cos = tl.load(
            COS, mask=(range_m[:, None] < seqlen_x) & (range_d_half_duplicate[None, :] < ro_dim_half), other=1.0
        ).to(tl.float32)
        sin = tl.load(
            SIN, mask=(range_m[:, None] < seqlen_x + seqlen_offset) & (range_d_half_duplicate[None, :] < ro_dim_half), other=0.0
        ).to(tl.float32)
        # if CONJUGATE:
        #     sin = -sin
        
        x0 = tl.load(x_ptr + x0_range, mask=x0_mask).to(tl.float32)
        x1 = tl.load(x_ptr + x1_range, mask=x1_mask).to(tl.float32)

        print('x0', x0)
        print('x1', x1)

        print('sin', sin)
        print('cos', cos)

        # Rotate corresponding elements in each half
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos

        # o0 = tl.join(o0, o0)
        # o0 = tl.permute(o0, 0, 2, 1)
        # o0 = tl.reshape(o0, BLOCK_M, BLOCK_K)

        # o1 = tl.join(o1, o1)
        # o1 = tl.permute(o1, 0, 2, 1)
        # o1 = tl.reshape(o1, BLOCK_M, BLOCK_K)

        print('o0', o0)
        print('o1', o1)

        # print('ro_dim_half: ', ro_dim_half)
        # print('range_d: ', range_d[None, :])
        # print('bool', range_d[None, :] // ro_dim_half == 0)
        out = tl.where(range_d[None, :] // ro_dim_half == 0, o0, o1)
        out = tl.where(range_d[None, :] < rotary_dim, out, 0.0)

        return out
        
    else:
        # Interleaved is slow due to x1 load
        range_d_swap = range_d + ((range_d + 1) % 2) * 2 - 1            # 1, 0, 3, 2, 5, 4, ...

        # X Range
        x0_range = range_m[:, None]*stride_m + range_d[None, :]         # 0, 1, 2, 3, 4, 5, ... (fast to load)
        x1_range = range_m[:, None]*stride_m + range_d_swap[None, :]    # 1, 0, 3, 2, 5, 4, ... (slow to load)
        
        # X Masks
        x0_mask = (range_m < seqlen_x)[:, None] & (range_d < rotary_dim)[None, :]                  # Mask for the first half
        x1_mask = (range_m < seqlen_x)[:, None] & (range_d_swap < rotary_dim)[None, :]    # Mask for the second half
        
        # Load COS/SIN
        range_d_repeat = tl.arange(0, BLOCK_K) // 2                # 0, 0, 1, 1, 2, 2, ...

        range_m_cos_sin = range_m + seqlen_offset
        COS = COS + (range_m_cos_sin[:, None] * ro_dim_half + range_d_repeat[None, :])
        SIN = SIN + (range_m_cos_sin[:, None] * ro_dim_half + range_d_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(range_m[:, None] < seqlen_x) & (range_d_repeat[None, :] < ro_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(range_m[:, None] < seqlen_x) & (range_d_repeat[None, :] < ro_dim_half),
            other=0.0,
        ).to(tl.float32)
        # if CONJUGATE:
        #     sin = -sin

        x0 = tl.load(x_ptr + x0_range, mask=x0_mask)
        x1 = tl.load(x_ptr + x1_range, mask=x1_mask)

        x0_cos = x0 * cos
        x1_sin = x1 * sin

        out = tl.where(range_d[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)

        return out

@triton.jit
def main(
    # Dimensions of X
    X,              # tensor being rotated. Has shape (batch (z), seqlen (s), group (g), head (h), head_dim (d))
    seqlen_x,       # seqlen of the x dim. shape is (batch (z), )
    rotary_dim: tl.constexpr,     # size of embedding space we end up rotating

    # COS/SIN and Offsetting Into It
    COS,            # pointer to tensor of shape (seqlen (m), ro_dim // 2)
    SIN,            # pointer to tensor of shape (seqlen (m), ro_dim // 2)
    SEQLEN_OFFSET,  # we use this as an offset into COS and SIN to apply the correct rotation
    SEQLEN_OFFSET_IS_TENSOR: tl.constexpr, # if seqlen_offset is a tensor it has shape (num_batch, )
    
    # PID Offsets
    batch_pid: tl.constexpr,      # pid for batch
    start_m: tl.constexpr,        # the token idx the current M_BLOCK starts at.
    group_pid: tl.constexpr,      # pid for group
    head_pid: tl.constexpr,       # pid to access head

    # Strides
    stride_batch: tl.constexpr,
    stride_m: tl.constexpr,
    stride_group: tl.constexpr,
    stride_head: tl.constexpr,
    stride_headdim: tl.constexpr,

    # Misc
    INTERLEAVED: tl.constexpr,

    # Meta-parameters
    BLOCK_M: tl.constexpr,     # block size to access chunks of tokens (# of tokens simultaneously)
    BLOCK_K: tl.constexpr,     # block size to access chunks of headdim (# of dimensions processed)

    # Split Parameters
    N_BLOCKS_PER_SPLIT: tl.constexpr,
):
    seqlen = seqlen_x

    # Below is the code for a single split
    lo = start_m
    hi = lo + BLOCK_M*N_BLOCKS_PER_SPLIT

    for start_n in tl.range(lo, hi, BLOCK_M):
        # Launch the kernel
        
        x = rotary_kernel_splitk(X=X,
                            seqlen_x=seqlen,
                            rotary_dim=rotary_dim,
                            COS=COS,
                            SIN=SIN,
                            SEQLEN_OFFSET=SEQLEN_OFFSET,
                            SEQLEN_OFFSET_IS_TENSOR=SEQLEN_OFFSET_IS_TENSOR,
                            batch_pid=batch_pid,      
                            start_m=start_n,          
                            group_pid=group_pid,      
                            head_pid=head_pid,          
                            stride_batch=stride_batch,
                            stride_m=stride_m,
                            stride_group=stride_group,
                            stride_head=stride_head,
                            stride_headdim=stride_headdim,
                            INTERLEAVED=False,
                            BLOCK_M=BLOCK_M,          
                            BLOCK_K=BLOCK_K,
                            )
        # pdb.set_trace()
        print(x)
        # x_ptrs = X + (tl.arange(0, BLOCK_M))
        # tl.store(X, x, )


def test_rotary_kernel_split():
    batch = 1
    seqlen = 8
    group = 1
    head = 1
    head_dim = 8
    assert head_dim >= 2
    
    device = "cuda"

    # Create a simple tensor with the specified shape
    X = torch.ones(batch, seqlen, group, head, head_dim, device='cuda')

    rotary_dim = head_dim

    # Compute theta for every dim in a token
    theta_base = 10_000
    numerator = torch.arange(0, rotary_dim, 2).float()[:rotary_dim//2]
    theta = 1.0 / theta_base**(numerator / rotary_dim)

    # pos/index in the sequence
    SEQLEN_OFFSET_IS_TENSOR = False
    if SEQLEN_OFFSET_IS_TENSOR:
        SEQLEN_OFFSET = torch.randint(low=0, high=4, size=(batch, ))
        max_offset = SEQLEN_OFFSET.max()
    else:
        max_offset = SEQLEN_OFFSET = 0

    print("SEQLEN_OFFSET", SEQLEN_OFFSET)

    m = torch.arange(seqlen + (max_offset), dtype=theta.dtype, device=theta.device)

    # outer product between m and theta
    angle = torch.outer(m, theta)

    # Computes the cos and sin tensors needed
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    # Define the grid and block sizes for the kernel launch
    BLOCK_M = seqlen

    # Rotary Block Metadata (compute BLOCK_K)
    BLOCK_K = (
        32
        if rotary_dim <= 32
        else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    )

    BLOCK_K = 8

    BLOCK_N = seqlen
    num_splits = 1
    N_BLOCKS_PER_SPLIT = (seqlen // BLOCK_N) // num_splits

    # for each chunk of tokens
    for start_m in range(0, seqlen, BLOCK_M):


        for off_zhg in range(0, batch*head*group):
            
            # zhg program ids
            batch_pid = off_zhg // (head * group)   # batch
            head_pid = (off_zhg // group) % head    # head
            group_pid = off_zhg % group             # group (gca / mqa)

            print("start_m: ", start_m, "| batch: ", batch_pid, "| head: ", head_pid, "| group: ", group_pid)

            # for each split
            BLOCK_SPLIT = BLOCK_N*N_BLOCKS_PER_SPLIT

            for lo in range(0, seqlen, BLOCK_SPLIT):
                grid = (1, 1, 1)
                main[grid](X=X,
                            seqlen_x=seqlen,
                            rotary_dim=rotary_dim,
                            COS=cos,
                            SIN=sin,
                            SEQLEN_OFFSET=SEQLEN_OFFSET,
                            SEQLEN_OFFSET_IS_TENSOR=isinstance(SEQLEN_OFFSET, torch.Tensor),
                            batch_pid=batch_pid,      
                            start_m=lo,          
                            group_pid=group_pid,      
                            head_pid=head_pid,          
                            stride_batch=X.stride(0),
                            stride_m=X.stride(-4),
                            stride_group=X.stride(-3),
                            stride_head=X.stride(-2),
                            stride_headdim=X.stride(-1),
                            INTERLEAVED=False,
                            BLOCK_M=BLOCK_N,          
                            BLOCK_K=BLOCK_K,
                            N_BLOCKS_PER_SPLIT=N_BLOCKS_PER_SPLIT
                            )
            print('\n')