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
    rotary_dim,     # size of embedding space we end up rotating

    # COS/SIN and Offsetting Into It
    COS,            # tensor of shape (seqlen (m), ro_dim // 2)
    SIN,            # tensor of shape (seqlen (m), ro_dim // 2)
    SEQLEN_OFFSET,  # we use this as an offset into COS and SIN to apply the correct rotation

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
    BLOCK_M: tl.constexpr,          # block size to access chunks of tokens (# of tokens simultaneously)
    BLOCK_DMODEL: tl.constexpr,     # block size to access chunks of headdim (# of dimensions processed)
):
    """
    Note: 
    - for K in splitk let BLOCK_M = BLOCK_N, and start_m=start_n
    """
    pdb.set_trace()
    range_m = start_m + tl.arange(0, BLOCK_M)
    range_d = tl.arange(0, BLOCK_DMODEL)

    x_ptr = X + (batch_pid * stride_batch) + (group_pid * stride_group) + (head_pid * stride_head)   # pointer to x block
    x_mask = (range_m < seqlen_x)[:, None] & (range_d < rotary_dim)[None, :]

    ro_dim_half = rotary_dim // 2       # length of cos/sin
    range_d_half = tl.arange(0, BLOCK_DMODEL // 2)

    # COS/SIN Range
    # cs_range = (SEQLEN_OFFSET + tl.arange(0, BLOCK_M))[:, None]*(ro_dim_half) + tl.arange(0, ro_dim_half)

    if not INTERLEAVED:
        x0_range = range_m[:, None]*stride_m + range_d_half[None, :]*stride_headdim                # BLOCK_M x 1st half of headdim (fast to load)
        x1_range = range_m[:, None]*stride_m + range_d_half[None, :]*stride_headdim + ro_dim_half  # BLOCK_M x 2nd half of headdim (fast to load)
        x0_mask = (range_m < seqlen_x)[:, None] & (range_d_half < rotary_dim)[None, :]                  # Mask for the first half
        x1_mask = (range_m < seqlen_x)[:, None] & (range_d_half + ro_dim_half < rotary_dim)[None, :]    # Mask for the second half
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
        range_d_repeat = tl.arange(0, BLOCK_DMODEL) // 2                # 0, 0, 1, 1, 2, 2, ...
        COS = COS + (range_m[:, None] * ro_dim_half + range_d_repeat[None, :])
        SIN = SIN + (range_m[:, None] * ro_dim_half + range_d_repeat[None, :])
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
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(range_d[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)

        pdb.set_trace()

        return out
    pdb.set_trace()
    x0 = tl.load(x_ptr + x0_range, mask=x0_mask)
    x1 = tl.load(x_ptr + x1_range, mask=x1_mask)

    

    pdb.set_trace()
    # x0 = x0*cos - x1*sin
    # x1 = x0*sin + x1*cos

@triton.jit
def split(
    # Dimensions of X
    X,              # tensor being rotated. Has shape (batch (z), seqlen (s), group (g), head (h), head_dim (d))
    seqlen_x,       # seqlen of the x dim. shape is (batch (z), )
    rotary_dim: tl.constexpr,     # size of embedding space we end up rotating

    # COS/SIN and Offsetting Into It
    COS,            # pointer to tensor of shape (seqlen (m), ro_dim // 2)
    SIN,            # pointer to tensor of shape (seqlen (m), ro_dim // 2)
    SEQLEN_OFFSET,  # we use this as an offset into COS and SIN to apply the correct rotation

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
    BLOCK_M: tl.constexpr,          # block size to access chunks of tokens (# of tokens simultaneously)
    BLOCK_DMODEL: tl.constexpr,     # block size to access chunks of headdim (# of dimensions processed)

    # Split Parameters
    N_BLOCKS_PER_SPLIT: tl.constexpr,
):
    # BLOCK_N = BLOCK_M
    seqlen = seqlen_x

    # load all of cos/sin ONCE for a single split (program)
    cs_range = (SEQLEN_OFFSET + tl.arange(0, BLOCK_M))[:, None]*(rotary_dim//2) + tl.arange(0, rotary_dim//2)
    cos = tl.load(COS + cs_range)
    sin = tl.load(SIN + cs_range)

    pdb.set_trace()
    
    # Below is the code for a single split
    lo = start_m
    hi = lo + BLOCK_M*N_BLOCKS_PER_SPLIT

    for start_n in tl.range(lo, hi, BLOCK_M):
        # Launch the kernel
        
        rotary_kernel_splitk(X=X,
                            seqlen_x=seqlen,
                            rotary_dim=rotary_dim,
                            COS=COS,
                            SIN=SIN,
                            SEQLEN_OFFSET=0,
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
                            BLOCK_DMODEL=BLOCK_DMODEL,
                            )


# @pytest.mark.parametrize('batch', [1])
# @pytest.mark.parametrize('seqlen', [1])
# @pytest.mark.parametrize('group', [1])
# @pytest.mark.parametrize('head', [1])
# @pytest.mark.parametrize('head_dim', [1])
#batch, seqlen, group, head, head_dim
def test_rotary_kernel_splitk():
    batch = 1
    seqlen = 8
    group = 2
    head = 1
    head_dim = 4
    assert head_dim >= 2
    
    device = "cuda"

    # Create a simple tensor with the specified shape
    X = torch.randn(batch, seqlen, group, head, head_dim, device='cuda')

    rotary_dim = head_dim

    # Compute theta for every dim in a token
    theta_base = 10_000
    numerator = torch.arange(0, rotary_dim, 2).float()[:rotary_dim//2]
    theta = 1.0 / theta_base**(numerator / rotary_dim)

    # pos/index in the sequence
    m = torch.arange(seqlen, dtype=theta.dtype, device=theta.device)

    # outer product between m and theta
    angle = torch.outer(m, theta)

    # Computes the cos and sin tensors needed
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    # Define the grid and block sizes for the kernel launch
    BLOCK_M = seqlen
    BLOCK_DMODEL = head_dim

    BLOCK_N = 2
    num_splits = 1
    N_BLOCKS_PER_SPLIT = (seqlen // BLOCK_N) // num_splits
    for start_m in range(0, seqlen, BLOCK_M):


        for off_zhg in range(0, batch*head*group):
            
            # zhg program ids
            batch_pid = off_zhg // (head * group)      # batch
            head_pid = (off_zhg // group) % head    # head
            group_pid = off_zhg % group             # group (gca / mqa)

            for lo in range(0, seqlen, BLOCK_N*N_BLOCKS_PER_SPLIT):
                # Below is the code for a single split
                hi = lo + BLOCK_N*N_BLOCKS_PER_SPLIT

                for start_n in range(lo, hi, BLOCK_N):
                    grid = (1, 1, 1)

                    pdb.set_trace()

                    # Launch the kernel
                    rotary_kernel_splitk[grid](X=X,
                                            seqlen_x=seqlen,
                                            rotary_dim=rotary_dim,
                                            COS=cos,
                                            SIN=sin,
                                            SEQLEN_OFFSET=0,
                                            batch_pid=batch_pid,      
                                            start_m=start_n,          
                                            group_pid=group_pid,      
                                            head_pid=head_pid,          
                                            stride_batch=X.stride(0),
                                            stride_m=X.stride(-4),
                                            stride_group=X.stride(-3),
                                            stride_head=X.stride(-2),
                                            stride_headdim=X.stride(-1),
                                            INTERLEAVED=False,
                                            BLOCK_M=BLOCK_N,          
                                            BLOCK_DMODEL=BLOCK_DMODEL,
                                            )

def test_rotary_kernel_split():
    batch = 1
    seqlen = 4
    group = 1
    head = 1
    head_dim = 4
    assert head_dim >= 2
    
    device = "cuda"

    # Create a simple tensor with the specified shape
    X = torch.randn(batch, seqlen, group, head, head_dim, device='cuda')

    rotary_dim = head_dim

    # Compute theta for every dim in a token
    theta_base = 10_000
    numerator = torch.arange(0, rotary_dim, 2).float()[:rotary_dim//2]
    theta = 1.0 / theta_base**(numerator / rotary_dim)

    # pos/index in the sequence
    SEQLEN_OFFSET = 0
    m = torch.arange(seqlen, dtype=theta.dtype, device=theta.device) + SEQLEN_OFFSET

    # outer product between m and theta
    angle = torch.outer(m, theta)

    # Computes the cos and sin tensors needed
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    # Define the grid and block sizes for the kernel launch
    BLOCK_M = seqlen
    BLOCK_DMODEL = head_dim

    BLOCK_N = seqlen
    num_splits = 1
    N_BLOCKS_PER_SPLIT = (seqlen // BLOCK_N) // num_splits
    for start_m in range(0, seqlen, BLOCK_M):


        for off_zhg in range(0, batch*head*group):
            
            # zhg program ids
            batch_pid = off_zhg // (head * group)      # batch
            head_pid = (off_zhg // group) % head    # head
            group_pid = off_zhg % group             # group (gca / mqa)

            for lo in range(0, seqlen, BLOCK_N*N_BLOCKS_PER_SPLIT):
                grid = (1, 1, 1)
                split[grid](X=X,
                            seqlen_x=seqlen,
                            rotary_dim=rotary_dim,
                            COS=cos,
                            SIN=sin,
                            SEQLEN_OFFSET=0,
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
                            BLOCK_DMODEL=BLOCK_DMODEL,
                            N_BLOCKS_PER_SPLIT=N_BLOCKS_PER_SPLIT
                            )