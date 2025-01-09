import torch
import triton
import triton.language as tl


@triton.jit
def matmul_fp8_kernel_no_loop(
    A_ptr,  # [M, K] in FP8
    B_ptr,  # [K, N] in FP8
    C_ptr,  # [M, N] in float32 (for storing the result)
    M, N, K,
    stride_am, stride_ak,  # strides for A
    stride_bk, stride_bn,  # strides for B
    stride_cm, stride_cn,  # strides for C
    BLOCK_M: tl.constexpr, # tile size along M
    BLOCK_N: tl.constexpr, # tile size along N
    BLOCK_K: tl.constexpr  # tile size along K (no loop: must cover entire K or partial only)
):
    """
    Simple matmul kernel that takes:
      - Two FP8 matrices A and B
      - Writes a float32 result C
    WITHOUT looping over K. Only one chunk of size BLOCK_K is processed.

    This kernel is for demonstration and testing only.
    If K > BLOCK_K, it accumulates only part of the product.
    """
    # 2D block indices along M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Each program instance computes a [BLOCK_M x BLOCK_N] tile in C
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # -----------------------
    # 1) Create an accumulator
    # -----------------------
    c_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -----------------------
    # 2) Load one slice of A and B
    # -----------------------
    # We skip the usual loop over K so we assume K <= BLOCK_K 
    # or we only compute partial coverage for K if K < BLOCK_K.
    k_offsets = tl.arange(0, BLOCK_K)
    
    # Addressing for A: A[row, k]
    a_ptrs = A_ptr + (row_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak)
    # Addressing for B: B[k, col]
    b_ptrs = B_ptr + (k_offsets[:, None] * stride_bk + col_offsets[None, :] * stride_bn)

    # Load from FP8 into float32
    # Here we do trivial boundary checks:
    a_mask = (row_offsets[:, None] < M) & (k_offsets[None, :] < K)
    b_mask = (k_offsets[:, None] < K) & (col_offsets[None, :] < N)

    A_tile_fp8 = tl.load(a_ptrs, mask=a_mask, other=0.0)
    B_tile_fp8 = tl.load(b_ptrs, mask=b_mask, other=0.0)

    print("A_tile_fp8:", A_tile_fp8)
    print("B_tile_fp8:", B_tile_fp8)

    # -----------------------
    # 3) Compute the dot-product
    # -----------------------
    c_tile += tl.dot(A_tile_fp8, B_tile_fp8)
    print("c_tile:", c_tile)

    # -----------------------
    # 4) Write results to C
    # -----------------------
    c_ptrs = C_ptr + (row_offsets[:, None] * stride_cm + col_offsets[None, :] * stride_cn)
    out_of_bounds = (row_offsets[:, None] >= M) | (col_offsets[None, :] >= N)
    tl.store(c_ptrs, c_tile, mask=~out_of_bounds)


def matmul_fp8_no_loop(A_fp8: torch.Tensor, B_fp8: torch.Tensor):
    """
    Minimal test function: 
    - A_fp8: [M, K] in FP8
    - B_fp8: [K, N] in FP8
    Returns C in float32, ignoring any leftover if K < BLOCK_K.
    """

    M, K = A_fp8.shape
    K2, N = B_fp8.shape
    assert K == K2, "Incompatible shapes for matmul!"
    
    # Pick block sizes. We want BLOCK_K >= K for a single slice coverage.
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64  # or something >= K

    # Allocate output
    C = torch.zeros((M, N), device=A_fp8.device, dtype=torch.float32)

    # Launch grid
    grid = (
        ( (M + BLOCK_M - 1) // BLOCK_M ),  # how many blocks in M
        ( (N + BLOCK_N - 1) // BLOCK_N ),  # how many blocks in N
    )

    # Grab strides (row-major).
    # (For FP8, these are still just standard strides in terms of # of elements.)
    stride_am = A_fp8.stride(0)
    stride_ak = A_fp8.stride(1)
    stride_bk = B_fp8.stride(0)
    stride_bn = B_fp8.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # If K > BLOCK_K, the result is only partial. 
    # For full correctness, K must be <= BLOCK_K (or block multiple).
    matmul_fp8_kernel_no_loop[grid](
        A_fp8, B_fp8, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    return C


# Suppose we have small M, N, K for demonstration
M, N, K = 2, 4, 16

# Create random FP8 input data
if True:
    A_fp8 = torch.arange(M, dtype=torch.float32, device='cuda').view(-1, 1).expand(-1, K).to(torch.float8_e4m3fnuz)
    B_fp8 = torch.arange(N, dtype=torch.float32, device='cuda').view(-1, 1).expand(-1, K).to(torch.float8_e4m3fnuz)
else:
    A_fp8 = torch.randn((M, K), device='cuda', dtype=torch.float32).to(torch.float8_e4m3fnuz)
    B_fp8 = torch.randn((K, N), device='cuda', dtype=torch.float32).to(torch.float8_e4m3fnuz)
print("A_fp8:", A_fp8, A_fp8.shape)
print("B_fp8:", B_fp8, B_fp8.shape)
C_out = matmul_fp8_no_loop(A_fp8, B_fp8.T)
print("C:", C_out, C_out.shape)