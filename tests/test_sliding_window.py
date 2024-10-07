import torch
import math

IS_LOCAL = False
IS_CASUAL = True

BLOCK_M = 16
BLOCK_N = 16

head_dim = 1
N_CTX_Q = 16
N_CTX_KV = 32

window = (float('inf'), float('inf'))

if IS_LOCAL:
    window = (2, 0)
if IS_CASUAL:
    window = (max(float('inf'), window[0]), 0) # if local enabled will choose the mask that is smallest (the intersection of both masks)

WINDOW_SIZE_LEFT = window[0]
WINDOW_SIZE_RIGHT = window[1]

qk = torch.randint(1, 9, (N_CTX_Q, N_CTX_KV))
print("QK:\n", qk)
print("WINDOW: ", window)

for start_m in range(math.ceil(N_CTX_Q / BLOCK_M)):
    for start_n in range(0, N_CTX_KV, BLOCK_N):
        row_idx = start_m * BLOCK_M + torch.arange(0, BLOCK_M) if N_CTX_Q >= (start_m+1) * BLOCK_M else start_m * BLOCK_M + torch.arange(0, N_CTX_Q % BLOCK_M)
        col_idx = start_n + torch.arange(0, BLOCK_N) if N_CTX_KV >= (start_n + BLOCK_N) else start_n + torch.arange(0, N_CTX_KV % BLOCK_N)

        mask = torch.full(((BLOCK_M if N_CTX_Q >= (start_m+1) * BLOCK_M else N_CTX_Q % BLOCK_M), (BLOCK_N if N_CTX_KV >= (start_n + BLOCK_N) else N_CTX_KV % BLOCK_N)), 1)

        col_offset = N_CTX_Q - N_CTX_KV

        if IS_LOCAL:
            local_mask = (-WINDOW_SIZE_LEFT <= (col_idx[None, :] + col_offset - row_idx[:, None])) & \
                         ((col_idx[None, :] + col_offset - row_idx[:, None]) <= WINDOW_SIZE_RIGHT)
            mask = mask * local_mask # apply local mask to mask
            print((col_idx[None, :] + col_offset - row_idx[:, None]))
            print("MASK:\n", mask)

            qk[row_idx[:, None], col_idx] = qk[row_idx[:, None], col_idx] * mask
            print("ROW: ", row_idx, "COL: ", col_idx, "COL_OFF: ", col_offset)
            print("MINI QK:\n", qk[row_idx[:, None], col_idx])
    
    print("QK after MASK:\n", qk)