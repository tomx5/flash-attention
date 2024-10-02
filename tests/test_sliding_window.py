import torch

IS_LOCAL = True

head_dim = 1
N_CTX_Q = 8
N_CTX_KV = 8

qk = torch.randn(N_CTX_Q, N_CTX_KV) + 10

col_idx = torch.arange(N_CTX_KV)
row_idx = torch.arange(N_CTX_Q)

print("QK:\n", qk)

window = (-2, 2)
print("WINDOW: ", window)

if IS_LOCAL:
    sliding_window_start = window[0]
    sliding_window_end = window[1]
    col_offset = N_CTX_Q - N_CTX_KV
    mask =  (sliding_window_start <= (col_idx[None, :] + col_offset - row_idx[:, None])) & \
            ((col_idx[None, :] + col_offset - row_idx[:, None]) <= sliding_window_end)
    print("MASK:\n", mask)
    qk = qk * mask

print("QK after MASK:\n", qk)