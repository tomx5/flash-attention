import math
import torch

DEBUG = False

def attention_forward_pytorch_ref_impl(q, k, v, sm_scale, causal, layout, use_exp2):
    """compute reference output and softmax_lse using PyTorch's built-in function"""

    # ensure the layout is 'bhsd'
    if layout == "bshd":
        if DEBUG:
            print("Changing layout to bhsd!")
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")

    # get seqlens
    N_CTX_Q = q.shape[2]
    N_CTX_K = k.shape[2]

    # compute attention scores
    attention_scores = torch.matmul(q.to(torch.float32), k.transpose(-2, -1).to(torch.float32))


    # scale score
    attention_scaled_scores = sm_scale * attention_scores

    # ===========================  Softmax ========================================
    # compute max for numerical stability
    max_scores = torch.max(attention_scaled_scores, dim=-1, keepdim=True)[0]

    # shift scores to by subtracing max
    attention_shifted_scaled_scores = attention_scaled_scores - max_scores

    # subtract max and exponentiate
    if use_exp2:
        RCP_LN = 1/ math.log(2)
        exp2_scores = torch.exp2(RCP_LN * attention_shifted_scaled_scores)
    else:
        exp_scores = torch.exp(attention_shifted_scaled_scores)

    # sum of exponentials
    if use_exp2:
        sum_exp2_scores = torch.sum(exp2_scores, dim=-1, keepdim=True)
    else:
        sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)

    # softmax probabilities
    if use_exp2:
        softmax_exp2 = exp2_scores / sum_exp2_scores
    else:
        softmax = exp_scores / sum_exp_scores
    
    # compute log-sum-exp and squeeze final dim which will be 1
    if use_exp2:
        LN2 = math.log(2)
        RCP_LN = 1/ math.log(2)
        # compute log-sum-exp in base 2 units
        max_scores_base2 = max_scores * RCP_LN
        softmax_exp2_lse_base2 = max_scores_base2 + torch.log2(sum_exp2_scores)
        # Convert back to natural units
        softmax_exp2_lse = softmax_exp2_lse_base2 * LN2
        softmax_exp2_lse.squeeze_(-1)
    else:
        softmax_lse = max_scores + torch.log(sum_exp_scores)
        softmax_lse.squeeze_(-1)
    
    # compute output
    if use_exp2:
        o_exp2 = torch.matmul(softmax_exp2, v.to(torch.float32)).to(torch.float16)
    else:
        o = torch.matmul(softmax, v.to(torch.float32)).to(torch.float16)


    # go back to original layout
    if layout == "bshd":
        if DEBUG:
            print("Changing back to bshd!")
        if use_exp2:
            o_exp2 = o_exp2.transpose(1, 2)
        else:
            o = o.transpose(1, 2)
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")

    if use_exp2:
        return o_exp2, softmax_exp2_lse, exp2_scores, softmax_exp2, attention_shifted_scaled_scores, attention_scores
    else:
        return o, softmax_lse, exp_scores, softmax, attention_shifted_scaled_scores, attention_scores


def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)