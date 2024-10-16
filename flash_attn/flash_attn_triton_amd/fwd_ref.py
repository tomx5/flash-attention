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

def attention_varlen_forward_pytorch_ref_impl(
    q,
    k,
    v,
    sm_scale,
    causal,
    layout,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    use_exp2  
):
    # Ensure the layout is 'thd'
    if layout != 'thd':
        raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")

    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    # Prepare lists to collect outputs
    o_list = []
    softmax_lse_list = []
    exp_scores_list = []
    softmax_list = []
    attention_shifted_scaled_scores_list = []
    attention_scores_list = []

    for i in range(batch_size):
        # Get the start and end indices for the current sequence
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()

        # Extract q_i, k_i, v_i
        q_i = q[start_q:end_q, :, :]  # [L_q_i, num_heads, head_dim]
        k_i = k[start_k:end_k, :, :]  # [L_k_i, num_heads, head_dim]
        v_i = v[start_k:end_k, :, :]  # [L_k_i, num_heads, head_dim]

        L_q_i = end_q - start_q
        L_k_i = end_k - start_k

        # Transpose to [num_heads, L_q_i, head_dim]
        q_i = q_i.permute(1, 0, 2).to(torch.float32)  # [num_heads, L_q_i, head_dim]
        k_i = k_i.permute(1, 0, 2).to(torch.float32)  # [num_heads, L_k_i, head_dim]
        v_i = v_i.permute(1, 0, 2).to(torch.float32)  # [num_heads, L_k_i, head_dim]

        # Compute attention scores: [num_heads, L_q_i, L_k_i]
        attention_scores = torch.bmm(q_i, k_i.transpose(1, 2))

        # Scale scores
        attention_scaled_scores = sm_scale * attention_scores

        # Apply causal mask if necessary
        if causal:
            causal_mask = torch.triu(
                torch.ones(L_q_i, L_k_i, dtype=torch.bool, device=attention_scaled_scores.device),
                diagonal=1
            )
            attention_scaled_scores = attention_scaled_scores.masked_fill(
                causal_mask.unsqueeze(0), float('-inf')
            )

        # Compute max_scores for numerical stability
        max_scores = torch.max(attention_scaled_scores, dim=-1, keepdim=True)[0]

        # Shift scores
        attention_shifted_scaled_scores = attention_scaled_scores - max_scores

        # Exponentiate
        if use_exp2:
            RCP_LN = 1 / math.log(2)
            exp_scores = torch.exp2(RCP_LN * attention_shifted_scaled_scores)
        else:
            exp_scores = torch.exp(attention_shifted_scaled_scores)

        # Sum of exponentials
        sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)

        # Compute softmax probabilities
        softmax = exp_scores / sum_exp_scores

        # Compute log-sum-exp
        if use_exp2:
            LN2 = math.log(2)
            RCP_LN = 1 / math.log(2)
            max_scores_base2 = max_scores * RCP_LN
            softmax_lse_base2 = max_scores_base2 + torch.log2(sum_exp_scores)
            softmax_lse = softmax_lse_base2 * LN2  # [num_heads, L_q_i, 1]
            softmax_lse = softmax_lse.squeeze(-1)  # [num_heads, L_q_i]
        else:
            softmax_lse = max_scores + torch.log(sum_exp_scores)  # [num_heads, L_q_i, 1]
            softmax_lse = softmax_lse.squeeze(-1)  # [num_heads, L_q_i]

        # Compute output
        o_i = torch.bmm(softmax, v_i)  # [num_heads, L_q_i, head_dim]

        # Convert back to 'thd' layout and float16
        o_i = o_i.permute(1, 0, 2).to(torch.float16)  # [L_q_i, num_heads, head_dim]

        # Collect outputs
        o_list.append(o_i)
        softmax_lse_list.append(softmax_lse.unsqueeze(0))
        exp_scores_list.append(exp_scores.unsqueeze(0))
        softmax_list.append(softmax.unsqueeze(0))
        attention_shifted_scaled_scores_list.append(attention_shifted_scaled_scores.unsqueeze(0))
        attention_scores_list.append(attention_scores.unsqueeze(0))

    # Concatenate outputs
    o = torch.cat(o_list, dim=0) 
    softmax_lse = torch.cat(softmax_lse_list, dim=0)
    exp_scores = torch.cat(exp_scores_list, dim=0)
    softmax = torch.cat(softmax_list, dim=0)
    attention_shifted_scaled_scores = torch.cat(attention_shifted_scaled_scores_list, dim=0)
    attention_scores = torch.cat(attention_scores_list, dim=0)

    return (
        o,
        softmax_lse,
        exp_scores,
        softmax,
        attention_shifted_scaled_scores,
        attention_scores
    )




def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)