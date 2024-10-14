import torch
import math

DEBUG=False

def attention_backward_pytorch_ref_impl(do, q, k, v, o, softmax_lse, sm_scale, causal, layout, use_exp2, bwd_preprocessing_use_o):
    # ensure the layout is 'bhsd'
    if layout == "bshd":
        print("Changing layout to bhsd!")
        do = do.transpose(1, 2).contiguous()
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        o = o.transpose(1, 2).contiguous()
        # softmax_lse = softmax_lse.transpose(1, 2).contiguous()
        # TODO: does L/M need to be transposed. possible to use strides
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")
    
    # recompute attention_scores
    attention_scores = torch.matmul(q, k.transpose(-2, -1))
    if DEBUG:
        print("attention_scores:", attention_scores)

    # scale scores
    attention_scaled_scores = sm_scale * attention_scores
    if DEBUG:
        print("attention_scaled_scores:", attention_scaled_scores)

    # compute probabilities using softmax_lse
    if use_exp2:
        RCP_LN = 1 / math.log(2)
        attention_scaled_scores_base2 = attention_scaled_scores * RCP_LN
        softmax_lse_base2 = softmax_lse * RCP_LN
        p = torch.exp2(attention_scaled_scores_base2 - softmax_lse_base2.unsqueeze(-1))
    else:
        p = torch.exp(attention_scaled_scores - softmax_lse.unsqueeze(-1))

    if DEBUG:
        print("p:", p)
    # compute gradient wrt v
    dv = torch.matmul(p.transpose(-2, -1), do.to(torch.float32))

    # compute dp
    dp = torch.matmul(do, v.transpose(-2, -1))

    # calculate ds
    if bwd_preprocessing_use_o:
        delta = torch.sum(o * do, axis=-1).unsqueeze(-1).to(torch.float32) # what oai kernel uses
    else:
        delta = torch.sum(p * dp, axis=-1).unsqueeze(-1) # what the math says you should use
    ds = (p * (dp - delta)) * sm_scale

    # compute gradient wrt k
    dk = torch.matmul(ds.transpose(-2, -1), q.to(torch.float32))

    # compute gradient wrt q
    dq = torch.matmul(ds, k.to(torch.float32))

    # cast back to original dtype
    dq = dq.to(q.dtype)
    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)

    # go back to original layout
    if layout == "bshd":
        print("Changing back to bshd!")
        dq = dq.transpose(1, 2)
        dk = dk.transpose(1, 2)
        dv = dv.transpose(1, 2)
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")

    return dq, dk, dv, delta.squeeze(-1)
