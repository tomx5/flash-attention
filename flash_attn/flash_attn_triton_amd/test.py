import torch
import pytest
import numpy as np
from flash_attn import (
    flash_attn_func, 
    flash_attn_kvpacked_func, 
    flash_attn_qkvpacked_func, 
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func
)

from .utils import DEBUG, DEBUG_TRITON, DEBUG_TRITON_DETAIL,\
    cast_to_fp8, decast_fp8, input_helper, varlen_input_helper, get_arch, arch_supports_fp8
from .fwd_ref import attention_forward_pytorch_ref_impl
from .fwd_prefill import attention_prefill_forward_triton_impl
from .bwd_prefill import attention_prefill_backward_triton_impl
from .bwd_prefill_split import attention_prefill_backward_triton_split_impl
from .bwd_ref import attention_backward_pytorch_ref_impl

# set print options
# torch.set_printoptions(linewidth=5e5, edgeitems=10, sci_mode=False)
# np.set_printoptions(linewidth=5000, threshold=1e4, suppress=True, precision=4)

# defailt fp16 tolerance is ATOL, RTOL = 1e-5, 1e-3. See table https://pytorch.org/docs/stable/testing.html
ATOL, RTOL = 1e-2, 1e-2 # old standard. maybe to lose. 
# ATOL, RTOL = 1e-3, 1e-3  # catchs fa mismatch issues
# ATOL, RTOL = 1e-4, 1e-3 # to strict. there will be small diffs
# ATOL, RTOL = 1e-5, 1e-3 # # default fp16. there will be small diffs
# ATOL_fp8, RTOL_fp8 = 1e-1, 1e-1 # to strict for larger tensors in fp8
ATOL_fp8, RTOL_fp8 = 2.5e-1, 2.5e-1 # test pass with dropout and causal in fp8
EQUAL_NAN = True

@pytest.mark.parametrize(
    "Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD",
    [
        (1, 1, 1, 1, 1, 1),
        (1, 1, 1, 2, 4, 16),
        (1, 2, 2, 2, 4, 16),
        (1, 4, 1, 2, 4, 16),
        (1, 4, 2, 2, 4, 16),
        (1, 1, 1, 4, 2, 16),
        (1, 1, 1, 4, 4, 16),
        (1, 2, 2, 4, 4, 16),
        (2, 1, 1, 4, 4, 16),
        (2, 2, 2, 4, 4, 16),
        (1, 1, 1, 128, 64, 16),
        (2, 2, 2, 2, 128, 1),
        (2, 3, 3, 2, 128, 16),
        (3, 2, 2, 256, 512, 16),
        (3, 3, 3, 128, 128, 64),
        (2, 4, 4, 1024, 1024, 64),
        (4, 6, 6, 108, 256, 224),
        (4, 8, 8, 2048, 2048, 128),
        (4, 16, 16, 4096, 4096, 64),
        (2, 4, 4, 8192, 8192, 32),
        # fa configs
        (4, 6, 1, 113, 203, 256),
        (4, 6, 1, 128, 217, 256),
        (4, 6, 2, 113, 211, 128),
        (4, 6, 2, 108, 256, 128),
        (4, 6, 1, 256, 512, 64),
        (4, 6, 1, 512, 256, 64),
        (4, 6, 2, 1024, 1024, 32),
        (4, 6, 2, 1023, 1024, 32),
        (4, 6, 6, 1024, 1023, 32),
        (4, 6, 6, 2048, 2048, 32),
    ],
)
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('layout', ["bhsd", "bshd", "thd"])
@pytest.mark.parametrize('use_exp2', [True, False]) # works when use_exp2 is false
@pytest.mark.parametrize('DEBUG_INPUT', [False]) # NOTE: debug input can overflow when the tensors are large. Just use to figure out issues
def test_op_prefill_fwd_impl(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, layout, use_exp2, DEBUG_INPUT):
    dtype = torch.float16
    torch.manual_seed(0)
    alibi_slopes = None
    device = "cuda"

    if layout == "thd":
        q, k, v, metadata = varlen_input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, device=device, DEBUG_INPUT=DEBUG_INPUT)
    else:
        q, k, v, metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, device=device, DEBUG_INPUT=DEBUG_INPUT)
    if DEBUG_INPUT:
        output_triton = torch.zeros_like(q).contiguous()
    else:
        output_triton = torch.empty_like(q)

    if DEBUG:
        if HQ // HK != 1:
            print("MQA/GQA")
        else:
            print("MHA")

    # update metadata
    metadata.use_exp2 = use_exp2
    if causal:
        metadata.need_causal()

    # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
    if dropout_p > 0.0:
        metadata.need_dropout(dropout_p)


    # call Triton's forward implementation directly
    output_triton, softmax_lse_triton, sd_mask_triton = attention_prefill_forward_triton_impl(
                                                q, 
                                                k, 
                                                v, 
                                                output_triton, 
                                                metadata.sm_scale, 
                                                metadata.alibi_slopes, 
                                                metadata.causal, 
                                                metadata.bias, 
                                                metadata.layout, 
                                                metadata.cu_seqlens_q, 
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q, 
                                                metadata.max_seqlens_k,
                                                metadata.dropout_p,
                                                metadata.philox_seed, 
                                                metadata.philox_offset, 
                                                metadata.return_scores, 
                                                metadata.use_exp2)

    output_ref, softmax_lse_ref, sd_mask_ref  = attention_forward_pytorch_ref_impl(
        q.clone(), 
        k.clone(), 
        v.clone(), 
        metadata.sm_scale, 
        causal, 
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        metadata.dropout_p,
        metadata.philox_seed, 
        metadata.philox_offset, 
        use_exp2
    )

    if DEBUG:
        print()
        print("Compare Triton Impl with refernce Pytorch Impl")

    # this can be set to true manually or when using dropout
    if metadata.return_scores:
        if DEBUG:
            print("sd_mask_triton:", sd_mask_triton, sd_mask_triton.shape)
            print("sd_mask_ref:", sd_mask_ref, sd_mask_ref.shape)
        torch.testing.assert_close(sd_mask_triton, sd_mask_ref, atol=ATOL, rtol=RTOL)

    if DEBUG:
        print("softmax_lse_triton:", softmax_lse_triton, softmax_lse_triton.shape)
        print("softmax_lse_ref:", softmax_lse_ref, softmax_lse_ref.shape)
    torch.testing.assert_close(softmax_lse_triton, softmax_lse_ref, atol=ATOL, rtol=RTOL)
    
    if DEBUG:
        print("output_triton:", output_triton, output_triton.shape)
        print("output_ref:", output_ref, output_ref.shape)
    torch.testing.assert_close(output_triton, output_ref, atol=ATOL, rtol=RTOL)

@pytest.mark.parametrize(
    "Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD", [
    (1, 1, 1, 1, 1, 1),
    (1, 1, 1, 4, 4, 4),
    (2, 1, 1, 4, 4, 16),
    (1, 2, 2, 4, 4, 16),
    (1, 4, 1, 2, 4, 16),
    (1, 8, 1, 2, 4, 16),
    (1, 16, 1, 2, 4, 16),
    (1, 32, 1, 2, 4, 16),
    (1, 64, 1, 2, 4, 16),
    (1, 4, 2, 2, 4, 16),
    (2, 2, 2, 4, 4, 16),
    (1, 1, 1, 4, 4, 16),
    (2, 1, 1, 4, 4 , 16),
    (4, 6, 6, 8, 8 , 16),
    (1, 1, 1, 4, 4, 32),
    (1, 1, 1, 16, 16, 16),
    (1, 1, 1, 32, 32, 16),
    (1, 1, 1, 64, 64, 16),
    (1, 1, 1, 64, 64, 16),
    (1, 1, 1, 64, 128, 16),
    (1, 1, 1, 64, 64, 32),
    (1, 1, 1, 64, 128, 32),
    (1, 1, 1, 128, 128, 64),
    (1, 1, 1, 128, 256, 45),
    (1, 1, 1, 113, 203, 192),
    (1, 1, 1, 256, 256, 64),
    (1, 1, 1, 256, 512, 16),
    (1, 1, 1, 512, 512, 64),
    (1, 1, 1, 1024, 1024, 64),
    # fa configs
    (2, 2, 2, 128, 128, 65),
    (2, 2, 2, 128, 128, 224),
    (4, 6, 6, 108, 256, 224),
    (1, 1, 1, 256, 512, 16),
    # old tests that work
    (4, 48, 6, 1024, 1024, 64),
    (4, 48, 12, 1024, 1024, 64),
    (4, 48, 24, 1024, 1024, 64),
    (4, 48, 48, 1024, 1024, 64),
    (4, 48, 48, 1024, 1024, 73),
    (4, 48, 48, 2048, 2048, 64),
    (1, 24, 24, 4096, 4096, 64),
    (1, 16, 16, 1024, 1024, 64),
    (1, 16, 16, 1024, 1024, 128),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('use_exp2', [False]) # FIXME: using exp2 causes issue when used with causal
@pytest.mark.parametrize('layout', ["bhsd", "bshd", "thd"])
@pytest.mark.parametrize('sequence_parallel', [True, False])
@pytest.mark.parametrize('DEBUG_INPUT', [False]) # debug output causes nans on larger tensors
def test_op_prefill_bwd_impl(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, use_exp2, layout, sequence_parallel, DEBUG_INPUT):
    if get_arch() == "gfx90a":
        if layout == "thd" and Z == 4 and HQ == 48 and HK == 48 and N_CTX_Q == 1024 and N_CTX_K == 1024:
            pytest.skip("This config doesnot work on MI200 Devices but works on MI300.")

    dtype = torch.float16
    torch.manual_seed(20) # seed from test_op_bwd

    alibi_slopes = None
    if layout == "thd":
        q, k, v, metadata = varlen_input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, DEBUG_INPUT=DEBUG_INPUT)
    else:
        q, k, v, metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, DEBUG_INPUT=DEBUG_INPUT)
    if DEBUG_INPUT:
        do = torch.ones_like(q).contiguous()
    else:
        do = torch.randn_like(q)

    # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
    if dropout_p > 0.0:
        metadata.need_dropout(dropout_p)

    # =============================================== Reference ==============================================================
    q_ref = q.clone() 
    k_ref = k.clone()
    v_ref = v.clone()    
    output_ref, softmax_lse_ref, sd_mask_ref = attention_forward_pytorch_ref_impl(
        q_ref,
        k_ref, 
        v_ref,
        metadata.sm_scale, 
        causal, 
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        metadata.dropout_p,
        metadata.philox_seed, 
        metadata.philox_offset, 
        use_exp2
    )


    if DEBUG:
        if HQ // HK != 1:
            print("MQA/GQA")
        else:
            print("MHA")

    dq = torch.zeros_like(q, dtype=q.dtype) # NOTE: the kernel does inplace accumlation on dq so dq has to be zeros
    if DEBUG_INPUT:
        dk = torch.zeros_like(k, dtype=k.dtype)
        dv = torch.zeros_like(v, dtype=v.dtype)
    else:
        dk = torch.empty_like(k, dtype=k.dtype)
        dv = torch.empty_like(v, dtype=v.dtype)

    do_ref = do.clone()
    dq_ref, dk_ref, dv_ref, delta_ref = attention_backward_pytorch_ref_impl(
        do_ref,
        q_ref,
        k_ref,
        v_ref,
        output_ref,
        softmax_lse_ref,
        metadata.sm_scale,
        causal,
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        metadata.dropout_p,
        metadata.philox_seed, 
        metadata.philox_offset, 
        use_exp2
    )

    # =============================================== Triton ==============================================================
    o = output_ref.clone().contiguous()
    softmax_lse = softmax_lse_ref.clone().contiguous()
    dq_triton, dk_triton, dv_triton, delta_triton, _, _ = attention_prefill_backward_triton_impl(
        do,
        q,
        k,
        v,
        o,
        softmax_lse,
        dq,
        dk,
        dv,
        metadata.sm_scale,
        alibi_slopes,
        causal,
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        metadata.dropout_p,
        metadata.philox_seed, 
        metadata.philox_offset, 
        use_exp2,
        sequence_parallel=sequence_parallel
    )

    # =============================================== Check ==============================================================
    if DEBUG:
        print()
    if DEBUG:
        print("delta_triton:", delta_triton, delta_triton.shape)
        print("delta_ref:", delta_ref, delta_ref.shape)
    torch.testing.assert_close(delta_triton, delta_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN)

    if DEBUG:
        print("dv_triton:", dv_triton, dv_triton.shape)
        print("dv_ref:", dv_ref, dv_ref.shape)
    torch.testing.assert_close(dv_triton, dv_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN)

    if DEBUG:
        print("dk_triton:", dk_triton, dk_triton.shape)
        print("dk_ref:", dk_ref, dk_ref.shape)
    torch.testing.assert_close(dk_triton, dk_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN)

    if DEBUG:
        print("dq_triton:", dq_triton, dq_triton.shape)
        print("dq_ref:", dq_ref, dq_ref.shape)
    torch.testing.assert_close(dq_triton, dq_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN)

@pytest.mark.parametrize(
    "Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD",
    [
        # head size
        # (1, 1, 1, 129, 129, 1),  # two blocks with 2nd block small enough to debug # fails
        # seqlen q == k
        # (1, 1, 1, 1, 1, 1),
        # (1, 1, 1, 2, 2, 2),  # small enough to debug
        (1, 1, 1, 4, 4, 16),
        # (1, 2, 2, 4, 4, 16),
        # (2, 1, 1, 4, 4, 16),
        # (2, 2, 2, 4, 4, 16),
        # (1, 1, 1, 128, 128, 32),  # only one block
        # (3, 3, 3, 128, 128, 64),
        # (1, 1, 1, 127, 127, 32),  # only one block but with masking
        # (1, 1, 1, 129, 129, 32),  # two blocks with 2nd block small enough to debug
        # (1, 1, 1, 350, 350, 1),  # two blocks with 2nd block small enough to debug
        # (1, 1, 1, 350, 350, 68),  # generic masking on q, k and head
        # (4, 1, 1, 512, 512, 128), # batch > 1
        # (4, 2, 2, 512, 512, 128),
        # (4, 2, 2, 512, 512, 68),
        # (4, 2, 2, 500, 500, 68),
        # (2, 4, 4, 1024, 1024, 64),
        # (4, 8, 8, 2048, 2048, 128),
        # (4, 16, 16, 4096, 4096, 64),
        # (2, 4, 4, 8192, 8192, 32),
        # # seqlen q > k
        # (1, 1, 1, 4, 2, 16),
        # (1, 1, 1, 64, 32, 8),
        # (1, 1, 1, 128, 64, 16),
        # (1, 1, 1, 192, 128, 32),
        # (1, 2, 2, 1024, 512, 68),
        # (1, 4, 4, 729, 516, 68),
        # (2, 4, 4, 2753, 1528, 68),  # a comprehensive seqlen_q > seqlen_k
        # # seqlen q < k
        # (1, 1, 1, 2, 4, 16),
        # (1, 2, 2, 2, 4, 16),
        # (1, 4, 1, 2, 4, 16),
        # (1, 4, 2, 2, 4, 16),
        # (2, 2, 2, 2, 128, 1),
        # (2, 3, 3, 2, 128, 16),
        # (1, 1, 1, 32, 64, 8),
        # (1, 1, 1, 128, 192, 32),
        # (4, 6, 6, 108, 256, 224),
        # (3, 2, 2, 256, 512, 16),
        # (2, 2, 2, 512, 1024, 68),
        # (1, 1, 1, 200, 413, 1),
        # (1, 1, 1, 782, 1546, 1),
        # # gqa/mqa
        # (4, 8, 2, 500, 500, 68), 
        # (4, 8, 2, 512, 512, 68),
        # (4, 8, 2, 512, 512, 128),
        # (4, 8, 2, 512, 1024, 68),
        # pytest.param(4, 8, 2, 1024, 512, 68, marks=pytest.mark.flaky(reruns=3, reason="Flaky config")),
        # (16, 16, 4, 1528, 2753, 68),
        # # fa configs
        # (4, 6, 1, 113, 203, 256),
        # (4, 6, 1, 128, 217, 256),
        # (4, 6, 2, 113, 211, 128),
        # (4, 6, 2, 108, 256, 128),
        # (4, 6, 1, 256, 512, 64),
        # (4, 6, 1, 512, 256, 64),
        # (4, 6, 2, 1024, 1024, 32),
        # (4, 6, 2, 1023, 1024, 32),
        # (4, 6, 6, 1024, 1023, 32),
        # (4, 6, 6, 2048, 2048, 32),
    ],
)
@pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('layout', ['bshd'])
@pytest.mark.parametrize('packing', ['none'])
@pytest.mark.parametrize('DEBUG_INPUT', [False])
@pytest.mark.skipif(not arch_supports_fp8(), reason="fp8 not supported on this device")
def test_fp8(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, layout, packing, DEBUG_INPUT):
    device = "cuda"
    window_size = (-1, -1)
    softcap = 0.0
    alibi_slopes = None
    deterministic = False
    fp8_dtype = torch.float8_e4m3fnuz
    ref_dtype = torch.float16
    is_varlen = True if layout == "thd" else False

    # skip QKV packing tests for uneven sequence lengths
    if packing == 'qkv':
        if N_CTX_Q != N_CTX_K:
            pytest.skip("QKV packing requires N_CTX_Q == N_CTX_K")
    # set seed
    torch.manual_seed(20)

    # choose input helper based on layout
    q, k, v, metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, ref_dtype, layout, device=device, DEBUG_INPUT=DEBUG_INPUT)
    
    # create gradient tensor
    if DEBUG_INPUT:
        do = torch.ones_like(q)
    else:
        do = torch.randn_like(q)

    # pack input tensors
    if packing == 'kv':
        if is_varlen:
            kv = torch.stack([k, v], dim=1)
        else:
            kv = torch.stack([k, v], dim=2)
    elif packing == 'qkv':
        if is_varlen:
            qkv = torch.stack([k, v], dim=1)
        else:
            qkv = torch.stack([k, v], dim=2)

    # ----------------------------------------------------------------
    # --- FP8 ---
    # ----------------------------------------------------------------
    # Cast to fp8 based on layout and packing
    if packing == 'none':
        if not is_varlen:
            q_fp8, descale_q = cast_to_fp8(q, fp8_dtype, layout)
            k_fp8, descale_k = cast_to_fp8(k, fp8_dtype, layout)
            v_fp8, descale_v = cast_to_fp8(v, fp8_dtype, layout)
        else:
            q_fp8, descale_q = cast_to_fp8(q, fp8_dtype, layout, cu_seqlens=metadata.cu_seqlens_q)
            k_fp8, descale_k = cast_to_fp8(k, fp8_dtype, layout, cu_seqlens=metadata.cu_seqlens_k)
            v_fp8, descale_v = cast_to_fp8(v, fp8_dtype, layout, cu_seqlens=metadata.cu_seqlens_k)
        # descale factors that are returned with kernel outputs
        descale_o = torch.zeros_like(descale_q)
        descale_dq = torch.zeros_like(descale_q)
        descale_dk = torch.zeros_like(descale_k)
        descale_dv = torch.zeros_like(descale_v)
    elif packing == 'kv':
        if not is_varlen:
            q_fp8, descale_q = cast_to_fp8(q, fp8_dtype, layout)
            kv_fp8, descale_kv = cast_to_fp8(kv, fp8_dtype, layout, packing=packing)
            # use same descale for k and v in kv-packing
            descale_k = descale_kv
            descale_v = descale_kv
        else:
            q_fp8, descale_q = cast_to_fp8(q, fp8_dtype, layout, cu_seqlens=metadata.cu_seqlens_q)
            kv_fp8, descale_kv = cast_to_fp8(kv, fp8_dtype, layout, cu_seqlens=metadata.cu_seqlens_k, packing=packing)
            # use same descale for k and v in kv-packing
            descale_k = descale_kv
            descale_v = descale_kv
    elif packing == 'qkv':
        qkv_fp8, descale_qkv = cast_to_fp8(qkv, fp8_dtype, layout, packing=packing)
        # use same descale for q, k, and v in qkv-packing
        descale_q = descale_qkv
        descale_k = descale_qkv
        descale_v = descale_qkv
    
    # Cast gradient
    if not is_varlen:
        do_fp8, descale_do = cast_to_fp8(do, fp8_dtype, layout)
    else:
        do_fp8, descale_do = cast_to_fp8(do, fp8_dtype, layout, cu_seqlens=metadata.cu_seqlens_q)
    
    # fp8 forward pass
    if packing == 'none':
        if not is_varlen:
            out_fp8, lse_fp8, S_dmask_fp8 = flash_attn_func(
                q_fp8,
                k_fp8,
                v_fp8,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do,
                descale_o=descale_o,
                descale_dq=descale_dq,
                descale_dk=descale_dk,
                descale_dv=descale_dv,
            )
        else:
            out_fp8, lse_fp8, S_dmask_fp8 = flash_attn_varlen_func(
                q_fp8,
                k_fp8,
                v_fp8,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do,
                descale_o=descale_o,
                descale_dq=descale_dq,
                descale_dk=descale_dk,
                descale_dv=descale_dv,
            )
    elif packing == 'kv':
        if not is_varlen:
            out_fp8, lse_fp8, S_dmask_fp8 = flash_attn_kvpacked_func(
                q_fp8,
                kv_fp8,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do
            )
        else:
            out_fp8, lse_fp8, S_dmask_fp8 = flash_attn_varlen_kvpacked_func(
                q_fp8,
                kv_fp8,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do
            )
    elif packing == 'qkv':
        if not is_varlen:
            out_fp8, lse_fp8, S_dmask_fp8 = flash_attn_qkvpacked_func(
                qkv_fp8,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do
            )
        else:
            out_fp8, lse_fp8, S_dmask_fp8 = flash_attn_varlen_qkvpacked_func(
                qkv_fp8,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do
            )
    
    # print("out_fp8:", out_fp8)
    # print("do_fp8 before grad:", do_fp8)


    # fp8 backward pass
    if packing == 'none':
        dq_fp8, dk_fp8, dv_fp8 = torch.autograd.grad(out_fp8, (q_fp8, k_fp8, v_fp8), do_fp8)
    elif packing == 'kv':
        dq_fp8, dkv_fp8 = torch.autograd.grad(out_fp8, (q_fp8, kv_fp8), do_fp8)
    elif packing == 'qkv':
        dqkv_fp8 = torch.autograd.grad(out_fp8, qkv_fp8, do_fp8)[0]

    # ----------------------------------------------------------------
    # --- Reference ---
    # ----------------------------------------------------------------
    # Prepare reference inputs by decasting FP8 tensors
    if packing == 'none':
        if not is_varlen:
            q_ref = decast_fp8(q_fp8, descale_q, ref_dtype, layout)
            k_ref = decast_fp8(k_fp8, descale_k, ref_dtype, layout)
            v_ref = decast_fp8(v_fp8, descale_v, ref_dtype, layout)
        else:
            q_ref = decast_fp8(q_fp8, descale_q, ref_dtype, layout, cu_seqlens=metadata.cu_seqlens_q)
            k_ref = decast_fp8(k_fp8, descale_k, ref_dtype, layout, cu_seqlens=metadata.cu_seqlens_k)
            v_ref = decast_fp8(v_fp8, descale_v, ref_dtype, layout, cu_seqlens=metadata.cu_seqlens_k)
    elif packing == 'kv':
        if not is_varlen:
            q_ref = decast_fp8(q_fp8, descale_q, ref_dtype, layout)
            kv_ref = decast_fp8(kv_fp8, descale_kv, ref_dtype, layout, packing=packing)
        else:
            q_ref = decast_fp8(q_fp8, descale_q, ref_dtype, layout, cu_seqlens=metadata.cu_seqlens_q)
            kv_ref = decast_fp8(kv_fp8, descale_kv, ref_dtype, layout, cu_seqlens=metadata.cu_seqlens_k, packing=packing)
    elif packing == 'qkv':
        qkv_ref = decast_fp8(qkv_fp8, descale_qkv, ref_dtype, layout, packing=packing)
    
    # Decast gradient tensor
    if not is_varlen:
        do_ref = decast_fp8(do_fp8, descale_do, ref_dtype, layout)
    else:
        do_ref = decast_fp8(do_fp8, descale_do, ref_dtype, layout, cu_seqlens=metadata.cu_seqlens_q)

    # reference forward pass
    if packing == 'none':
        if not is_varlen:
            out_ref, lse_ref, S_dmask_ref = flash_attn_func(
                q_ref,
                k_ref,
                v_ref,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
        else:
            out_ref, lse_ref, S_dmask_ref = flash_attn_varlen_func(
                q_ref,
                k_ref,
                v_ref,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
    elif packing == 'kv':
        if not is_varlen:
            out_ref, lse_ref, S_dmask_ref = flash_attn_kvpacked_func(
                q_ref,
                kv_ref,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
        else:
            out_ref, lse_ref, S_dmask_ref = flash_attn_varlen_kvpacked_func(
                q_ref,
                kv_ref,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
    elif packing == 'qkv':
        if not is_varlen:
            out_ref, lse_ref, S_dmask_ref = flash_attn_qkvpacked_func(
                qkv_ref,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
        else:
            out_ref, lse_ref, S_dmask_ref = flash_attn_varlen_qkvpacked_func(
                qkv_ref,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )

    # reference backward pass
    if packing == 'none':
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), do_ref)
    elif packing == 'kv':
        dq_ref, dkv_ref = torch.autograd.grad(out_ref, (q_ref, kv_ref), do_ref)
    elif packing == 'qkv':
        dqkv_ref = torch.autograd.grad(out_ref, qkv_ref, do_ref)[0]


    # ----------------------------------------------------------------
    # --- Compare ---
    # ----------------------------------------------------------------
    # compare forward
    if DEBUG:
        print()
        print(f"Compare fp8 against ref with dtype {ref_dtype}")

    # convert the fp8 output to ref type
    if not is_varlen:
        out_fp8_decast = decast_fp8(out_fp8, descale_o, ref_dtype, layout)
    else:
        out_fp8_decast = decast_fp8(out_fp8, descale_o, ref_dtype, layout, cu_seqlens=metadata.cu_seqlens_q)

    if DEBUG:
        print("out_ref:", out_ref, out_ref.shape)
        print("out_fp8:", out_fp8, out_fp8.shape)
    torch.testing.assert_close(out_ref, out_fp8_decast, atol=ATOL_fp8, rtol=RTOL_fp8)

    if DEBUG:
        print("lse_ref:", lse_ref, lse_ref.shape)
        print("lse_fp8:", lse_fp8, lse_fp8.shape)
    torch.testing.assert_close(lse_ref, lse_fp8, atol=ATOL_fp8, rtol=RTOL_fp8)

    if dropout_p > 0.0:
        if DEBUG:
            print("S_dmask_ref:", S_dmask_ref, S_dmask_ref.shape)
            print("S_dmask_fp8:", S_dmask_fp8, S_dmask_fp8.shape)
        torch.testing.assert_close(S_dmask_ref, S_dmask_fp8, atol=ATOL_fp8, rtol=RTOL_fp8)
 
    # compare backward gradients
    if packing == 'none':
        # convert the fp8 grads to ref type
        if not is_varlen:
            dv_fp8_decast = decast_fp8(dv_fp8, descale_dv, ref_dtype, layout)
            dk_fp8_decast = decast_fp8(dk_fp8, descale_dk, ref_dtype, layout)
            dq_fp8_decast = decast_fp8(dq_fp8, descale_dq, ref_dtype, layout)
        else:
            dv_fp8_decast = decast_fp8(dv_fp8, descale_dv, ref_dtype, layout, cu_seqlens=metadata.cu_seqlens_k)
            dk_fp8_decast = decast_fp8(dk_fp8, descale_dk, ref_dtype, layout, cu_seqlens=metadata.cu_seqlens_k)
            dq_fp8_decast = decast_fp8(dq_fp8, descale_dq, ref_dtype, layout, cu_seqlens=metadata.cu_seqlens_q)
        
        if DEBUG:
            print("dv_ref:", dv_ref, dv_ref.shape)
            print("dv_fp8_decast:", dv_fp8_decast, dv_fp8_decast.shape)
        
        torch.testing.assert_close(dv_ref, dv_fp8_decast, atol=ATOL_fp8, rtol=RTOL_fp8, equal_nan=EQUAL_NAN)

        if DEBUG:
            print("dk_ref:", dk_ref, dk_ref.shape)
            print("dk_fp8_decast:", dk_fp8_decast, dk_fp8_decast.shape)
        torch.testing.assert_close(dk_ref, dk_fp8_decast, atol=ATOL_fp8, rtol=RTOL_fp8, equal_nan=EQUAL_NAN)

        if DEBUG:
            print("dq_ref:", dq_ref, dq_ref.shape)
            print("dq_fp8_decast:", dq_fp8_decast, dq_fp8_decast.shape)
        torch.testing.assert_close(dq_ref, dq_fp8_decast, atol=ATOL_fp8, rtol=RTOL_fp8, equal_nan=EQUAL_NAN)
    # elif packing == 'kv':
    #     if DEBUG:
    #         print("dq_ref:", dq_ref, dq_ref.shape)
    #         print("dq_fp8_decast:", dq_fp8_decast, dq_fp8_decast.shape)
    #     torch.testing.assert_close(dq_ref, dq_fp8_decast, atol=ATOL_fp8, rtol=RTOL_fp8, equal_nan=EQUAL_NAN)
    #     if DEBUG:
    #         print("dkv_ref:", dkv_ref, dkv_ref.shape)
    #         print("dkv_fp8_decast:", dkv_fp8_decast, dkv_fp8_decast.shape)
    #     torch.testing.assert_close(dkv_ref, dkv_fp8_decast, atol=ATOL_fp8, rtol=RTOL_fp8, equal_nan=EQUAL_NAN)
    # elif packing == 'qkv':
    #     if DEBUG:
    #         print("dqkv_ref:", dqkv_ref, dqkv_ref.shape)
    #         print("dqkv_fp8_decast:", dqkv_fp8_decast, dqkv_fp8_decast.shape)
    #     torch.testing.assert_close(dqkv_ref, dqkv_fp8_decast, atol=ATOL_fp8, rtol=RTOL_fp8, equal_nan=EQUAL_NAN)


@pytest.mark.parametrize(
    "Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD", [
    (1, 1, 1, 1, 1, 1),
    (1, 1, 1, 4, 4, 4),
    (2, 1, 1, 4, 4, 16),
    (1, 2, 2, 4, 4, 16),
    (1, 4, 1, 2, 4, 16),
    (1, 8, 1, 2, 4, 16),
    (1, 16, 1, 2, 4, 16),
    (1, 32, 1, 2, 4, 16),
    (1, 64, 1, 2, 4, 16),
    (1, 4, 2, 2, 4, 16),
    (2, 2, 2, 4, 4, 16),
    (1, 1, 1, 4, 4, 16),
    (2, 1, 1, 4, 4 , 16),
    (4, 6, 6, 8, 8 , 16),
    (1, 1, 1, 4, 4, 32),
    (1, 1, 1, 16, 16, 16),
    (1, 1, 1, 32, 32, 16),
    (1, 1, 1, 64, 64, 16),
    (1, 1, 1, 64, 64, 16),
    (1, 1, 1, 64, 128, 16),
    (1, 1, 1, 64, 64, 32),
    (1, 1, 1, 64, 128, 32),
    (1, 1, 1, 128, 128, 64),
    (1, 1, 1, 128, 256, 45),
    (1, 1, 1, 113, 203, 192),
    (1, 1, 1, 256, 256, 64),
    (1, 1, 1, 256, 512, 16),
    (1, 1, 1, 512, 512, 64),
    (1, 1, 1, 1024, 1024, 64),
    # fa configs
    (2, 2, 2, 128, 128, 65),
    (2, 2, 2, 128, 128, 224),
    (4, 6, 6, 108, 256, 224),
    (1, 1, 1, 256, 512, 16),
    # old tests that work
    (4, 48, 6, 1024, 1024, 64),
    (4, 48, 12, 2048, 1024, 64),
    (4, 48, 24, 1024, 1024, 64),
    (4, 48, 48, 1024, 1024, 64),
    (4, 48, 48, 1024, 1024, 73),
    (4, 48, 48, 2048, 2048, 64),
    (1, 24, 24, 4096, 4096, 64),
    (1, 16, 16, 1024, 1024, 64),
    (1, 16, 16, 1024, 1024, 128),
    # testcase new
    # seqlen q == k
    (1, 1, 1, 2, 2, 2),  # small enough to debug
    (1, 1, 1, 128, 128, 32),  # only one block
    (1, 1, 1, 127, 127, 32),  # only one block but with masking
    (1, 1, 1, 129, 129, 1),  # two blocks with 2nd block small enough to debug
    (1, 1, 1, 350, 350, 1),  # two blocks with 2nd block small enough to debug
    (1, 1, 1, 350, 350, 68),  # generic masking on q, k and head
    (4, 1, 1, 512, 512, 128),  # batch > 1
    (4, 8, 2, 512, 512, 128),  # GQA
    (4, 8, 2, 512, 512, 68),   # non-power-of-2 head_dim
    (4, 8, 2, 500, 500, 68),  # comprehensive case for seqlen q == k
    # seqlen q > k
    (1, 1, 1, 64, 32, 8),  # seqlen_q > seqlen_k
    (1, 1, 1, 192, 128, 32),  # seqlen_q > seqlen_k
    (4, 8, 2, 1024, 512, 68),  # seqlen_q < seqlen_k
    (1, 1, 1, 729, 516, 68),  # seqlen_q > seqlen_k
    (16, 16, 4, 2753, 1528, 68),  # a comprehensive seqlen_q > seqlen_k
    # seqlen q < k
    (1, 1, 1, 32, 64, 8),  # seqlen_q > seqlen_k
    (1, 1, 1, 128, 192, 32),  # seqlen_q < seqlen_k
    (4, 8, 2, 512, 1024, 68),  # seqlen_q < seqlen_k
    (1, 1, 1, 200, 413, 1),  # seqlen_q < seqlen_k
    (1, 1, 1, 782, 1546, 1),  # seqlen_q < seqlen_k
    (16, 16, 4, 1528, 2753, 68),  # a comprehensive seqlen_q < seqlen_k

# varlen
# dropout
# direct comparison among tutorial, Michael's implementation bwd and this one

])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('dropout_p', [0.0, 0.2])
@pytest.mark.parametrize('use_exp2', [True, False]) # FIXME: using exp2 causes issue when used with causal
# @pytest.mark.parametrize('layout', ["bhsd"])
@pytest.mark.parametrize('layout', ["bhsd", "thd"])
@pytest.mark.parametrize('sequence_parallel', [True])
@pytest.mark.parametrize('DEBUG_INPUT', [False]) # debug output causes nans on larger tensors
def test_op_prefill_bwd_split_impl(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dropout_p, use_exp2, layout, sequence_parallel, DEBUG_INPUT):
    dtype = torch.float16
    torch.manual_seed(20) # seed from test_op_bwd

    alibi_slopes = None
    if layout == "thd":
        q, k, v, metadata = varlen_input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, DEBUG_INPUT=DEBUG_INPUT)
    else:
        q, k, v, metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout, DEBUG_INPUT=DEBUG_INPUT)
    if DEBUG_INPUT:
        do = torch.ones_like(q).contiguous()
    else:
        do = torch.randn_like(q)

    # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
    if dropout_p > 0.0:
        metadata.need_dropout(dropout_p)

    # print("from the very beginning")
    # print("q:", q.shape)
    # print("k:", k.shape)
    # print("v:", v.shape)

    # =============================================== Reference ==============================================================
    q_ref = q.clone()
    k_ref = k.clone()
    v_ref = v.clone()
    output_ref, softmax_lse_ref, sd_mask_ref = attention_forward_pytorch_ref_impl(
        q_ref,
        k_ref,
        v_ref,
        metadata.sm_scale,
        causal,
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        metadata.dropout_p,
        metadata.philox_seed,
        metadata.philox_offset,
        use_exp2
    )


    if DEBUG:
        if HQ // HK != 1:
            print("MQA/GQA")
        else:
            print("MHA")

    dq = torch.zeros_like(q, dtype=q.dtype) # NOTE: the kernel does inplace accumlation on dq so dq has to be zeros
    if DEBUG_INPUT:
        dk = torch.zeros_like(k, dtype=k.dtype)
        dv = torch.zeros_like(v, dtype=v.dtype)
    else:
        dk = torch.empty_like(k, dtype=k.dtype)
        dv = torch.empty_like(v, dtype=v.dtype)

    do_ref = do.clone()
    dq_ref, dk_ref, dv_ref, delta_ref = attention_backward_pytorch_ref_impl(
        do_ref,
        q_ref,
        k_ref,
        v_ref,
        output_ref,
        softmax_lse_ref,
        metadata.sm_scale,
        causal,
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        metadata.dropout_p,
        metadata.philox_seed,
        metadata.philox_offset,
        use_exp2
    )
    # =============================================== Triton ==============================================================
    o = output_ref.clone().contiguous()
    softmax_lse = softmax_lse_ref.clone().contiguous()
    dq_triton, dk_triton, dv_triton, delta_triton, _, _ = attention_prefill_backward_triton_split_impl(
        do,
        q,
        k,
        v,
        o,
        softmax_lse,
        dq,
        dk,
        dv,
        metadata.sm_scale,
        alibi_slopes,
        causal,
        layout,
        metadata.cu_seqlens_q,
        metadata.cu_seqlens_k,
        metadata.max_seqlens_q,
        metadata.max_seqlens_k,
        metadata.dropout_p,
        metadata.philox_seed,
        metadata.philox_offset,
        use_exp2,
        DEBUG_TRITON=DEBUG_TRITON,
        DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL,
    )

    # =============================================== Check ==============================================================
    if DEBUG:
        print()
    if DEBUG:
        print("delta_triton:", delta_triton, delta_triton.shape)
        print("delta_ref:", delta_ref, delta_ref.shape)
    if DEBUG:
        dim_names = ["batch", "qhead", "seqlen_kv", "head_dim"]
        mismatch = torch.where(torch.isclose(dv_triton, dv_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN) != 1)
        num_error_dv = mismatch[0].numel()
        if num_error_dv > 0:
            print(f"\nnumber of mismatch in dv: {num_error_dv}")
            for m, name in zip(mismatch, dim_names):
                print(f"{name}: {m.unique().cpu()}")
        dim_names = ["batch", "kvhead", "seqlen_kv", "head_dim"]
        mismatch = torch.where(torch.isclose(dk_triton, dk_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN) != 1)
        num_error_dk = mismatch[0].numel()
        if num_error_dk > 0:
            print(f"\nnumber of mismatch in dk: {num_error_dk}")
            for m, name in zip(mismatch, dim_names):
                print(f"{name}: {m.unique().cpu()}")
        dim_names = ["batch", "qhead", "seqlen_q", "head_dim"]
        mismatch = torch.where(torch.isclose(dq_triton, dq_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN) != 1)
        num_error_dq = mismatch[0].numel()
        if num_error_dq > 0:
            print(f"\nnumber of mismatch in dq: {num_error_dq}")
            for m, name in zip(mismatch, dim_names):
                print(f"{name}: {m.unique().cpu()}")

    torch.testing.assert_close(delta_triton, delta_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN)
    torch.testing.assert_close(dv_triton, dv_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN)
    torch.testing.assert_close(dk_triton, dk_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN)
    torch.testing.assert_close(dq_triton, dq_ref, atol=ATOL, rtol=RTOL, equal_nan=EQUAL_NAN)
