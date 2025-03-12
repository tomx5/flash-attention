"""
Lean Attention
===============

This is a Triton implementation of the Lean Attention algorithm from https://arxiv.org/abs/2405.10480

Status:

TO be added features:
- Batch size > 1 and different context length per request in a batch
- Causal = True for Prefill
- Add MQA and GQA support
"""

import pytest
import torch
import sys

import triton
import triton.language as tl

def get_num_splits_and_buffer_sizes(batch_size, max_seqlen_q, max_seqlen_k, num_heads, num_heads_k, head_size, BLOCK_M, BLOCK_N, num_SMs):

        ##### Lean Atteion: Calculate Splits and Tile Sizes #####
        ## based on onnxruntime/contrib_ops/cuda/bert/lean_attention
        num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
        num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

        #TODO: Support Grouped-Query Attention
        max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

        #print(f"block_m: {BLOCK_M}, block_n: {BLOCK_N} ")
        #print(f"num_m_block: {num_m_blocks}, num_n_block: {num_n_blocks} ")
        #print(f"max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
        #print(f"num_heads: {num_heads}, num_heads_k: {num_heads_k} ")

        # Decode or Not Causal
        # TODO: Add Causal case, fixed to False for now
        is_causal=False
        if max_seqlen_q == 1:
            is_causal = False
        
        tiles_per_head = 0
        if is_causal:
            # Prefill - Causal
            for i in range (0, num_m_blocks):
                tiles_per_head += (((i + 1) * block_m) + block_n - 1) // block_n
            
        else:
            # Decode or Not Causal
            tiles_per_head = num_m_blocks * num_n_blocks
        
        tiles_per_head = num_m_blocks * num_n_blocks
        total_tiles = tiles_per_head * batch_size * num_heads_k # Total tiles across all heads

        #StreamK Lean has as many threadblocks as SMs
        # This should be a function of tile size and number of scratchpad space
        # LeanAttention assign 2 tiles per CTA and 2 CTAs per SM
        lean_griddimz = num_SMs  #CTA launch grid
        #if (total_tiles <= 2 * 2 * num_SMs):
        #    lean_griddimz = min((total_tiles + 1) / 2, (32 * total_tiles + num_n_blocks - 1) / num_n_blocks)
        #else:
        #    lean_griddimz = min(2 * num_SMs, 32 * num_heads_k * batch_size * num_m_blocks)

        # Max number lean tiles per task block (CTA)
        max_tiles_per_tb = (total_tiles + lean_griddimz - 1) // lean_griddimz

        #Find max number of splits
        num_splits = 0
        even_split = False
        if (total_tiles % lean_griddimz == 0):
            even_split = True
            num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) // (max_tiles_per_tb))
        else:
            even_split = False
            num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1))

        #high_load_tbs is the remainder of total_tile / num_cta
        high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)

        return total_tiles, high_load_tbs, max_tiles_per_tb, tiles_per_head, lean_griddimz, num_splits, even_split

@triton.jit
def _attn_fwd_persistent(
    Q, K, V, qk_scale, Mp, Lp, Op,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_oph, stride_opm, stride_opn,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    locks,
    # leanAttention params
    num_wgs: tl.constexpr,
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
    even_split: tl.constexpr,
):
    current_pid = tl.program_id(0)

    if current_pid < high_load_wgs:
        iter = max_tiles_per_wg * current_pid
        cta_end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (current_pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter + (max_tiles_per_wg - 1)


    #qk_scale = sm_scale * 1.44269504

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)

        #Loop context length
    while iter < cta_end_tile_gid:
        #Index of current output tile
        tile_idx = iter // tiles_per_head

        Q_base = Q + tile_idx * stride_qh
        #To generate an otuput tile, a loop over [tile_iter, tile_iter_end) lean tiles is needed
        #[tile_iter, tile_iter_end) are in the form of global tile id
        tile_iter = tile_idx * tiles_per_head
        tile_iter_end = tile_iter + tiles_per_head
        #Local lean tile ID within a loop of an output tile
        local_iter = iter - tile_iter
        local_iter_end = tl.minimum(tile_iter_end,cta_end_tile_gid) - tile_iter

        if iter == tile_iter:
            host_block = True
        else:
            host_block = False
        # finishing_block: the output tile is finished within this block
        if cta_end_tile_gid >= tile_iter_end:
            finishing_block = True
        else:
            finishing_block = False

        kv_block_shape = (local_iter_end - local_iter) * BLOCK_N
        kv_offset = iter * BLOCK_N * stride_kn
        K_base = K + kv_offset
        V_base = V + kv_offset


        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        acc, l_i, m_i = _attn_lean_tile(acc, l_i, m_i, Q_base, stride_qm, stride_qk, kv_block_shape, K_base, V_base, stride_kn, stride_kk, stride_vn, stride_vk, qk_scale,  #
                                        BLOCK_M, BLOCK_N, HEAD_DIM, tile_idx, local_iter, local_iter_end, #
                                        )

        # initialize pointer to m and l
        m_cta = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_cta = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc_cta = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        
        # lean output tile epilogue
        if not host_block:
            # Update pointers of partial results M[cta], L[cta], O[cta]
            mp_ptrs = Mp + current_pid * BLOCK_M + tl.arange(0, BLOCK_M)
            lp_ptrs = Lp + current_pid * BLOCK_M + tl.arange(0, BLOCK_M)
            op_ptrs = Op + current_pid * stride_oph + offs_m[:,None]*stride_opm + offs_k[None,:]*stride_opn

            tl.store(mp_ptrs, m_i, cache_modifier=".wt")
            tl.store(lp_ptrs, l_i, cache_modifier=".wt")
            tl.store(op_ptrs, acc, cache_modifier=".wt")
            tl.debug_barrier()
            tl.atomic_xchg(locks + current_pid, 1)
        if host_block and finishing_block:
            # A host block that is also a finishing block completes all the LeanTile iterations for its output tile
            # in a single CTA and so can directly store its results from LeanTile() in global memory without any reduction
            o_h_offs = Out + tile_idx * stride_oh
            o_ptrs = o_h_offs + offs_m[:,None]*stride_om + offs_k[None,:]*stride_on
            acc = acc / l_i[:, None]
            tl.store(o_ptrs, acc.to(Out.type.element_ty))

        if host_block and not finishing_block:
            # Only host_block update the output tile
            o_h_offs = Out + tile_idx * stride_oh
            o_ptrs = o_h_offs + offs_m[:,None]*stride_om + offs_k[None,:]*stride_on
            #last_cta = tile_iter_end // tiles_per_head
            last_cta = current_pid + 1
            temp_end_gid = cta_end_tile_gid
            split = 1
            while (split < num_splits) and (temp_end_gid < tile_iter_end):
                if last_cta < high_load_wgs:
                    if (tile_iter_end-temp_end_gid) < max_tiles_per_wg:
                        temp_end_gid += (tile_iter_end-temp_end_gid)
                    else:
                        temp_end_gid += max_tiles_per_wg
                else:
                    if (tile_iter_end-temp_end_gid) < (max_tiles_per_wg-1):
                        temp_end_gid += (tile_iter_end-temp_end_gid)
                    else:
                        temp_end_gid += (max_tiles_per_wg-1)

                last_cta += 1
                split += 1

            for cta in range((current_pid+1), last_cta):
                while tl.atomic_cas(locks + cta, 1, 1) != 1:
                    pass
                offs_mplp = cta*BLOCK_M + tl.arange(0, BLOCK_M)
                mp_ptrs = Mp + offs_mplp
                lp_ptrs = Lp + offs_mplp
                op_h_offs = Op + cta * stride_oph
                op_ptrs = op_h_offs + offs_m[:,None]*stride_opm + offs_k[None,:]*stride_opn
                m_cta = tl.load(mp_ptrs)
                l_cta = tl.load(lp_ptrs)
                acc_cta = tl.load(op_ptrs)

                #m_i is the host_block's m, m_cta is other non-host block's m
                m_new = tl.maximum(m_cta, m_i)
                alpha = tl.math.exp2(m_cta - m_new)
                alpha1 = tl.math.exp2(m_i - m_new)
                l_new = alpha * l_cta + alpha1 * l_i
                acc = acc_cta * alpha[:,None] + acc * alpha1[:,None]
                #update m, l
                m_i = m_new
                l_i = l_new
            #host_block (finishing OR non-finishing) write final result to memory
            # A host block that is also a finishing block completes all the LeanTile iterations for its output tile
            # in a single CTA and so can directly store its results from LeanTile() in global memory without any reduction
            acc = acc / l_i[:, None]
            tl.store(o_ptrs, acc.to(Out.type.element_ty))

        # update iter
        iter = iter + (local_iter_end - local_iter)


@triton.jit
def _attn_fwd(
    Q, K, V, qk_scale, Mp, Lp, Op,Mph, Lph, Oph,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_oph, stride_opm, stride_opn,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    #locks,
    # leanAttention params
    num_wgs: tl.constexpr,
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
    even_split: tl.constexpr,
):
    current_pid = tl.program_id(0)

    if current_pid < high_load_wgs:
        iter = max_tiles_per_wg * current_pid
        cta_end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (current_pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter + (max_tiles_per_wg - 1)


    #qk_scale = sm_scale * 1.44269504
    
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)

    #Loop context length
    while iter < cta_end_tile_gid:
        #Index of current output tile
        tile_idx = iter // tiles_per_head
        
        #To generate an otuput tile, a loop over [tile_iter, tile_iter_end) lean tiles is needed
        #[tile_iter, tile_iter_end) are in the form of global tile id
        tile_iter = tile_idx * tiles_per_head
        tile_iter_end = tile_iter + tiles_per_head
        #Local lean tile ID within a loop of an output tile
        local_iter = iter - tile_iter
        local_iter_end = tl.minimum(tile_iter_end,cta_end_tile_gid) - tile_iter

        if iter == tile_iter:
            host_block = True
        else:
            host_block = False
        # finishing_block: the output tile is finished within this block
        if cta_end_tile_gid >= tile_iter_end:
            finishing_block = True
        else:
            finishing_block = False

        kv_block_shape = (local_iter_end - local_iter) * BLOCK_N
        kv_offset = iter * BLOCK_N * stride_kn
        K_base = K + kv_offset
        V_base = V + kv_offset
        
        Q_base = Q + tile_idx * stride_qh
                
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        
        acc, l_i, m_i = _attn_lean_tile(acc, l_i, m_i, Q_base, stride_qm, stride_qk, kv_block_shape, K_base, V_base, stride_kn, stride_kk, stride_vn, stride_vk, qk_scale,  #
                                        BLOCK_M, BLOCK_N, HEAD_DIM, tile_idx, local_iter, local_iter_end, #
                                        )

        # initialize pointer to m and l
        #m_cta = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        #l_cta = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        #acc_cta = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # lean output tile epilogue
        if not host_block:
            # Update pointers of partial results M[cta], L[cta], O[cta]
            #offs_mplp = 2 * current_pid * BLOCK_M + tl.arange(0, BLOCK_M)
            mp_ptrs = Mp + current_pid * BLOCK_M + tl.arange(0, BLOCK_M) 
            lp_ptrs = Lp + current_pid * BLOCK_M + tl.arange(0, BLOCK_M)
            op_ptrs = Op + current_pid * stride_oph + offs_m[:,None]*stride_opm + offs_k[None,:]*stride_opn  

            tl.store(mp_ptrs, m_i, cache_modifier=".wt")
            tl.store(lp_ptrs, l_i, cache_modifier=".wt")
            tl.store(op_ptrs, acc, cache_modifier=".wt")
            #tl.debug_barrier()
            #tl.atomic_xchg(locks + current_pid, 1)
        #else: # host_block
        if host_block and not finishing_block:
            #if not finishing_block: # another CTA is processing the end of the output tile and store partial results
            #offs_mplp = (2 * current_pid + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
            mph_ptrs = Mph + current_pid * BLOCK_M + tl.arange(0, BLOCK_M)
            lph_ptrs = Lph + current_pid * BLOCK_M + tl.arange(0, BLOCK_M)
            oph_ptrs = Oph + current_pid * stride_oph + offs_m[:,None]*stride_opm + offs_k[None,:]*stride_opn 
            
            tl.store(mph_ptrs, m_i, cache_modifier=".wt")
            tl.store(lph_ptrs, l_i, cache_modifier=".wt")
            tl.store(oph_ptrs, acc, cache_modifier=".wt")
        #    else:
        if host_block and finishing_block:
            # A host block that is also a finishing block completes all the LeanTile iterations for its output tile
            # in a single CTA and so can directly store its results from LeanTile() in global memory without any reduction
            o_h_offs = Out + tile_idx * stride_oh
            o_ptrs = o_h_offs + offs_m[:,None]*stride_om + offs_k[None,:]*stride_on
            acc = acc / l_i[:, None]
            tl.store(o_ptrs, acc.to(Out.type.element_ty))

        # update iter
        iter = iter + (local_iter_end - local_iter)


@triton.jit
def _attn_reduce(
    Q, K, V, Mp, Lp, Op, Mph, Lph, Oph,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_oph, stride_opm, stride_opn,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    #locks,
    # leanAttention params
    num_wgs: tl.constexpr,
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
    even_split: tl.constexpr,
):
    current_pid = tl.program_id(0)

    if current_pid < high_load_wgs:
        cta_start_tile_gid = max_tiles_per_wg * current_pid
        cta_end_tile_gid = cta_start_tile_gid + max_tiles_per_wg
    else:
        cta_start_tile_gid = (max_tiles_per_wg - 1) * (current_pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = cta_start_tile_gid + (max_tiles_per_wg - 1)

    start_tile_hid = cta_start_tile_gid // tiles_per_head


    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)


    iter = cta_start_tile_gid
    #last_tile_idx = start_tile_hid

    #Loop context length
    while iter < cta_end_tile_gid:
        #Index of current output tile
        tile_idx = iter // tiles_per_head
        #To generate an otuput tile, a loop over [tile_iter, tile_iter_end) lean tiles is needed
        #[tile_iter, tile_iter_end) are in the form of global tile id
        tile_iter = tile_idx * tiles_per_head
        tile_iter_end = tile_iter + tiles_per_head

        #Local lean tile ID within a loop of an output tile
        local_iter = iter - tile_iter
        local_iter_end = tl.minimum(tile_iter_end,cta_end_tile_gid) - tile_iter
        # host_block: this WG contains a lean tile that is a start of the loop for a new output tile
        if iter == tile_iter:
            host_block = True
        else:
            host_block = False
        # finishing_block: the output tile is finished within this block
        if cta_end_tile_gid >= tile_iter_end:
            finishing_block = True
        else:
            finishing_block = False

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        
        m_cta = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_cta = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc_cta = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        
        # Only host, non-finishing CTA performs final accumulation
        if host_block and not finishing_block:
            o_h_offs = Out + tile_idx * stride_oh
            o_ptrs = o_h_offs + offs_m[:,None]*stride_om + offs_k[None,:]*stride_on

            #Load Host-nonFinishing partial result first
            offs_mplp = current_pid*BLOCK_M + tl.arange(0, BLOCK_M)
            mp_ptrs = Mph + offs_mplp
            lp_ptrs = Lph + offs_mplp
            op_h_offs = Oph + current_pid * stride_oph
            op_ptrs = op_h_offs + offs_m[:,None]*stride_opm + offs_k[None,:]*stride_opn
            #while tl.load(locks + (current_pid*2+1), cache_modifier=".cv", volatile=True) != 1:
            #    pass
            m_i = tl.load(mp_ptrs)
            l_i = tl.load(lp_ptrs)
            acc = tl.load(op_ptrs)
            
            last_cta = current_pid + 1
            temp_end_gid = cta_end_tile_gid
            split = 1
            while (split < num_splits) and (temp_end_gid < tile_iter_end):
                if last_cta < high_load_wgs:
                    if (tile_iter_end-temp_end_gid) < max_tiles_per_wg:
                        temp_end_gid += (tile_iter_end-temp_end_gid)
                    else:
                        temp_end_gid += max_tiles_per_wg
                else:
                    if (tile_iter_end-temp_end_gid) < (max_tiles_per_wg-1):
                        temp_end_gid += (tile_iter_end-temp_end_gid)
                    else:
                        temp_end_gid += (max_tiles_per_wg-1)

                last_cta += 1
                split += 1
            #Next, load nonHost partial restult
            for cta in range((current_pid+1), last_cta):
                #while tl.atomic_cas(locks + cta, 1, 1) != 1:
                #while tl.load(locks + cta*2, cache_modifier=".cv", volatile=True) != 1:
                #    pass
                #Partial results are stored in [nonHost, Host-nonFinishing] layout
                offs_mplp = cta*BLOCK_M + tl.arange(0, BLOCK_M)
                mp_ptrs = Mp + offs_mplp
                lp_ptrs = Lp + offs_mplp
                op_h_offs = Op + cta * stride_oph
                op_ptrs = op_h_offs + offs_m[:,None]*stride_opm + offs_k[None,:]*stride_opn
                m_cta = tl.load(mp_ptrs)
                l_cta = tl.load(lp_ptrs)
                acc_cta = tl.load(op_ptrs)

                #m_i is the host CTA's m, m_cta is other nonHost CTA's m
                m_new = tl.maximum(m_cta, m_i)
                alpha = tl.math.exp2(m_cta - m_new)
                alpha1 = tl.math.exp2(m_i - m_new)
                l_new = alpha * l_cta + alpha1 * l_i
                acc = acc_cta * alpha[:,None] + acc * alpha1[:,None]
                #update m, l
                m_i = m_new
                l_i = l_new
            #host non-finishing CTA write final result to memory
            acc = acc / l_i[:, None]
            tl.store(o_ptrs, acc.to(Out.type.element_ty))

        # update iter
        iter = iter + (local_iter_end - local_iter)


@triton.jit
def _attn_lean_tile(acc, l_i, m_i, Q_base, stride_qm, stride_qk, kv_block_shape, K_base, V_base, stride_kn, stride_kk, stride_vn, stride_vk, qk_scale,  #
                                        BLOCK_M, BLOCK_N, HEAD_DIM, tile_idx, local_iter, local_iter_end): #
    
    Q_block_ptr = tl.make_block_ptr(
                base=Q_base,
                shape=(BLOCK_M, HEAD_DIM),
                strides=(stride_qm, stride_qk),
                offsets=(0, 0),
                block_shape=(BLOCK_M, HEAD_DIM),
                order=(1, 0),
                )

    q = tl.load(Q_block_ptr)
    #q = tl.load(tl.multiple_of(Q_block_ptr, (1,16)), cache_modifier=".cg")

    K_block_ptr = tl.make_block_ptr(
                base=K_base,
                shape=(HEAD_DIM, kv_block_shape),
                strides=(stride_kk, stride_kn),
                offsets=(0, 0),
                block_shape=(HEAD_DIM, BLOCK_N),
                order=(0, 1) #K parent tensor shape [Z, H, CTX, HEAD_DIM]
                )
    V_block_ptr = tl.make_block_ptr(
                base=V_base,
                shape=(kv_block_shape, HEAD_DIM),
                strides=(stride_vn, stride_vk),
                offsets=(0, 0),
                block_shape=(BLOCK_N, HEAD_DIM),
                order=(1, 0),
                )
   
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for iter in range(local_iter,local_iter_end):
        # -- compute qk ----
        k = tl.load(K_block_ptr,  cache_modifier=".cg")
        #k = tl.load(tl.multiple_of(K_block_ptr, (16,1)), cache_modifier=".cg")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k)
        qk = (qk * qk_scale)
        #m_ij = tl.maximum(m_i, tl.max(qk, 1)*qk_scale)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk) #p.shape = [BLOCK_M, BLOCK_N]
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] #Scale each row of acc by the corresponding elements in alpha
        v = tl.load(V_block_ptr, cache_modifier=".cg") #v.shape = [BLOCK_N, HEAD_DIM]
        #v = tl.load(tl.multiple_of(V_block_ptr, (1,16)), cache_modifier=".cg")
        acc += tl.dot(p.to(v.dtype), v) #acc.shape = [BLOCK_M, HEAD_DIM]
        # -- update l_i
        l_ij = tl.sum(p, 1) #rowsum(p)
        l_i = l_i * alpha + l_ij
        # update m_i
        m_i = m_ij.to(m_i.dtype)
        # update k/v pointer
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i

class _attention(torch.autograd.Function):
    
    _debug = False

    @staticmethod
    def set_debug(debug: bool):
        _attention._debug = debug

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, Mp, Lp, Op,  Mph, Lph, Oph, total_sm):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
        assert  HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        BLOCK_M=64 
        BLOCK_N=64
        #total_sm =152 
        BATCH=k.shape[0]
        N_CTX_Q=q.shape[2]
        N_CTX_K=k.shape[2]
        H=q.shape[1]
        
        persistent_kernel = False 
        
        qk_scale = sm_scale * 1.44269504
        #if _attention._debug:
            #print(f"attention(): BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}\n")

        kernel_timing = {
            "attn_fwd": {
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": torch.cuda.Event(enable_timing=True),
                "ms": 0,
                "experiments": 0,
            },
            "reduce": {
                "start_event": torch.cuda.Event(enable_timing=True),
                "end_event": torch.cuda.Event(enable_timing=True),
                "ms": 0,
                "experiments": 0,
            },
        }


        total_lean_tiles, high_load_wgs, max_tiles_per_wg, tiles_per_head, total_programs, num_splits, even_split = \
            get_num_splits_and_buffer_sizes(BATCH,N_CTX_Q,N_CTX_K,H,H,HEAD_DIM_Q,BLOCK_M, BLOCK_N, total_sm)
        if _attention._debug:
            print(f"total_programs={total_programs}, total_lean_tiles={total_lean_tiles}, high_load_wgs={high_load_wgs}, \
                max_tiles_per_wg={max_tiles_per_wg}, tiles_per_head={tiles_per_head}, even_split={even_split}\n")
        
        # This is the grid for FlashAttention
        #grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        
        # LeanAttention grid
        grid = (total_sm, 1, 1)

        o = torch.empty_like(q, dtype=v.dtype)
        
        #IMPORTANT: Each CTA has at most 1 split of tiles between 2 heads
        #Allocate memory partial result m, l, o. 
        #Mp = torch.empty((total_programs * 2, N_CTX_Q), device=q.device, dtype=torch.float32)
        #Lp = torch.empty((total_programs * 2, N_CTX_Q), device=q.device, dtype=torch.float32)
        #Op = torch.empty((total_programs * 2, BLOCK_M, HEAD_DIM_Q), device=q.device, dtype=torch.float32)
       
        # Allocate separate memory region to store partial result for host CTAs and non-host CTAs
        #Mp = torch.empty((total_programs, N_CTX_Q), device=q.device, dtype=torch.float32)
        #Lp = torch.empty((total_programs, N_CTX_Q), device=q.device, dtype=torch.float32)
        #Op = torch.empty((total_programs, BLOCK_M, HEAD_DIM_Q), device=q.device, dtype=torch.float32)

        #Mph = torch.empty((total_programs, N_CTX_Q), device=q.device, dtype=torch.float32)
        #Lph = torch.empty((total_programs, N_CTX_Q), device=q.device, dtype=torch.float32)
        #Oph = torch.empty((total_programs, BLOCK_M, HEAD_DIM_Q), device=q.device, dtype=torch.float32)

        if persistent_kernel:
            locks = torch.zeros((total_sm,), device = q.device, dtype = torch.int32)
            kernel_timing["attn_fwd"]["start_event"].record()
            
            compiled_kernel=_attn_fwd_persistent[grid](
                q, k, v, qk_scale, Mp, Lp, Op, 
                o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                Op.stride(0), Op.stride(1), Op.stride(2),
                HEAD_DIM=HEAD_DIM_K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                locks=locks,

                # leanAttention params
                num_wgs = total_programs,
                high_load_wgs = high_load_wgs,
                max_tiles_per_wg = max_tiles_per_wg,
                tiles_per_head = tiles_per_head,
                num_splits = num_splits,
                even_split = even_split,

                waves_per_eu = 1,
                num_warps = 4,

            )
            kernel_timing["attn_fwd"]["end_event"].record()
            torch.cuda.synchronize()
            for k in ["attn_fwd" ]:
                ms = kernel_timing[k]["start_event"].elapsed_time(
                    kernel_timing[k]["end_event"]
                )
                kernel_timing[k]["ms"] += ms
            total_ms = kernel_timing["attn_fwd"]["ms"]
        else:
            kernel_timing["attn_fwd"]["start_event"].record()
            compiled_kernel=_attn_fwd[grid](
                q, k, v, qk_scale, Mp, Lp, Op,  Mph, Lph, Oph,
                o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                Op.stride(0), Op.stride(1), Op.stride(2),
                HEAD_DIM=HEAD_DIM_K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                #locks=locks,

                # leanAttention params
                num_wgs = total_programs,
                high_load_wgs = high_load_wgs,
                max_tiles_per_wg = max_tiles_per_wg,
                tiles_per_head = tiles_per_head,
                num_splits = num_splits,
                even_split = even_split,

                waves_per_eu = 2,
                num_warps = 4,

            )
            kernel_timing["attn_fwd"]["end_event"].record()
            torch.cuda.synchronize()

            #print(f"fwd kernel {compiled_kernel.n_regs} registers used, {compiled_kernel.n_spills} spills")
            #print("IR",compiled_kernel.asm['ttir'])
            #print("TTGIR", compiled_kernel.asm['ttgir'])
            #print("AMDGCN", compiled_kernel.asm['amdgcn'])

            kernel_timing["reduce"]["start_event"].record()         
            reduce=_attn_reduce[grid](
                q, k, v, Mp, Lp, Op, Mph, Lph, Oph,
                o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                Op.stride(0), Op.stride(1), Op.stride(2),
                HEAD_DIM=HEAD_DIM_K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                #locks=locks,

                # leanAttention params
                num_wgs = total_programs,
                high_load_wgs = high_load_wgs,
                max_tiles_per_wg = max_tiles_per_wg,
                tiles_per_head = tiles_per_head,
                num_splits = num_splits,
                even_split = even_split,

                waves_per_eu = 1,
                num_warps = 4,

            )
            kernel_timing["reduce"]["end_event"].record()
            torch.cuda.synchronize()
            for k in ["attn_fwd","reduce" ]:
               ms = kernel_timing[k]["start_event"].elapsed_time(
                   kernel_timing[k]["end_event"]
               )
               kernel_timing[k]["ms"] += ms
            total_ms = kernel_timing["attn_fwd"]["ms"] + kernel_timing["reduce"]["ms"]
            #print("IR",compiled_kernel.asm['ttir'])
            #print("TTGIR", fwd.asm['ttgir'])
            #print(f"reduce kernel {reduce.n_regs} registers used, {reduce.n_spills} spills")
            
        return o, total_ms 


lean_attention = _attention.apply

@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX, D_HEAD, dtype',
                         [(*shape, dtype)
                          #for shape in [(4, 48, 1024, 128),
                          #              (4, 48, 2048, 128),
                          #              (4, 48, 4096, 128)]
                          #for dtype in ['fp16', 'bf16', 'fp8']])
                          for shape in [
                 #               (1, 128, 131072, 64)
                (1, 64, 64, 32768, 64),
                (1, 64, 64, 65536, 64),
                (1, 64, 64, 131072, 64),
                (1, 64, 64, 262144, 64),
                (1, 64, 64, 524288, 64),
                (1, 96, 64, 32768, 64),
                (1, 96, 64, 65536, 64),
                (1, 96, 64, 131072, 64),
                (1, 96, 64, 262144, 64),
                #(1, 96, 64, 524288, 64),
                (1, 128, 64, 32768, 64),
                (1, 128, 64, 65536, 64),
                (1, 128, 64, 131072, 64),
                (1, 128, 64, 262144, 64),

                            ]
                          #for shape in [(1, 32, 1024, 64)]
                          for dtype in ['bf16']])
def test_op_fwd(Z, H, N_CTX_Q, N_CTX, D_HEAD, dtype):
    torch.manual_seed(20)
    init_dtype = torch.float16
    
    q = (
        torch.empty((Z, H, N_CTX_Q, D_HEAD), dtype=init_dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        #.requires_grad_()
        )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        #.requires_grad_()
        )
    v = (
        #torch.empty((Z, H, D_HEAD, N_CTX), dtype=init_dtype, device="cuda")
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda")
        .normal_(mean=0., std=0.5)
        #.requires_grad_()
        )
    sm_scale = 0.5

    total_sm = 304
    
    # Allocate separate memory region to store partial result for host CTAs and non-host CTAs
    Mp = torch.empty((total_sm, N_CTX_Q), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_sm, N_CTX_Q), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_sm, N_CTX_Q, D_HEAD), device=q.device, dtype=torch.float32)

    Mph = torch.empty((total_sm, N_CTX_Q), device=q.device, dtype=torch.float32)
    Lph = torch.empty((total_sm, N_CTX_Q), device=q.device, dtype=torch.float32)
    Oph = torch.empty((total_sm, N_CTX_Q, D_HEAD), device=q.device, dtype=torch.float32)


    # reference implementation
    #M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    ref_out = torch.matmul(p, v)
    # triton implementation
    # q,k casting for partial fp8
    #q = q.to(name_to_torch_types[dtype])
    #k = k.to(name_to_torch_types[dtype])
    #dout = torch.randn_like(q, dtype=torch.float16)
    tri_out, ms = lean_attention(q, k, v, sm_scale, Mp, Lp, Op, Mph, Lph, Oph, total_sm) 
    # compare
    atol = 1.4e-1 if dtype == 'fp8' else 1e-2
    rtol = 1e-2 if dtype == 'fp8' else 3e-3
    torch.testing.assert_close(ref_out, tri_out, atol=atol, rtol=rtol)

configs = []
for inf in [True]:
    configs.append(triton.testing.Benchmark(
        x_names=['BATCH', 'H', 'N_CTX_Q', 'N_CTX', 'D_HEAD'],
        x_vals=[
                #(1, 32, 64, 1024, 64),
                (1, 64, 64, 32768, 64),
                (1, 64, 64, 65536, 64),
                (1, 64, 64, 131072, 64),
                (1, 64, 64, 262144, 64),
                (1, 64, 64, 524288, 64),
                (1, 96, 64, 32768, 64),
                (1, 96, 64, 65536, 64),
                (1, 96, 64, 131072, 64),
                (1, 96, 64, 262144, 64),
                #(1, 96, 64, 524288, 64),
                (1, 128, 64, 32768, 64),
                (1, 128, 64, 65536, 64),
                (1, 128, 64, 131072, 64),
                (1, 128, 64, 262144, 64),
                #(1, 128, 64, 524288, 64),


               ],
        line_arg='provider',
        line_vals=['triton'],
        line_names=['Triton'],
        #styles=[('red', '-'), ('blue', '-')],
        ylabel='ms',
        plot_name=f'lean-attention-',
        args={
            'inf': inf,
             })
    )

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX_Q, N_CTX, D_HEAD, inf, provider, device="cuda"):

    init_dtype = torch.float16
    
    q = torch.randn((BATCH, H, N_CTX_Q, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    sm_scale = 1.3
  
    total_sm = 608 
    warmup = 25
    rep = 100
    # Allocate separate memory region to store partial result for host CTAs and non-host CTAs
    Mp = torch.empty((total_sm, N_CTX_Q), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_sm, N_CTX_Q), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_sm, N_CTX_Q, D_HEAD), device=q.device, dtype=torch.float32)

    Mph = torch.empty((total_sm, N_CTX_Q), device=q.device, dtype=torch.float32)
    Lph = torch.empty((total_sm, N_CTX_Q), device=q.device, dtype=torch.float32)
    Oph = torch.empty((total_sm, N_CTX_Q, D_HEAD), device=q.device, dtype=torch.float32)

    #print(f"Q shape={q.shape}")
    #print(f"Q stride0={q.stride(0)}, stride1={q.stride(1)}, stride2={q.stride(2)}, stride3={q.stride(3)}")
    #print(f"K shape={k.shape}")
    #print(f"K stride0={k.stride(0)}, stride1={k.stride(1)}, stride2={k.stride(2)}, stride3={k.stride(3)}")
    #print(f"V shape={v.shape}")
    #print(f"V stride0={v.stride(0)}, stride1={v.stride(1)}, stride2={v.stride(2)}, stride3={v.stride(3)}")
    #print(f"N_CTX_Q={N_CTX_Q}, N_CTX={N_CTX}, H={H}")

    #tri_out = attention(q, k, v, sm_scale)
    fn = lambda: lean_attention(q, k, v, sm_scale, Mp, Lp, Op, Mph, Lph, Oph, total_sm)
    ms = triton.testing.do_bench(fn,warmup=warmup, rep=rep)

    tri_out,kernel_ms = lean_attention(q, k, v, sm_scale, Mp, Lp, Op, Mph, Lph, Oph, total_sm)

    print(f"kernel_ms = {kernel_ms}")
    #flops_per_matmul = 2. * BATCH * H * N_CTX_Q * N_CTX * D_HEAD
    #total_flops = 2 * flops_per_matmul
    #return total_flops / ms * 1e-9
    return ms

def main():
    bench_flash_attention.run(save_path='.', print_data=True)

if __name__ == '__main__':
    sys.exit(main())


