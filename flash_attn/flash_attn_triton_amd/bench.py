import argparse
import torch
import triton
import time
import pandas as pd
from logging import warning
from flash_attn import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_with_kvcache,
    flash_attn_fp8_func,
    flash_attn_qkvpacked_fp8_func,
    flash_attn_varlen_fp8_func,
    flash_attn_varlen_qkvpacked_fp8_func,
)

from flash_attn.flash_attn_triton_amd.utils import input_helper
from typing import Literal, Optional
from functools import lru_cache

FUNCTIONS = {
    "flash_attn_with_kvcache": flash_attn_with_kvcache,
    "flash_attn_func": flash_attn_func,
    "flash_attn_fp8_func": flash_attn_fp8_func,
    "flash_attn_kvpacked_func": flash_attn_kvpacked_func,
    "flash_attn_varlen_func": flash_attn_varlen_func,
    "flash_attn_varlen_fp8_func": flash_attn_varlen_fp8_func,
    "flash_attn_varlen_kvpacked_func": flash_attn_varlen_kvpacked_func,
    "flash_attn_qkvpacked_func": flash_attn_qkvpacked_func,
    "flash_attn_qkvpacked_fp8_func": flash_attn_qkvpacked_fp8_func,
    "flash_attn_varlen_qkvpacked_func": flash_attn_varlen_qkvpacked_func,
    "flash_attn_varlen_qkvpacked_fp8_func": flash_attn_varlen_qkvpacked_fp8_func,
}

def estimate_memory(config):
    batch, hq, hk, sq, sk, d_head, causal, dropout = config
    memory_estimate = batch * (hq * sq + hk * sk) * d_head * 4  # bytes
    return memory_estimate

@lru_cache(maxsize=100)
def generate_benchmark_configs(is_varlen: bool, packing: Optional[Literal["kv", "qkv"]]):
    """
    generates a small number of configs that cover the parameter space well
    """

    # define all parameter options as lists
    batch_sizes = [1, 64]
    if packing == "qkv":
        hq_values = hk_values = [2, 8]
        sq_values = sk_values = [256, 8192]
    else:
        hq_values = [64, 128] # test mqa/gqa
        hk_values = [16, 64]
        if is_varlen: # make sure the seqlen is greater than the batchsize so that subsequences are greater than 0
            sq_values = [128, 512]
            sk_values = [512, 2024]
        else:
            sq_values = [4, 4096]
            sk_values = [4096, 16384] # test large k values for inference perf
    d_head_values = [64, 128]
    causal_values = [False]
    dropout_values = [0.0]
    
    # generate all possible configs
    configs = []
    
    # one big loop to generate configs
    for batch in batch_sizes:
        for hq in hq_values:
            for hk in hk_values:
                for sq in sq_values:
                    for sk in sk_values:
                        for d_head in d_head_values:
                            for causal in causal_values:
                                for dropout in dropout_values:
                                    # filter configs
                                    config = (batch, hq, hk, sq, sk, d_head, causal, dropout)

                                    # skip if memory usage would be too high
                                    if estimate_memory(config) > 8 * 1024 * 1024 * 1024:  # 8 GB limit
                                        continue

                                    # we need hq to be a multiple of hk
                                    if hq % hk != 0:
                                        continue

                                    # for qkvpacked functions, q and k must have same dimensions
                                    if packing == "qkv" and (sq != sk or hq != hk):
                                        continue
                                        
                                    # add config
                                    configs.append(config)
    
    # sort by memory usage (smallest to largest) for better visualization
    configs.sort(key=estimate_memory)
    
    return configs


def create_benchmark_fn(
    fn_name: str,
    BATCH: int,
    HQ: int,
    HK: int,
    N_CTX_Q: int,
    N_CTX_K: int,
    D_HEAD: int,
    causal: bool,
    dropout_p: float,
    dtype: torch.dtype,
    mode: Literal["fwd", "full"],
    device: Literal["cpu", "cuda"],
):
    if fn_name == "flash_attn_func":
        q, k, v, do, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="bshd", device=device)
        def flash_attn_bench_fn():
            out, lse, S_dmask = flash_attn_func(
                q,
                k,
                v,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)

        return flash_attn_bench_fn

    elif fn_name == "flash_attn_kvpacked_func":
        q, kv, do, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="bshd", packing="kv", device=device)
        def flash_attn_kvpacked_bench_fn():
            out, lse, S_dmask = flash_attn_kvpacked_func(
                q,
                kv,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dq, dkv = torch.autograd.grad(out, (q, kv), do)

        return flash_attn_kvpacked_bench_fn
    elif fn_name == "flash_attn_qkvpacked_func":
        qkv, do, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="bshd", packing="qkv", device=device)
        def flash_attn_qkvpacked_bench_fn():
            out, lse, S_dmask = flash_attn_qkvpacked_func(
                qkv,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dqkv = torch.autograd.grad(out, (qkv), do)

        return flash_attn_qkvpacked_bench_fn   
    elif fn_name == "flash_attn_varlen_func":
        q_unpad, k_unpad, v_unpad, do_unpad, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="thd", device=device)
        def flash_attn_varlen_bench_fn():
            out_unpad, lse, S_dmask = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), do_unpad)
        return flash_attn_varlen_bench_fn
    elif fn_name == "flash_attn_varlen_kvpacked_func":
        q_unpad, kv_unpad, do_unpad, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="thd", packing="kv", device=device)
        def flash_attn_varlen_kvpacked_bench_fn():
            out_unpad, lse, S_dmask = flash_attn_varlen_kvpacked_func(
                q_unpad,
                kv_unpad,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dq_unpad, dkv_unpad = torch.autograd.grad(out_unpad, (q_unpad, kv_unpad), do_unpad)
        return flash_attn_varlen_kvpacked_bench_fn
    elif fn_name == "flash_attn_varlen_qkvpacked_func":
        qkv_unpad, do_unpad, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="thd", packing="qkv", device=device)
        def flash_attn_varlen_qkvpacked_bench_fn():
            out_unpad, lse, S_dmask = flash_attn_varlen_qkvpacked_func(
                qkv_unpad,
                metadata.cu_seqlens_q,
                metadata.max_seqlens_q,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dqkv_unpad = torch.autograd.grad(out_unpad, (qkv_unpad), do_unpad)
        return flash_attn_varlen_qkvpacked_bench_fn
    elif fn_name == "flash_attn_with_kvcache":
        q, k_cache, v_cache, _, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="bshd", device=device)
        def flash_attn_with_kvcache_bench_fn():
            out = flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                None,
                None,
                rotary_cos=None,
                rotary_sin=None,
                cache_seqlens=None,
                cache_batch_idx=None,
                cache_leftpad=None,
                block_table=None,
                causal=causal,
                window_size=(-1, -1),
                rotary_interleaved=False,
                alibi_slopes=None,
                num_splits=0,
            )
        return flash_attn_with_kvcache_bench_fn
    elif fn_name == "flash_attn_fp8_func":
        (q, descale_q), (k, descale_k), (v, descale_v), (do, descale_do), metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="bshd", device=device)
        def flash_attn_f8_bench_fn():
            out, lse, S_dmask = flash_attn_fp8_func(
                q,
                k,
                v,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do,
            )
            if mode == "full":
                dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)

        return flash_attn_f8_bench_fn
    elif fn_name == "flash_attn_qkvpacked_fp8_func":
        qkv, do, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="bshd", packing="qkv", device=device)
        def flash_attn_qkvpacked_fp8_bench_fn():
            out, lse, S_dmask = flash_attn_qkvpacked_fp8_func(
                qkv,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dqkv = torch.autograd.grad(out, (qkv), do)

        return flash_attn_qkvpacked_fp8_bench_fn   
    elif fn_name == "flash_attn_varlen_fp8_func":
        (q_unpad, descale_q), (k_unpad, descale_k), (v_unpad, descale_v), (do_unpad, descale_do), metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="thd", device=device)
        def flash_attn_varlen_fp8_bench_fn():
            out_unpad, lse, S_dmask = flash_attn_varlen_fp8_func(
                q_unpad,
                k_unpad,
                v_unpad,
                metadata.cu_seqlens_q,
                metadata.cu_seqlens_k,
                metadata.max_seqlens_q,
                metadata.max_seqlens_k,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), do_unpad)
        return flash_attn_varlen_fp8_bench_fn
    elif fn_name == "flash_attn_varlen_qkvpacked_fp8_func":
        qkv_unpad, do_unpad, metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout="thd", packing="qkv", device=device)
        def flash_attn_varlen_qkvpacked_fp8_bench_fn():
            out_unpad, lse, S_dmask = flash_attn_varlen_qkvpacked_fp8_func(
                qkv_unpad,
                metadata.cu_seqlens_q,
                metadata.max_seqlens_q,
                dropout_p,
                causal=causal,
                window_size=(-1, -1),
                softcap=0.0 ,
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=True,
            )
            if mode == "full":
                dqkv_unpad = torch.autograd.grad(out_unpad, (qkv_unpad), do_unpad)
        return flash_attn_varlen_qkvpacked_fp8_bench_fn
    else:
        valid_fn_names = ", ".join(FUNCTIONS.keys())
        raise ValueError(f"{fn_name} should be one of the following functions. {valid_fn_names}")


def get_packing_type(fn_name: str) -> Optional[Literal["kv", "qkv"]]:
    if "_kvpacked" in fn_name:
        packing = "kv"
    elif "_qkvpacked" in fn_name:
        packing = "qkv"
    else:
        packing = None

    return packing

def run_benchmark(fn_name, configs, dtype, mode):
    """
    Runs the benchmark for the provided function based on the provided arguments.
    """
    # start timing the benchmark
    start_time = time.time()

    # check that we have at least one config
    assert len(configs) > 0

    # print bench fn
    mode_text = "forward" if mode == "fwd" else "forward and backward"
    print(f"\nBenchmarking {fn_name} {mode_text} in {dtype} with {len(configs)} configs...")

    # Setup benchmark configurations
    bench_configs = [
        triton.testing.Benchmark(
            x_names=["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K", "D_HEAD", "CAUSAL", "DROPOUT"],
            x_vals=configs,
            line_arg="provider",
            line_vals=["triton"],
            line_names=["Time (ms)"],
            styles=[("red", "-")],
            ylabel="ms",
            plot_name=f"benchmark-{fn_name}-{mode}-{dtype}",
            args={
                "dtype": dtype,
                "mode": mode
            },
        )
    ]

    @triton.testing.perf_report(bench_configs)
    def bench_function(
        BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT, dtype, mode, provider, device="cuda"
    ):
        benchmark_fn = create_benchmark_fn(fn_name,
                                           BATCH, 
                                           HQ, 
                                           HK, 
                                           N_CTX_Q, 
                                           N_CTX_K, 
                                           D_HEAD,
                                           CAUSAL,
                                           DROPOUT,
                                           dtype,
                                           mode,
                                           device
                                           )

        # run the benchmark
        ms = triton.testing.do_bench(benchmark_fn, warmup=25, rep=100)
        return ms

    df = bench_function.run(save_path=".", print_data=True, return_df=True)[0]
    
    # calculate and print elapsed time
    elapsed_time = time.time() - start_time
    print(f"Total time for benchmarking {fn_name} in {mode} mode: {elapsed_time:.2f} seconds")

    return df

def process_args():
    """
    Parses command-line arguments.
    """
    # create parser
    parser = argparse.ArgumentParser(
        prog="Benchmark FlashAttention",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-benchmark_fn",
        type=str,
        nargs="*",
        choices=FUNCTIONS.keys(),
        help=f"Function(s) to benchmark",
    )
    parser.add_argument("-b", type=int, default=0, help=f"Batch size")
    parser.add_argument("-hq", type=int, default=0, help=f"Q Number of heads")
    parser.add_argument("-hk", type=int, default=0, help=f"K and V Number of heads")
    parser.add_argument("-sq", type=int, default=0, help=f"Q Sequence Length")
    parser.add_argument("-sk", type=int, default=0, help=f"K and V Sequence Length")
    parser.add_argument("-d", type=int, default=0, help=f"Head Dimension")
    parser.add_argument("-causal", action="store_true", default=False, help=f"Causal")
    parser.add_argument("-dropout", type=float, default=0.0, help=f"Dropout")

    # parse args
    args = parser.parse_args()

    # determine the functions to benchmark
    if args.benchmark_fn is None or len(args.benchmark_fn) == 0:
        benchmark_fns = FUNCTIONS.keys()
    else:
        for fn_name in args.benchmark_fn:
            if fn_name not in FUNCTIONS:
                raise ValueError(f"invalid benchmark function specified: {fn_name}")
        benchmark_fns = args.benchmark_fn

    # get configs
    configs = {}
    for fn_name in benchmark_fns:
        # get info about fn to benchmark
        packing = get_packing_type(fn_name)
        is_varlen = True if "varlen" in fn_name else False
        is_fp8 = True if "fp8" in fn_name else False
        supports_backward = False if fn_name in ["flash_attn_with_kvcache"] else True

        # get dtype
        dtype = torch.float8_e4m3fnuz if is_fp8 else torch.float16
  
        # check backward pass support
        if supports_backward:
            mode = "full"
        else:
            mode = "fwd"
            warning(f"{fn_name} does not have a backward pass so benching forward pass only.")

        if args.b or args.hq or args.hk or args.sq or args.sk or args.d:
            assert args.b and args.hq and args.sq and args.d, (
                "if custom config is specified, please provide at least batch, number of Q heads, Q sequence length, and head size."
            )
            configs[fn_name] = [(args.b, 
                        args.hq, 
                        args.hk if args.hk is not None else args.hq, 
                        args.sq, 
                        args.sk if args.sk is not None else args.sq,  
                        args.d, 
                        args.causal, 
                        args.dropout)], dtype, mode
        else:
            configs[fn_name] = generate_benchmark_configs(is_varlen, packing), dtype, mode

    return configs

def main():
    """
    Main function to run benchmarks.
    """
    # start timing the entire benchmarking process
    total_start_time = time.time()

    # process args
    bench_fn_configs = process_args()
    has_multiple_fns = True if len(bench_fn_configs) > 1 else False
    combined_df = None

    # run benchmarks
    for fn_name, (configs, dtype, mode) in bench_fn_configs.items():
        # run bench mark
        df = run_benchmark(fn_name, configs, dtype, mode)
        config_cols = [col for col in df.columns if col != "Time (ms)"]
        df = df.rename(columns={"Time (ms)": f"{fn_name}_{mode}_{dtype}_ms"})

        # merge into one final dataframe
        if has_multiple_fns:
            combined_df = df if combined_df is None else combined_df.merge(df, on=config_cols, how="outer") 
    
    # print total time for all benchmarks
    total_elapsed_time = time.time() - total_start_time
    print(f"\nTotal time for all benchmarks: {total_elapsed_time:.2f} seconds")

    # save combined data
    if has_multiple_fns:
        combined_filename = f"combined.csv"
        combined_df.to_csv(combined_filename, index=False)
        print(f"\nCombined data saved to {combined_filename}")
        print(combined_df)

if __name__ == "__main__":
    main()
