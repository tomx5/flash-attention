import argparse
import torch
import triton
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_kvpacked_func, \
    flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func, \
    flash_attn_with_kvcache
from flash_attn.flash_attn_triton_amd.utils import input_helper

ARGS_TO_TORCH_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

FUNCTIONS = {
    "flash_attn": flash_attn_func,
    "flash_attn_kvpacked": flash_attn_kvpacked_func,
    "flash_attn_qkvpacked": flash_attn_qkvpacked_func,
    "flash_attn_varlen": flash_attn_varlen_func,
    # "flash_attn_varlen_kvpacked": flash_attn_varlen_kvpacked_func,
    # "flash_attn_varlen_qkvpacked": flash_attn_varlen_qkvpacked_func,
    "flash_attn_with_kvcache": flash_attn_with_kvcache,
}

MODES = {
        "fwd": "for forward pass only", 
        "full": "for forward and backward pass"
         }

def get_benchmark_configs(args, fn_name):
    """
    returns benchmark configurations based on whether variable-length sequences are used.
    """
    if args.custom_config:
        hk = args.hq if not args.hk else args.hk
        sk = args.sq if not args.sk else args.sk
        return [(args.b, args.hq, hk, args.sq, sk, args.d)]
    elif fn_name in ["flash_attn_varlen"]:
        return [
            (2, 16, 4, 1024, 1024, 128, False, 0.0),
            (8, 16, 2, 2048, 2048, 128, False, 0.0),
            (4, 16, 8, 4096, 4096, 128, False, 0.0),
            (2, 16, 4, 8192, 8192, 128, False, 0.0),
            (2, 16, 8, 16384, 16384, 128, False, 0.0),
            (2, 48, 12, 1024, 1024, 128, False, 0.0),
            (2, 48, 24, 2048, 2048, 128, False, 0.0),
            (2, 48, 8, 4096, 4096, 128, False, 0.0),
            (2, 48, 4, 8192, 8192, 128, False, 0.0),
            (2, 48, 2, 16384, 16384, 128, False, 0.0),
            (2, 64, 32, 1024, 1024, 128, False, 0.0),
            (4, 64, 16, 2048, 2048, 128, False, 0.0),
            (4, 64, 8, 4096, 4096, 128, False, 0.0),
            (4, 64, 32, 8192, 8192, 128, False, 0.0),
            (4, 128, 16, 16384, 16384, 128, False, 0.0),
        ]
    elif fn_name in ["flash_attn", "flash_attn_kvpacked", "flash_attn_with_kvcache"]:
        return [
            (16, 16, 16, 1024, 1024, 128, False, 0.0),
            (8, 16, 16, 2048, 2048, 128, False, 0.0),
            (4, 16, 16, 4096, 4096, 128, False, 0.0),
            (1, 8, 8, 8192, 8192, 128, False, 0.0),
            (1, 2, 2, 16384, 16384, 128, False, 0.0),
            (2, 48, 48, 1024, 1024, 128, False, 0.0),
            (2, 48, 48, 2048, 1024, 128, False, 0.0),
            (1, 8, 8, 4096, 8192, 128, False, 0.0),
            (1, 8, 8, 8192, 4096, 128, False, 0.0),
            (2, 4, 4, 16384, 8192, 128, False, 0.0),
            (2, 8, 8, 1989, 15344, 128, False, 0.0),
            (4, 16, 16, 4097, 163, 128, False, 0.0),
            (2, 16, 16, 8122, 2159, 128, False, 0.0),
            (1, 16, 16, 16281, 7, 128, False, 0.0),
            (2, 48, 48, 1021, 1020, 128, False, 0.0),
            (2, 48, 48, 2001, 2048, 128, False, 0.0),
            (2, 8, 8, 3996, 9639, 128, False, 0.0),
            (2, 8, 8, 8181, 1021, 128, False, 0.0),
        ]
    elif fn_name in ["flash_attn_qkvpacked"]:
        return [
            (16, 16, 16, 1024, 1024, 128, False, 0.0),
            (8, 16, 16, 2048, 2048, 128, False, 0.0),
            (4, 16, 16, 4096, 4096, 128, False, 0.0),
            (1, 8, 8, 8192, 8192, 128, False, 0.0),
            (1, 2, 2, 16384, 16384, 128, False, 0.0),
            (2, 48, 48, 1024, 1024, 128, False, 0.0),
            (2, 48, 48, 2048, 2048, 128, False, 0.0),
            (1, 8, 8, 4096, 4096, 128, False, 0.0),
            (1, 8, 8, 8192, 8192, 128, False, 0.0),
            (2, 4, 4, 8192, 8192, 128, False, 0.0),
            (2, 8, 8, 1989, 1989, 128, False, 0.0),
            (4, 16, 16, 163, 163, 128, False, 0.0),
            (2, 16, 16, 8122, 8122, 128, False, 0.0),
            (1, 16, 16, 1021, 1021, 128, False, 0.0),
            (2, 48, 48, 1020, 1020, 128, False, 0.0),
            (2, 48, 48, 2001, 2001, 128, False, 0.0),
            (2, 8, 8, 3996, 3996, 128, False, 0.0),
            (2, 8, 8, 1021, 1021, 128, False, 0.0),
        ]

def create_benchmark_fn(fn_name, BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, device, dropout_p, causal, mode):
    if fn_name == "flash_attn":
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
    
    elif fn_name == "flash_attn_kvpacked":
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
    elif fn_name == "flash_attn_qkvpacked":
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
    elif fn_name == "flash_attn_varlen":
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
    else:
        raise ValueError(f"{fn_name} is not supported for benchmarking")

def run_benchmark(args, fn_name, fn, mode):
    """
    Runs the benchmark for the provided function based on the provided arguments.
    """
    # check mode
    if mode not in MODES:
        raise ValueError(f"{mode} not in {MODES}")

    # print bench fn
    if fn_name in ["flash_attn_with_kvcache"] and mode == "full":
        print(f"Benchmarking {fn_name} in {mode} mode. It does not have a backward pass so we will be running the forward pass only.")
    else:
        print(f"Benchmarking {fn_name} in {mode} mode ...")

    # get configs
    dtype = ARGS_TO_TORCH_DTYPE[args.dtype]
    configs = get_benchmark_configs(args, fn_name)

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
            plot_name=f"benchmark-{fn_name}-{mode}",
            args={
                "dtype": dtype,
                "mode": mode
            },
        )
    ]

    @triton.testing.perf_report(bench_configs)
    def bench_function(
        BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, CAUSAL, DROPOUT,  dtype, mode, provider, device="cuda"
    ):
        warmup = 25
        rep = 100

        benchmark_fn = create_benchmark_fn(fn_name,
                                           BATCH, 
                                           HQ, 
                                           HK, 
                                           N_CTX_Q, 
                                           N_CTX_K, 
                                           D_HEAD, 
                                           dtype, 
                                           device,
                                           CAUSAL,
                                           DROPOUT,
                                           mode)

        # run the benchmark
        ms = triton.testing.do_bench(benchmark_fn, warmup=warmup, rep=rep)
        return ms

    bench_function.run(save_path=".", print_data=True)


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="Benchmark FlashAttention",
        allow_abbrev=False,
    )
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-sk", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-causal", action="store_true", default=False)
    parser.add_argument("-dtype", default="fp16")
    parser.add_argument(
        "-benchmark_fn",
        type=str,
        nargs="*",
        choices=FUNCTIONS.keys(),
        help=f"Function(s) to benchmark: {FUNCTIONS.keys()}",
    )
    mode_help_str = ". ".join(f'"{k}" {v.replace(" only", "")}' for k, v in MODES.items())
    parser.add_argument(
        "-mode",
        type=str,
        nargs='*',
        default= ["full"],
        choices= MODES.keys(),
        help=f"Mode(s) to run: {mode_help_str}",
    )
    return parser.parse_args()

def main():
    """
    Main function to run benchmarks.
    """
    args = parse_args()

    # Validate arguments
    args.custom_config = False
    if args.b or args.hq or args.hk or args.sq or args.sk or args.d:
        args.custom_config = True
        assert args.b and args.hq and args.sq and args.d, (
            "if custom config is specified, please provide batch, number of Q heads, Q sequence length, and head size."
        )
    assert args.dtype in ARGS_TO_TORCH_DTYPE, "only fp16, bf16 and fp32 types currently supported."

    # determine the functions to benchmark
    if args.benchmark_fn is None or len(args.benchmark_fn) == 0:
        bench_fn_list = FUNCTIONS.keys()
    else:
        bench_fn_list = args.benchmark_fn

    # run benchmarks
    for fn_name in bench_fn_list:
        if fn_name not in FUNCTIONS:
            raise ValueError(f"invalid benchmark function specified: {fn_name}")
        for mode in args.mode:
            run_benchmark(args, fn_name, FUNCTIONS[fn_name], mode)

if __name__ == "__main__":
    main()
