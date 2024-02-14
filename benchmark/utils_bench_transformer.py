from typing import Callable
import time
import torch

torch.set_grad_enabled(False)
import transformers
import numpy as np
import itertools
import numpy as np
from tqdm import tqdm


def benchmark(
    f: Callable,
    *,
    f_setup: Callable | None = None,
    min_repeat: int,
    min_secs: float,
    tqdm_kwargs: dict | None = None
) -> np.ndarray:
    latency = []

    # First run, ignore min_secs
    if f_setup is not None:
        f_setup()
    st = time.perf_counter_ns()
    f()
    ed = time.perf_counter_ns()
    latency.append((ed - st) / 1e9)

    # Subsequent runs, until reaching both min_repeat and min_secs
    min_nanos = int(min_secs * 1e9)
    start_nanos = time.perf_counter_ns()
    while True:
        now_nanos = time.perf_counter_ns()
        if len(latency) > min_repeat and now_nanos - start_nanos > min_nanos:
            break
        if f_setup is not None:
            f_setup()
        st = time.perf_counter_ns()
        f()
        ed = time.perf_counter_ns()
        latency.append((ed - st) / 1e9)
    return np.array(latency)


def tail_mean(xs: np.ndarray, skip=0.2):
    return xs[int(len(xs) * skip) :].mean()


def benchmark_dense(out, nd_list, seqlen_list, bs_list):
    seqlen_list = [1] + seqlen_list
    total = len(list(itertools.product(nd_list, seqlen_list, bs_list)))
    pbar = tqdm(total=total)
    for (n, d), seqlen in reversed(
        list(itertools.product(nd_list, seqlen_list))
    ):
        h = n * d
        maxbs = max(bs_list)
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda:0")
        X = torch.rand(
            (maxbs, seqlen, h), dtype=torch.bfloat16, device="cuda:0"
        )
        W = torch.rand((h, h), dtype=torch.bfloat16, device="cuda:0")
        torch.cuda.synchronize()
        for bs in reversed(bs_list):
            pbar.set_postfix(n=n, h=h, d=d, seqlen=seqlen, bs=bs)

            def run():
                torch.matmul(X[:bs], W)
                torch.cuda.synchronize()

            def clear_cache():
                cache.zero_()
                torch.cuda.synchronize()

            latency = benchmark(
                run, f_setup=clear_cache, min_repeat=20, min_secs=2
            )
            l = tail_mean(latency)
            out.append(
                {"n": n, "d": d, "seqlen": seqlen, "bs": bs, "latency": l}
            )
            pbar.update()
        del cache, X, W
        torch.cuda.empty_cache()
    pbar.close()


def benchmark_qk_init(out, nd_list, seqlen_list, bs_list):
    total = len(list(itertools.product(nd_list, seqlen_list, bs_list)))
    pbar = tqdm(total=total)
    for (n, d), seqlen in reversed(
        list(itertools.product(nd_list, seqlen_list))
    ):
        h = n * d
        try:
            maxbs = max(
                b
                for b in bs_list
                if b * n * seqlen * d * 2 * 2 + b * n * seqlen**2 * 2 < 80e9
            )
        except ValueError:
            pbar.update(len(bs_list))
            continue
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda:0")
        Qmax = torch.rand(
            (maxbs, n, seqlen, d), dtype=torch.bfloat16, device="cuda:0"
        )
        Kmax = torch.rand(
            (maxbs, n, seqlen, d), dtype=torch.bfloat16, device="cuda:0"
        )
        torch.cuda.synchronize()
        for bs in reversed(bs_list):
            pbar.set_postfix(n=n, h=h, d=d, seqlen=seqlen, bs=bs)
            if bs > maxbs:
                pbar.update()
                continue
            Q = Qmax[:bs]
            K = Kmax[:bs]

            def run():
                torch.bmm(
                    Q.view(bs * n, seqlen, d),
                    K.view(bs * n, seqlen, d).transpose(1, 2),
                )
                torch.cuda.synchronize()

            def clear_cache():
                cache.zero_()
                torch.cuda.synchronize()

            latency = benchmark(
                run, f_setup=clear_cache, min_repeat=20, min_secs=2
            )
            l = tail_mean(latency)
            out.append(
                {"n": n, "d": d, "seqlen": seqlen, "bs": bs, "latency": l}
            )
            pbar.update()
        del cache, Q, K, Qmax, Kmax
        torch.cuda.empty_cache()
    pbar.close()


def benchmark_qk_ar(out, nd_list, seqlen_list, bs_list):
    total = len(list(itertools.product(nd_list, seqlen_list, bs_list)))
    pbar = tqdm(total=total)
    for (n, d), seqlen in reversed(
        list(itertools.product(nd_list, seqlen_list))
    ):
        h = n * d
        try:
            maxbs = max(
                b
                for b in bs_list
                if b * n * (1 + seqlen) * d * 2 + b * n * seqlen * 2 < 80e9
            )
        except ValueError:
            pbar.update(len(bs_list))
            continue
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda:0")
        Qmax = torch.rand(
            (maxbs, n, 1, d), dtype=torch.bfloat16, device="cuda:0"
        )
        Kmax = torch.rand(
            (maxbs, n, seqlen, d), dtype=torch.bfloat16, device="cuda:0"
        )
        torch.cuda.synchronize()
        for bs in reversed(bs_list):
            pbar.set_postfix(n=n, h=h, d=d, seqlen=seqlen, bs=bs)
            if bs > maxbs:
                pbar.update()
                continue
            Q = Qmax[:bs]
            K = Kmax[:bs]

            def run():
                torch.bmm(
                    Q.view(bs * n, 1, d),
                    K.view(bs * n, seqlen, d).transpose(1, 2),
                )
                torch.cuda.synchronize()

            def clear_cache():
                cache.zero_()
                torch.cuda.synchronize()

            latency = benchmark(
                run, f_setup=clear_cache, min_repeat=20, min_secs=2
            )
            l = tail_mean(latency)
            out.append(
                {"n": n, "d": d, "seqlen": seqlen, "bs": bs, "latency": l}
            )
            pbar.update()
        del cache, Q, K, Qmax, Kmax
        torch.cuda.empty_cache()
    pbar.close()


def greedy_sample_one(
    model, input_ids, attention_mask=None, past_key_values=None
):
    bs, tgt_len = input_ids.shape
    if past_key_values is not None:
        _bs, _num_heads, src_len, _head_dims = past_key_values[0][0].shape
        assert bs == _bs
    else:
        src_len = 0
    if attention_mask is None:
        attention_mask = torch.ones(
            (bs, src_len + tgt_len), device=model.device
        )
    ret = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=False,
        return_dict=True,
    )
    return ret


def time_greedy_generate(model, input_ids, new_tokens):
    ts = []
    output = input_ids
    past_key_values = None
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=model.device)
    attention_mask = torch.ones(input_ids.shape, device=model.device)
    for _ in range(new_tokens):
        cache.zero_()
        torch.cuda.synchronize()
        st = time.perf_counter_ns()

        ret = greedy_sample_one(
            model, input_ids, attention_mask, past_key_values
        )
        input_ids = torch.argmax(ret.logits[:, -1, :], axis=-1)[:, None]
        output = torch.cat([output, input_ids], axis=1)
        past_key_values = ret.past_key_values
        attention_mask = torch.cat(
            [
                attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], 1)),
            ],
            dim=-1,
        )

        torch.cuda.synchronize()
        ed = time.perf_counter_ns()
        ts.append((ed - st) / 1e9)
    return np.array(ts)


def _gen_opt_cfg(
    n_layers: int, d_model: int, n_heads: int, **kwargs
) -> transformers.OPTConfig:
    return transformers.OPTConfig(
        num_hidden_layers=n_layers,
        hidden_size=d_model,
        ffn_dim=d_model * 4,
        num_attention_heads=n_heads,
        **kwargs
    )


def gen_opt_cfg():
    optcfg = {
        # https://arxiv.org/pdf/2205.01068.pdf   Table 2.1
        "125m": _gen_opt_cfg(12, 768, 12),
        "350m": _gen_opt_cfg(24, 1024, 16),
        "760m": _gen_opt_cfg(24, 1536, 16),
        "1.3b": _gen_opt_cfg(24, 2048, 32),
        "2.7b": _gen_opt_cfg(32, 2560, 32),
        "6.7b": _gen_opt_cfg(32, 4096, 32),
        "13b": _gen_opt_cfg(40, 5120, 40),
        "13b_1layer": _gen_opt_cfg(1, 5120, 40),
        "30b": _gen_opt_cfg(48, 7168, 56),
        "66b": _gen_opt_cfg(64, 9216, 72),
        "175b": _gen_opt_cfg(96, 12288, 96),
        "175b_1layer": _gen_opt_cfg(1, 12288, 96),
    }
    return optcfg
