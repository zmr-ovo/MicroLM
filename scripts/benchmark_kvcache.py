"""KV Cache 性能基准测试脚本。

本脚本对比同一模型在「无 KV Cache」和「有 KV Cache」两种解码模式下的
生成速度，量化 KV Cache 带来的加速效果。

测试矩阵（20 种配置组合）：
  prompt 长度 × 生成长度 = [16, 32, 64, 128, 256] × [32, 64, 128, 256]

每种配置执行流程：
  1. 预热（warmup）：跑 2 次短生成（8 token），消除 GPU 冷启动抖动
  2. 正式跑（bench）：跑 5 次完整生成，取平均值
  3. 记录指标：无 cache 耗时、有 cache 总耗时/prefill 耗时/decode 耗时、加速比

输出产物：
  - kvcache_benchmark.csv  原始数据表（每个配置一行）
  - kvcache_benchmark.json 结构化结果 + 测试配置元数据

关键指标解读：
  - speedup：无 cache 耗时 / 有 cache 总耗时，越大说明 KV Cache 越有价值
  - decode_tps：decode 阶段每秒生成的 token 数（cache 模式的核心吞吐指标）
  - prefill_time vs decode_time：prefill 一次 O(T²)，decode 每步 O(T)，
    生成长度越大，decode 省的时间越多

使用方式：
    python scripts/benchmark_kvcache.py \
        --checkpoint outputs/sft_baseline/ckpt_final.pt \
        --out-dir results/
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from microlm.model import TransformerLM
from microlm.model.transformer import KVCache
from microlm.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    关键参数：
      --warmup-runs 2  预热次数（消除 GPU 冷启动抖动）
      --bench-runs  5  正式测试次数（取平均）
      --dtype           默认 float32，GPU 可用 float16 加速
    """
    p = argparse.ArgumentParser(description="Benchmark KV Cache for MicroLM")
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/sft_baseline/ckpt_final.pt"))
    p.add_argument("--vocab-path", type=Path, default=Path("outputs/tokenizer_full_clean/vocab.json"))
    p.add_argument("--merges-path", type=Path, default=Path("outputs/tokenizer_full_clean/merge.txt"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    p.add_argument("--warmup-runs", type=int, default=2)
    p.add_argument("--bench-runs", type=int, default=5)
    p.add_argument("--eos-token", type=str, default="</s>")
    return p.parse_args()


# ─── Model loading (reused from run_eval_prompts.py) ────────────────────────

def load_model(checkpoint_path: Path, device: str, dtype: torch.dtype) -> TransformerLM:
    """从检查点加载模型，自动处理 LoRA checkpoint 的 key 映射。

    兼容三种检查点格式：
      - 预训练裸 state_dict
      - 训练检查点（含 model_state_dict 包装）
      - LoRA 检查点（自动提取 original.weight → weight，跳过 lora_A/lora_B）

    Args:
        checkpoint_path: 检查点 .pt 文件路径
        device:          "cuda" 或 "cpu"
        dtype:           torch.float32 或 torch.float16

    Returns:
        已加载权重并设为 eval() 模式的 TransformerLM 模型
    """
    config_path = checkpoint_path.parent / "model_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    # Handle LoRA checkpoint keys
    is_lora_ckpt = any("original.weight" in k for k in cleaned)
    if is_lora_ckpt:
        remapped = {}
        for k, v in cleaned.items():
            if k.endswith(".original.weight"):
                remapped[k.replace(".original.weight", ".weight")] = v
            elif ".lora_" in k:
                continue
            else:
                remapped[k] = v
        cleaned = remapped

    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=float(cfg.get("rope_theta", 1000000.0)),
        use_rms_norm=True,
        norm_mode="pre",
        ffn_type="swiglu",
        device=device,
        dtype=dtype,
    ).to(device)
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


# ─── Generation with NO cache (recompute full sequence each step) ───────────

@torch.no_grad()
def generate_no_cache(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, float]:
    """无 KV Cache 的自回归解码 —— 每步重新计算完整序列。

    实现方式：每轮循环把整个 generated 序列（截断到 context_length）
    送入模型，只取最后一个位置的 logit 采样。复杂度 O(T²·L)：
    生成第 N 个 token 时要重新算前 N-1 个 token 的所有前向计算。

    Args:
        model:          已加载的 TransformerLM 模型
        prompt_ids:     输入 prompt token IDs [1, prompt_len]
        max_new_tokens: 生成多少个新 token

    Returns:
        (完整序列 token IDs, 总耗时秒数)
    """
    generated = prompt_ids.clone()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        # Feed entire sequence, only take last logit
        input_ids = generated[:, -model.context_length:]
        logits = model(input_ids)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    return generated, elapsed


# ─── Generation WITH KV Cache (prefill + incremental decode) ────────────────

@torch.no_grad()
def generate_with_cache(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[torch.Tensor, float, float, float]:
    """带 KV Cache 的自回归解码（prefill + decode 两阶段）。

    Phase 1 — Prefill：prompt 全部 token 一次性送入，填充 K/V Cache。
             复杂度 O(T²)，但只做一次。
    Phase 2 — Decode：每步只输入 1 个新 token，K/V 来自 cache 拼接。
             每步复杂度 O(T)，而非无 cache 时的 O(T²)。

    第一步采样放在 prefill 内完成（不单独计 decode），所以 decode 循环
    实际跑 max_new_tokens - 1 步。

    Args:
        model:          已加载的 TransformerLM 模型
        prompt_ids:     输入 prompt token IDs [1, prompt_len]
        max_new_tokens: 生成多少个新 token

    Returns:
        (完整序列, 总耗时, prefill耗时, decode耗时)
    """
    generated = prompt_ids.clone()

    # Phase 1: Prefill
    kv_cache = KVCache(len(model.layers))
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_prefill_start = time.perf_counter()

    logits, kv_cache = model.forward(
        prompt_ids, kv_cache=kv_cache, use_cache=True, start_pos=0,
    )
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated = torch.cat((generated, next_token), dim=1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    prefill_time = time.perf_counter() - t_prefill_start

    # Phase 2: Decode
    t_decode_start = time.perf_counter()

    for _ in range(max_new_tokens - 1):
        cur_pos = generated.shape[1] - 1
        logits, kv_cache = model.forward(
            next_token, kv_cache=kv_cache, use_cache=True, start_pos=cur_pos,
        )
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    decode_time = time.perf_counter() - t_decode_start
    total_time = prefill_time + decode_time

    return generated, total_time, prefill_time, decode_time


# ─── 基准测试运行器 ─────────────────────────────────────────────────────

def run_benchmark(
    model: TransformerLM,
    prompt_lengths: list[int],
    gen_lengths: list[int],
    vocab_size: int,
    warmup_runs: int,
    bench_runs: int,
    device: str,
) -> list[dict]:
    """运行完整基准测试矩阵。

    对每对 (prompt_len, gen_len) 组合：
      1. 随机生成 prompt token IDs（seed=42 保证可复现）
      2. 安全检查：prompt+gen 不超出 context_length
      3. 预热：各跑 warmup_runs 次短生成（8 token），消除冷启动
      4. 正式测试：各跑 bench_runs 次完整生成，记录每次耗时
      5. 计算平均值 + 加速比 + 吞吐量

    Args:
        model:          已加载的模型
        prompt_lengths: 要测试的 prompt 长度列表
        gen_lengths:    要测试的生成长度列表
        vocab_size:     模型词表大小（用于随机生成 token）
        warmup_runs:    预热次数
        bench_runs:     正式测试次数
        device:         "cuda" 或 "cpu"

    Returns:
        结果列表，每个元素为一个 dict，包含 prompt_len, gen_len,
        no_cache_time_s, cache_total_time_s, speedup 等字段
    """
    results = []

    for prompt_len in prompt_lengths:
        # Create a random prompt token sequence
        torch.manual_seed(42)
        prompt_ids = torch.randint(
            0, vocab_size, (1, prompt_len), device=device, dtype=torch.long,
        )

        for gen_len in gen_lengths:
            # Safety: prompt + gen must fit in context_length
            if prompt_len + gen_len > model.context_length:
                continue

            # ---- 预热：消除 GPU 冷启动抖动 ----
            for _ in range(warmup_runs):
                generate_no_cache(model, prompt_ids.clone(), min(gen_len, 8))
                generate_with_cache(model, prompt_ids.clone(), min(gen_len, 8))

            # ---- 正式测试：记录每次耗时 ----
            no_cache_times = []
            cache_times = []
            cache_prefill_times = []
            cache_decode_times = []

            for run_idx in range(bench_runs):
                torch.manual_seed(42 + run_idx)

                # No cache
                _, nc_time = generate_no_cache(model, prompt_ids.clone(), gen_len)
                no_cache_times.append(nc_time)

                # With cache
                torch.manual_seed(42 + run_idx)
                _, c_total, c_prefill, c_decode = generate_with_cache(
                    model, prompt_ids.clone(), gen_len,
                )
                cache_times.append(c_total)
                cache_prefill_times.append(c_prefill)
                cache_decode_times.append(c_decode)

            nc_avg = sum(no_cache_times) / len(no_cache_times)
            c_avg = sum(cache_times) / len(cache_times)
            c_prefill_avg = sum(cache_prefill_times) / len(cache_prefill_times)
            c_decode_avg = sum(cache_decode_times) / len(cache_decode_times)
            speedup = nc_avg / c_avg if c_avg > 0 else float("inf")
            decode_tps = gen_len / c_decode_avg if c_decode_avg > 0 else 0
            no_cache_tps = gen_len / nc_avg if nc_avg > 0 else 0

            row = {
                "prompt_len": prompt_len,
                "gen_len": gen_len,
                "no_cache_time_s": round(nc_avg, 4),
                "no_cache_tps": round(no_cache_tps, 1),
                "cache_total_time_s": round(c_avg, 4),
                "cache_prefill_time_s": round(c_prefill_avg, 4),
                "cache_decode_time_s": round(c_decode_avg, 4),
                "cache_decode_tps": round(decode_tps, 1),
                "speedup": round(speedup, 2),
            }
            results.append(row)
            print(
                f"  prompt={prompt_len:>3d}  gen={gen_len:>3d}  "
                f"no_cache={nc_avg:.3f}s ({no_cache_tps:.0f} tok/s)  "
                f"cache={c_avg:.3f}s ({decode_tps:.0f} tok/s decode)  "
                f"speedup={speedup:.2f}x"
            )

    return results


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    """基准测试主入口。

    流程：
      1. 解析参数 + 加载模型
      2. 定义测试矩阵（5 prompt × 4 gen = 20 组）
      3. 运行 benchmark
      4. 保存 CSV + JSON 结果
      5. 打印汇总统计（平均加速比、decode 吞吐量）
    """
    args = parse_args()
    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device, dtype)
    print(f"  d_model={model.token_embeddings.weight.shape[1]}, "
          f"layers={len(model.layers)}, ctx={model.context_length}")
    print(f"  device={args.device}, dtype={args.dtype}")

    vocab_size = model.token_embeddings.weight.shape[0]

    # Test matrix
    prompt_lengths = [16, 32, 64, 128, 256]
    gen_lengths = [32, 64, 128, 256]

    print(f"\nRunning benchmark (warmup={args.warmup_runs}, runs={args.bench_runs})...")
    print(f"  prompt_lengths={prompt_lengths}")
    print(f"  gen_lengths={gen_lengths}\n")

    results = run_benchmark(
        model, prompt_lengths, gen_lengths, vocab_size,
        args.warmup_runs, args.bench_runs, args.device,
    )

    # Save CSV
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "kvcache_benchmark.csv"
    if results:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV saved to {csv_path}")

    # Save JSON with metadata
    json_path = args.out_dir / "kvcache_benchmark.json"
    output = {
        "benchmark_config": {
            "checkpoint": str(args.checkpoint),
            "device": args.device,
            "dtype": args.dtype,
            "vocab_size": vocab_size,
            "d_model": model.token_embeddings.weight.shape[1],
            "num_layers": len(model.layers),
            "context_length": model.context_length,
            "warmup_runs": args.warmup_runs,
            "bench_runs": args.bench_runs,
            "prompt_lengths": prompt_lengths,
            "gen_lengths": gen_lengths,
        },
        "results": results,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"JSON saved to {json_path}")

    # Print summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        max_speedup = max(r["speedup"] for r in results)
        min_speedup = min(r["speedup"] for r in results)
        avg_decode_tps = sum(r["cache_decode_tps"] for r in results) / len(results)
        print(f"  Speedup range: {min_speedup:.2f}x ~ {max_speedup:.2f}x")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Average decode throughput (cache): {avg_decode_tps:.0f} tok/s")


if __name__ == "__main__":
    main()
