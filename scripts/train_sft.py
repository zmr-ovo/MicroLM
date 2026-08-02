"""SFT（Supervised Fine-Tuning）训练脚本 —— 指令微调完整流水线。

本脚本在预训练基座模型上进行监督微调，让模型学会"对话"能力：

  1. 参数解析：命令行参数 > JSON 配置文件 > 代码默认值（三级优先级）
  2. 模型初始化：加载预训练 checkpoint 作为基座，可选 LoRA 低秩适配
  3. 数据加载：SFTDataset 按行偏移量加载 JSONL 多轮对话，只对 assistant 回复算损失
  4. 训练循环：前向 → masked 交叉熵损失 → 反向 → 梯度裁剪 → AdamW 权重更新
  5. 验证 & 日志：定期在验证集评估，WandB 实时记录
  6. 检查点保存：定期保存模型 + 优化器状态 + LoRA 适配器，支持中断续训

训练循环每步做的事：
    input_ids, labels = next(train_iter)   # 从 DataLoader 取一个 batch
    logits = model(input_ids)              # 前向传播
    loss = masked_cross_entropy(...)       # 只对 assistant 回复位置算损失（labels=-100 忽略）
    loss.backward()                        # 反向传播
    clip_grad_norm(max_norm=1.0)           # 梯度裁剪（防爆炸）
    optimizer.step()                       # AdamW 权重更新

与预训练 (train_pretrain.py) 的关键区别：
  - 数据：JSONL 多轮对话（按偏移量加载）vs .npy 长 token 数组（随机截取）
  - Loss：选择性损失（只学 assistant 回复）vs 全部 token 参与
  - 学习率：恒定 1e-5 vs cosine schedule 6e-4→6e-5
  - 初始化：加载预训练权重 vs 随机初始化
  - LoRA：可选（冻结基座只训练低秩适配器）vs 无

使用方式：
    python train_sft.py --config configs/sft.json
    python train_sft.py --train-data-path data/sft_train.jsonl --vocab-path ... --merges-path ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from microlm.model import TransformerLM
from microlm.model.lora import (
    apply_lora_to_model,
    get_lora_params,
    get_lora_state_dict,
    load_lora_state_dict,
    print_trainable_params,
)
from microlm.tokenizer import BPETokenizer
from microlm.training import AdamW
from microlm.training import SFTDataset
from microlm.training import load_model_state, load_checkpoint, masked_cross_entropy, save_checkpoint


def load_config_defaults(config_path: str | None) -> dict[str, object]:
    """从 JSON 配置文件加载所有超参数默认值。

    配置文件结构（六个 section）：
      - tokenizer: vocab_path, merges_path, special_tokens
      - model:     context_length, d_model, num_layers, num_heads, d_ff, vocab_size,
                    rope_theta, use_rms_norm, norm_mode, ffn_type
      - optimizer: lr, weight_decay
      - training:  batch_size, max_steps, eval_interval, save_interval, device,
                    seed, out_dir, init_checkpoint, resume
      - data:      train_data_path, valid_data_path, system_prompt_ratio, eos_token
      - logging:   wandb_project, run_name, mode
      - lora:      enabled, r, alpha, targets

    Path 类型的值在加载时直接转为 pathlib.Path 对象，方便后续使用。

    Args:
        config_path: JSON 配置文件路径，传 None 时返回空 dict（全用命令行默认值）

    Returns:
        过滤掉 None 值后的参数字典，作为 argparse 的 defaults 基准
    """
    if config_path is None:
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer = config.get("tokenizer", {})
    model = config.get("model", {})
    optimizer = config.get("optimizer", {})
    training = config.get("training", {})
    data = config.get("data", {})
    logging = config.get("logging", {})
    lora_cfg = config.get("lora", {})

    defaults: dict[str, object] = {
        "vocab_path": Path(tokenizer["vocab_path"]) if tokenizer.get("vocab_path") else None,
        "merges_path": Path(tokenizer["merges_path"]) if tokenizer.get("merges_path") else None,
        "special_tokens": tokenizer.get("special_tokens"),
        "context_length": model.get("context_length"),
        "d_model": model.get("d_model"),
        "num_heads": model.get("num_heads"),
        "num_layers": model.get("num_layers"),
        "d_ff": model.get("d_ff"),
        "vocab_size": model.get("vocab_size"),
        "rope_theta": model.get("rope_theta"),
        "use_rms_norm": model.get("use_rms_norm"),
        "norm_mode": model.get("norm_mode"),
        "ffn_type": model.get("ffn_type"),
        "lr": optimizer.get("lr"),
        "weight_decay": optimizer.get("weight_decay"),
        "batch_size": training.get("batch_size"),
        "max_steps": training.get("max_steps"),
        "eval_interval": training.get("eval_interval"),
        "save_interval": training.get("save_interval"),
        "device": training.get("device"),
        "seed": training.get("seed"),
        "out_dir": Path(training["out_dir"]) if training.get("out_dir") else None,
        "init_checkpoint": Path(training["init_checkpoint"]) if training.get("init_checkpoint") else None,
        "resume": training.get("resume"),
        "train_data_path": Path(data["train_data_path"]) if data.get("train_data_path") else None,
        "valid_data_path": Path(data["valid_data_path"]) if data.get("valid_data_path") else None,
        "system_prompt_ratio": data.get("system_prompt_ratio"),
        "eos_token": data.get("eos_token"),
        "wandb_project": logging.get("wandb_project"),
        "run_name": logging.get("run_name"),
        "wandb_mode": logging.get("mode"),
        "use_lora": lora_cfg.get("enabled", False),
        "lora_r": lora_cfg.get("r", 8),
        "lora_alpha": lora_cfg.get("alpha", 16.0),
        "lora_targets": lora_cfg.get("targets"),
    }
    return {key: value for key, value in defaults.items() if value is not None}


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    """构建命令行参数解析器，defaults 来自 JSON 配置文件。

    参数优先级：命令行 > JSON 配置文件 > 代码硬编码默认值。
    每个 add_argument 的 default= 取 defaults 中的值（若存在），
    否则使用代码硬编码的兜底值。

    参数分七组：
      - Tokenizer：  vocab_path, merges_path, special_tokens
      - 模型结构：   context_length, d_model, num_layers, num_heads, d_ff,
                      vocab_size, rope_theta, use_rms_norm, norm_mode, ffn_type
      - 优化器：     lr, weight_decay
      - 训练控制：   batch_size, max_steps, eval_interval, save_interval,
                      device, seed, out_dir, init_checkpoint, resume
      - 数据 & 对话：train_data_path, valid_data_path, system_prompt_ratio, eos_token
      - 日志：       wandb_project, run_name, wandb_mode
      - LoRA：       use_lora, lora_r, lora_alpha, lora_targets

    Args:
        defaults: load_config_defaults() 返回的配置文件参数字典

    Returns:
        配置好的 ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(description="Train MicroLM on SFT conversations.")
    parser.add_argument("--config", type=str, default=defaults.get("config"))

    parser.add_argument("--vocab-path", type=Path, default=defaults.get("vocab_path"))
    parser.add_argument("--merges-path", type=Path, default=defaults.get("merges_path"))
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=defaults.get("special_tokens"),
        help="Special token reserved while loading the tokenizer. May be passed multiple times.",
    )

    parser.add_argument("--context-length", type=int, default=defaults.get("context_length", 512))
    parser.add_argument("--d-model", type=int, default=defaults.get("d_model", 512))
    parser.add_argument("--num-heads", type=int, default=defaults.get("num_heads", 8))
    parser.add_argument("--num-layers", type=int, default=defaults.get("num_layers", 8))
    parser.add_argument("--d-ff", type=int, default=defaults.get("d_ff", 1344))
    parser.add_argument("--vocab-size", type=int, default=defaults.get("vocab_size", 6400))
    parser.add_argument("--rope-theta", type=float, default=defaults.get("rope_theta", 1000000.0))
    parser.add_argument(
        "--use-rms-norm",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("use_rms_norm", True),
    )
    parser.add_argument(
        "--norm-mode",
        type=str,
        default=defaults.get("norm_mode", "pre"),
        choices=["pre", "post"],
    )
    parser.add_argument(
        "--ffn-type",
        type=str,
        default=defaults.get("ffn_type", "swiglu"),
        choices=["swiglu", "silu"],
    )

    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-5))
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 0.1))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 2))
    parser.add_argument("--max-steps", type=int, default=defaults.get("max_steps", 100))
    parser.add_argument("--eval-interval", type=int, default=defaults.get("eval_interval", 10))
    parser.add_argument("--save-interval", type=int, default=defaults.get("save_interval", 50))
    parser.add_argument("--device", type=str, default=defaults.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--out-dir", type=Path, default=defaults.get("out_dir", Path("outputs/sft")))
    parser.add_argument("--init-checkpoint", type=Path, default=defaults.get("init_checkpoint"))
    parser.add_argument("--resume", action="store_true", default=defaults.get("resume", False))

    parser.add_argument("--train-data-path", type=Path, default=defaults.get("train_data_path"))
    parser.add_argument("--valid-data-path", type=Path, default=defaults.get("valid_data_path"))
    parser.add_argument("--system-prompt-ratio", type=float, default=defaults.get("system_prompt_ratio", 0.0))
    parser.add_argument("--eos-token", type=str, default=defaults.get("eos_token", "<|endoftext|>"))

    parser.add_argument("--wandb-project", type=str, default=defaults.get("wandb_project", "micro-lm"))
    parser.add_argument("--run-name", type=str, default=defaults.get("run_name"))
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=defaults.get("wandb_mode", "disabled"),
        choices=["online", "offline", "disabled"],
    )

    # LoRA
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=defaults.get("use_lora", False),
    )
    parser.add_argument("--lora-r", type=int, default=defaults.get("lora_r", 8))
    parser.add_argument("--lora-alpha", type=float, default=defaults.get("lora_alpha", 16.0))
    parser.add_argument(
        "--lora-targets",
        nargs="*",
        default=defaults.get("lora_targets"),
        help="Linear layer names to apply LoRA to (default: q/k/v/output_proj).",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """解析命令行参数（三级优先级：命令行 > JSON 配置 > 代码默认值）。

    分两趟解析：
      1. 先解析 --config，确定 JSON 配置文件路径
      2. 从 JSON 加载默认值，注入 ArgumentParser
      3. 再解析其余命令行参数（覆盖 JSON 默认值）

    Returns:
        合并后的参数命名空间，可直接用 args.vocab_size 等方式访问
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args()

    defaults = load_config_defaults(config_args.config)
    defaults["config"] = config_args.config
    parser = build_parser(defaults)
    return parser.parse_args(remaining)


def set_seed(seed: int) -> None:
    """固定所有随机种子，保证实验可复现。

    同时设置 PyTorch CPU 和所有 GPU 的随机种子。
    注意：SFT 没有设置 numpy 种子，因为 DataLoader 的 shuffle
    由 PyTorch 的 generator 单独控制。

    Args:
        seed: 随机种子整数
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_configured_tokenizer(args: argparse.Namespace) -> BPETokenizer:
    """从配置加载 BPE 分词器。

    从文件加载词表 (vocab) 和合并规则 (merges)，并注册特殊 token。
    默认的特殊 token 为 <|endoftext|>，命令行可通过 --special-token 多次追加。

    Args:
        args: 解析后的命令行参数，需包含 vocab_path, merges_path, special_tokens

    Returns:
        配置好的 BPETokenizer 实例

    Raises:
        ValueError: vocab_path 或 merges_path 为空
    """
    special_tokens = args.special_tokens or ["<|endoftext|>"]
    if args.vocab_path is None or args.merges_path is None:
        raise ValueError("Tokenizer vocab/merges paths are required")
    return BPETokenizer.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        special_tokens=special_tokens,
    )


def build_model(args: argparse.Namespace, device: str) -> TransformerLM:
    """根据参数实例化 TransformerLM 模型并移动到目标设备。

    模型结构参数全部从命令行/配置文件传入，不做硬编码。
    架构与预训练完全一致：RMSNorm pre-norm + SwiGLU FFN + RoPE + causal LM head。

    Args:
        args:   解析后的命令行参数，包含所有模型结构配置
        device: 目标设备字符串，如 "cuda" 或 "cpu"

    Returns:
        已移至目标设备的 TransformerLM 实例
    """
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rms_norm=bool(args.use_rms_norm),
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type,
        device=device,
    ).to(device)
    return model


def evaluate(model: TransformerLM, loader: DataLoader, device: str) -> float:
    """在验证集上评估模型损失。

    遍历整个 DataLoader，计算所有 batch 的加权平均损失。
    只对非 padding / 非 -100 的位置计算损失（与训练时一致的选择性损失）。

    和预训练 evaluate 的关键区别：这里用 masked_cross_entropy 而非
    全序列交叉熵——因为 SFT 的 labels 中 user/system 部分是 -100，
    直接算 cross_entropy 会把 -100 当成合法类别索引导致错误。

    Args:
        model:  正在训练的 TransformerLM 模型
        loader:  验证集 DataLoader
        device:  计算设备

    Returns:
        加权平均验证损失（标量 float），若全部被 mask 掉则返回 nan
    """
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss_mask = (shift_labels != -100).long()
            weight = float(loss_mask.sum().item())
            if weight == 0:
                continue
            loss = masked_cross_entropy(shift_logits, shift_labels, loss_mask)
            total_loss += loss.item() * weight
            total_weight += weight
    model.train()
    return total_loss / total_weight if total_weight > 0 else float("nan")


def iter_batches(loader: DataLoader):
    """将有限 DataLoader 包装为无限迭代器。

    训练循环需要持续不断的 batch 流，而 DataLoader 遍历完一轮后会 StopIteration。
    用这个生成器包装后，DataLoader 耗尽时自动从头再来，训练循环只需 `next(train_iter)`。

    注意：每轮 epoch 之间没有显式边界（主循环只记 step 不记 epoch），
    这对 SFT 来说没问题——数据增强（system prompt 随机注入）提供了足够多样性。

    Args:
        loader: PyTorch DataLoader 实例

    Yields:
        (input_ids, labels) 元组，流式无限产出
    """
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    """SFT 训练主入口。

    完整流程：
      1. 解析参数，创建输出目录，固定随机种子
      2. 加载 BPE 分词器（词表 + 合并规则 + 特殊 token）
      3. 构建 TransformerLM 模型，加载预训练 checkpoint 作为基座
      4. 若词表大小超过模型 vocab_size，自动扩展 embedding 和 lm_head（resize）
      5. 可选应用 LoRA：冻结基座权重，只训练低秩适配器（大幅减少显存）
      6. 构建 SFTDataset + DataLoader（训练集和验证集）
      7. 初始化 AdamW 优化器（LoRA 模式下只优化适配器参数）
      8. 若存在检查点则恢复（支持中断续训）
      9. 初始化 WandB 日志（可选）
     10. 训练循环（见下方详细注释）
     11. 保存最终检查点 + LoRA 适配器，关闭 WandB

    训练循环每步做的事：
      - 从无限 DataLoader 取一个 batch（input_ids, labels）
      - labels 中 user/system 部分 = -100，只 assistant 回复区域有值
      - 前向 → shift → masked 交叉熵损失（自动忽略 -100 位置）
      - 反向传播 → 梯度裁剪 → AdamW 权重更新
      - 每 eval_interval 步在验证集评估 + 日志输出
      - 每 save_interval 步保存检查点（模型+优化器+LoRA 适配器）
    """
    args = parse_args()
    if args.train_data_path is None or args.valid_data_path is None:
        raise ValueError("Both train-data-path and valid-data-path are required")
    if not args.train_data_path.exists():
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not args.valid_data_path.exists():
        raise FileNotFoundError(f"Validation data not found at {args.valid_data_path}")

    # ---- 基础设置：创建输出目录 + 固定随机种子 ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # ---- 加载 Tokenizer + 构建模型 ----
    tokenizer = load_configured_tokenizer(args)
    model = build_model(args, args.device)

    # ═══════════════════════════════════════════════════════════
    # 加载预训练基座模型权重（必须在 LoRA 之前——checkpoint 里是原始 Linear 层参数）
    # ═══════════════════════════════════════════════════════════
    if args.init_checkpoint is not None and not args.resume:
        load_model_state(str(args.init_checkpoint), model)
        print(f"Loaded init checkpoint from {args.init_checkpoint}")

        # 词表扩容：tokenizer 包含特殊 token 后词表可能大于模型的 vocab_size
        # 此时需扩展 embedding 和 lm_head（新 token 的 embedding 初始化为 0）
        actual_vocab_size = len(tokenizer.id_to_vocab)
        if actual_vocab_size > args.vocab_size:
            old_emb = model.token_embeddings.weight.data  # [old_vocab, d_model]
            new_emb = torch.zeros(actual_vocab_size, args.d_model, device=old_emb.device, dtype=old_emb.dtype)
            new_emb[:old_emb.shape[0]] = old_emb
            model.token_embeddings.weight = nn.Parameter(new_emb)

            old_head = model.lm_head.weight.data  # [old_vocab, d_model]
            new_head = torch.zeros(actual_vocab_size, args.d_model, device=old_head.device, dtype=old_head.dtype)
            new_head[:old_head.shape[0]] = old_head
            model.lm_head.weight = nn.Parameter(new_head)
            print(f"Resized vocab: {args.vocab_size} -> {actual_vocab_size} (for special tokens)")

    # ═══════════════════════════════════════════════════════════
    # LoRA 低秩适配（可选）：冻结基座权重，只训练低秩矩阵 A 和 B
    # 参数量从 ~31.7M 降到 ~300K，显存和训练时间大幅减少
    # ═══════════════════════════════════════════════════════════
    if args.use_lora:
        apply_lora_to_model(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            target_names=args.lora_targets,
        )
        print(f"LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}")
        if args.lora_targets:
            print(f"LoRA targets: {args.lora_targets}")
        print_trainable_params(model)
    else:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {num_params:,}")

    # ---- 保存模型配置和训练参数（用于推理时重建模型） ----
    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "use_rms_norm": bool(args.use_rms_norm),
        "norm_mode": args.norm_mode,
        "ffn_type": args.ffn_type,
    }
    with (args.out_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    resolved_config = vars(args).copy()
    resolved_config["train_data_path"] = str(args.train_data_path)
    resolved_config["valid_data_path"] = str(args.valid_data_path)
    resolved_config["out_dir"] = str(args.out_dir)
    resolved_config["vocab_path"] = str(args.vocab_path) if args.vocab_path is not None else None
    resolved_config["merges_path"] = str(args.merges_path) if args.merges_path is not None else None
    resolved_config["init_checkpoint"] = str(args.init_checkpoint) if args.init_checkpoint is not None else None
    resolved_config["use_lora"] = args.use_lora
    resolved_config["lora_r"] = args.lora_r
    resolved_config["lora_alpha"] = args.lora_alpha
    resolved_config["lora_targets"] = args.lora_targets
    with (args.out_dir / "resolved_train_config.json").open("w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2, ensure_ascii=False)

    # ---- 构建 SFT 数据集和 DataLoader ----
    # 训练集：按 system_prompt_ratio 随机注入 system prompt
    train_ds = SFTDataset(
        args.train_data_path,
        tokenizer=tokenizer,
        max_length=args.context_length,
        system_prompt_ratio=args.system_prompt_ratio,
        seed=args.seed,
        eos_token=args.eos_token,
    )
    # 验证集：system_prompt_ratio=0（不做随机增强，评估更稳定）
    valid_ds = SFTDataset(
        args.valid_data_path,
        tokenizer=tokenizer,
        max_length=args.context_length,
        system_prompt_ratio=0.0,
        seed=args.seed,
        eos_token=args.eos_token,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- 初始化优化器：LoRA 模式下只优化适配器参数 ----
    optimizer = AdamW(
        get_lora_params(model) if args.use_lora else model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- 检查是否有中断的检查点，恢复训练 ----
    ckpt_path = args.out_dir / "ckpt.pt"
    start_step = 0
    if args.resume and ckpt_path.exists():
        start_step = load_checkpoint(str(ckpt_path), model, optimizer)
        print(f"Resuming SFT from step {start_step}")
        if args.use_lora:
            lora_path = args.out_dir / "lora_adaptor.pt"
            if lora_path.exists():
                load_lora_state_dict(model, torch.load(lora_path, map_location=args.device, weights_only=True))
                print(f"Loaded LoRA adaptor from {lora_path}")
    # init_checkpoint already loaded above (before LoRA application)

    wandb = None
    if args.wandb_mode != "disabled":
        import wandb as wandb_module

        wandb = wandb_module
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            mode=args.wandb_mode,
            config=resolved_config,
        )

    # ---- 训练日志：JSONL 格式，每行一次评估记录 ----
    log_path = args.out_dir / "train_log.jsonl"
    # 无限 batch 流：训练循环用 next(train_iter) 持续取数据
    train_iter = iter_batches(train_loader)

    # ═══════════════════════════════════════════════════════════
    # 训练循环
    # ═══════════════════════════════════════════════════════════
    for step in range(start_step, args.max_steps):
        model.train()

        # ① 取一个 batch：input_ids 和 labels 均被 padding 到 context_length
        #    labels 中 user/system 部分 = -100，只 assistant 回复区域有值
        input_ids, labels = next(train_iter)
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # ② 前向传播 + 选择性损失计算
        logits = model(input_ids)                                          # 前向传播
        shift_logits = logits[:, :-1, :]                                   # 预测位置 1..T
        shift_labels = labels[:, 1:]                                       # 标签位置 1..T
        loss_mask = (shift_labels != -100).long()                          # 只对 assistant 区域算损失
        loss = masked_cross_entropy(shift_logits, shift_labels, loss_mask) # 自动忽略 -100 位置

        # ③ 反向传播 + 梯度裁剪 + 权重更新
        optimizer.zero_grad(set_to_none=True)                              # 清空上轮梯度（set_to_none 比 zero_() 更高效）
        loss.backward()                                                    # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)            # 梯度裁剪（防爆炸）
        optimizer.step()                                                   # AdamW 权重更新

        completed_step = step + 1

        # ④ 定期在验证集评估 + 日志
        if completed_step % args.eval_interval == 0 or completed_step == args.max_steps:
            val_loss = evaluate(model, valid_loader, args.device)
            print(f"Step {completed_step}: train_loss {loss.item():.4f}, val_loss {val_loss:.4f}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "step": completed_step,
                            "train_loss": float(loss.item()),
                            "val_loss": float(val_loss),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if wandb is not None:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "val/loss": val_loss,
                        "step": completed_step,
                    }
                )

        # ⑤ 定期保存检查点（模型+优化器）+ LoRA 适配器
        if completed_step % args.save_interval == 0 or completed_step == args.max_steps:
            save_checkpoint(model, optimizer, iteration=completed_step, out=str(ckpt_path))
            if args.use_lora:
                torch.save(get_lora_state_dict(model), args.out_dir / "lora_adaptor.pt")

    save_checkpoint(model, optimizer, iteration=args.max_steps, out=str(args.out_dir / "ckpt_final.pt"))
    if args.use_lora:
        torch.save(get_lora_state_dict(model), args.out_dir / "lora_adaptor.pt")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
