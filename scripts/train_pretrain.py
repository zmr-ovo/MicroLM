"""语言模型预训练脚本 —— 完整的训练流水线。

本脚本整合项目的所有组件，构成一条端到端的预训练流水线：

  1. 参数解析：命令行参数 > JSON 配置文件 > 代码默认值（三级优先级）
  2. 数据加载：读取 token 化语料的 .npy 或 .memmap 文件
  3. 模型构建：根据参数实例化 TransformerLM（31.7M 参数）
  4. 训练循环：随机采样 → 前向 → 交叉熵损失 → 反向传播 → 权重更新
  5. 验证 & 日志：每 100 步在验证集评估，WandB 实时记录
  6. 检查点保存：每 1000 步保存模型+优化器状态，支持中断续训

训练循环的核心结构（每一步 iteration）：
    lr = cosine_schedule(iter)              # 更新学习率
    x, y = get_batch(train_data)            # 随机采样一个批次
    logits = model(x)                       # 前向传播
    loss = cross_entropy(logits, y)         # 下一个 token 预测损失
    loss.backward()                         # 反向传播
    clip_gradient_norm(max_norm=1.0)        # 梯度裁剪（防爆炸）
    optimizer.step()                        # AdamW 权重更新

使用方式：
    python train_pretrain.py --config configs/base.json
    python train_pretrain.py --train_data_path data/train.npy --valid_data_path data/valid.npy
"""

import argparse
import json
import os

import torch
import numpy as np
import wandb

from microlm.model import TransformerLM
from microlm.training import AdamW
from microlm.training import gradient_clipping as clip_gradient_norm
from microlm.training import learning_rate_schedule as get_lr_cosine_schedule
from microlm.training import get_batch
from microlm.training import save_checkpoint, load_checkpoint
from microlm.training import cross_entropy


def load_config_defaults(config_path: str | None) -> dict[str, object]:
    """从 JSON 配置文件加载所有超参数默认值。

    配置文件结构（五个 section）：
      - model:    vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta, ...
      - optimizer: lr, warmup_iters, min_lr, max_norm, weight_decay
      - training:  batch_size, max_iters, out_dir, device, seed
      - data:      train_data_path, valid_data_path
      - logging:   run_name, wandb_project, mode

    特殊处理：
      - use_rms_norm=False → 转为布尔开关 --no_rms_norm
      - rope_theta=None    → 转为布尔开关 --no_rope
      - 值为 None 的键不会进入 defaults（由 argparse 的 default 兜底）

    Args:
        config_path: JSON 配置文件路径，传 None 时返回空 dict（全用命令行默认值）

    Returns:
        过滤掉 None 值后的参数字典，作为 argparse 的 defaults 基准
    """
    if config_path is None:
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = config.get("model", {})
    optimizer = config.get("optimizer", {})
    training = config.get("training", {})
    data = config.get("data", {})
    logging = config.get("logging", {})

    defaults: dict[str, object] = {
        "batch_size": training.get("batch_size"),
        "context_length": model.get("context_length"),
        "d_model": model.get("d_model"),
        "num_heads": model.get("num_heads"),
        "num_layers": model.get("num_layers"),
        "d_ff": model.get("d_ff"),
        "vocab_size": model.get("vocab_size"),
        "norm_mode": model.get("norm_mode"),
        "ffn_type": model.get("ffn_type"),
        "lr": optimizer.get("lr"),
        "max_iters": training.get("max_iters"),
        "warmup_iters": optimizer.get("warmup_iters"),
        "min_lr": optimizer.get("min_lr"),
        "max_norm": optimizer.get("max_norm"),
        "weight_decay": optimizer.get("weight_decay"),
        "train_data_path": data.get("train_data_path"),
        "valid_data_path": data.get("valid_data_path"),
        "out_dir": training.get("out_dir"),
        "device": training.get("device"),
        "run_name": logging.get("run_name"),
        "wandb_project": logging.get("wandb_project"),
        "wandb_mode": logging.get("mode"),
        "seed": training.get("seed"),
        "rope_theta": model.get("rope_theta"),
    }

    # 布尔型参数的映射：JSON 中的 False/None → argparse 的 --no_xxx 开关
    if model.get("use_rms_norm") is False:
        defaults["no_rms_norm"] = True
    if model.get("rope_theta") is None:
        defaults["no_rope"] = True

    # 剔除 None 值，让 argparse 的 default= 参数接管
    return {key: value for key, value in defaults.items() if value is not None}


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    """构建命令行参数解析器，defaults 来自 JSON 配置文件。

    参数优先级：命令行 > JSON 配置文件 > 代码硬编码默认值。
    每个 add_argument 的 default= 取 defaults 中的值（若存在），
    否则使用代码硬编码的兜底值。

    参数分四组：
      - 模型结构：vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta, ...
      - 优化器：   lr, warmup_iters, min_lr, max_norm, weight_decay
      - 数据 & 输出：train_data_path, valid_data_path, out_dir, device
      - 日志：     run_name, wandb_project, wandb_mode

    Args:
        defaults: load_config_defaults() 返回的配置文件参数字典

    Returns:
        配置好的 ArgumentParser 实例
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=defaults.get("config"))

    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 32))
    parser.add_argument("--context_length", type=int, default=defaults.get("context_length", 256))
    parser.add_argument("--d_model", type=int, default=defaults.get("d_model", 512))
    parser.add_argument("--num_heads", type=int, default=defaults.get("num_heads", 8))
    parser.add_argument("--num_layers", type=int, default=defaults.get("num_layers", 4))
    parser.add_argument("--d_ff", type=int, default=defaults.get("d_ff", 2048))
    parser.add_argument("--vocab_size", type=int, default=defaults.get("vocab_size", 10000))

    parser.add_argument(
        "--no_rms_norm",
        action="store_true",
        default=defaults.get("no_rms_norm", False),
        help="Disable RMSNorm completely",
    )
    parser.add_argument(
        "--norm_mode",
        type=str,
        default=defaults.get("norm_mode", "pre"),
        choices=["pre", "post"],
        help="Normalization placement",
    )
    parser.add_argument(
        "--no_rope",
        action="store_true",
        default=defaults.get("no_rope", False),
        help="Disable Rotary Positional Embeddings",
    )
    parser.add_argument(
        "--ffn_type",
        type=str,
        default=defaults.get("ffn_type", "swiglu"),
        choices=["swiglu", "silu"],
        help="Type of Feed_Forward Network",
    )
    parser.add_argument("--rope_theta", type=float, default=defaults.get("rope_theta", 10000.0))

    parser.add_argument("--lr", type=float, default=defaults.get("lr", 6e-4))
    parser.add_argument("--max_iters", type=int, default=defaults.get("max_iters", 10000))
    parser.add_argument("--warmup_iters", type=int, default=defaults.get("warmup_iters", 1000))
    parser.add_argument("--min_lr", type=float, default=defaults.get("min_lr", 6e-5))
    parser.add_argument("--max_norm", type=float, default=defaults.get("max_norm", 1.0))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 0.1))

    parser.add_argument("--train_data_path", type=str, required="train_data_path" not in defaults, default=defaults.get("train_data_path"))
    parser.add_argument("--valid_data_path", type=str, required="valid_data_path" not in defaults, default=defaults.get("valid_data_path"))
    parser.add_argument("--out_dir", type=str, default=defaults.get("out_dir", "out"))
    parser.add_argument("--device", type=str, default=defaults.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))

    parser.add_argument("--run_name", type=str, default=defaults.get("run_name"), help="WandB run name")
    parser.add_argument("--wandb_project", type=str, default=defaults.get("wandb_project", "micro-lm"))
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=defaults.get("wandb_mode", "online"),
        choices=["online", "offline", "disabled"],
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

    同时设置 numpy、PyTorch CPU 和所有 GPU 的随机种子。
    cudnn 的确定性未强制（会损失性能），仅保证 Python 层采样一致。

    Args:
        seed: 随机种子整数
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_token_data(path: str) -> np.ndarray:
    """加载 token 化语料文件，返回 numpy 数组（支持 .npy 和 .memmap）。

    大语料（GB 级）使用 np.memmap 做内存映射——数据留在磁盘上，
    只在切片访问时才读入对应页面，避免撑爆 RAM。小语料直接用 np.load。

    .npy 文件从扩展名判别；无扩展名或其他后缀一律按 uint16 memmap 读取。

    Args:
        path: token 化数据的文件路径

    Returns:
        token ID 的一维 numpy 数组，形状 (total_tokens,)
    """
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.uint16, mode="r")


def main():
    """预训练主入口。

    完整流程：
      1. 解析参数，创建输出目录，固定随机种子
      2. 加载训练/验证数据（numpy memmap 避免大语料撑爆内存）
      3. 构建 TransformerLM 模型，打印参数量并保存配置
      4. 若存在检查点则恢复（支持中断续训）
      5. 初始化 AdamW 优化器 + WandB 日志
      6. 训练循环（见下方详细注释）
      7. 保存最终检查点，关闭 WandB

    训练循环每步做的事：
      - 学习率按 cosine schedule 衰减（前 warmup_iters 步线性预热）
      - 随机采样 → 前向 → 交叉熵损失 → 反向 → 梯度裁剪（不是截断单个数，是整体等比缩放） → 权重更新
      - 每 100 步评估验证损失 + 日志输出
      - 每 1000 步保存检查点（ckpt.pt）
    """
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"Validation data not found at {args.valid_data_path}")

    train_data = load_token_data(args.train_data_path)
    val_data = load_token_data(args.valid_data_path)

    print(f"训练集大小： {len(train_data)} tokens")
    print(f"验证集大小 {len(val_data)} tokens")

    actual_rope_theta = None if args.no_rope else args.rope_theta
    use_rms_norm = not args.no_rms_norm

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=actual_rope_theta,
        use_rms_norm=use_rms_norm,
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type,
        device=args.device,
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")
    print(f"Model Config: Norm={args.norm_mode}, UseNorm={use_rms_norm}, FFN={args.ffn_type}, RoPE={not args.no_rope}")

    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": actual_rope_theta if actual_rope_theta is not None else 10000.0,
        "use_rms_norm": use_rms_norm,
        "norm_mode": args.norm_mode,
        "ffn_type": args.ffn_type,
    }
    with open(os.path.join(args.out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    resolved_config = vars(args).copy()
    with open(os.path.join(args.out_dir, "resolved_train_config.json"), "w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- 检查是否有中断的检查点，恢复训练 ----
    start_iter = 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter = load_checkpoint(ckpt_path, model, optimizer)
        print(f"Resuming from iteration {start_iter}")

    # ---- 初始化 WandB 在线实验追踪 ----
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode,
        config=resolved_config,
    )

    # ═══════════════════════════════════════════════════════════
    # 训练循环
    # ═══════════════════════════════════════════════════════════
    for it in range(start_iter, args.max_iters):
        # ① 余弦学习率调度：前 warmup_iters 步从 0 线性升至 lr，之后余弦衰减至 min_lr
        lr = get_lr_cosine_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ② 训练一步：采样 → 前向 → 损失 → 反向 → 梯度裁剪 → 权重更新
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)                                                # 前向传播
        loss = cross_entropy(logits, y)                                  # 下一个 token 预测损失
        optimizer.zero_grad()                                            # 清空上轮梯度
        loss.backward()                                                  # 反向传播
        clip_gradient_norm(model.parameters(), args.max_norm)            # 梯度裁剪（防爆炸）
        optimizer.step()                                                 # AdamW 权重更新

        # ③ 每 100 步在验证集评估 + 日志
        if it % 100 == 0 or it == args.max_iters - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, args.batch_size, args.context_length, args.device)
                v_logits = model(vx)
                v_loss = cross_entropy(v_logits, vy)
                print(f"Iter {it}: train_loss {loss.item():.4f}, val_loss {v_loss.item():.4f}, lr {lr:.2e}")
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "val/loss": v_loss.item(),
                        "lr": lr,
                        "iter": it + 1,
                    }
                )

        # ④ 每 1000 步保存检查点（覆盖写入，始终保留最新）
        if it % 1000 == 0 and it > 0:
            save_checkpoint(model, optimizer, it, ckpt_path)

    # 训练结束：保存最终检查点
    save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "ckpt_final.pt"))
    wandb.finish()

if __name__ == "__main__":
    main()
