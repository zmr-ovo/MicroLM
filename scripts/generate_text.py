"""文本生成推理脚本 —— 从训练好的 MicroLM checkpoint 生成文本。

本脚本整合 tokenizer + 模型 + prompt 构建，构成完整推理管线：

  1. 参数解析：支持 JSON 配置文件或命令行直接指定模型超参数
  2. 模型重建：根据 model_config.json 实例化 TransformerLM，加载 checkpoint 权重
  3. Prompt 构建：支持纯文本续写 / 命令行 JSON 对话 / 文件对话 三种输入
  4. 生成：支持两种解码策略——
       - greedy（temperature=0）：每步取概率最大的 token，确定性输出
       - temperature + top-p 采样：通过 KV Cache 加速的自回归生成
  5. 输出：完整文本 / 仅新生成部分，可选打印 token ID

与 chat.py（交互式多轮对话）的区别：
  - generate_text.py：单次生成，批处理友好，适合脚本调用和评测
  - chat.py：交互式 REPL，维护对话历史，多轮上下文自动拼接

使用方式：
    python generate_text.py --checkpoint-path ckpt.pt --config-path model_config.json --prompt "你好"
    python generate_text.py --checkpoint-path ckpt.pt --prompt "从前有座山" --temperature 0.8 --top-p 0.9
    python generate_text.py --checkpoint-path ckpt.pt --conversations-json '[{"role":"user","content":"你好"}]'
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

from microlm.model import TransformerLM
from microlm.inference import resolve_generation_prompt
from microlm.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    模型超参数可从两个来源获取：
      - --config-path：JSON 配置文件（model_config.json），包含所有模型结构参数
      - 命令行直接指定：--d-model 512 --num-layers 8 ...
      两者不能同时缺失，命令行值会覆盖配置文件中的对应值。

    推理特有参数：
      - --temperature 0：greedy decoding（确定性），>0：随机采样
      - --top-p 0.9：nucleus sampling，保留累积概率不超过 0.9 的 token
      - --print-new-text-only：只打印新生成部分（不打印 prompt）
      - --show-token-ids：同时输出 token ID（调试用）

    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(
        description="Generate text from a trained MicroLM checkpoint."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to a model checkpoint or raw state_dict file.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Optional JSON config file with model hyperparameters.",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("output/tinystories_bpe_10k/vocab.json"),
    )
    parser.add_argument(
        "--merges-path",
        type=Path,
        default=Path("output/tinystories_bpe_10k/merge.txt"),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Prompt string used for generation.",
    )
    parser.add_argument(
        "--conversations-json",
        type=str,
        default=None,
        help="JSON string containing a chat-style conversations list.",
    )
    parser.add_argument(
        "--conversations-path",
        type=Path,
        default=None,
        help="Path to a JSON file containing a chat-style conversations list.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of newly generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling threshold.",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=None,
        help="Special token reserved by the tokenizer. May be passed multiple times.",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="Optional special token string that stops generation when produced.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on. Use 'auto' to prefer CUDA and fall back to CPU.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Model parameter dtype used at inference time.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--show-token-ids",
        action="store_true",
        help="Print prompt and generated token ids alongside decoded text.",
    )
    parser.add_argument(
        "--print-new-text-only",
        action="store_true",
        help="Print only the newly generated suffix instead of the full decoded sequence.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        help="Override context length when no config file is provided.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        help="Override d_model when no config file is provided.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Override num_layers when no config file is provided.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        help="Override num_heads when no config file is provided.",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        help="Override d_ff when no config file is provided.",
    )
    parser.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help="Override RoPE theta when no config file is provided.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """将 dtype 名称字符串转换为 PyTorch dtype 对象。

    Args:
        dtype_name: "float32" / "float16" / "bfloat16"

    Returns:
        对应的 torch.dtype 枚举值
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_name]


def resolve_model_dtype(dtype_name: str, device: str) -> torch.dtype:
    """根据设备能力解析实际可用的 dtype。

    CPU 平台对 float16/bfloat16 支持有限，需要自动降级：
      - float16 on CPU → 强制降为 float32（CPU 不支持 float16 推理）
      - bfloat16 on CPU → 若无 MKL-DNN 支持则降为 float32
      - GPU 上所有 dtype 原样使用

    Args:
        dtype_name: 用户指定的 dtype 名称 ("float32"/"float16"/"bfloat16")
        device:     目标设备 ("cuda"/"cpu")

    Returns:
        实际可用的 torch.dtype
    """
    dtype = get_torch_dtype(dtype_name)
    if dtype == torch.float16 and device == "cpu":
        return torch.float32
    if dtype == torch.bfloat16 and device == "cpu" and not torch.backends.mkldnn.is_available():
        return torch.float32
    return dtype


def resolve_device(device_arg: str) -> str:
    """解析设备参数，自动检测可用设备。

    "auto" 模式下的探测逻辑：
      1. CUDA 不可用（无 GPU 驱动 / 无 CUDA toolkit）→ "cpu"
      2. CUDA 可用但显存不足（分配 1 元素 tensor 失败）→ "cpu"
      3. CUDA 可用且正常 → "cuda"

    Args:
        device_arg: "auto" / "cuda" / "cpu"

    Returns:
        实际使用的设备字符串
    """
    if device_arg != "auto":
        return device_arg

    if not torch.cuda.is_available():
        return "cpu"

    try:
        torch.empty(1, device="cuda")     # 试探性分配显存
        return "cuda"
    except Exception:
        return "cpu"


def load_model_config(args: argparse.Namespace, vocab_size: int) -> dict[str, int | float]:
    """加载模型结构配置，支持 JSON 文件或命令行参数两种来源。

    优先使用 JSON 配置文件（model_config.json，由训练脚本自动生成），
    缺失时从命令行参数拼接。vocab_size 由 tokenizer 词表大小决定，
    不来自配置文件（因为 tokenizer 可能被扩展过如加了特殊 token）。

    Args:
        args:       解析后的命令行参数
        vocab_size: tokenizer 的实际词表大小

    Returns:
        模型构造参数字典，包含 vocab_size, context_length, d_model,
        num_layers, num_heads, d_ff, rope_theta

    Raises:
        ValueError: 既没有配置文件也没有足够的命令行参数
    """
    if args.config_path is not None:
        with args.config_path.open("r", encoding="utf-8") as f:
            raw_config = json.load(f)
        return {
            "vocab_size": int(raw_config.get("vocab_size", vocab_size)),
            "context_length": int(raw_config["context_length"]),
            "d_model": int(raw_config["d_model"]),
            "num_layers": int(raw_config["num_layers"]),
            "num_heads": int(raw_config["num_heads"]),
            "d_ff": int(raw_config["d_ff"]),
            "rope_theta": float(raw_config.get("rope_theta", 10000.0)),
        }

    required_fields = {
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
    }
    missing = [name for name, value in required_fields.items() if value is None]
    if missing:
        raise ValueError(
            "Missing model hyperparameters. Provide --config-path or all of: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )

    return {
        "vocab_size": vocab_size,
        "context_length": int(args.context_length),
        "d_model": int(args.d_model),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "d_ff": int(args.d_ff),
        "rope_theta": float(args.rope_theta),
    }


def normalize_state_dict_keys(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """规范化 state_dict 的 key 名称。

    处理 torch.compile 产生的 "_orig_mod." 前缀：
      - 编译保存的检查点中 key 形如 "_orig_mod.lm_head.weight"
      - 去掉前缀恢复为 "lm_head.weight"，匹配未编译模型的参数名

    Args:
        state_dict: 可能带 _orig_mod 前缀的 state_dict

    Returns:
        key 名称规范化后的 state_dict
    """
    normalized = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]       # 去掉前缀
        normalized[key] = value
    return normalized


def load_state_dict(checkpoint_path: Path, device: str) -> OrderedDict[str, torch.Tensor]:
    """从检查点文件加载模型权重。

    兼容两种检查点格式：
      - 训练检查点（含 model_state_dict / optimizer_state_dict / iteration）：
        自动提取 model_state_dict 部分
      - 裸 state_dict 文件（只有模型权重）：直接使用

    Args:
        checkpoint_path: 检查点文件路径（.pt）
        device:          加载到的目标设备

    Returns:
        规范化 key 后的 state_dict

    Raises:
        TypeError: 文件格式无法识别
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 训练检查点：提取 model_state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint                      # 裸 state_dict

    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}")
    return normalize_state_dict_keys(OrderedDict(state_dict))


def sample_greedy_or_temperature(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """选择解码策略：greedy（确定性）或 temperature+top-p 采样。

    两种模式：
      - temperature == 0.0 → Greedy：每步取 argmax(logits)，无随机性。
        不需要 KV Cache（每次拼接完整历史重新前向），简单但慢。
      - temperature > 0 → 调用 model.generate()：内部使用 KV Cache 加速
        的 temperature + top-p 采样，效率更高。

    Args:
        model:          已加载权重的 TransformerLM 模型
        prompt_ids:     输入 prompt 的 token ID 张量 [1, seq_len]
        max_new_tokens: 最多生成多少个新 token
        eos_token_id:   EOS token ID，遇到则提前停止
        temperature:     采样温度（0 = greedy）
        top_p:           nucleus sampling 阈值

    Returns:
        完整序列 token IDs [1, prompt_len + 生成长度]
    """
    if temperature == 0.0:
        model.eval()
        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            idx_cond = generated[:, -model.context_length :]
            logits = model(idx_cond)[:, -1, :]
            if top_p < 1.0:
                logits = model._top_p_filter(logits, top_p)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return generated

    return model.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
    )


def main() -> None:
    """文本生成主入口。

    完整流程：
      1. 解析参数 + 固定随机种子
      2. 自动检测设备（GPU/CPU）
      3. 加载 BPE 分词器（词表 + 合并规则 + 特殊 token）
      4. 重建模型结构（从 model_config.json 或命令行参数）
      5. 加载检查点权重（兼容训练检查点和裸 state_dict）
      6. 解析 EOS token ID
      7. 构建生成 prompt（支持纯文本 / JSON 对话 / 文件对话）
      8. Tokenize prompt → 送入模型生成
      9. Decode → 输出文本（完整或仅新生成部分）

    输出控制：
      - --print-new-text-only：只输出模型生成的部分（不含 prompt）
      - --show-token-ids：同时打印 token ID 列表（调试用）
    """
    args = parse_args()
    torch.manual_seed(args.seed)

    # ---- 设备检测 + Tokenizer 加载 ----
    device = resolve_device(args.device)
    special_tokens = args.special_tokens or ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        special_tokens=special_tokens,
    )

    # ---- 重建模型结构 + 加载权重 ----
    config = load_model_config(args, vocab_size=len(tokenizer.id_to_vocab))
    dtype = resolve_model_dtype(args.dtype, device)

    model = TransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
        device=device,
        dtype=dtype,
    ).to(device)

    state_dict = load_state_dict(args.checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    # ---- 解析 EOS token（用于自动停止生成） ----
    eos_token_id = None
    if args.eos_token is not None:
        eos_token_bytes = args.eos_token.encode("utf-8")
        if eos_token_bytes not in tokenizer.vocab_to_id:
            raise ValueError(f"EOS token {args.eos_token!r} is not in the tokenizer vocab")
        eos_token_id = tokenizer.vocab_to_id[eos_token_bytes]

    # ---- 构建 prompt → tokenize → tensor ----
    generation_prompt = resolve_generation_prompt(
        prompt=args.prompt,
        conversations_json=args.conversations_json,
        conversations_path=args.conversations_path,
    )

    prompt_token_ids = tokenizer.encode(generation_prompt)
    if not prompt_token_ids:
        raise ValueError("Prompt encodes to an empty token sequence. Provide a non-empty prompt.")

    prompt_tensor = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)

    # ---- 生成 ----
    with torch.no_grad():
        generated = sample_greedy_or_temperature(
            model=model,
            prompt_ids=prompt_tensor,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    # ---- Decode + 输出 ----
    full_ids = generated[0].tolist()
    new_ids = full_ids[len(prompt_token_ids) :]          # 只取新生成的部分
    full_text = tokenizer.decode(full_ids)
    new_text = tokenizer.decode(new_ids)

    if args.show_token_ids:
        print(f"prompt_token_ids={prompt_token_ids}")
        print(f"generated_token_ids={new_ids}")

    if args.print_new_text_only:
        print(new_text)
    else:
        print(full_text)


if __name__ == "__main__":
    main()
