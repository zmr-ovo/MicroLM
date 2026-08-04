"""MicroLM 交互式多轮对话聊天（REPL）。

本脚本构建完整的聊天体验，支持：
  - SFT / LoRA 检查点自动加载与模型重建
  - 多轮对话历史管理（自动拼接上下文 + context-length 感知截断）
  - 运行时参数调整（/temp、/topp、/system、/clear 等 REPL 命令）
  - 会话日志持久化（JSONL 格式记录每轮对话）
  - Unicode surrogate 字符过滤（小模型可能生成无效码点导致 re-encode 崩溃）

与 generate_text.py 的关键区别：
  - generate_text.py：单次生成，无状态，适合脚本调用和批处理
  - chat.py：交互式 REPL，维护完整对话历史，每轮自动拼接上下文重新生成

核心类 ChatSession 封装了完整的对话生命周期：
  1. 用户输入 → 追加到 conversations 历史
  2. 历史 + system prompt → render → tokenize
  3. 超长时自动截断最早轮次（保留用户轮配对删除）
  4. 送入模型生成 → decode → 清洗 → 追加到历史
  5. 记录到日志文件（可选）

使用方式：
    # SFT checkpoint
    python scripts/chat.py \
        --checkpoint-path outputs/sft_baseline/ckpt_final.pt \
        --config-path outputs/sft_baseline/model_config.json \
        --vocab-path outputs/tokenizer_full_clean/vocab.json \
        --merges-path outputs/tokenizer_full_clean/merge.txt \
        --eos-token "</s>"

    # LoRA checkpoint
    python scripts/chat.py \
        --checkpoint-path outputs/pretrain_full_corpus/ckpt_final.pt \
        --lora-path outputs/sft_lora/lora_adaptor.pt \
        --config-path outputs/sft_baseline/model_config.json \
        --vocab-path outputs/tokenizer_full_clean/vocab.json \
        --merges-path outputs/tokenizer_full_clean/merge.txt \
        --eos-token "</s>"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch

from microlm.model import TransformerLM, apply_lora_to_model, load_lora_state_dict, merge_lora
from microlm.tokenizer import BPETokenizer
from microlm.training.sft import ROLE_MARKERS


def _remove_surrogates(text: str) -> str:
    """移除 Unicode surrogate 字符（U+D800–U+DFFF），防止 re-encode 崩溃。

    小模型可能生成 token 序列解码出无意义的 surrogate 码点。
    这些字符在下轮拼接上下文、重新 encode 时会抛出 UnicodeEncodeError。
    它们不携带语义信息，安全移除即可。
    """
    # Python's regex module supports \p{Cs} (surrogate code points)
    import regex as re
    return re.sub(r"[\ud800-\udfff]", "", text)


# ---------------------------------------------------------------------------
# 模型 / Tokenizer 加载辅助函数（与 generate_text.py 共用部分逻辑）
# ---------------------------------------------------------------------------

def resolve_device(device_arg: str) -> str:
    """解析设备参数，"auto" 时自动检测 GPU/CPU 可用性。

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


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """dtype 名称字符串 → PyTorch dtype 对象。

    Args:
        dtype_name: "float32" / "float16" / "bfloat16"

    Returns:
        对应的 torch.dtype
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_name]


def resolve_model_dtype(dtype_name: str, device: str) -> torch.dtype:
    """根据设备能力解析实际可用的 dtype（CPU 不支持 float16/bf16 时降级）。"""
    dtype = get_torch_dtype(dtype_name)
    if dtype == torch.float16 and device == "cpu":
        return torch.float32
    if dtype == torch.bfloat16 and device == "cpu" and not torch.backends.mkldnn.is_available():
        return torch.float32
    return dtype


def load_model_config(config_path: Path, vocab_size: int) -> dict:
    """从 JSON 配置文件加载模型结构参数（model_config.json）。

    与 generate_text.py 同名函数的关键区别：
      - chat.py 中 config_path 是必需参数（通过 --config-path），不兜底命令行
      - vocab_size 由 tokenizer 决定（可能含特殊 token 导致词表扩容）

    Args:
        config_path: model_config.json 文件路径
        vocab_size:  tokenizer 的实际词表大小

    Returns:
        模型构造参数字典
    """
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "vocab_size": int(raw.get("vocab_size", vocab_size)),
        "context_length": int(raw["context_length"]),
        "d_model": int(raw["d_model"]),
        "num_layers": int(raw["num_layers"]),
        "num_heads": int(raw["num_heads"]),
        "d_ff": int(raw["d_ff"]),
        "rope_theta": float(raw.get("rope_theta", 10000.0)),
    }


def normalize_state_dict_keys(state_dict: OrderedDict) -> OrderedDict:
    """去掉 torch.compile 产生的 "_orig_mod." 前缀。"""
    normalized = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        normalized[key] = value
    return normalized


def load_state_dict(checkpoint_path: Path, device: str) -> OrderedDict:
    """从检查点加载模型权重，兼容训练检查点和裸 state_dict。"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}")
    return normalize_state_dict_keys(OrderedDict(state_dict))


# ---------------------------------------------------------------------------
# 聊天会话（ChatSession）
# ---------------------------------------------------------------------------

class ChatSession:
    """管理多轮对话状态和模型交互的核心类。

    职责：
      - 维护对话历史列表 self.conversations
      - 每次 chat() 调用时：拼接历史 → render → tokenize → 生成 → decode → 追加到历史
      - 超长时自动截断最早轮次（保留 system prompt，按用户轮配对删除）
      - 运行时参数可调（temperature、top_p 通过 REPL 命令实时修改）
      - Unicode surrogate 过滤（小模型可能解码出无效码点）
      - 可选日志持久化（JSONL 格式）
    """

    def __init__(
        self,
        model: TransformerLM,
        tokenizer: BPETokenizer,
        eos_token: str,
        context_length: int,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
        system_prompt: str | None = None,
        log_path: str | None = None,
    ) -> None:
        """初始化聊天会话。

        Args:
            model:           已加载权重的 TransformerLM 模型
            tokenizer:        BPE 分词器实例
            eos_token:        EOS token 字符串（生成到此时停止）
            context_length:   模型最大上下文长度
            max_new_tokens:   每轮最多生成多少个新 token
            temperature:      采样温度（0=贪婪解码，>0=随机采样）
            top_p:            nucleus sampling 阈值
            system_prompt:    可选系统提示词（每轮自动拼接）
            log_path:         可选日志文件路径（JSONL 格式）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token = eos_token
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.conversations: list[dict[str, str]] = []
        self.log_path = log_path
        self._log_file = None

        # 解析 EOS token ID（只在模型词表范围内有效）
        eos_bytes = eos_token.encode("utf-8")
        model_vocab_size = model.token_embeddings.weight.shape[0]
        if eos_bytes in tokenizer.vocab_to_id:
            eid = tokenizer.vocab_to_id[eos_bytes]
            self.eos_token_id = eid if eid < model_vocab_size else None
        else:
            self.eos_token_id = None

        if log_path:
            self._log_file = open(log_path, "a", encoding="utf-8")

        if system_prompt:
            self._log_entry({"role": "system", "content": system_prompt})

    # ---- 对话历史管理 ---------------------------------------------------

    def _build_prompt_conversations(self) -> list[dict[str, str]]:
        """构建当前轮次的对话列表（system prompt + 历史对话）。

        每次 chat() 调用时都重新构建——因为可能有运行时 system prompt 变更。
        """
        convs = []
        if self.system_prompt:
            convs.append({"role": "system", "content": self.system_prompt})
        convs.extend(self.conversations)
        return convs

    def _truncate_conversations(self, prompt_ids: list[int]) -> list[dict[str, str]]:
        """超长时截断最早轮次，保留 system prompt + 最近轮次。

        预算 = context_length - max_new_tokens - 16（16 是安全余量，给特殊 token）。
        策略：从对话开头删轮次（不管 user 还是 assistant），每次删一轮后
        重新 encode 检查长度。无法 encode 时（Unicode 异常等）直接退出循环。

        注意：system prompt 始终保留不删。
        """
        budget = self.context_length - self.max_new_tokens - 16
        if len(prompt_ids) <= budget:
            return self._build_prompt_conversations()

        convs = self._build_prompt_conversations()
        has_system = convs and convs[0]["role"] == "system"
        system_part = [convs[0]] if has_system else []
        dialogue = convs[1:] if has_system else convs

        while dialogue and len(prompt_ids) > budget:
            dialogue.pop(0)                              # 删除最早的一轮
            trial = system_part + dialogue
            if dialogue:
                try:
                    rendered = self._render_prompt(trial)
                    prompt_ids = self.tokenizer.encode(rendered)
                except Exception:
                    break
            else:
                break

        return system_part + dialogue

    def _render_prompt(self, convs: list[dict[str, str]]) -> str:
        """将对话列表渲染为模型可接受的纯文本 prompt。

        格式必须与训练时的 render_chat_prompt (sft.py) 严格一致：
          <|role|>\ncontent\n<|endoftext|>\n  （assistant 轮追加 eos）
        末尾追加 "<|assistant|>\n" 引导模型开始生成回复。

        与训练时的重要区别：EOS 只在 eos_token_id 有效时才拼接——
        因为 chat.py 使用的 EOS token 可能不在模型词表范围内（如 "</s>"）。
        """
        parts: list[str] = []
        for message in convs:
            role = message["role"]
            content = message["content"]
            parts.append(ROLE_MARKERS[role])
            parts.append(content)
            parts.append("\n")
            if role == "assistant" and self.eos_token_id is not None:
                parts.append(self.eos_token)
                parts.append("\n")
        parts.append(ROLE_MARKERS["assistant"])
        return "".join(parts)

    def chat(self, user_input: str) -> str:
        """处理一轮用户输入，返回模型生成的回复。

        完整流程（每轮都执行，无缓存）：
          1. 用户输入追加到 conversations 历史
          2. 历史 + system prompt → render → tokenize
          3. token ID 安全裁剪（超出模型词表大小的 ID 被丢弃）
          4. 超长截断：token 数超过预算时，从最早轮次删对话
          5. 硬裁剪兜底：截断后还超长则只保留最后 max_prompt_len 个 token
          6. 送入模型生成（greedy 或 temperature+top-p）
          7. decode → 清洗 surrogate 字符 → 去除尾部 EOS
          8. 追加到历史 → 记录日志

        Args:
            user_input: 用户输入的原始文本

        Returns:
            模型生成的回复文本
        """
        # ① 用户输入追加到历史
        self.conversations.append({"role": "user", "content": user_input})
        self._log_entry({"role": "user", "content": user_input})

        # ② 构建 prompt → render → tokenize
        convs = self._build_prompt_conversations()
        prompt_text = self._render_prompt(convs)
        prompt_ids = self.tokenizer.encode(prompt_text)

        # ③ token ID 安全裁剪：丢弃超出模型 vocab_size 的 ID
        #    （如 EOS 在 tokenizer 词表 index 6400，但模型 embedding 只有 6400 行）
        model_vocab_size = self.model.token_embeddings.weight.shape[0]
        prompt_ids = [tid for tid in prompt_ids if tid < model_vocab_size]

        # ④ 超长截断（避免上下文溢出）
        budget = self.context_length - self.max_new_tokens - 16
        if len(prompt_ids) > budget:
            convs = self._truncate_conversations(prompt_ids)
            prompt_text = self._render_prompt(convs)
            prompt_ids = self.tokenizer.encode(prompt_text)

        if not prompt_ids:
            reply = "[Error: prompt is empty after truncation]"
            self.conversations.append({"role": "assistant", "content": reply})
            return reply

        # ⑤ 硬裁剪兜底：截断后仍超长则只取最后 max_prompt_len 个 token
        max_prompt_len = self.context_length - self.max_new_tokens
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        prompt_tensor = torch.tensor(
            [prompt_ids], dtype=torch.long, device=next(self.model.parameters()).device
        )

        # ⑥ 生成
        with torch.no_grad():
            if self.temperature == 0.0:
                # Greedy：确定性解码，无 KV Cache
                generated = prompt_tensor.clone()
                for _ in range(self.max_new_tokens):
                    idx_cond = generated[:, -self.model.context_length:]
                    logits = self.model(idx_cond)[:, -1, :]
                    if self.top_p < 1.0:
                        logits = self.model._top_p_filter(logits, self.top_p)
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    generated = torch.cat((generated, next_token), dim=1)
                    if self.eos_token_id is not None and (next_token == self.eos_token_id).all():
                        break
                full_ids = generated[0].tolist()
            else:
                # temperature + top-p 采样：KV Cache 加速
                output = self.model.generate(
                    prompt_ids=prompt_tensor,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.eos_token_id,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                full_ids = output[0].tolist()

        # ⑦ decode → 清洗
        new_ids = full_ids[len(prompt_ids):]
        reply = self.tokenizer.decode(new_ids).strip()
        reply = _remove_surrogates(reply)              # 移除无效 Unicode 码点

        # ⑧ 去掉尾部 EOS token
        if self.eos_token and reply.endswith(self.eos_token):
            reply = reply[: -len(self.eos_token)].strip()

        # ⑨ 追加到历史 + 日志
        self.conversations.append({"role": "assistant", "content": reply})
        self._log_entry({"role": "assistant", "content": reply})

        return reply

    # ---- 日志持久化 ------------------------------------------------------

    def _log_entry(self, entry: dict[str, str]) -> None:
        """写入一条带时间戳的对话日志（JSONL 格式，立即 flush 防丢失）。

        Args:
            entry: 包含 "role" 和 "content" 的字典
        """
        if self._log_file is None:
            return
        entry_with_ts = {
            "role": entry["role"],
            "content": entry["content"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._log_file.write(json.dumps(entry_with_ts, ensure_ascii=False) + "\n")
        self._log_file.flush()                          # 立即写盘，防止崩溃丢失

    def save_log(self, path: str | None = None) -> None:
        """将会话完整记录保存为 JSONL 文件。

        Args:
            path: 目标文件路径，None 时使用初始化时的 log_path 或提示需要 /save
        """
        target = path or self.log_path
        if not target:
            print("[No log path specified. Use /save <path>]")
            return
        with open(target, "w", encoding="utf-8") as f:
            if self.system_prompt:
                entry = {
                    "role": "system",
                    "content": self.system_prompt,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            for conv in self.conversations:
                entry = {
                    "role": conv["role"],
                    "content": conv["content"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[Session saved to {target}]")

    def close(self) -> None:
        """关闭日志文件句柄（会话结束时调用）。"""
        if self._log_file:
            self._log_file.close()

    # ---- REPL 命令处理 ----------------------------------------------------

    def clear_history(self) -> None:
        """清空对话历史（system prompt 不受影响）。"""
        self.conversations.clear()
        print("[Conversation history cleared]")

    def show_history(self) -> None:
        """打印当前对话历史（超长内容截断为 100 字符）。"""
        if self.system_prompt:
            print(f"  system: {self.system_prompt}")
        for i, conv in enumerate(self.conversations):
            role = conv["role"]
            content = conv["content"]
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"  [{role}]: {content}")

    def set_temperature(self, value: str) -> None:
        """运行时调整 temperature（REPL 命令 /temp）。

        设为 0 切换到 greedy 解码（确定性输出），设为正数增加随机性。
        """
        try:
            self.temperature = float(value)
            print(f"[temperature set to {self.temperature}]")
        except ValueError:
            print(f"[Invalid value: {value}]")

    def set_top_p(self, value: str) -> None:
        """运行时调整 top-p 阈值（REPL 命令 /topp）。

        必须在 [0.0, 1.0] 范围内。
        """
        try:
            v = float(value)
            if not (0.0 <= v <= 1.0):
                raise ValueError
            self.top_p = v
            print(f"[top_p set to {self.top_p}]")
        except ValueError:
            print(f"[Invalid value: {value}. Must be in [0.0, 1.0]]")

    def set_system_prompt(self, text: str) -> None:
        """运行时设置或清除 system prompt（REPL 命令 /system）。

        传入空字符串清除 system prompt，之后轮次不再自动拼接。
        """
        self.system_prompt = text if text else None
        if self.system_prompt:
            print(f"[system prompt set: {self.system_prompt}]")
        else:
            print("[system prompt cleared]")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_chat_model(
    checkpoint_path: Path,
    config_path: Path,
    vocab_path: Path,
    merges_path: Path,
    eos_token: str,
    lora_path: Path | None = None,
    dtype: str = "float32",
    device: str = "auto",
) -> tuple[TransformerLM, BPETokenizer, dict]:
    """一次性加载模型、分词器和配置。

    完整加载链路：
      1. 设备检测 (GPU/CPU) + dtype 解析
      2. BPETokenizer 从 vocab/merges 文件加载
      3. 从 model_config.json 读取模型结构参数
      4. 实例化 TransformerLM → 加载预训练权重
      5. 可选 LoRA：apply_lora → load_lora_state → merge（熔合到基座）
      6. model.eval()（关 Dropout 等）

    LoRA 加载后直接 merge 到基座权重，推理时无额外开销。
    后续如需继续训练，需先 unmerge 恢复原始权重。

    Args:
        checkpoint_path: 预训练检查点 .pt 文件路径
        config_path:     model_config.json 路径
        vocab_path:      BPE 词表文件路径
        merges_path:     BPE 合并规则文件路径
        eos_token:       EOS token 字符串
        lora_path:       可选 LoRA 适配器 .pt 文件路径
        dtype:           "float32" / "float16" / "bfloat16"
        device:          "auto" / "cuda" / "cpu"

    Returns:
        (model, tokenizer, config) 三元组
    """
    device = resolve_device(device)
    torch_dtype = resolve_model_dtype(dtype, device)

    special_tokens = [eos_token] if eos_token else []
    tokenizer = BPETokenizer.from_files(
        str(vocab_path),
        str(merges_path),
        special_tokens=special_tokens,
    )

    config = load_model_config(config_path, vocab_size=len(tokenizer.id_to_vocab))

    model = TransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
        device=device,
        dtype=torch_dtype,
    ).to(device)

    state_dict = load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict)

    # LoRA：加载适配器 → 熔合到基座（推理无额外开销）
    if lora_path is not None:
        apply_lora_to_model(model)
        lora_state = torch.load(lora_path, map_location=device, weights_only=False)
        load_lora_state_dict(model, lora_state)
        merge_lora(model)                      # 熔合后 = 普通模型

    model.eval()
    return model, tokenizer, config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 命令行参数 + REPL 交互循环
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    必需参数（5 个）：
      --checkpoint-path  预训练检查点路径
      --config-path      model_config.json 路径
      --vocab-path       BPE 词表文件
      --merges-path      BPE 合并规则文件
      --eos-token        EOS token（如 "</s>"）

    可选参数：
      --lora-path        LoRA 适配器路径
      --system-prompt    初始 system prompt
      --log              会话日志路径（JSONL 格式）
      --temperature / --top-p / --max-new-tokens / --dtype / --device / --seed
    """
    parser = argparse.ArgumentParser(
        description="MicroLM interactive multi-turn chat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
REPL commands:
  /temp <value>       Set sampling temperature
  /topp <value>       Set top-p (nucleus sampling threshold)
  /system <text>      Set or clear system prompt
  /clear              Clear conversation history
  /history            Show current conversation history
  /save [path]        Save session to JSONL file
  /help               Show this help
  /quit               Exit
""",
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, default=None, help="Optional LoRA adaptor path")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--log", type=str, default=None, help="Path to save session log (JSONL)")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eos-token", type=str, default="</s>")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


HELP_TEXT = """\
REPL commands:
  /temp <value>       Set sampling temperature
  /topp <value>       Set top-p (nucleus sampling threshold)
  /system <text>      Set or clear system prompt (empty to clear)
  /clear              Clear conversation history
  /history            Show current conversation history
  /save [path]        Save session to JSONL file
  /help               Show this help
  /quit               Exit

Type your message to chat with the model. Press Ctrl+C or /quit to exit.
"""


def repl(session: ChatSession) -> None:
    """交互式 REPL（Read-Eval-Print Loop）主循环。

    循环流程：
      1. 显示启动信息（temperature、top_p、system prompt）
      2. 等待用户输入 → 识别 REPL 命令（以 / 开头）或普通聊天消息
      3. REPL 命令：/temp、/topp、/system、/clear、/history、/save、/help、/quit
      4. 普通消息：session.chat(user_input) → 打印回复 + 耗时
      5. Ctrl+C 或 Ctrl+D 或 /quit 退出

    Args:
        session: 已初始化好的 ChatSession 实例
    """
    print("=" * 60)
    print("  MicroLM Chat  (type /help for commands, /quit to exit)")
    print("=" * 60)
    print(f"  temperature={session.temperature}  top_p={session.top_p}  "
          f"max_new_tokens={session.max_new_tokens}")
    if session.system_prompt:
        print(f"  system: {session.system_prompt}")
    print()

    try:
        while True:
            try:
                user_input = input("你> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # ---- REPL 命令处理 ----
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit", "/q"):
                    break
                elif cmd == "/help":
                    print(HELP_TEXT)
                elif cmd == "/temp":
                    if arg:
                        session.set_temperature(arg)
                    else:
                        print(f"[current temperature: {session.temperature}]")
                elif cmd == "/topp":
                    if arg:
                        session.set_top_p(arg)
                    else:
                        print(f"[current top_p: {session.top_p}]")
                elif cmd == "/system":
                    session.set_system_prompt(arg)
                elif cmd == "/clear":
                    session.clear_history()
                elif cmd == "/history":
                    session.show_history()
                elif cmd == "/save":
                    session.save_log(arg if arg else None)
                else:
                    print(f"[Unknown command: {cmd}. Type /help for available commands]")
                continue

            # ---- 普通聊天 ----
            start = time.time()
            reply = session.chat(user_input)
            elapsed = time.time() - start
            print(f"\nAI> {reply}")
            print(f"    [{elapsed:.1f}s]")
            print()

    except KeyboardInterrupt:
        print()
    finally:
        session.close()
        print("[Session ended]")


def main() -> None:
    """聊天入口：加载模型 → 创建会话 → 启动 REPL 交互循环。

    完整流程（4 步）：
      1. 解析命令行参数 + 固定随机种子（seed=42 保证可复现）
      2. 调用 load_chat_model() 一次性完成：
           - 设备检测（GPU/CPU 自动选择）
           - BPE 分词器加载（vocab + merges + 特殊 token）
           - 从 model_config.json 重建模型结构
           - 加载预训练 checkpoint 权重
           - 可选加载 LoRA 适配器并 merge 到基座（推理无额外开销）
      3. 创建 ChatSession 实例，配置：
           - 生成参数（temperature、top_p、max_new_tokens）
           - system prompt（可选，每轮自动拼到对话开头）
           - 日志路径（可选，JSONL 格式持久化）
      4. 启动 repl(session) 进入交互循环

    启动后用户看到的信息：
      - 设备名 + context_length
      - LoRA 适配器路径（如使用）
      - 当前 temperature、top_p、max_new_tokens
      - system prompt 内容（如有）
      - > 提示符等待输入
    """
    args = parse_args()
    torch.manual_seed(args.seed)

    # ---- 步骤 2：加载模型（设备 + tokenizer + 模型结构 + 权重 + LoRA） ----
    print("Loading model...")
    model, tokenizer, config = load_chat_model(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        eos_token=args.eos_token,
        lora_path=args.lora_path,
        dtype=args.dtype,
        device=args.device,
    )
    device = resolve_device(args.device)
    print(f"Model loaded on {device} (context_length={config['context_length']})")
    if args.lora_path:
        print(f"LoRA adaptor loaded from {args.lora_path} (merged)")

    # ---- 步骤 3：创建聊天会话 ----
    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        eos_token=args.eos_token,
        context_length=int(config["context_length"]),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system_prompt,
        log_path=args.log,
    )

    # ---- 步骤 4：进入 REPL 交互循环 ----
    repl(session)


if __name__ == "__main__":
    main()
