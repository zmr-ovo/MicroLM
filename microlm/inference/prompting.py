"""推理时的 prompt 构建与解析模块。

本模块解决一个问题：推理时用户可能以多种方式提供输入——
  1. 纯文本字符串（单轮生成 / 续写）
  2. JSON 字符串（多轮对话，命令行 --conversations-json '[...]'）
  3. JSONL 文件路径（多轮对话，命令行 --conversations-path chat.jsonl）

无论哪种输入，本模块统一解析为对话列表，然后调用 build_generation_prompt
渲染为带生成引导的纯文本 prompt（末尾追加 "<|assistant|>\n"）。

核心函数 resolve_generation_prompt 就是 chat.py / generate_text.py 推理脚本的入口，
它屏蔽了输入来源的差异，让推理脚本只拿到最终可直接 tokenize 的字符串。
"""

from __future__ import annotations

import json
from pathlib import Path

from microlm.training import build_generation_prompt


def _normalize_conversations(raw_conversations: object) -> list[dict[str, str]]:
    """将原始 JSON 解析结果规范化为标准对话列表。

    校验内容：
      - 必须是 list 且非空
      - 每个元素必须是 dict，且 role/content 都是字符串
      - 空 content 不跳过（与训练时的 normalize_conversations 不同——推理时
        用户传什么就是什么，不做内容过滤）

    Args:
        raw_conversations: json.loads() 解析后的对象，应为对话列表

    Returns:
        标准格式的对话列表 [{"role": "user", "content": "你好"}, ...]

    Raises:
        ValueError: 格式不合法（非 list、空列表、缺少 role/content 等）
    """
    if not isinstance(raw_conversations, list) or not raw_conversations:
        raise ValueError("conversations must be a non-empty list")

    conversations: list[dict[str, str]] = []
    for index, message in enumerate(raw_conversations):
        if not isinstance(message, dict):
            raise ValueError(f"conversation turn {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"conversation turn {index} must contain string role/content")
        conversations.append({"role": role, "content": content})
    return conversations


def load_conversations_from_json(raw_json: str) -> list[dict[str, str]]:
    """从 JSON 字符串解析多轮对话。

    支持命令行直接传入对话 JSON：
        --conversations-json '[{"role":"user","content":"你好"}]'

    Args:
        raw_json: JSON 字符串，解析后应为对话列表

    Returns:
        标准格式的对话列表
    """
    parsed = json.loads(raw_json)
    return _normalize_conversations(parsed)


def load_conversations_from_path(path: str | Path) -> list[dict[str, str]]:
    """从 JSON 文件加载多轮对话。

    文件内容应为 JSON 数组（而非 JSONL），如一个 .json 文件：
        [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]

    Args:
        path: JSON 文件路径

    Returns:
        标准格式的对话列表
    """
    conversation_path = Path(path)
    return load_conversations_from_json(conversation_path.read_text(encoding="utf-8"))


def resolve_generation_prompt(
    prompt: str | None,
    conversations_json: str | None,
    conversations_path: str | Path | None,
    eos_token: str = "<|endoftext|>",
) -> str:
    """统一解析多种输入方式，返回可直接 tokenize 的生成 prompt 字符串。

    优先级（互斥，只能提供一个）：
      1. conversations_json：命令行直接传入对话 JSON 字符串
         → 解析 → build_generation_prompt → "<|user|>\n你好\n<|assistant|>\n"
      2. conversations_path：JSON 文件路径
         → 读文件 → 解析 → build_generation_prompt → 同上
      3. prompt：纯文本字符串（兜底），原样返回不做任何处理

    对话输入会通过 build_generation_prompt 追加 "<|assistant|>\n"，
    引导模型开始生成回复。纯文本 prompt 不追加任何东西——适用于续写场景。

    Args:
        prompt:              纯文本 prompt 字符串（单轮续写用）
        conversations_json:  JSON 格式的多轮对话字符串
        conversations_path:  多轮对话 JSON 文件路径
        eos_token:           EOS token（传给 build_generation_prompt）

    Returns:
        可直接传入 tokenizer.encode() 的 prompt 字符串

    Raises:
        ValueError: 同时提供了 conversations_json 和 conversations_path，
                    或者三种输入全为空

    示例：
        # 多轮对话 JSON
        resolve_generation_prompt(
            prompt=None,
            conversations_json='[{"role":"user","content":"你好"}]',
            conversations_path=None,
        )
        → "<|user|>\n你好\n<|assistant|>\n"

        # 纯文本续写
        resolve_generation_prompt(
            prompt="中国的首都是",
            conversations_json=None,
            conversations_path=None,
        )
        → "中国的首都是"
    """
    # 互斥检查：只能提供一个
    if conversations_json is not None and conversations_path is not None:
        raise ValueError("Provide only one of conversations_json or conversations_path")

    # 方式 1：命令行 JSON 字符串
    if conversations_json is not None:
        conversations = load_conversations_from_json(conversations_json)
        return build_generation_prompt(conversations, eos_token=eos_token)

    # 方式 2：JSON 文件路径
    if conversations_path is not None:
        conversations = load_conversations_from_path(conversations_path)
        return build_generation_prompt(conversations, eos_token=eos_token)

    # 方式 3：纯文本 prompt（兜底）
    if prompt is None or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string")

    return prompt
