"""SFT（Supervised Fine-Tuning）数据集构建模块。

本模块将多轮对话 JSONL 语料转换为模型训练所需的 (input_ids, labels) 对，
核心流程：

  1. 读取 JSONL，每行为一条多轮对话 {"conversations": [{"role":"user", "content":"..."}, ...]}
  2. 规范化角色名、去空白、可选注入 system prompt
  3. 渲染为带特殊标记的纯文本（如 "<|user|>\n你好\n<|assistant|>\n你好！<|endoftext|>\n"）
  4. tokenize + padding，只对 assistant 回复部分计算损失（user/system 部分 labels = -100）

关键设计：labels 只在 assistant 的回复区域有值（= input_ids），其余为 -100，
确保模型只学习"如何回答"，不学习"用户说了什么"。

聊天格式示例：
    <|system|>\n你是一个AI助手\n
    <|user|>\n你好\n
    <|assistant|>\n你好！有什么可以帮你的？<|endoftext|>\n
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ---- 默认 system prompt 池，maybe_add_system_prompt 中随机抽取 ----
DEFAULT_CHAT_SYSTEM_PROMPTS = [
    "你是一个知识丰富的AI助手，请尽力给出准确、简洁的回答。",
    "你是一个可靠的中文助手，请根据用户问题给出有帮助的回复。",
    "You are a helpful AI assistant.",
    "You are a knowledgeable and concise assistant.",
]

# ---- 角色 → 特殊标记的映射，render_chat_prompt 中拼接使用 ----
ROLE_MARKERS = {
    "system": "<|system|>\n",
    "user": "<|user|>\n",
    "assistant": "<|assistant|>\n",
    "tool": "<|tool|>\n",
}


def normalize_conversations(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    """规范化多轮对话：角色名转小写、去空白、过滤空内容、校验合法性。

    输入可能来自各种数据源（ShareGPT、Alpaca 等），格式不统一：
      - 角色名可能大小写混用（"User" / "USER" / "Human"）
      - content 可能首尾有空白
      - 某些轮 content 为空（数据脏）

    统一转为 {"role": "user", "content": "你好"} 的标准格式，
    过滤掉 content 为空的轮次，只保留 role 在 ROLE_MARKERS 中的合法角色。

    Args:
        conversations: 原始对话列表

    Returns:
        规范化后的对话列表（可能比输入少——空 content 被过滤）
    """
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations must be a non-empty list")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(conversations):
        if not isinstance(message, dict):
            raise ValueError(f"conversation turn {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"conversation turn {index} must contain string role/content")
        role = role.strip().lower()       # "USER" → "user"
        if role not in ROLE_MARKERS:
            raise ValueError(f"unsupported conversation role {role!r}")
        content = content.strip()
        if not content:                   # 空内容直接跳过（不报错）
            continue
        normalized.append({"role": role, "content": content})

    if not normalized:
        raise ValueError("conversation list becomes empty after normalization")

    return normalized


def maybe_add_system_prompt(
    conversations: list[dict[str, str]],
    rng: random.Random,
    system_prompt_ratio: float,
    system_prompts: list[str] | None = None,
) -> list[dict[str, str]]:
    """按概率随机插入 system prompt 到对话开头。

    训练时通过 system_prompt_ratio 控制有多少比例的样本带 system prompt：
      - ratio = 0.0：全都不加
      - ratio = 1.0：全部加上
      - ratio = 0.5：一半加一半不加（"混合训练"）

    这样做的好处：模型同时学会"有 system prompt 时听指令"和"没有时正常聊天"，
    推理时两种用法都能支持。

    system prompt 从池中随机抽取一条。如果对话已经有 system 开头的第一轮，
    则不再重复注入（尊重数据原始设定）。

    Args:
        conversations:  规范化后的对话列表
        rng:            独立随机数生成器（种子 = 全局 seed + 样本 index，保证复现）
        system_prompt_ratio: 0~1，注入 system prompt 的概率
        system_prompts: 候选 system prompt 列表，None 时用 DEFAULT_CHAT_SYSTEM_PROMPTS

    Returns:
        可能插入了 system prompt 的对话列表
    """
    if not conversations:
        return conversations
    if conversations[0]["role"] == "system":    # 已有 system prompt，不重复加
        return conversations
    if system_prompt_ratio <= 0.0:
        return conversations
    prompts = system_prompts or DEFAULT_CHAT_SYSTEM_PROMPTS
    if not prompts:
        return conversations
    if rng.random() >= system_prompt_ratio:     # 随机决定：这次不加
        return conversations
    injected = {"role": "system", "content": rng.choice(prompts)}  # 随机抽取一条
    return [injected, *conversations]


def render_chat_prompt(
    conversations: list[dict[str, str]],
    eos_token: str = "<|endoftext|>",
    add_generation_prompt: bool = False,
) -> str:
    """将结构化对话列表渲染为模型可接受的纯文本字符串。

    拼接规则：
      - 每轮格式：角色标记 + 内容 + 换行
      - assistant 轮额外追加 eos_token + 换行（标记回复结束）
      - user/system/tool 轮不加 eos（因为后面还有 assistant 回复）

    示例输出：
        <|system|>\n你是一个AI助手\n
        <|user|>\n你好\n
        <|assistant|>\n你好！<|endoftext|>\n

    add_generation_prompt=True（推理时使用）：
      在末尾追加 "<|assistant|>\n"，引导模型开始生成回复。

    Args:
        conversations:       规范化后的对话列表
        eos_token:           用于标记 assistant 回复结束的特殊 token
        add_generation_prompt: 是否追加 assistant 标记（推理时用）

    Returns:
        渲染后的纯文本字符串
    """
    parts: list[str] = []
    for message in conversations:
        role = message["role"]
        content = message["content"]
        parts.append(ROLE_MARKERS[role])       # "<|assistant|>\n"
        parts.append(content)
        parts.append("\n")
        if role == "assistant":
            parts.append(eos_token)            # 表示这一轮回复结束了
            parts.append("\n")

    if add_generation_prompt:
        parts.append(ROLE_MARKERS["assistant"]) # 推理时：让模型开始"写回复"

    return "".join(parts)


def _find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
    """在 token 序列中查找子序列 pattern，返回首次出现的位置。

    朴素线性扫描（不用 KMP——pattern 很短，开销可忽略）。
    build_loss_labels 用这个来定位 "<|assistant|>\n" 和 "<|endoftext|>\n"
    的 token 边界，从而确定哪些 token 属于 assistant 回复（需要计算损失）。

    Args:
        sequence: token ID 列表
        pattern:  需要查找的 token ID 子序列
        start:    从哪个位置开始查找

    Returns:
        首次匹配的起始索引，未找到返回 -1
    """
    if not pattern:
        return start
    limit = len(sequence) - len(pattern) + 1
    for index in range(start, max(limit, start)):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    return -1


def build_loss_labels(
    input_ids: list[int],
    tokenizer,
    max_length: int,
    assistant_header_ids: list[int],
    eos_boundary_ids: list[int],
    pad_token_id: int,
) -> list[int]:
    """构建 labels 序列：只在 assistant 回复区域赋值为 input_ids，其余为 -100。

    这是 SFT 最关键的设计——"选择性损失"：

    完整 token 序列：  [system]...[user]...[assistant]回复内容<|endoftext|>...[padding]
    对应的 labels：    -100  -100  -100  -100    345,678,901,...     -100   -100  -100
                      ↑ 不学      ↑ 不学          ↑ 只学这部分！       ↑ padding不学

    算法：
      1. 找到 assistant 标记 "<|assistant|>\n" 的位置
      2. 从标记后一位开始，找到 "<|endoftext|>\n" 作为回复结束边界
      3. 边界内的 token 的 label = input_id（标准语言模型"预测下一个 token"）
      4. 边界外的 token 的 label = -100（PyTorch CrossEntropyLoss 默认忽略值）
      5. padding 区域的 token 也不计入损失

    这样模型只学习 assistant 的回复内容，不学习 user 的问题或 system prompt。

    Args:
        input_ids:           完整 token 序列（已 padding 到 max_length）
        tokenizer:           分词器（未直接使用，保留参数以便未来扩展）
        max_length:          序列最大长度
        assistant_header_ids: "<|assistant|>\n" 的 token ID 列表
        eos_boundary_ids:     "<|endoftext|>\n" 的 token ID 列表
        pad_token_id:        padding 用的 token ID

    Returns:
        与 input_ids 等长的 labels 列表（值 = input_id 或 -100）
    """
    labels = [-100] * len(input_ids)                  # 默认全部忽略
    index = 0
    while index < len(input_ids):
        # 找下一个 assistant 标记
        header_index = _find_subsequence(input_ids, assistant_header_ids, start=index)
        if header_index < 0:
            break                                      # 没有更多 assistant 轮了
        start = header_index + len(assistant_header_ids)  # 回复内容的起始位置
        # 找这个回复的结束标记
        end = _find_subsequence(input_ids, eos_boundary_ids, start=start)
        if end < 0:
            end = len(input_ids)
            boundary = end
        else:
            boundary = min(end + len(eos_boundary_ids), max_length)
        # 回复区域内的 token：label = input_id（正常预测下一个 token）
        for position in range(start, min(boundary, len(input_ids))):
            if input_ids[position] != pad_token_id:    # padding 不参与损失
                labels[position] = input_ids[position]
        index = boundary if end >= 0 else len(input_ids)
    return labels


class SFTDataset(Dataset):
    """SFT 数据集类 —— 从 JSONL 文件按需加载多轮对话样本。

    每条 JSONL 行格式：
        {"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}

    加载策略：不把整个 JSONL 读进内存，而是记录每行的文件偏移量（_offsets），
    __getitem__ 时 seek 到对应位置读一行 —— 百万级样本也不撑爆 RAM。

    __getitem__ 完整流程：
      1. seek 读取 JSONL 中的一行
      2. 规范化 + 可选随机注入 system prompt
      3. 渲染为纯文本 → tokenize → padding 到 max_length
      4. 构建 labels（只对 assistant 回复计算损失）
      5. 返回 (input_ids 张量, labels 张量)
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        max_length: int = 1024,
        system_prompt_ratio: float = 0.0,
        seed: int = 42,
        eos_token: str = "<|endoftext|>",
        system_prompts: list[str] | None = None,
    ) -> None:
        """初始化 SFT 数据集。

        Args:
            jsonl_path:          对话 JSONL 文件路径
            tokenizer:           BPETokenizer 实例
            max_length:          最大序列长度（超出截断，不足用 pad_token 补齐）
            system_prompt_ratio: 随机注入 system prompt 的比例（0.0~1.0）
            seed:                随机种子（用于确定性 system prompt 注入）
            eos_token:           标记回复结束的特殊 token
            system_prompts:      自定义 system prompt 池，None 用 DEFAULT_CHAT_SYSTEM_PROMPTS

        Raises:
            ValueError: EOS token 不在词表中，或 JSONL 无有效样本
        """
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt_ratio = system_prompt_ratio
        self.seed = seed
        self.eos_token = eos_token
        self.system_prompts = system_prompts or DEFAULT_CHAT_SYSTEM_PROMPTS

        # 预编码两个关键标记（用于 build_loss_labels 定位 assistant 回复边界）
        self.assistant_header_ids = tokenizer.encode(ROLE_MARKERS["assistant"])
        self.eos_boundary_ids = tokenizer.encode(f"{eos_token}\n")

        # 校验 EOS token 在词表中存在
        eos_token_bytes = eos_token.encode("utf-8")
        if eos_token_bytes not in tokenizer.vocab_to_id:
            raise ValueError(f"EOS token {eos_token!r} is not in the tokenizer vocabulary")
        self.pad_token_id = tokenizer.vocab_to_id[eos_token_bytes]  # 用 EOS 做 padding

        # 建立行偏移量索引：只记每行起始字节位置，不读内容
        self._offsets: list[int] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            while True:
                offset = f.tell()         # 当前字节位置
                line = f.readline()
                if not line:
                    break
                if line.strip():          # 跳过空行
                    self._offsets.append(offset)

        if not self._offsets:
            raise ValueError(f"No usable SFT samples found in {self.jsonl_path}")

    def __len__(self) -> int:
        """返回数据集样本数。"""
        return len(self._offsets)

    def _read_sample(self, index: int) -> dict[str, object]:
        """按偏移量定位并读取一行 JSON 样本。每次读取都打开-关闭文件，
        避免文件句柄泄漏；由 PyTorch DataLoader 的多进程安全读取。
        """
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            f.seek(self._offsets[index])
            return json.loads(f.readline())

    def _prepare_conversations(self, sample: dict[str, object], index: int) -> list[dict[str, str]]:
        """提取对话列表，规范化，可选注入 system prompt。

        rng 使用 self.seed + index 创建独立随机数生成器，
        保证同一样本在各 epoch 中是否获得 system prompt 是一致的。
        """
        conversations = sample.get("conversations")
        if not isinstance(conversations, list):
            raise ValueError("SFT sample must contain a conversations list")
        normalized = normalize_conversations(conversations)
        rng = random.Random(self.seed + index)
        return maybe_add_system_prompt(
            normalized,
            rng=rng,
            system_prompt_ratio=self.system_prompt_ratio,
            system_prompts=self.system_prompts,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取第 index 条样本。

        完整管道：
          JSONL 一行 → normalized conversations → 纯文本 → token IDs →
          padding 到 max_length → 构建选择性 labels

        Returns:
            (input_ids, labels)，形状均为 [max_length]，labels 中非 assistant
            回复区域为 -100
        """
        sample = self._read_sample(index)
        conversations = self._prepare_conversations(sample, index)

        # 对话 → 纯文本 → tokenize → 截断
        rendered = render_chat_prompt(conversations, eos_token=self.eos_token, add_generation_prompt=False)
        input_ids = self.tokenizer.encode(rendered)[: self.max_length]

        # padding：不足 max_length 的部分用 pad_token 补齐
        input_ids += [self.pad_token_id] * (self.max_length - len(input_ids))

        # 构建 labels：只对 assistant 回复区域计算损失
        labels = build_loss_labels(
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            assistant_header_ids=self.assistant_header_ids,
            eos_boundary_ids=self.eos_boundary_ids,
            pad_token_id=self.pad_token_id,
        )
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def build_generation_prompt(
    conversations: list[dict[str, str]],
    eos_token: str = "<|endoftext|>",
) -> str:
    """构建推理时的生成 prompt。

    和训练时 render 的区别：末尾追加 "<|assistant|>\n"，引导模型开始生成回复。
    同时校验对话最后不能以 assistant 结尾（问完问题才能让模型回答）。

    输入示例：
        [{"role":"user", "content":"1+1=?"}]

    输出示例：
        "<|user|>\n1+1=?\n<|assistant|>\n"
                                        ↑ 模型从这里开始生成

    Args:
        conversations: 对话列表（最后一条应为 user 或 system）
        eos_token:     EOS token

    Returns:
        末尾带 assistant 标记的生成用文本
    """
    normalized = normalize_conversations(conversations)
    if normalized[-1]["role"] == "assistant":
        raise ValueError("generation prompt should end with user/system turns, not assistant")
    return render_chat_prompt(normalized, eos_token=eos_token, add_generation_prompt=True)
