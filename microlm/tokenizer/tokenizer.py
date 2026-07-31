"""BPE 分词器推理模块。

提供 BPETokenizer 类，包含三项核心能力：
1. 编码（encode）：将文本按 GPT-2 正则预分词后，依训练产出的合并规则逐对合并，
   最终映射为 token ID 序列；支持特殊 token 保护，不被拆分。
2. 解码（decode）：将 token ID 序列反向查表还原为 UTF-8 字节串，再解码为文本。
3. 流式编码（encode_iterable）：逐块接收文本输入，在换行或空格等安全边界处截断
   编码，避免 token 在跨块边界处被错误拆分，适用于大文件或流式数据。
与训练模块 bpe.py 的 vocab.json 和 merge.txt 产出格式直接对接。
"""

import regex as re
from collections.abc import Iterable
import json
from .bpe import bytes_to_unicode


class BPETokenizer:
    """BPE 分词器，负责文本与 token ID 序列之间的双向转换。"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],  special_tokens:list[str]|None=None):
        """初始化分词器。

        构建三项核心映射：
        - id_to_vocab / vocab_to_id：token ID 与字节串的双向查找表
        - merges_id：合并规则 → 优先级编号（数字越小越优先）
        并将特殊 token（如 <pad>、<bos>）注册到词表中。
        """
        # ---- 1. 构建双向词表 ----
        self.id_to_vocab = dict(vocab)
        self.vocab_to_id = {v: id for id, v in self.id_to_vocab.items()}

        # ---- 2. 构建合并规则优先级表（按 merge.txt 行号 = 优先级） ----
        self.merges_id = {m: id for id, m in enumerate(merges)}
        self.special_tokens = special_tokens or []

        # ---- 3. 将特殊 token 注册到词表（分配新的 token ID） ----
        next_id = max(self.id_to_vocab.keys()) + 1 if self.id_to_vocab else 0
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.vocab_to_id:
                self.id_to_vocab[next_id] = tok_bytes
                self.vocab_to_id[tok_bytes] = next_id
                next_id += 1

        # ---- 4. 编译特殊 token 匹配正则（按长度降序，长匹配优先） ----
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_regex = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_regex)
        else:
            self.special_regex = None

        # ---- 5. 编译 GPT-2 预分词正则 ----
        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """从训练产出的 vocab.json 和 merge.txt 文件加载分词器。

        处理流程：
        1. 读取 vocab.json，自动识别两种格式（{id: token} 或 {token: id}）
        2. 通过 bytes_to_unicode 反向映射将可见字符还原为原始字节
        3. 逐行读取 merge.txt，每行 "左token 右token" 解析为合并元组
        4. 调用 __init__ 构建完整分词器实例
        """
        # ---- 1. 读取并解析词表文件 ----
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        # 构建 Unicode 字符 → 原始字节的反向映射
        byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

        # 自动适配两种 vocab.json 格式
        if not raw_vocab:
            vocab_items: list[tuple[int, str]] = []
        elif all(isinstance(k, str) and k.isdigit() and isinstance(v, str) for k, v in raw_vocab.items()):
            # 格式 1: {"0": "a", "1": "b", ...} → key 是字符串数字
            vocab_items = [(int(k), v) for k, v in raw_vocab.items()]
        elif all(isinstance(k, str) and isinstance(v, int) for k, v in raw_vocab.items()):
            # 格式 2: {"a": 0, "b": 1, ...} → value 是数字
            vocab_items = [(v, k) for k, v in raw_vocab.items()]
        else:
            raise ValueError("Unsupported vocab.json format")

        # 将可见字符表示的 token 还原为原始字节串
        vocab = {
            token_id: bytes(byte_decoder[ch] for ch in token_text)
            for token_id, token_text in vocab_items
        }

        # ---- 2. 逐行读取合并规则文件 ----
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append(
                        (
                            bytes(byte_decoder[ch] for ch in parts[0]),
                            bytes(byte_decoder[ch] for ch in parts[1]),
                        )
                    )

        return cls(vocab, merges, special_tokens)

    def encode(self, text)->list[int]:
        """将文本编码为 token ID 序列。

        若存在特殊 token，先用特殊 token 正则将文本切成普通片段和特殊 token：
        - 普通片段 → 走 _encode_text_segment 做 BPE 编码
        - 特殊 token → 直接查 vocab_to_id 映射为 ID，保证不被拆分
        无特殊 token 时整段文本直接走 BPE 编码。
        """
        if not text:
            return []

        # 无特殊 token：整段直接 BPE 编码
        if self.special_regex == None:
            return self._encode_text_segment(text)

        # 有特殊 token：按特殊 token 边界切分，分段处理
        tokens = []
        last_pos = 0
        for match in self.special_regex.finditer(text):
            # 特殊 token 之前的普通文本 → BPE 编码
            pre_text = text[last_pos:match.start()]
            if pre_text:
                tokens.extend(self._encode_text_segment(pre_text))
            # 特殊 token 本身 → 直接映射为 ID
            special_token = match.group()
            special_token = self.vocab_to_id[special_token.encode("utf-8")]
            tokens.append(special_token)
            last_pos = match.end()

        # 最后一段普通文本（尾部）
        if text[last_pos:]:
            tokens.extend(self._encode_text_segment(text[last_pos:]))
        return tokens

    def _encode_text_segment(self, text:str)-> list[int]:
        """对一段普通文本（不含特殊 token）执行 BPE 编码。

        步骤：
        1. GPT-2 正则预分词，将文本切为单词片段
        2. 每个片段编码为单字节序列
        3. 贪心合并：每轮找优先级最高（merges_id 值最小）的可合并相邻对，
           将该对替换为合并后的 token，重复直到无可合并对
        4. 最终 token 序列查 vocab_to_id 映射为 ID 列表
        """
        ids = []
        # ---- 1. GPT-2 正则预分词 ----
        pre_tokens = self.gpt2_pat.findall(text)
        for word in pre_tokens:
            # ---- 2. 初始化为单字节序列 ----
            tokens = [bytes([b]) for b in word.encode("utf-8")]

            # ---- 3. 按合并规则优先级贪心合并 ----
            while(len(tokens) > 1):
                # 找当前序列中优先级最高（merges_id 最小）的相邻对
                best_pair = None
                best_id = float("inf")
                for i in range(len(tokens)-1):
                    pair = (tokens[i], tokens[i+1])
                    if pair in self.merges_id and self.merges_id[pair] < best_id:
                        best_pair = pair
                        best_id = self.merges_id[pair]

                # 没有可合并的对了，停止
                if best_pair == None:
                    break

                # 扫描整个序列，将 best_pair 全部替换为合并后的 token
                i = 0
                new_tokens = []
                while(i<len(tokens)):
                    if i<len(tokens) -1 and (tokens[i], tokens[i+1]) == best_pair:
                        new_tokens.append(best_pair[0] + best_pair[1])
                        i+=2  # 跳过被合并的两个 token
                    else:
                        new_tokens.append(tokens[i])
                        i+=1
                tokens = new_tokens

            # ---- 4. token 字节串 → token ID ----
            for t in tokens:
                ids.append(self.vocab_to_id[t])
        return ids
    
    def decode(self, ids:list[int]) ->str:
        """将 token ID 序列解码回文本。

        每个 ID 查 id_to_vocab 获得对应字节串，拼接后以 UTF-8 解码。
        非法字节序列用 replacement character (U+FFFD) 替代，不抛异常。
        """
        byte_segments = [self.id_to_vocab[i] for i in ids]
        full_bytes = b"".join(byte_segments)
        return full_bytes.decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """流式编码：逐块接收文本，按安全边界截断后分批编码。

        适用场景：文件太大无法一次读入内存，或数据是实时流式到达的。

        安全边界策略：
        在 buffer 中找最后一个换行符或空格作为截断点——GPT-2 正则天然在这些
        位置切分，因此绝不会把一个 token 从中间切断。未达边界的尾部留到下一
        轮与新的 chunk 拼接，确保跨块边界 token 的完整性。
        """
        buffer = ""
        for chunk in iterable:
            buffer += chunk

            # 找安全截断边界：优先换行，其次空格
            safe_idx = max(buffer.rfind('\n'), buffer.rfind(' '))

            if safe_idx != -1:
                # 安全部分立即编码产出
                safe_text = buffer[:safe_idx + 1]
                yield from self.encode(safe_text)
                # 尾部留到下一轮
                buffer = buffer[safe_idx + 1:]

        # 最后一批残留文本
        if buffer:
            yield from self.encode(buffer)
