"""将 JSONL 语料转换为预训练数据。

读取每行为 {"text": "..."} 格式的 JSONL 文件，经过可配置的清洗管线，生成训练集
和验证集文本文件。支持控制字符去除、HTML 标签清理、空白压缩、长度过滤、字面量
替换规则以及精确去重。输出带文档分隔符的训练/验证集、分词器训练语料，以及包含
完整统计信息的 metadata JSON，确保数据处理过程可复现。

用法示例：
    python prepare_pretrain_jsonl.py --input-path data/raw.jsonl \
        --output-dir data/pretrain --min-length 50 --clean-html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import codecs
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。

    定义所有 CLI 参数，包括输入输出路径、数据集划分比例、文档分隔符、
    字面量替换规则以及清洗/过滤相关配置。
    """
    parser = argparse.ArgumentParser(
        description="将逐行 {'text': ...} 格式的 JSONL 语料转换为训练/验证集文本文件，支持增强数据清洗。"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="输入的 JSONL 源文件路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pretrain"),
        help="生成的训练/验证集文本文件输出目录。",
    )
    parser.add_argument(
        "--text-key",
        type=str,
        default="text",
        help="JSON 对象中包含文档原始文本的字段名。",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.01,
        help="通过确定性文本哈希分配到验证集的文档比例。",
    )
    parser.add_argument(
        "--document-separator",
        type=str,
        default="###",
        help="在训练/验证集输出中插入文档之间的特殊分隔符。",
    )
    parser.add_argument(
        "--replace-literal",
        action="append",
        default=[],
        help="字面量替换规则，格式为 旧=新。右侧支持 \\n 等转义序列。",
    )
    # --- 清洗参数 ---
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="清洗后文档的最小字符数，短于此值的文档被丢弃。设为 0 关闭此限制。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="文档最大字符数限制。0 表示不限制。",
    )
    parser.add_argument(
        "--max-length-action",
        choices=["drop", "truncate"],
        default="drop",
        help="超过 --max-length 的文档处理方式：drop（丢弃）或 truncate（截断）。",
    )
    parser.add_argument(
        "--clean-html",
        action="store_true",
        default=False,
        help="去除文本中的 HTML 风格标签（<...>）。",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        default=False,
        help="跳过精确去重步骤。",
    )
    return parser


def should_use_valid_split(text: str, valid_ratio: float) -> bool:
    """判断一篇文档是否应归入验证集。

    对文档文本做 SHA-1 哈希，映射到 [0, 2^64) 的桶中。归一化后的值
    若小于 valid_ratio 则返回 True。每篇文档的分配结果完全由内容决定——
    不依赖数据集顺序、随机种子或相邻文档，同一篇文档在任何运行中都会被
    分到同一个集合中。
    """
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return bucket / 2**64 < valid_ratio


def parse_replacement_rules(raw_rules: list[str]) -> list[tuple[str, str]]:
    """将 "旧=新" 格式的替换规则解析为 (旧文本, 新文本) 元组列表。

    每条规则只按第一个 "=" 分割，因此右侧可以包含等号。
    右侧部分通过 unicode_escape 编解码器处理，将字面转义序列（如 "\\n"）
    转换为真正的控制字符。
    """
    rules: list[tuple[str, str]] = []
    for raw_rule in raw_rules:
        if "=" not in raw_rule:
            raise ValueError(f"Invalid --replace-literal rule {raw_rule!r}; expected old=new")
        old, new = raw_rule.split("=", 1)
        new = codecs.decode(new, "unicode_escape")
        rules.append((old, new))
    return rules


# ---------- 清洗辅助函数 ----------

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_control_chars(text: str) -> tuple[str, bool]:
    """去除控制字符（保留 \\n 和 \\t）。返回 (清洗后文本, 是否被修改过)。"""
    cleaned = _CONTROL_CHAR_RE.sub("", text)
    return cleaned, len(cleaned) != len(text)


def compress_whitespace(text: str) -> tuple[str, bool]:
    """压缩连续空格/Tab 为单个空格；将 3 个以上连续换行压缩为 2 个。"""
    cleaned = _MULTI_SPACE_RE.sub(" ", text)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned, cleaned != text


def clean_html_tags(text: str) -> tuple[str, bool]:
    """去除 HTML 风格标签（<...>）。返回 (清洗后文本, 是否被修改过)。"""
    cleaned = _HTML_TAG_RE.sub("", text)
    return cleaned, len(cleaned) != len(text)


def compute_length_stats(lengths: list[int]) -> dict:
    """计算文档长度的描述性统计指标，包括均值和各分位数。"""
    if not lengths:
        return {"count": 0}
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    total = sum(sorted_lengths)
    mean = total / n
    if n % 2 == 0:
        median = (sorted_lengths[n // 2 - 1] + sorted_lengths[n // 2]) / 2
    else:
        median = sorted_lengths[n // 2]

    def percentile(p: float) -> int:
        k = (n - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_lengths[int(k)]
        return int(sorted_lengths[f] * (c - k) + sorted_lengths[c] * (k - f))

    return {
        "count": n,
        "min": sorted_lengths[0],
        "max": sorted_lengths[-1],
        "mean": round(mean, 1),
        "median": int(median),
        "p25": percentile(25),
        "p75": percentile(75),
        "p95": percentile(95),
        "p99": percentile(99),
        "total_chars": total,
    }


def main() -> None:
    """执行完整的 JSONL 转预训练数据流水线。

    编排参数解析、文档读取、清洗管线（去除首尾空白、字面量替换、
    控制字符清理、HTML 标签去除、空白压缩、长度过滤、去重）、
    训练/验证集划分，并写入输出文件及记录配置与统计信息的 metadata JSON。
    """
    args = build_parser().parse_args()
    if not 0.0 <= args.valid_ratio < 1.0:
        raise ValueError("--valid-ratio 必须在 [0.0, 1.0) 区间内")
    replacement_rules = parse_replacement_rules(args.replace_literal)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.txt"
    valid_path = args.output_dir / "valid.txt"
    tokenizer_corpus_path = args.output_dir / "tokenizer_corpus.txt"
    metadata_path = args.output_dir / "metadata.json"

    # 计数器
    total_raw = 0
    skipped_empty = 0
    skipped_short = 0
    skipped_long = 0
    cleaned_control = 0
    cleaned_html_count = 0
    compressed_ws = 0
    duplicates_removed = 0

    train_docs = 0
    valid_docs = 0
    train_chars = 0
    valid_chars = 0

    all_lengths: list[int] = []
    seen_hashes: set[str] = set() if not args.no_dedup else set()

    with (
        args.input_path.open("r", encoding="utf-8") as src,
        train_path.open("w", encoding="utf-8") as train_f,
        valid_path.open("w", encoding="utf-8") as valid_f,
        tokenizer_corpus_path.open("w", encoding="utf-8") as tokenizer_f,
    ):
        for line_number, raw_line in enumerate(src, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = record.get(args.text_key)
            if not isinstance(text, str):
                continue

            total_raw += 1

            # 1. 去除首尾空白
            text = text.strip()
            if not text:
                skipped_empty += 1
                continue

            # 2. 字面量替换
            for old, new in replacement_rules:
                text = text.replace(old, new)

            # 3. 控制字符清洗
            text, ctrl_modified = clean_control_chars(text)
            if ctrl_modified:
                cleaned_control += 1

            # 4. HTML 标签清洗
            if args.clean_html:
                text, html_modified = clean_html_tags(text)
                if html_modified:
                    cleaned_html_count += 1

            # 5. 空白压缩
            text, ws_modified = compress_whitespace(text)
            if ws_modified:
                compressed_ws += 1

            # 6. 再次去除首尾空白
            text = text.strip()
            if not text:
                skipped_empty += 1
                continue

            # 7. 长度过滤
            text_len = len(text)
            if args.min_length > 0 and text_len < args.min_length:
                skipped_short += 1
                continue
            if args.max_length > 0 and text_len > args.max_length:
                if args.max_length_action == "drop":
                    skipped_long += 1
                    continue
                else:  # 截断
                    text = text[:args.max_length]
                    text_len = args.max_length

            # 8. 去重
            if not args.no_dedup:
                doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if doc_hash in seen_hashes:
                    duplicates_removed += 1
                    continue
                seen_hashes.add(doc_hash)

            all_lengths.append(text_len)

            tokenizer_f.write(text)
            tokenizer_f.write("\n")

            target = valid_f if should_use_valid_split(text, args.valid_ratio) else train_f
            target.write(text)
            target.write("\n")
            target.write(args.document_separator)
            target.write("\n")

            if target is train_f:
                train_docs += 1
                train_chars += text_len
            else:
                valid_docs += 1
                valid_chars += text_len

    total_kept = train_docs + valid_docs

    filter_stats = {
        "total_raw_documents": total_raw,
        "skipped_empty": skipped_empty,
        "skipped_short": skipped_short,
        "skipped_long": skipped_long,
        "cleaned_html": cleaned_html_count,
        "cleaned_control_chars": cleaned_control,
        "compressed_whitespace": compressed_ws,
        "duplicates_removed": duplicates_removed,
        "total_kept": total_kept,
        "filter_rate": f"{(1 - total_kept / total_raw) * 100:.2f}%" if total_raw > 0 else "0.00%",
    }

    metadata = {
        "source_path": str(args.input_path),
        "text_key": args.text_key,
        "document_separator": args.document_separator,
        "valid_ratio": args.valid_ratio,
        "replacement_rules": [
            {"old": old, "new": new} for old, new in replacement_rules
        ],
        "cleaning_config": {
            "min_length": args.min_length,
            "max_length": args.max_length,
            "max_length_action": args.max_length_action,
            "clean_html": args.clean_html,
            "dedup_enabled": not args.no_dedup,
        },
        "filter_stats": filter_stats,
        "length_stats": compute_length_stats(all_lengths),
        "train": {
            "path": str(train_path),
            "documents": train_docs,
            "characters": train_chars,
            "avg_doc_length": round(train_chars / train_docs, 1) if train_docs > 0 else 0,
        },
        "valid": {
            "path": str(valid_path),
            "documents": valid_docs,
            "characters": valid_chars,
            "avg_doc_length": round(valid_chars / valid_docs, 1) if valid_docs > 0 else 0,
        },
        "tokenizer_corpus": {
            "path": str(tokenizer_corpus_path),
            "documents": total_kept,
        },
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"wrote train split to {train_path}")
    print(f"wrote valid split to {valid_path}")
    print(f"wrote tokenizer corpus to {tokenizer_corpus_path}")
    print(f"saved metadata to {metadata_path}")
    print(
        "documents: "
        f"raw={total_raw}, kept={total_kept}, "
        f"empty={skipped_empty}, short={skipped_short}, "
        f"long={skipped_long}, dupes={duplicates_removed}"
    )


if __name__ == "__main__":
    main()
