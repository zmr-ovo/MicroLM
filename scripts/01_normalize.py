"""数据清洗管线 — 第 1 阶段：字段结构归一化。

本脚本是清洗管线的第一步，将原始 instructie 数据集从「多格式混用」的
JSON 数组转换为「字段统一、命名规范」的 JSONL 格式，为后续过滤和分层做准备。

原始数据的不一致性（需要归一化的原因）：
  - 字段名混用：有的样本用 "text"，有的用 "input"，语义相同但 key 不同
  - relation 结构不齐：部分样本含 head_type/tail_type，部分没有
  - cate 命名冗余：原始类别名 "建筑结构" 过长，统一为 "建筑"
  - 文件格式：原始为 JSON 数组（一个文件一大坨），转为 JSONL（每行一条）
  - 缺少溯源标记：新增 source 字段标识数据来源（train/valid/test）

输入（来自 conf.py 路径常量）：
  - data/instructie/train_zh.json  原始训练集（JSON 数组）
  - data/instructie/valid_zh.json  原始验证集
  - data/instructie/test_zh.json   原始测试集

输出（路径常量见 conf.py）：
  - data/processed/normalized_train.jsonl  标准化训练集（JSONL）
  - data/processed/normalized_valid.jsonl  标准化验证集
  - data/processed/normalized_test.jsonl   标准化测试集
  - reports/normalize_report.json          归一化报告（样本数 + cate 映射）

使用方式：
    python scripts/01_normalize.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from conf import (
    RAW_TRAIN, RAW_VALID, RAW_TEST, RAW_SCHEMA,
    NORM_TRAIN, NORM_VALID, NORM_TEST,
    CATE_MAP, PROC_DIR, REPORT_DIR,
)


def load_jsonl(path):
    """从 JSONL 文件加载数据（每行一条完整 JSON）。

    instructie 原始数据虽然文件后缀是 .json，内部实际是 JSONL 格式——
    每行一个独立 JSON 对象，没有外层的 `[...]` 数组包裹。因此逐行
    json.loads(line) 即可解析，无需 json.load() 整文件加载。

    Args:
        path: 文件路径（.json 或 .jsonl，内部必须为 JSONL 格式）

    Returns:
        list[dict]：解析后的样本列表
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    """将数据保存为 JSONL 格式（每行一个完整 JSON 对象）。

    与原始 JSON 数组格式相比，JSONL 的优势：
      - 可按行追加（适合流式处理）
      - 每行独立（部分损坏不影响其余行）
      - 方便 grep/head/tail 等命令行工具直接查看

    Args:
        data: 样本列表（list[dict]）
        path: 输出文件路径（自动创建父目录）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_item(item, source):
    """将单条原始样本标准化为清洗管线统一格式。

    执行四项归一化操作：
      1. 字段名统一：text → input（两者语义相同，统一为 "input"）
      2. cate 规范化：通过 CATE_MAP 替换冗余命名（如 "建筑结构"→"建筑"）
      3. relation 结构对齐：提取 head/relation/tail 三字段，去除空值前后空白，
         可选字段 head_type/tail_type 独立保存（不丢弃训练时标注的类型信息）
      4. 追加 source 标记：标识数据来源（train/valid/test），方便后续溯源

    Args:
        item: 原始样本 dict，可能包含 text/input/cate/relation/id 等字段
        source: 数据来源标记，值为 "train" / "valid" / "test"

    Returns:
        标准化 dict，结构如下：
        {
          "id": str,
          "cate": str,            # 规范化后的类别名
          "input": str,           # 输入文本（原 text 字段已合并）
          "relation": [{"head": str, "relation": str, "tail": str}, ...],
          "head_types": [str],    # 可选，仅原始数据有类型信息时出现
          "tail_types": [str],    # 可选
          "source": "train"|"valid"|"test"
        }
    """
    # ① 字段名统一：原始数据可能用 "text" 或 "input"，统一为 "input"
    text = item.get("text") or item.get("input", "")
    text = text.strip()

    # ② 类别名规范化：通过 conf.CATE_MAP 替换
    cate = item.get("cate", "")
    cate = CATE_MAP.get(cate, cate)

    # ③ relation 结构对齐：三元组提取 + 保留类型信息
    raw_relations = item.get("relation", [])
    relations = []
    head_types = []
    tail_types = []

    for r in raw_relations:
        rel = {
            "head": r.get("head", "").strip(),
            "relation": r.get("relation", "").strip(),
            "tail": r.get("tail", "").strip(),
        }
        relations.append(rel)
        # head_type/tail_type 是可选字段（词性/实体类型标注），
        # 即使为空字符串也保留，保证与 relation 一一对应
        head_types.append(r.get("head_type", ""))
        tail_types.append(r.get("tail_type", ""))

    # ④ 组装标准格式
    result = {
        "id": str(item.get("id", "")),
        "cate": cate,
        "input": text,
        "relation": relations,
        "source": source,
    }

    # 仅当至少有一个非空类型信息时才附加（纯空数组无意义）
    if any(t for t in head_types):
        result["head_types"] = head_types
        result["tail_types"] = tail_types

    return result


def main():
    """归一化主入口：加载 → 标准化 → 验证 → 保存 → 出报告。

    执行流程（5 步）：
      1. 加载原始 JSON 文件（train/valid/test 三份）
      2. 逐条调用 normalize_item() 标准化字段结构
      3. 断言验证（确保 text 已变为 input、cate 已规范化、relation 为列表）
      4. 保存为 JSONL + 输出 cate 分布统计
      5. 生成 normalize_report.json（供后续脚本和人工审查参考）

    无命令行参数——所有路径和阈值从 conf.py 读取，保证全管线一致。
    """
    print("=" * 60)
    print("01_normalize.py - 标准化原始数据")
    print("=" * 60)

    # ---- 步骤 1：加载原始 JSON 文件 ----
    print("\n加载原始文件...")
    train = load_jsonl(RAW_TRAIN)
    valid = load_jsonl(RAW_VALID)
    test = load_jsonl(RAW_TEST)
    print(f"  train: {len(train)}")
    print(f"  valid: {len(valid)}")
    print(f"  test:  {len(test)}")

    # ---- 步骤 2：逐条标准化 ----
    print("\n标准化...")
    norm_train = [normalize_item(item, "train") for item in train]
    norm_valid = [normalize_item(item, "valid") for item in valid]
    norm_test = [normalize_item(item, "test") for item in test]

    # ---- 步骤 3：断言验证（防止回归） ----
    for name, data in [("train", norm_train), ("valid", norm_valid), ("test", norm_test)]:
        # 确认 text 字段已全部替换为 input
        assert all("input" in d for d in data), f"{name} 仍有 text 字段"
        # 确认 relation 字段存在且为列表
        assert all("relation" in d for d in data), f"{name} 缺 relation"
        assert all(isinstance(d["relation"], list) for d in data), f"{name} relation 不是 list"
        # 确认 cate 映射已生效（不应再出现旧名称）
        cates = set(d["cate"] for d in data)
        assert "建筑结构" not in cates, f"{name} 仍有 建筑结构 cate"

    # ---- 步骤 4：统计 cate 分布 + 保存 ----
    print("\n标准化后 cate 分布:")
    for name, data in [("train", norm_train), ("valid", norm_valid), ("test", norm_test)]:
        from collections import Counter
        cate_dist = Counter(d["cate"] for d in data)
        print(f"\n  {name} ({len(data)} 条):")
        for cate, cnt in cate_dist.most_common():
            print(f"    {cate}: {cnt}")

    print("\n保存标准化文件...")
    save_jsonl(norm_train, NORM_TRAIN)
    save_jsonl(norm_valid, NORM_VALID)
    save_jsonl(norm_test, NORM_TEST)
    print(f"  {NORM_TRAIN}")
    print(f"  {NORM_VALID}")
    print(f"  {NORM_TEST}")

    # ---- 步骤 5：生成归一化报告 ----
    report = {
        "step": "normalize",
        "input_counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "output_counts": {"train": len(norm_train), "valid": len(norm_valid), "test": len(norm_test)},
        "cate_map_applied": CATE_MAP,       # 记录应用了哪些名称映射
    }
    report_path = os.path.join(REPORT_DIR, "normalize_report.json")
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {report_path}")

    print("\n[01_normalize] 完成.")


if __name__ == "__main__":
    main()
