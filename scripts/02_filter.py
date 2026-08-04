"""数据清洗管线 — 第 2 阶段：两层过滤。

本脚本对归一化后的数据执行两级过滤，逐层剔除不可用样本：

  硬过滤（Hard Filter）—— 规则驱动，一刀切：
    基于绝对阈值（conf.HARD_FILTER），不满足条件的直接丢弃。
    检查项：空关系、超短/超长输入、超复杂输出、过长实体名、train/valid/test 泄漏。

  软过滤（Soft Filter）—— 统计驱动，自适应：
    基于 per-topic 分位数阈值（conf.SOFT_FILTER），剔除每个主题内部的
    极端离群样本。与硬过滤不同，软过滤的阈值来自数据本身（如 P99 分位数），
    而非预设的绝对值，因此能自适应不同主题的数据分布差异。

为什么需要 per-topic（按主题）而非全局分位数：
  「建筑」主题的文本通常较短、关系数少（2-5 个）；
  「医学」主题可能更长、关系更密集（5-15 个）。
  用全局 P99 会把建筑主题的"正常但略多"样本误杀，
  而把医学主题的"异常多"样本漏掉。按主题单独计算阈值避免了跨领域偏差。

输入：
  - data/processed/normalized_train.jsonl  归一化训练集
  - data/processed/normalized_valid.jsonl  归一化验证集（仅用于泄漏检测）
  - data/processed/normalized_test.jsonl   归一化测试集（仅用于泄漏检测）

输出：
  - data/processed/filtered_train.jsonl    过滤后训练集
  - reports/filter_report.json             过滤报告（剔除明细 + 分位数表）

阈值来源：conf.py 的 HARD_FILTER 和 SOFT_FILTER 字典。

使用方式：
    python scripts/02_filter.py
"""

import json
import sys
import os
import statistics
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from conf import (
    NORM_TRAIN, NORM_VALID, NORM_TEST,
    FILTERED_TRAIN, FILTER_REPORT,
    HARD_FILTER, SOFT_FILTER,
)


def load_jsonl(path):
    """加载 JSONL 文件，返回样本列表。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def save_jsonl(data, path):
    """将样本列表保存为 JSONL 格式（每行一个 JSON 对象）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def compute_output_json_len(item):
    """计算样本 relation 字段序列化为 JSON 字符串后的长度。

    这个指标衡量输出标签的复杂度——三元组越多、实体名越长，
    relation JSON 就越长。过长的输出可能超出模型生成长度限制。

    Args:
        item: 单条样本 dict

    Returns:
        relation 的 JSON 字符串长度（字符数）
    """
    return len(json.dumps(item["relation"], ensure_ascii=False))


def compute_max_head_tail_len(item):
    """计算样本中所有 head/tail 实体的最大字符数。

    三元组的 head 和 tail 应该是紧凑的实体名（如"张三"、"北京"）。
    如果某个实体名异常长（如 200+ 字符），通常是数据标注错误。
    本函数取所有 relation 中 head 和 tail 长度的最大值，用于硬过滤。

    Args:
        item: 单条样本 dict

    Returns:
        最大的 head 或 tail 字符数
    """
    max_len = 0
    for r in item["relation"]:
        max_len = max(max_len, len(r["head"]), len(r["tail"]))
    return max_len


def compute_per_topic_quantiles(data, fields, pcts):
    """计算每个 topic 在各统计字段上的分位数阈值。

    分位数计算流程：
      1. 按 topic 分组，收集每个样本的各字段值
      2. 每个 topic 内部对字段值排序
      3. 对每个目标分位数（如 P75、P90、P95、P99），取排序后对应位置的值

    例如，P99=150 意味着该 topic 下 99% 的样本该字段值 ≤ 150。

    Args:
        data:   训练集样本列表
        fields: 统计字段定义 {"field_name": lambda item: value, ...}
        fields = {
            "input_len":      lambda item: len(item["input"]),        # 样本输入文本长度
            "output_len":     lambda item: len(json.dumps(item["relation"])),  # relation JSON 长度
            "relation_count": lambda item: len(item["relation"]),     # 三元组个数
            "head_tail_len":  lambda item: max(len(r["head"]), len(r["tail"])),  # 最长实体名
        }
        pcts:   目标分位数列表，如 [75, 90, 95, 99]

        Returns:
            嵌套字典 {topic: {field_name: {"P75": val, "P90": val, ...}}}
            最终返回的结构

            {
                "建筑": {
                    "input_len":      {"P75": 150, "P90": 200, "P95": 230, "P99": 250},
                    "output_len":     {"P75": 60,  "P90": 80,  "P95": 95,  "P99": 110},
                    "relation_count": {"P75": 3,   "P90": 5,   "P95": 6,   "P99": 8},
                    "head_tail_len":  {"P75": 4,   "P90": 6,   "P95": 7,   "P99": 10},
                },
                "医学": {
                    "input_len":      {"P75": 400, "P90": 520, "P95": 600, "P99": 650},
                    ...
                },
            }
    """
    # 步骤 1：按 topic 分组收集字段值
    topic_values = defaultdict(lambda: defaultdict(list))
    for item in data:
        topic = item["cate"]
        for fname, fn in fields.items():
            topic_values[topic][fname].append(fn(item))

    # 步骤 2+3：排序后按分位点取值
    result = {}
    for topic in sorted(topic_values.keys()):
        result[topic] = {}
        for fname in sorted(fields.keys()):
            vals = sorted(topic_values[topic][fname])
            result[topic][fname] = {}
            for p in pcts:
                idx = min(int(len(vals) * p / 100), len(vals) - 1)
                result[topic][fname][f"P{p}"] = vals[idx]
    return result


def hard_filter(item, reasons):
    """硬过滤：基于绝对阈值的规则过滤，不满足任一条件即丢弃。

    检查项（阈值来自 conf.HARD_FILTER）：
      1. min_relations：至少 1 个三元组（空关系无训练价值）
      2. max_relations：不超过 25 个三元组（过于复杂可能是噪音）
      3. min_input_len：输入文本至少 15 字符（太短无上下文）
      4. max_input_len：输入文本不超过 800 字符（太长可能截断）
      5. max_output_json_len：relation JSON 不超过 2500 字符
      6. max_head_tail_len：单个实体名不超过 100 字符

    Args:
        item:    单条样本 dict
        reasons: 外部传入的列表，不通过时追加剔除原因（用于统计报告）

    Returns:
        True 表示保留（通过所有检查），False 表示丢弃
    """
    rels = item["relation"]
    n_rel = len(rels)
    input_len = len(item["input"])

    if n_rel < HARD_FILTER["min_relations"]:
        reasons.append("empty_relation")
        return False
    if n_rel > HARD_FILTER["max_relations"]:
        reasons.append("too_many_relations")
        return False
    if input_len < HARD_FILTER["min_input_len"]:
        reasons.append("too_short_input")
        return False
    if input_len > HARD_FILTER["max_input_len"]:
        reasons.append("too_long_input")
        return False
    if compute_output_json_len(item) > HARD_FILTER["max_output_json_len"]:
        reasons.append("too_long_output")
        return False
    if compute_max_head_tail_len(item) > HARD_FILTER["max_head_tail_len"]:
        reasons.append("too_long_head_tail")
        return False

    return True


def soft_filter(item, topic_thresholds, reasons):
    """软过滤：基于 per-topic 分位数阈值的自适应过滤。

    对每个样本，用其所属 topic 的分位数上限（P99）做检查。
    超过 P99 的视为该主题下的离群值并丢弃。

    与硬过滤的关键区别：
      - 硬过滤对所有主题一刀切（如 input_len > 800 全丢）；
      - 软过滤不同主题有不同阈值——建筑主题的 P99 可能是 500，
        而医学主题的 P99 可能是 700。

    Args:
        item:              单条样本 dict
        topic_thresholds:  per-topic 分位数阈值，
                           {topic: {field: {"soft_max": value}}}
        reasons:           外部传入的列表，不通过时追加剔除原因

    Returns:
        True 表示保留（未超任何阈值），False 表示丢弃
    """
    topic = item["cate"]
    if topic not in topic_thresholds:
        return True  # 无阈值则不过滤（罕见 topic 不丢弃）

    thresholds = topic_thresholds[topic]

    # 输入长度检查
    input_len = len(item["input"])
    if "input_len" in thresholds:
        max_val = thresholds["input_len"]["soft_max"]
        if input_len > max_val:
            reasons.append(f"soft_input_len_exceed")
            return False

    # 关系数量检查
    n_rel = len(item["relation"])
    if "relation_count" in thresholds:
        max_val = thresholds["relation_count"]["soft_max"]
        if n_rel > max_val:
            reasons.append("soft_relation_count_exceed")
            return False

    # 输出长度检查
    output_len = compute_output_json_len(item)
    if "output_len" in thresholds:
        max_val = thresholds["output_len"]["soft_max"]
        if output_len > max_val:
            reasons.append("soft_output_len_exceed")
            return False

    # 实体名长度检查
    ht_len = compute_max_head_tail_len(item)
    if "head_tail_len" in thresholds:
        max_val = thresholds["head_tail_len"]["soft_max"]
        if ht_len > max_val:
            reasons.append("soft_head_tail_len_exceed")
            return False

    return True


def main():
    """两层过滤主入口：加载 → 泄漏检测 → 计算分位数 → 硬过滤 → 软过滤 → 保存。

    执行流程（6 步）：
      1. 加载归一化后的 train/valid/test 数据
      2. 构建 valid + test 文本集合，用于泄漏检测（train 不能含它们的 text）
      3. 计算 train 的 per-topic 分位数（P75/P90/P95/P99），为软过滤准备阈值
      4. 硬过滤：绝对阈值规则 + 泄漏检测，不通过的丢弃
      5. 软过滤：per-topic P99 阈值，超过的丢弃
      6. 保存过滤后数据（JSONL）+ 输出过滤报告（JSON）

    为什么只对 train 做过滤：
      valid 和 test 用于评估，不应做任何过滤——保留真实分布才能准确
      衡量模型在原始数据上的表现。过滤只在训练集上做。
    """
    print("=" * 60)
    print("02_filter.py - 两层过滤")
    print("=" * 60)

    # ---- 步骤 1：加载标准化数据 ----
    print("\n加载标准化数据...")
    train = load_jsonl(NORM_TRAIN)
    valid = load_jsonl(NORM_VALID)
    test = load_jsonl(NORM_TEST)
    print(f"  train: {len(train)}")
    print(f"  valid: {len(valid)}")
    print(f"  test:  {len(test)}")

    # ---- 步骤 2：泄漏检测 ----
    # 确保训练集中不包含与 valid/test 完全相同的文本
    # （同一文本出现在不同 split 中会导致评估结果虚高）
    print("\n[泄漏检测]")
    valid_texts = set(item["input"] for item in valid)
    test_texts = set(item["input"] for item in test)
    leak_texts = valid_texts | test_texts
    print(f"  valid 唯一文本: {len(valid_texts)}")
    print(f"  test 唯一文本: {len(test_texts)}")

    # ---- 步骤 3：计算 per-topic 分位数 ----
    print("\n[计算 per-topic 分位数]")
    pct = SOFT_FILTER["input_len_pct"]  # 默认 99（即 P99）

    # 四个统计字段及其提取函数
    fields = {
        "input_len": lambda item: len(item["input"]),
        "output_len": lambda item: compute_output_json_len(item),
        "relation_count": lambda item: len(item["relation"]),
        "head_tail_len": lambda item: compute_max_head_tail_len(item),
    }

    # 计算 P75/P90/P95/P99（用于报告展示 + 软过滤阈值）
    quantiles = compute_per_topic_quantiles(train, fields, [75, 90, 95, 99])

    # 打印分位数表（快速了解各主题的数据分布差异）
    print(f"\n  分位数统计 (P{pct} 用作软上限):")
    print(f"  {'Topic':<10} {'输入长度P99':>12} {'输出长度P99':>12} {'关系数P99':>10} {'ht长度P99':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
    for topic in sorted(quantiles.keys()):
        q = quantiles[topic]
        print(f"  {topic:<10} {q['input_len'][f'P{pct}']:>12} "
              f"{q['output_len'][f'P{pct}']:>12} "
              f"{q['relation_count'][f'P{pct}']:>10} "
              f"{q['head_tail_len'][f'P{pct}']:>10}")

    # 从分位数表提取软过滤阈值（只用 P99）
    topic_thresholds = {}
    for topic in quantiles:
        topic_thresholds[topic] = {}
        for fname in quantiles[topic]:
            soft_max = quantiles[topic][fname][f"P{pct}"]
            topic_thresholds[topic][fname] = {"soft_max": soft_max}

    # ---- 步骤 4：硬过滤（绝对阈值 + 泄漏检测） ----
    print(f"\n[硬过滤] 开始...")
    hard_reason_counter = Counter()
    hard_pass = []
    for item in train:
        reasons = []
        if hard_filter(item, reasons):
            # 硬规则通过后，额外做泄漏检测
            # （泄漏检测放在硬过滤阶段而非单独一步，复用循环减少遍历）
            if item["input"] in leak_texts:
                reasons.append("leak_with_valid_test")
                hard_reason_counter["leak_with_valid_test"] += 1
            else:
                hard_pass.append(item)
        for r in reasons:
            hard_reason_counter[r] += 1

    print(f"  硬过滤前: {len(train)}")
    print(f"  硬过滤后: {len(hard_pass)}")
    print(f"  剔除明细:")
    for reason, cnt in hard_reason_counter.most_common():
        print(f"    {reason}: {cnt}")
    total_hard = sum(hard_reason_counter.values())
    print(f"    总剔除: {total_hard} (含交叉，一条样本可能命中多条规则)")

    # ---- 步骤 5：软过滤（per-topic 分位数阈值） ----
    print(f"\n[软过滤] 开始 (per-topic P{pct} 阈值)...")
    soft_reason_counter = Counter()
    soft_pass = []
    for item in hard_pass:
        reasons = []
        if soft_filter(item, topic_thresholds, reasons):
            soft_pass.append(item)
        for r in reasons:
            soft_reason_counter[r] += 1

    print(f"  软过滤前: {len(hard_pass)}")
    print(f"  软过滤后: {len(soft_pass)}")
    print(f"  剔除明细:")
    for reason, cnt in soft_reason_counter.most_common():
        print(f"    {reason}: {cnt}")

    # ---- 步骤 6：统计 + 保存 + 出报告 ----
    print(f"\n[过滤后 topic 分布]")
    filtered_cate = Counter(item["cate"] for item in soft_pass)
    for cate, cnt in filtered_cate.most_common():
        orig = sum(1 for t in train if t["cate"] == cate)
        print(f"  {cate}: {cnt} / {orig} (保留 {cnt/orig*100:.1f}%)")

    # 保存过滤后的训练集
    print(f"\n保存过滤后数据...")
    save_jsonl(soft_pass, FILTERED_TRAIN)
    print(f"  {FILTERED_TRAIN}")

    # 生成过滤报告（记录每个阶段淘汰的样本数及原因，供审查）
    report = {
        "step": "filter",
        "input_count": len(train),
        "after_hard_filter": len(hard_pass),
        "after_soft_filter": len(soft_pass),
        "hard_filter_reasons": dict(hard_reason_counter),
        "soft_filter_reasons": dict(soft_reason_counter),
        "per_topic_quantiles": {
            topic: {fname: vals for fname, vals in topic_data.items()}
            for topic, topic_data in quantiles.items()
        },
        "filtered_topic_dist": dict(filtered_cate),
    }
    os.makedirs(os.path.dirname(FILTER_REPORT), exist_ok=True)
    with open(FILTER_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  {FILTER_REPORT}")

    print(f"\n[02_filter] 完成. {len(train)} -> {len(soft_pass)} (保留 {len(soft_pass)/len(train)*100:.1f}%)")


if __name__ == "__main__":
    main()
