"""数据清洗管线 — 第 3 阶段：质量打分与分层。

本脚本对过滤后的训练数据逐条打分，分为 high / medium / low 三个质量档位。
后续阶段可以利用档位信息——例如数据衍生只用 high 档样本作为模板，
采样时优先使用 high 和 medium 档的样本。

质量分层依据三个维度（阈值来自 conf.QUALITY）：

  1. 匹配度（match_ratio）：head/tail 实体在原文中能原文匹配的比例。
     匹配度 = 1.0 表示所有三元组的 head 和 tail 都能在 input 原文中找到。
     匹配度低意味着实体可能是模型"脑补"出来的，标注不可靠。

  2. 关系数量（relation count）：三元组个数是否在理想区间 [2, 10]。
     太少（≤1）：信息量不足，训练信号弱。
     太多（≥11）：标注可能过于复杂，含噪音。

  3. 输入长度（input length）：原文长度是否在理想区间 [30, 400] 字符。
     太短：缺乏上下文，抽取任务退化为填空。
     太长：超出模型处理偏好区域。

三档判定规则：
  high：   匹配度 = 1.0（全部匹配），且至少 2 个维度达标
  medium： 匹配度 ≥ 0.8，且至少 1 个维度达标
  low：    其余（匹配度 < 0.8，或多维度不达标）

输入：
  - data/processed/filtered_train.jsonl  过滤后训练集

输出：
  - data/processed/tiered_train.jsonl    带 quality_tier/match_ratio/quality_dims 字段的训练集
  - reports/quality_report.json          质量分层报告（三档分布 + per-topic 质量分布）

使用方式：
    python scripts/03_quality_tier.py
"""

import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from conf import (
    FILTERED_TRAIN, TIERED_TRAIN, QUALITY_REPORT,
    QUALITY,
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


def compute_match_ratio(item):
    """计算样本中 head/tail 实体在原文中的匹配率。

    对每个三元组，检查 head 和 tail 是否都能在 item["input"] 原文中
    通过子串匹配找到。两者都在 → 该三元组为"已匹配"。
    匹配率 = 已匹配三元组数 / 三元组总数。

    匹配率是衡量标注质量的核心指标——人工标注的三元组，
    其 head 和 tail 应该是从原文中直接取出的实体名。
    如果很多实体在原文中找不到，说明标注可能不可靠（填写了原文无关的实体）。

    Args:
        item: 单条样本 dict（含 input 和 relation 字段）

    Returns:
        0.0~1.0 之间的浮点数，0.0 表示全部不匹配，1.0 表示全部匹配
    """
    text = item["input"]
    rels = item["relation"]
    if not rels:
        return 0.0
    matched = 0
    for r in rels:
        if r["head"] in text and r["tail"] in text:
            matched += 1
    return matched / len(rels)


def assign_quality_tier(item):
    """为单条样本打分并分配质量档位（high / medium / low）。

    打分流程：
      1. 计算匹配度（compute_match_ratio）
      2. 三个维度各自判定达标/不达标（bool）
      3. 计数达标维度数（score = 0~3）
      4. 按规则分档

    分档规则（config.QUALITY 中配置）：
      high：   匹配度 ≥ 1.0 且 score ≥ 2（全匹配 + 至少 2 维达标）
      medium：匹配度 ≥ 0.8 且 score ≥ 1（80%匹配 + 至少 1 维达标）
      low：    其余情况

    Args:
        item: 单条样本 dict

    Returns:
        (tier, match_ratio, dims) 三元组：
        - tier:        "high" / "medium" / "low"
        - match_ratio: 匹配度浮点数（0.0~1.0）
        - dims:        三个维度的达标判定 {"match": bool, "rel_count": bool, "input_len": bool}
    """
    match_ratio = compute_match_ratio(item)
    n_rel = len(item["relation"])
    input_len = len(item["input"])

    ideal_rel = QUALITY["ideal_relation_range"]    # (2, 10)
    ideal_len = QUALITY["ideal_input_len_range"]   # (30, 400)

    # 三个维度各自的达标判定
    dims = {
        "match": match_ratio >= QUALITY["match_ratio_high"],             # 匹配度 ≥ 1.0
        "rel_count": ideal_rel[0] <= n_rel <= ideal_rel[1],              # 关系数 ∈ [2, 10]
        "input_len": ideal_len[0] <= input_len <= ideal_len[1],          # 长度 ∈ [30, 400]
    }

    score = sum(dims.values())   # 0 ~ 3

    # 按规则分档
    if match_ratio >= QUALITY["match_ratio_high"] and score >= 2:
        tier = "high"
    elif match_ratio >= QUALITY["match_ratio_medium"] and score >= 1:
        tier = "medium"
    else:
        tier = "low"

    return tier, match_ratio, dims


def main():
    """质量分层主入口：加载 → 逐条打分 → 统计 → 保存。

    执行流程（5 步）：
      1. 加载过滤后的训练数据
      2. 逐条调用 assign_quality_tier() 打分，追加 quality_tier/match_ratio/quality_dims 字段
      3. 统计三档分布 + 匹配率 + 各维度达标率
      4. 统计 per-topic 质量分布（检查是否有主题质量系统性偏低）
      5. 保存带档位标签的 JSONL + 质量报告

    生成的 quality_tier 字段供后续阶段使用：
      - 数据衍生（04_derive.py）：只在 high 档样本上做模板派生
      - 采样（05_sample.py）：优先保留 high/medium 档样本
    """
    print("=" * 60)
    print("03_quality_tier.py - 质量打分与分层")
    print("=" * 60)

    # ---- 步骤 1：加载过滤后数据 ----
    data = load_jsonl(FILTERED_TRAIN)
    print(f"\n加载过滤后数据: {len(data)} 条")

    # ---- 步骤 2：逐条打分 ----
    print("\n质量分层中...")
    tier_counter = Counter()                    # 三档计数
    match_ratios = []                           # 所有样本的匹配度（用于算均值）
    dim_counters = {"match": 0, "rel_count": 0, "input_len": 0}  # 各维度达标计数

    for item in data:
        tier, match_ratio, dims = assign_quality_tier(item)
        # 将打分结果写入样本，后续阶段可直接读取
        item["quality_tier"] = tier              # "high" / "medium" / "low"
        item["match_ratio"] = round(match_ratio, 4)  # 保留 4 位小数
        item["quality_dims"] = dims              # {"match": bool, ...}
        tier_counter[tier] += 1
        match_ratios.append(match_ratio)
        for k, v in dims.items():
            if v:
                dim_counters[k] += 1

    # ---- 步骤 3：统计三档分布 + 匹配率 + 维度达标率 ----
    print(f"\n质量分层结果:")
    for tier in ["high", "medium", "low"]:
        cnt = tier_counter[tier]
        print(f"  {tier}: {cnt} ({cnt/len(data)*100:.1f}%)")

    print(f"\n匹配率统计:")
    print(f"  平均: {sum(match_ratios)/len(match_ratios)*100:.1f}%")
    full_match = sum(1 for r in match_ratios if r == 1.0)  # 100% 匹配的样本数
    print(f"  100%匹配: {full_match} ({full_match/len(data)*100:.1f}%)")

    print(f"\n各维度达标率:")
    for k in ["match", "rel_count", "input_len"]:
        cnt = dim_counters[k]
        print(f"  {k}: {cnt}/{len(data)} ({cnt/len(data)*100:.1f}%)")

    # ---- 步骤 4：per-topic 质量分布 ----
    # 检查是否某个主题的质量系统性偏低（如法律类全是 low）
    # 如果某个主题 low 占比过高，说明该主题标注质量需要关注
    print(f"\n各 topic 质量分布:")
    topic_tier = {}
    for item in data:
        t = item["cate"]
        q = item["quality_tier"]
        if t not in topic_tier:
            topic_tier[t] = Counter()
        topic_tier[t][q] += 1

    print(f"  {'Topic':<10} {'high':>8} {'medium':>8} {'low':>8} {'high%':>8}")
    for topic in sorted(topic_tier.keys()):
        tc = topic_tier[topic]
        total = sum(tc.values())
        print(f"  {topic:<10} {tc['high']:>8} {tc['medium']:>8} {tc['low']:>8} {tc['high']/total*100:>7.1f}%")

    # ---- 步骤 5：保存 + 报告 ----
    save_jsonl(data, TIERED_TRAIN)
    print(f"\n保存: {TIERED_TRAIN}")

    report = {
        "step": "quality_tier",
        "input_count": len(data),
        "tier_counts": dict(tier_counter),                         # {"high": N, "medium": M, "low": K}
        "dim_pass_rates": {k: f"{v/len(data)*100:.1f}%" for k, v in dim_counters.items()},
        "avg_match_ratio": f"{sum(match_ratios)/len(match_ratios)*100:.1f}%",
        "full_match_count": full_match,                             # 100% 匹配的样本数
        "topic_tier_dist": {t: dict(c) for t, c in topic_tier.items()},  # per-topic 质量分布
    }
    with open(QUALITY_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告: {QUALITY_REPORT}")

    print(f"\n[03_quality_tier] 完成.")


if __name__ == "__main__":
    main()
