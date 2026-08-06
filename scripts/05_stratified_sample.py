"""数据清洗管线 — 第 5 阶段：分层采样。

本脚本是清洗管线的最后一步，从四类派生任务中按多层策略采样，
形成最终的 SFT 训练集。采样不是在数量上随便砍一刀，而是在四个维度
上同时做平衡控制：

  1. 任务类型配比（task_type ratio）—— conf.SAMPLE["task_ratio"]
     ie_extraction 50% / text_to_json 25% / format_following 15% / schema_repair 10%

  2. 质量优先级（quality priority）—— high > medium > low
     优先从高质量样本中选取，确保训练信号可靠

  3. topic 均衡（topic balance）—— 每个 topic 等比例分配采样配额
     防止某个主题（如"建筑"）因样本多而挤占其他主题的空间

  4. 复杂度控制（complexity control）—— medium > simple > complex
     优先选中等复杂度样本（关系数适中、输入长度适中），兼顾学习效率和泛化能力

采样算法（per task_type）：
  a. 按 (quality_tier, topic, complexity) 三维分桶
  b. 每个 topic 分配等量配额（per_topic_target = task_target / n_topics）
  c. 桶内按 quality → complexity 优先级顺序取样本
  d. 不足时从该 task_type 的剩余样本中随机补充
  e. 超出时随机裁剪

复杂度分桶规则（classify_complexity）：
  simple：  关系数 ≤ 3 且 输入长度 < 100
  medium：  关系数 ≤ 6 且 输入长度 < 250
  complex： 其余（高关系数或长文本）

输入：
  - data/processed/derived_all.jsonl  所有派生任务样本

输出：
  - data/processed/sampled_train.jsonl  采样后的候选训练集
  - reports/sample_report.json          采样报告（四维分布统计）

配置来源：conf.py 的 SAMPLE 字典。

使用方式：
    python scripts/05_stratified_sample.py
"""

import json
import sys
import os
import random
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from conf import DERIVED_ALL, SAMPLED_TRAIN, SAMPLE_REPORT, SAMPLE


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


def classify_complexity(item):
    """按关系数和输入长度将样本分为三个复杂度桶。

    分桶阈值（经验值，配合 01~04 阶段的过滤阈值设计）：
      simple：  关系数 ≤ 3 且 输入 < 100 字符
                —— 抽取任务简单，适合低资源场景下的基础学习
      medium：  关系数 ≤ 6 且 输入 < 250 字符
                —— 适中的抽取难度，模型训练的主要信号来源
      complex： 关系数 > 6 或 输入 ≥ 250 字符
                —— 高难度样本，用于提升模型上限但不宜过多

    采样时优先选 medium（兼顾学习效率和泛化），simple 次之，
    complex 最后（难样本太多会导致训练不稳定）。

    Args:
        item: 单条派生任务 dict（含 n_relations 和 input_len 字段）

    Returns:
        "simple" / "medium" / "complex"
    """
    n_rel = item["n_relations"]
    input_len = item["input_len"]

    if n_rel <= 3 and input_len < 100:
        return "simple"
    elif n_rel <= 6 and input_len < 250:
        return "medium"
    else:
        return "complex"


def main():
    """分层采样主入口：加载 → 计算配额 → 分组 → 分层采样 → 保存。

    执行流程（6 步）：
      1. 加载派生数据 + 计算各 task_type 的目标采样数
      2. 按 task_type 分组
      3. 每个 task_type 内做四维分层采样（quality × topic × complexity）
      4. 不足时从剩余池补充，超出时随机裁剪
      5. 统计四维分布（任务/主题/质量/复杂度）
      6. 保存采样结果 + 报告

    采样优先级链（在每个 topic 内部）：
      quality: high → medium → low
      complexity: medium → simple → complex
      先按 quality 遍历，同 quality 内按 complexity 遍历，
      形成 high-medium > high-simple > high-complex > medium-medium > ... 的优先级。
    """
    print("=" * 60)
    print("05_stratified_sample.py - 分层采样")
    print("=" * 60)

    random.seed(SAMPLE["random_seed"])

    # ---- 步骤 1：加载数据 + 计算任务配额 ----
    data = load_jsonl(DERIVED_ALL)
    print(f"\n加载派生数据: {len(data)} 条")

    target = SAMPLE["candidate_target"]    # 候选集目标条数（默认 30000）
    task_ratio = SAMPLE["task_ratio"]
    print(f"目标采样数: {target}")
    print(f"任务配比: {task_ratio}")

    # 计算每个 task_type 的目标数量
    # 例：target=30000, ie_extraction=0.5 → 15000 条
    task_targets = {}
    for tt, ratio in task_ratio.items():
        task_targets[tt] = int(target * ratio)

    print(f"\n各任务目标数:")
    for tt, t in task_targets.items():
        print(f"  {tt}: {t}")

    # ---- 步骤 2：按 task_type 分组 ----
    by_task = defaultdict(list)
    for item in data:
        by_task[item["task_type"]].append(item)

    print(f"\n各任务可用数:")
    for tt, items in by_task.items():
        print(f"  {tt}: {len(items)}")

    # ---- 步骤 3+4：每个 task_type 内分层采样 ----
    # 采样优先级（按此顺序从桶中取）：
    #   quality:  high → medium → low
    #   complexity: medium → simple → complex
    #   topic:    等比例分配
    sampled = []

    for tt, items in by_task.items():
        tt_target = task_targets.get(tt, 0)
        if tt_target == 0:
            continue

        print(f"\n采样 {tt} (目标 {tt_target})...")

        # ---- 步骤 3a：三维分桶 (quality_tier, topic, complexity) ----
        buckets = defaultdict(list)
        for item in items:
            quality = item.get("quality_tier", "medium")
            topic = item["cate"]
            complexity = classify_complexity(item)
            key = (quality, topic, complexity)
            buckets[key].append(item)

        # ---- 步骤 3b：计算 per-topic 等比例配额 ----
        topic_counts = Counter(item["cate"] for item in items)
        topics = sorted(topic_counts.keys())
        # 每个 topic 至少分 1 条，总配额在各 topic 间均分
        per_topic_target = max(1, tt_target // len(topics))

        quality_order = ["high", "medium", "low"]
        complexity_order = ["medium", "simple", "complex"]

        tt_sampled = []

        # ---- 步骤 3c：按 quality → complexity 优先级采样 ----
        for topic in topics:
            topic_collected = []
            remaining = per_topic_target

            for q in quality_order:
                if remaining <= 0:
                    break
                for c in complexity_order:
                    if remaining <= 0:
                        break
                    key = (q, topic, c)
                    if key in buckets and buckets[key]:
                        pool = buckets[key]
                        n_take = min(remaining, len(pool))
                        random.shuffle(pool)                 # 桶内随机打散
                        topic_collected.extend(pool[:n_take])
                        remaining -= n_take

            tt_sampled.extend(topic_collected)

        # ---- 步骤 4a：不足时从剩余池随机补充 ----
        # 某些 task_type × topic 组合可能样本不够，差额从该 task_type
        # 全量未使用样本中随机补
        if len(tt_sampled) < tt_target:
            sampled_ids = set(
                (d["original_id"], d["task_type"]) for d in tt_sampled
            )
            remaining_pool = [
                d for d in items
                if (d["original_id"], d["task_type"]) not in sampled_ids
            ]
            random.shuffle(remaining_pool)
            n_more = min(tt_target - len(tt_sampled), len(remaining_pool))
            tt_sampled.extend(remaining_pool[:n_more])

        # ---- 步骤 4b：超出时随机裁剪 ----
        if len(tt_sampled) > tt_target:
            random.shuffle(tt_sampled)
            tt_sampled = tt_sampled[:tt_target]

        sampled.extend(tt_sampled)
        print(f"  采样结果: {len(tt_sampled)}")

    # ---- 步骤 5：四维分布统计 ----
    print(f"\n{'='*60}")
    print(f"采样结果汇总")
    print(f"{'='*60}")
    print(f"总采样数: {len(sampled)}")

    # 任务类型分布
    print(f"\n任务分布:")
    tt_dist = Counter(d["task_type"] for d in sampled)
    for tt in ["ie_extraction", "text_to_json", "format_following", "schema_repair"]:
        cnt = tt_dist.get(tt, 0)
        print(f"  {tt}: {cnt} ({cnt/len(sampled)*100:.1f}%)")

    # topic 分布（检查是否有 topic 被过度/不足采样）
    print(f"\nTopic 分布:")
    topic_dist = Counter(d["cate"] for d in sampled)
    for topic, cnt in topic_dist.most_common():
        print(f"  {topic}: {cnt} ({cnt/len(sampled)*100:.1f}%)")

    # 质量分布（应呈现 high > medium > low 的递减趋势）
    print(f"\n质量分布:")
    q_dist = Counter(d["quality_tier"] for d in sampled)
    for q in ["high", "medium", "low"]:
        cnt = q_dist.get(q, 0)
        print(f"  {q}: {cnt} ({cnt/len(sampled)*100:.1f}%)")

    # 复杂度分布（medium 应占主导）
    print(f"\n复杂度分布:")
    c_dist = Counter(classify_complexity(d) for d in sampled)
    for c in ["simple", "medium", "complex"]:
        cnt = c_dist.get(c, 0)
        print(f"  {c}: {cnt} ({cnt/len(sampled)*100:.1f}%)")

    # ---- 步骤 6：保存 ----
    save_jsonl(sampled, SAMPLED_TRAIN)
    print(f"\n保存: {SAMPLED_TRAIN}")

    report = {
        "step": "stratified_sample",
        "input_count": len(data),
        "sampled_count": len(sampled),
        "target": target,
        "task_targets": task_targets,                     # 配置的目标配比
        "actual_task_dist": dict(tt_dist),                # 实际采样后的任务分布
        "topic_dist": dict(topic_dist),                   # 各主题采样数
        "quality_dist": dict(q_dist),                     # 各质量档位采样数
        "complexity_dist": dict(c_dist),                  # 各复杂度桶采样数
    }
    with open(SAMPLE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告: {SAMPLE_REPORT}")

    print(f"\n[05_stratified_sample] 完成. {len(data)} -> {len(sampled)}")




if __name__ == "__main__":
    main()
