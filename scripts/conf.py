"""数据清洗管线的全局配置文件。

本文件为所有清洗脚本（归一化、过滤、质量分层、数据衍生、采样）提供统一的
路径映射和阈值参数，确保各阶段产物的输入/输出路径一致，阈值修改只需改一处。

清洗管线全景（从原始数据到 SFT 训练集）：

  原始 JSON（train/valid/test_zh.json）
    │
    ▼ normalize（结构归一化）
  标准化 JSONL（normalized_*.jsonl）
    │
    ▼ filter（硬过滤：空关系、超短/超长文本；软过滤：分位数截尾）
  过滤后 JSONL（filtered_*.jsonl） + filter_report.json
    │
    ▼ quality_tiering（质量分层：high / medium / low）
  分层 JSONL（tiered_train.jsonl） + quality_report.json
    │
    ▼ derive（数据衍生：schema_repair / format_following）
  衍生增强 JSONL（derived_all.jsonl） + derive_report.json
    │
    ▼ sample（采样 + 任务配比）
  最终训练集（train.jsonl / valid.jsonl） + metadata.json

每阶段都有独立脚本，通过本文件的路径常量连接成完整管线。
修改数据源路径或阈值时，只需改本文件，无需逐个脚本改动。
"""

import os

# ── 路径常量 ──────────────────────────────────────────────────────────────────
# 所有路径基于 scripts/ 的上级目录（项目根）拼接，保证跨环境可移植。

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "instructie")      # 原始数据目录
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")      # 中间产物目录
CAND_DIR = os.path.join(BASE_DIR, "data", "sft_candidate")  # 最终候选 SFT 数据集
REPORT_DIR = os.path.join(BASE_DIR, "reports")              # 各阶段清洗报告

# ---- 阶段 0：原始数据 ----
# 数据集来源：https://huggingface.co/datasets/michaelfeil/instructie
# 原始格式为 JSON 数组，每条样本包含 input/output/task_type/category 字段
RAW_TRAIN = os.path.join(RAW_DIR, "train_zh.json")        # 原始训练集（JSON 数组）
RAW_VALID = os.path.join(RAW_DIR, "valid_zh.json")        # 原始验证集
RAW_TEST = os.path.join(RAW_DIR, "test_zh.json")          # 原始测试集
RAW_SCHEMA = os.path.join(RAW_DIR, "schema_zh.json")      # 预定义 schema 模板

# ---- 阶段 1：结构归一化 ----
# 将原始 JSON 数组转为 JSONL（每行一条），统一 role/content 字段命名，
# 将原始 task_type 映射为标准任务名（ie_extraction / text_to_json / ...）
NORM_TRAIN = os.path.join(PROC_DIR, "normalized_train.jsonl")
NORM_VALID = os.path.join(PROC_DIR, "normalized_valid.jsonl")
NORM_TEST = os.path.join(PROC_DIR, "normalized_test.jsonl")

# ---- 阶段 2：硬/软过滤 ----
# 剔除脏数据：空关系、超短/超长文本、异常输出长度等
FILTERED_TRAIN = os.path.join(PROC_DIR, "filtered_train.jsonl")
FILTER_REPORT = os.path.join(REPORT_DIR, "filter_report.json")   # 统计被剔除数量及原因

# ---- 阶段 3：质量分层 ----
# 根据 head/tail 在原文中的匹配率 + 关系数量 + 输入长度，分 high / medium / low 三档
TIERED_TRAIN = os.path.join(PROC_DIR, "tiered_train.jsonl")
QUALITY_REPORT = os.path.join(REPORT_DIR, "quality_report.json")  # 各质量档位的样本分布

# ---- 阶段 4：数据衍生 ----
# 从高质量样本派生新训练数据（schema_repair：错误 schema → 修复；
# format_following：指令 → 格式化输出），增加任务多样性
DERIVED_ALL = os.path.join(PROC_DIR, "derived_all.jsonl")
DERIVE_REPORT = os.path.join(REPORT_DIR, "derive_report.json")

# ---- 阶段 5：采样 ----
# 按 task_ratio 配比采样，控制最终训练集规模
SAMPLED_TRAIN = os.path.join(PROC_DIR, "sampled_train.jsonl")
SAMPLE_REPORT = os.path.join(REPORT_DIR, "sample_report.json")

# ---- 最终交付 ----
# 供 train_sft.py 直接使用的 SFT 训练/验证集
FINAL_TRAIN = os.path.join(CAND_DIR, "train.jsonl")              # SFT 训练集
FINAL_VALID = os.path.join(CAND_DIR, "valid.jsonl")              # SFT 验证集
FINAL_METADATA = os.path.join(CAND_DIR, "metadata.json")         # 数据集统计信息

# ── 类别名称映射 ──────────────────────────────────────────────────────────────
# 将原始数据中的类别名统一为更简洁的标准名，避免后续脚本做多对一匹配
CATE_MAP = {
    "建筑结构": "建筑",
}

# ── 硬过滤阈值 ────────────────────────────────────────────────────────────────
# 「硬过滤」：不满足条件的样本直接丢弃，无论整体分布如何。
# 用于剔除明显不可用的数据（空样本、极端值），控制训练数据的质量下限。
HARD_FILTER = {
    "min_relations": 1,        # 最少关系数：空关系（0 个三元组）的样本无训练价值
    "max_relations": 25,       # 最多关系数：超过 25 个三元组的样本过于复杂，可能含噪音
    "min_input_len": 15,       # 最短输入字符数：过短的文本缺乏上下文，抽取无意义
    "max_input_len": 800,      # 最长输入字符数：过长的文本可能截断或超出模型处理能力
    "max_output_json_len": 2500,  # 输出 JSON 最大长度：防止异常巨大的标签
    "max_head_tail_len": 100,  # 单个 head/tail 实体最大字符数：过长的实体名可能是噪音
}

# ── 软过滤阈值（分位数）─────────────────────────────────────────────────────────
# 「软过滤」：基于数据整体分布，剔除落在极端分位数之外的离群样本。
# 与硬过滤不同，软过滤的阈值是统计量（分位数），而非绝对数值。
# 例如 input_len_pct=99 表示：输入长度超过 99% 样本的 1% 极端值会被剔除。
SOFT_FILTER = {
    "input_len_pct": 99,          # 输入长度分位数：截断顶部 1%
    "output_len_pct": 99,         # 输出长度分位数：截断顶部 1%
    "relation_count_pct": 99,     # 关系数量分位数：截断顶部 1%
    "head_tail_len_pct": 99,      # 实体长度分位数：截断顶部 1%
}

# ── 质量分层 ───────────────────────────────────────────────────────────────────
# 将过滤后的数据按质量分为 high / medium / low 三档。
# 分档依据三个维度：
#   1. 匹配度（match_ratio）：head/tail 实体在原文中能原文匹配的比例
#   2. 关系数（relation count）：三元组数量是否在理想区间内
#   3. 输入长度（input length）：输入文本长度是否在理想区间内
#
# 三档判定规则（满足越多维度 → 档次越高）：
#   high：   匹配度 ≥ 1.0（全部匹配），且关系数和长度都在理想区间
#   medium： 匹配度 ≥ 0.8（80% 以上匹配），其他维度暂不要求
#   low：    其余样本（匹配度 < 0.8）
QUALITY = {
    "match_ratio_high": 1.0,        # high 档要求 100% 实体在原文中能找到
    "match_ratio_medium": 0.8,      # medium 档要求 ≥80% 匹配
    "ideal_relation_range": (2, 10),   # 理想关系数区间（太少信息不足，太多可能含噪）
    "ideal_input_len_range": (30, 400),  # 理想输入长度区间（以字符计）
}

# ── 采样策略 ───────────────────────────────────────────────────────────────────
# 从清洗后的数据中按任务类型配比采样，形成最终的 SFT 训练集。
# 采样分两阶段：
#   1. 候选采样：从全量清洗数据中采样 candidate_target 条（任务配比）
#   2. 最终划分：从候选集中按 internal_valid_ratio 划分 train/valid
#
# 任务类型说明：
#   ie_extraction      信息抽取（从文本中提取三元组）—— 核心任务，占比最高
#   text_to_json       文本转 JSON（结构化输出）
#   format_following   格式遵循（按指定格式输出）
#   schema_repair      模式修复（修正错误的 schema）
SAMPLE = {
    "candidate_target": 30000,       # 候选集目标条数（全量清洗后的采样上限）
    "final_target": 15000,           # 最终训练集目标条数
    "internal_valid_ratio": 0.05,    # 内部验证集比例（从候选集中划分 5% 为 valid）
    "random_seed": 42,               # 采样随机种子（保证可复现）
    "task_ratio": {                  # 各任务类型的采样配比（总和 = 1.0）
        "ie_extraction": 0.50,       # 信息抽取占 50%
        "text_to_json": 0.25,        # 文本转 JSON 占 25%
        "format_following": 0.15,    # 格式遵循占 15%
        "schema_repair": 0.10,       # 模式修复占 10%
    },
}
