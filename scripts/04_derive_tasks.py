"""数据清洗管线 — 第 4 阶段：从高质量样本派生四类 SFT 任务。

本脚本的核心思路：原始 instructie 数据集只提供了标准信息抽取（ie_extraction）
一种任务格式，但实际 SFT 训练需要多种任务形态来增强指令跟随的泛化能力。
因此，从过滤后数据中选取高质量样本作为"模板"，变换 prompt 和输出格式，
派生出另外三类辅助任务。

四类任务：

  1. ie_extraction（信息抽取）—— 基础任务，所有质量等级的样本都派生。
     Prompt 格式：「根据 schema 从文本中抽取信息，以 JSON 输出。」

  2. text_to_json（文本转 JSON）—— 强调结构化输出的格式正确性。
     Prompt 中包含更详细的 JSON 格式约束说明，仅 medium 及以上样本派生。

  3. format_following（格式遵循）—— 强调严格按指令输出，禁止多余文字。
     Prompt 首句随机选择一个"只输出 JSON"的变体约束，仅 high 样本派生。

  4. schema_repair（模式修复）—— 对正确输出做可控扰动（拼写错误/缺失字段/
     幻觉字段/类型错误），构造"找出并修正错误"的纠错任务。
     仅 high 样本且关系数 ≥ 3 时派生（太少关系的样本扰动空间不足）。

派生策略（按质量档位逐级放宽）：
  high 样本：   派生全部 4 类（ie + json + format + repair）
  medium 样本： 派生 2 类（ie + json）
  low 样本：    仅派生 1 类（ie）

这样设计的好处：
  - 高质量样本承载更多任务类型，充分利用可靠标注
  - 低质量样本只做基础 IE，确保每条数据至少被用一次
  - 自然形成任务数量梯度（IE > JSON > Format > Repair），
    后续采样时可在 05_sample.py 中精确控制配比

扰动类型（schema_repair 的 4 种可控错误，见 perturb_output 函数）：
  1. 字段名拼写错误：替换目标字段名中的一个字符
  2. 缺失字段：从一个实体的输出中删除一个字段
  3. 幻觉字段：添加一个 schema 中存在但原文不存在的字段
  4. 类型错误：将字符串值替换为列表

输入：
  - data/processed/tiered_train.jsonl   带质量档位的训练集
  - data/instructie/schema_zh.json      预定义 schema 模板（短字段名列表）

输出：
  - data/processed/derived_all.jsonl    所有派生样本（含原始 IE 复制）
  - reports/derive_report.json          派生报告（四类任务数量 + topic 分布）

使用方式：
    python scripts/04_derive_tasks.py
"""

import json
import sys
import os
import random
import copy
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from conf import (
    TIERED_TRAIN, RAW_SCHEMA, DERIVED_ALL, DERIVE_REPORT,
    SAMPLE, CATE_MAP,
)

# 类别反向映射：规范名 → 原始名（用于在 schema.json 中查找）
# CATE_MAP 是 {"建筑结构": "建筑"}，反向得到 {"建筑": "建筑结构"}
SCHEMA_KEY_MAP = {v: k for k, v in CATE_MAP.items()}


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


def load_schema(path):
    """加载预定义 schema 模板文件。

    schema_zh.json 结构：
      {"建筑结构": [["head_type", "tail_type"], ["field1", "field2", ...]], ...}
      每个 topic 对应一个元组：[类型定义, 短字段名列表]。
      本项目中只用第二个元素（短字段名列表）。

    Args:
        path: schema JSON 文件路径

    Returns:
        schema 字典 {topic: [types_list, field_names_list]}
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def relations_to_output_json(relations):
    """将三元组列表按实体分组，转为紧凑的嵌套 JSON 字符串。

    原始 relation 是平铺的三元组列表：
        [{"head":"张三","relation":"出生于","tail":"北京"},
         {"head":"张三","relation":"任职于","tail":"阿里巴巴"}]

    转换后按 head 实体分组，同关系的多个值聚合：
        {"张三": {"出生于": "北京", "任职于": "阿里巴巴"}}

    格式规则：
      - 同一 head 的多个三元组聚合为一个对象
      - 同一 relation 下的多个 tail → 列表；单个 tail → 展开为字符串
      - 输出为紧凑 JSON（无缩进），适合作为模型标签

    Args:
        relations: 三元组列表 [{"head": str, "relation": str, "tail": str}, ...]

    Returns:
        按实体分组的 JSON 字符串
    """
    # 步骤 1：按 head 实体分组
    entity_map = defaultdict(list)
    for r in relations:
        entity_map[r["head"]].append({"relation": r["relation"], "tail": r["tail"]})

    # 步骤 2：每个实体内按 relation 聚合 tail 值
    result = {}
    for entity, rels in entity_map.items():
        rel_dict = defaultdict(list)
        for r in rels:
            rel_dict[r["relation"]].append(r["tail"])
        # 单值展开：只有一个 tail 的 relation 不需要用列表包裹
        for k, v in rel_dict.items():
            if len(v) == 1:
                rel_dict[k] = v[0]
        result[entity] = dict(rel_dict)

    return json.dumps(result, ensure_ascii=False, indent=None)


def get_schema_for_topic(schema, cate):
    """获取指定 topic 的 schema 短字段名列表。

    schema 中的 key 是原始 topic 名（如"建筑结构"），而数据中的 cate
    已经过 CATE_MAP 标准化（如"建筑"）。本函数先直接匹配，不命中时
    用 SCHEMA_KEY_MAP 反向查找。

    Args:
        schema: load_schema() 返回的 schema 字典
        cate:   标准化的类别名（如 "建筑"）

    Returns:
        短字段名列表（如 ["地址", "建筑面积", "结构类型", ...]），
        找不到时返回空列表
    """
    if cate in schema:
        return schema[cate][1]           # [types, field_names] → 取 field_names
    schema_key = SCHEMA_KEY_MAP.get(cate)
    if schema_key and schema_key in schema:
        return schema[schema_key][1]
    return []


def make_ie_extraction(item, schema_fields):
    """任务 A：标准信息抽取 —— 根据 schema 从文本中抽取信息。

    最基础的 IE 任务格式，与原始 instructie 数据集风格一致。
    所有质量等级的样本都会生成这类任务。

    Prompt 模板：
        "你是一个信息抽取助手。请根据给定的 schema 从文本中抽取信息，
         并以 JSON 格式输出。\nSchema: [...] \n文本: ..."

    Args:
        item:          单条训练样本 dict
        schema_fields: 该 topic 的 schema 短字段名列表

    Returns:
        派生任务 dict（含 task_type/prompt/output 等字段）
    """
    schema_str = json.dumps(schema_fields, ensure_ascii=False)
    prompt = (
        "你是一个信息抽取助手。请根据给定的 schema 从文本中抽取信息，并以 JSON 格式输出。\n"
        f"Schema: {schema_str}\n"
        f"文本: {item['input']}"
    )
    output = relations_to_output_json(item["relation"])
    return {
        "task_type": "ie_extraction",
        "original_id": item["id"],
        "cate": item["cate"],
        "quality_tier": item["quality_tier"],
        "n_relations": len(item["relation"]),
        "input_len": len(item["input"]),
        "prompt": prompt,
        "output": output,
    }


def make_text_to_json(item, schema_fields):
    """任务 B：文本转结构化 JSON —— 强调输出格式的正确性。

    与 ie_extraction 的核心区别：prompt 中显式描述 JSON 结构要求
    （字段名 = 关系类型、单值展开、多值列表、按实体分组），
    训练模型关注输出的结构正确性而不仅是内容正确。

    仅对 medium 及以上质量样本生成。

    Args:
        item:          单条训练样本 dict
        schema_fields: 该 topic 的 schema 短字段名列表

    Returns:
        派生任务 dict
    """
    schema_str = json.dumps(schema_fields, ensure_ascii=False)
    prompt = (
        "请将以下文本中的信息按照指定 schema 转换为结构化 JSON 对象。\n"
        f"Schema 字段: {schema_str}\n"
        f"要求: 输出合法 JSON，字段名为 schema 中定义的关系类型，值为抽取结果。"
        f"同一实体的多个关系值用列表表示，单个值直接使用字符串。按实体分组。\n"
        f"文本: {item['input']}"
    )
    output = relations_to_output_json(item["relation"])
    return {
        "task_type": "text_to_json",
        "original_id": item["id"],
        "cate": item["cate"],
        "quality_tier": item["quality_tier"],
        "n_relations": len(item["relation"]),
        "input_len": len(item["input"]),
        "prompt": prompt,
        "output": output,
    }


def make_format_following(item, schema_fields):
    """任务 C：格式遵循 —— 强调严格按指令输出，禁止额外文字。

    与 text_to_json 的进一步区别：prompt 首句从 4 种"只输出 JSON"
    的约束变体中随机选取一条（增加指令多样性），训练模型学会在
    不同措辞下都遵守格式约束。

    仅对 high 质量样本生成。

    Args:
        item:          单条训练样本 dict
        schema_fields: 该 topic 的 schema 短字段名列表

    Returns:
        派生任务 dict
    """
    # 4 种语义等价但措辞不同的格式约束，随机选取增加多样性
    constraints = [
        "只输出 JSON，不要附加任何解释文字。",
        "只输出 JSON 格式的结果，不要包含任何额外说明。",
        "严格按照 JSON 格式输出，不要在 JSON 前后添加任何文字。",
        "仅输出结构化 JSON 数据，禁止附加解释、标注或格式化标记。",
    ]
    constraint = random.choice(constraints)

    schema_str = json.dumps(schema_fields, ensure_ascii=False)
    prompt = (
        f"{constraint}\n"
        f"Schema: {schema_str}\n"
        f"从文本中抽取信息并输出 JSON: {item['input']}"
    )
    output = relations_to_output_json(item["relation"])
    return {
        "task_type": "format_following",
        "original_id": item["id"],
        "cate": item["cate"],
        "quality_tier": item["quality_tier"],
        "n_relations": len(item["relation"]),
        "input_len": len(item["input"]),
        "prompt": prompt,
        "output": output,
    }


def perturb_output(output_str, schema_fields, relations):
    """对正确的 JSON 输出做可控扰动，生成 schema_repair 纠错任务。

    四种扰动按优先级依次尝试，成功一种就立即返回（不叠加多种错误）：
      优先 1：字段名拼写错误 —— 随机替换目标字段名中一个字符的码点
      优先 2：缺失一个字段 —— 从使用了 schema 字段的实体中删除一个字段
      优先 3：添加幻觉字段 —— 在 schema 中存在但样本未使用的字段中挑一个插入
      优先 4：类型错误 —— 将某个字符串值替换为列表（如 "北京" → ["北京","多余值"]）

    优先级的设计逻辑：拼写错误和缺失字段更接近真实 error 场景，
    幻觉字段和类型错误是更强的对抗样本，排在后面兜底。

    为什么低关系数样本不做扰动：三元组 < 3 个时，输出的 JSON 结构很浅，
    可选的扰动点太少，扰动后要么没变化要么完全破坏结构。

    Args:
        output_str:    正确的 JSON 输出字符串（relations_to_output_json 的结果）
        schema_fields: 该 topic 的 schema 短字段名列表
        relations:     原始三元组列表（暂未使用，预留给更复杂的扰动逻辑）

    Returns:
        (perturbed_json_str, perturbation_desc) 元组，
        扰动成功时 desc 为中文错误描述，扰动失败时返回 (None, None)
    """
    try:
        output_obj = json.loads(output_str)    # 解析为 dict 以便修改
    except json.JSONDecodeError:
        return None, None

    perturbation_types = []

    # 分析当前输出的字段使用情况
    used_rel_types = set()
    for entity, rels in output_obj.items():
        if isinstance(rels, dict):
            used_rel_types.update(rels.keys())

    available_schema = set(schema_fields)
    used_in_schema = used_rel_types & available_schema     # 使用了且属于 schema 的字段
    not_used_in_schema = available_schema - used_rel_types  # schema 中存在但未使用的（幻觉候选）

    # ---- 扰动 1：字段名拼写错误 ----
    if used_in_schema:
        target_field = random.choice(list(used_in_schema))
        perturbed = copy.deepcopy(output_obj)
        for entity in perturbed:
            if isinstance(perturbed[entity], dict) and target_field in perturbed[entity]:
                # 随机替换字段名中的一个字符（偏移 ±1 或 ±2 码点）
                field_chars = list(target_field)
                if len(field_chars) > 1:
                    idx = random.randint(0, len(field_chars) - 1)
                    field_chars[idx] = chr(ord(field_chars[idx]) + random.choice([1, -1, 2]))
                wrong_field = "".join(field_chars)
                perturbed[entity][wrong_field] = perturbed[entity].pop(target_field)
                perturbation_desc = f"字段名 '{target_field}' 被错误写成了 '{wrong_field}'"
                return json.dumps(perturbed, ensure_ascii=False), perturbation_desc

    # ---- 扰动 2：缺失一个字段 ----
    if used_in_schema and len(used_in_schema) > 1:
        target_field = random.choice(list(used_in_schema))
        perturbed = copy.deepcopy(output_obj)
        for entity in perturbed:
            if isinstance(perturbed[entity], dict) and target_field in perturbed[entity]:
                del perturbed[entity][target_field]
        perturbation_desc = f"缺少字段 '{target_field}'"
        return json.dumps(perturbed, ensure_ascii=False), perturbation_desc

    # ---- 扰动 3：添加幻觉字段 ----
    if not_used_in_schema:
        fake_field = random.choice(list(not_used_in_schema))
        perturbed = copy.deepcopy(output_obj)
        entities = list(perturbed.keys())
        if entities:
            target_entity = random.choice(entities)
            if isinstance(perturbed[target_entity], dict):
                perturbed[target_entity][fake_field] = "这是一个不正确的值"
                perturbation_desc = (
                    f"实体 '{target_entity}' 中添加了不在原文中的幻觉字段 '{fake_field}'"
                )
                return json.dumps(perturbed, ensure_ascii=False), perturbation_desc

    # ---- 扰动 4：类型错误（字符串 → 列表） ----
    for entity, rels in output_obj.items():
        if isinstance(rels, dict):
            for field, val in rels.items():
                if isinstance(val, str):
                    perturbed = copy.deepcopy(output_obj)
                    perturbed[entity][field] = [val, "多余值"]
                    perturbation_desc = f"字段 '{field}' 的值应该是字符串，但被错误地写成了列表"
                    return json.dumps(perturbed, ensure_ascii=False), perturbation_desc

    return None, None


def make_schema_repair(item, schema_fields):
    """任务 D：schema 纠错 —— 在正确输出中注入错误，让模型找出并修正。

    先调用 relations_to_output_json 获得正确答案，再通过 perturb_output
    注入一个可控错误，构造"有错误的输出 → 请修正"的任务对。

    Prompt 中包含错误类型描述（perturbation_desc），模型需要结合原文和
    schema 信息，定位错误并输出修正后的正确 JSON。

    仅对 high 质量且 relation ≥ 3 的样本生成（参见 perturb_output 注释）。

    Args:
        item:          单条训练样本 dict
        schema_fields: 该 topic 的 schema 短字段名列表

    Returns:
        派生任务 dict（含 perturbation 字段，记录注入了什么错误），
        扰动失败时返回 None
    """
    output_str = relations_to_output_json(item["relation"])
    perturbed_output, perturbation_desc = perturb_output(
        output_str, schema_fields, item["relation"]
    )

    if perturbed_output is None:
        return None

    schema_str = json.dumps(schema_fields, ensure_ascii=False)
    prompt = (
        "以下信息抽取结果存在错误，请根据 schema 和原文找出并修正错误。\n"
        f"Schema: {schema_str}\n"
        f"原文: {item['input']}\n"
        f"有错误的抽取结果: {perturbed_output}\n"
        f"错误类型: {perturbation_desc}\n"
        f"请输出修正后的正确 JSON。"
    )

    return {
        "task_type": "schema_repair",
        "original_id": item["id"],
        "cate": item["cate"],
        "quality_tier": item["quality_tier"],
        "n_relations": len(item["relation"]),
        "input_len": len(item["input"]),
        "prompt": prompt,
        "output": output_str,              # 正确答案（未扰动版本）
        "perturbation": perturbation_desc,  # 记录注入的错误类型
    }


def main():
    """任务派生主入口：加载 → 按质量档位派生 → 统计 → 保存。

    执行流程（5 步）：
      1. 加载分层训练数据 + schema 模板
      2. 遍历每条样本，按质量档位决定派生哪些任务类型
      3. 每个任务类型调用对应的 make_*() 函数构造 prompt/output 对
      4. 统计四类任务的数量分布 + per-topic 派生量
      5. 保存派生样本 JSONL + 派生报告

    派生策略矩阵：
      ┌────────────────┬───────┬───────┬───────┐
      │ quality_tier   │  IE   │  JSON │ Format│ Repair│
      ├────────────────┼───────┼───────┼───────┤
      │ high           │  ✅   │  ✅   │  ✅   │  ✅¹  │
      │ medium         │  ✅   │  ✅   │       │       │
      │ low            │  ✅   │       │       │       │
      └────────────────┴───────┴───────┴───────┘
      ¹ Repair 仅对 relation ≥ 3 的样本生成
    """
    print("=" * 60)
    print("04_derive_tasks.py - 派生四类 SFT 任务")
    print("=" * 60)

    random.seed(42)

    # ---- 步骤 1：加载数据 + schema ----
    data = load_jsonl(TIERED_TRAIN)
    schema = load_schema(RAW_SCHEMA)
    print(f"\n加载分层后数据: {len(data)} 条")
    print(f"Schema topics: {list(schema.keys())}")

    # ---- 步骤 2+3：按质量档位派生 ----
    # 派生策略（在模块 docstring 中有完整说明）：
    #   high:    全部 4 类
    #   medium:  IE + text_to_json
    #   low:     仅 IE
    derived = []
    task_counter = Counter()     # 各任务类型派生数量
    skip_counter = Counter()     # 跳过原因计数

    for i, item in enumerate(data):
        cate = item["cate"]
        schema_fields = get_schema_for_topic(schema, cate)

        # 查不到 schema 的 topic 跳过（无法构建带 schema 的 prompt）
        if not schema_fields:
            skip_counter["no_schema"] += 1
            continue

        tier = item["quality_tier"]

        # ---- 所有质量等级：信息抽取 ----
        d = make_ie_extraction(item, schema_fields)
        if d:
            derived.append(d)
            task_counter["ie_extraction"] += 1

        # ---- high + medium：文本转 JSON ----
        if tier in ("high", "medium"):
            d = make_text_to_json(item, schema_fields)
            if d:
                derived.append(d)
                task_counter["text_to_json"] += 1

        # ---- 仅 high：格式遵循 + schema 纠错 ----
        if tier == "high":
            d = make_format_following(item, schema_fields)
            if d:
                derived.append(d)
                task_counter["format_following"] += 1

            # schema_repair 需要至少 3 个三元组才有足够的扰动空间
            if len(item["relation"]) >= 3:
                d = make_schema_repair(item, schema_fields)
                if d:
                    derived.append(d)
                    task_counter["schema_repair"] += 1
                else:
                    skip_counter["schema_repair_failed"] += 1

        if i % 50000 == 0 and i > 0:
            print(f"  已处理 {i}/{len(data)}...")

    # ---- 步骤 4：统计报告 ----
    print(f"\n派生结果:")
    print(f"  输入样本: {len(data)}")
    print(f"  派生样本总数: {len(derived)}")
    total = len(derived)
    for tt in ["ie_extraction", "text_to_json", "format_following", "schema_repair"]:
        cnt = task_counter[tt]
        print(f"  {tt}: {cnt} ({cnt/total*100:.1f}%)")

    if skip_counter:
        print(f"\n跳过统计:")
        for k, v in skip_counter.items():
            print(f"  {k}: {v}")

    # per-topic 派生量（检查是否有 topic 完全没派生）
    print(f"\n各 topic 派生数量:")
    topic_counter = Counter(d["cate"] for d in derived)
    for topic, cnt in topic_counter.most_common():
        print(f"  {topic}: {cnt}")

    # ---- 步骤 5：保存 ----
    save_jsonl(derived, DERIVED_ALL)
    print(f"\n保存: {DERIVED_ALL}")

    report = {
        "step": "derive_tasks",
        "input_count": len(data),
        "derived_count": len(derived),
        "task_counts": dict(task_counter),
        "skip_counts": dict(skip_counter),
        "task_ratios": {k: f"{v/total*100:.1f}%" for k, v in task_counter.items()},
        "topic_dist": dict(topic_counter),
    }
    with open(DERIVE_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告: {DERIVE_REPORT}")

    print(f"\n[04_derive_tasks] 完成. {len(data)} 条原始样本 -> {len(derived)} 条派生样本")


if __name__ == "__main__":
    main()
