"""LoRA (Low-Rank Adaptation) 低秩适配模块。

LoRA 的核心思想：预训练权重已经学到了通用知识，微调时不需要修改它。
只需在原始 Linear 层旁边挂一对低秩矩阵 A 和 B，训练时只更新 A 和 B：

    output = W·x + (α/r) · B·A·x
             ↑                  ↑
        原始输出（冻结）    LoRA 增量（只训练这个）

A 的形状为 [r, in_features]，B 为 [out_features, r]，r 是低秩维度。
r 通常取 8~64，远小于 in/out_features（512~2048），因此可训练参数极少。

本模块提供六个核心 API：

  apply_lora_to_model()  —— 把普通 Linear 层替换为 LoRALinear（训练前调用）
  get_lora_params()       —— 获取所有 LoRA 参数（传给优化器）
  get_lora_state_dict()   —— 提取 LoRA 权重用于保存（几 MB，不是几百 MB）
  load_lora_state_dict()  —— 加载保存的 LoRA 权重
  merge_lora()            —— 把 LoRA 权重"熔合"进原始 Linear（推理加速）
  unmerge_lora()          —— 恢复熔合前的原始权重

完整使用流程：
  训练时：
    1. apply_lora_to_model(model, r=8, alpha=16)
    2. optimizer = AdamW(get_lora_params(model), ...)
    3. 正常训练（原始权重冻结，只更新 A/B）
    4. torch.save(get_lora_state_dict(model), "lora.pt")

  推理时：
    1. apply_lora_to_model(model, r=8, alpha=16)
    2. load_lora_state_dict(model, torch.load("lora.pt"))
    3. merge_lora(model)  → 熔合后等价于普通模型，无额外开销
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from .transformer import Linear

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from .transformer import Linear


class LoRALinear(nn.Module):
    """LoRA 低秩适配器 —— 替代原始 Linear 层，冻结原权重，只训练低秩矩阵 A 和 B。

    结构：
        input x (in_features) → A [r, in] → r 维压缩 → B [out, r] → out 维展开
        ↓
        原始 W [out, in] 直接乘 x
        ↓
        output = W·x + (α/r)·B·A·x

    Args:
        original: 原始 Linear 层（权重被冻结）
        r:        低秩维度（压缩瓶颈的大小，默认 8）
        alpha:    缩放系数（控制 LoRA 修正力度，默认 16）
    """

    def __init__(
        self,
        original: Linear,
        r: int = 8,
        alpha: float = 16.0,
    ) -> None:
        """初始化 LoRA 适配器。

        A 用 kaiming_uniform 随机初始化（提供初始随机扰动），
        B 初始化为全零（保证训练开始时 LoRA 输出 = 0，等价于原始模型）。
        """
        super().__init__()
        self.original = original
        self.original.weight.requires_grad_(False)          # 冻结原始权重

        out_features, in_features = original.weight.shape   # 如 [512, 512]
        self.r = r
        self.scaling = alpha / r                            # 缩放因子（默认 16/8=2）
        device = original.weight.device

        # A [r, in_features] 压缩矩阵：把输入从 in_features 维压到 r 维
        # B [out_features, r] 展开矩阵：把 r 维扩回 out_features 维
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # A 随机，B 全零

        self._merged = False

    # ---- forward --------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：原始输出 + LoRA 增量。

        计算过程：
          1. W·x          → 原始输出（冻结，无梯度）
          2. x @ A.T      → [..., r]     压缩到瓶颈维
          3. ... @ B.T    → [..., out]   展开回输出维
          4. × (α/r)     → 缩放
          5. W·x + 增量   → 最终输出

        若已调用 merge()（熔合模式），跳过步骤 2-4，直接返回 W'·x，
        其中 W' = W + (α/r)·B·A 已事先算好。

        Args:
            x: 输入张量 [..., in_features]

        Returns:
            输出张量 [..., out_features]
        """
        original_out = torch.einsum("... i, o i -> ... o", x, self.original.weight)
        if self._merged:
            return original_out                                # 熔合模式：无额外计算
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return original_out + lora_out                          # 训练模式：W·x + 增量

    # ---- merge / unmerge ------------------------------------------------

    @torch.no_grad()
    def merge(self) -> None:
        """将 LoRA 权重熔合进原始权重：W' = W + (α/r)·B·A。

        熔合后的好处：
          - 前向传播不再需要计算 A 和 B，速度与原始 Linear 完全一样
          - 适合推理部署——既保留了 SFT 的对话能力，又没有额外计算开销

        原理：
          W·x + (α/r)·B·A·x = (W + (α/r)·B·A)·x = W'·x
          提前算出 W'，推理时只做一次矩阵乘法。

        熔合后可调用 unmerge() 恢复原始权重（继续训练或换 LoRA 权重）。
        """
        if self._merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling  # [out, in]，完整增量矩阵
        self.original.weight.add_(delta)                      # W = W + 增量
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """撤销熔合，恢复原始权重：W' - (α/r)·B·A = W。

        用于以下场景：
          - 训练中断续训前需要恢复（训练时要求原始权重不变）
          - 切换不同 LoRA 适配器前需要先还原基座

        注意：必须在 A/B 未被修改的情况下调用，否则恢复不完整。
        """
        if not self._merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling  # [out, in]
        self.original.weight.sub_(delta)                      # W = W - 增量
        self._merged = False

    @property
    def merged(self) -> bool:
        return self._merged


# ---- apply LoRA to a TransformerLM ----------------------------------------

_DEFAULT_TARGETS = {"q_proj", "k_proj", "v_proj", "output_proj"}


def _replace_module(root: nn.Module, full_name: str, new_module: nn.Module) -> None:
    """在模型树中替换指定路径的模块。

    通过点号分隔的路径名（如 "blocks.0.attn.q_proj"），
    递归定位到目标模块的父节点，然后用新模块替换旧模块。

    Args:
        root:       模型根节点
        full_name:  点号分隔的模块路径，如 "transformer.blocks.0.attn.q_proj"
        new_module: 替换上去的新模块
    """
    parts = full_name.split(".")
    parent = root
    for part in parts[:-1]:          # 沿路径下钻到父节点
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)  # 替换叶子节点


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_names: Iterable[str] | None = None,
) -> None:
    """将模型中匹配的 Linear 层替换为 LoRALinear，冻结所有原始参数。

    遍历模型的每一层，找到名称末尾匹配 target_names 的自定义 Linear 模块，
    替换为 LoRALinear（原始权重冻结，新增 A/B 矩阵可训练）。

    默认目标层：q_proj, k_proj, v_proj, output_proj（注意力投影层）。
    不改造 FFN 层和 embedding——attention 对指令跟随最重要，且参数量已足够少。

    必须在训练前调用。若模型已加载预训练权重，先加载权重再调用此函数。

    Args:
        model:        TransformerLM 模型实例
        r:            低秩维度（瓶颈大小，默认 8）
        alpha:        缩放系数（控制 LoRA 修正力度，默认 16）
        target_names: 要替换的层名集合，None 使用默认值

    示例：
        model = TransformerLM(...)
        load_model_state("pretrain.pt", model)     # 先加载预训练权重
        apply_lora_to_model(model, r=8, alpha=16)  # 再应用 LoRA
    """
    # 第一步：冻结所有参数
    for p in model.parameters():
        p.requires_grad_(False)

    if target_names is None:
        target_names = _DEFAULT_TARGETS
    target_set = set(target_names)

    # 先收集匹配列表，再替换（不能在迭代 named_modules() 时直接修改）
    replacements: list[tuple[str, LoRALinear]] = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]           # 取层名最后一段（如 "q_proj"）
        if leaf in target_set and isinstance(module, Linear):
            replacements.append((name, LoRALinear(module, r=r, alpha=alpha)))

    for name, lora_layer in replacements:
        _replace_module(model, name, lora_layer)


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """获取模型中所有 LoRA 参数（A 和 B 矩阵）。

    遍历模型所有子模块，收集 LoRALinear 的 lora_A 和 lora_B 参数。
    返回的列表直接传给优化器，确保只训练 LoRA 参数，原始权重保持冻结。

    Args:
        model: 已应用 LoRA 的模型

    Returns:
        LoRA A 和 B 矩阵的 Parameter 列表

    示例：
        optimizer = AdamW(get_lora_params(model), lr=1e-4)
    """
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """提取模型中所有 LoRA 权重，用于保存。

    只导出 A 和 B 矩阵（已移到 CPU），不包含冻结的原始权重。
    因此保存的文件只有几 MB，远小于全量检查点（几百 MB）。

    key 的命名格式：
        "transformer.blocks.0.attn.q_proj.lora_A"
        "transformer.blocks.0.attn.q_proj.lora_B"
        ...

    Args:
        model: 已应用 LoRA 的模型

    Returns:
        {完整路径.lora_A: 张量, 完整路径.lora_B: 张量, ...}

    示例：
        torch.save(get_lora_state_dict(model), "lora_adapter.pt")
    """
    sd: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.lora_A"] = module.lora_A.data.cpu()
            sd[f"{name}.lora_B"] = module.lora_B.data.cpu()
    return sd


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """加载保存的 LoRA 权重到模型中。

    按 key 匹配：找到模型中的 LoRALinear 层，将 state_dict 中对应的
    lora_A / lora_B 拷贝过去。不存在的 key 静默跳过（兼容不同 target_names 配置）。

    必须先调用 apply_lora_to_model() 建立 LoRA 结构，再调用本函数加载权重。

    Args:
        model:      已应用 LoRA 的模型（结构必须与保存时一致）
        state_dict: get_lora_state_dict() 或 torch.load() 返回的字典

    示例：
        apply_lora_to_model(model, r=8, alpha=16)
        load_lora_state_dict(model, torch.load("lora_adapter.pt"))
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in state_dict:
                module.lora_A.data.copy_(state_dict[a_key])
            if b_key in state_dict:
                module.lora_B.data.copy_(state_dict[b_key])


def merge_lora(model: nn.Module) -> None:
    """熔合模型中所有 LoRA 层（W' = W + (α/r)·B·A），用于推理部署。

    熔合后：
      - 可移除 LoRA 参数，模型变回普通 TransformerLM（无额外计算开销）
      - 推理速度与未使用 LoRA 的模型完全一致

    通常在推理脚本中调用一次，熔合后即可正常 generate()。

    Args:
        model: 已应用并加载了 LoRA 权重的模型
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora(model: nn.Module) -> None:
    """撤销所有 LoRA 层的熔合，恢复原始权重（W' - (α/r)·B·A = W）。

    用于：
      - 推理后再恢复，继续用不同数据训练
      - 切换 LoRA 适配器：unmerge 上一个 → 加载新的 → merge

    Args:
        model: 已熔合 LoRA 的模型
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def print_trainable_params(model: nn.Module) -> None:
    """打印参数概览：总参数量 vs 可训练（LoRA）参数量。

    输出示例：
        Total params: 31,745,536 | Trainable (LoRA): 294,912 (0.93%)

    用于确认 LoRA 应用成功：可训练参数应该只占总参数的 1% 左右。
    如果这个比例接近 100%，说明 LoRA 没有正确应用。

    Args:
        model: 模型实例（LoRA 应用前后均可）
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total * 100 if total > 0 else 0
    print(f"Total params: {total:,} | Trainable (LoRA): {trainable:,} ({ratio:.2f}%)")
