"""训练数据加载器 —— 随机截取 token 序列片段的批采样。

语言模型训练的核心是"下一个 token 预测"：
  给定前文 token₀, token₁, ..., tokenₙ，预测 tokenₙ₊₁。

本模块的函数将一整篇 token 化后的语料视为一条长数组，每轮随机选取起点，
截取 context_length 个连续 token 作为输入 x，同一起点右移一位截取同样长度
作为标签 y。每个 token 位置都参与损失计算，causal mask 确保只看上文。

使用方式：
    x, y = get_batch(dataset, batch_size=32, context_length=256, device="cuda")
    logits = model(x)                # (batch, seq, vocab_size)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size), # 展平所有位置
        y.view(-1)                   # 展平所有标签
    )
"""

import torch
import numpy as np
import numpy.typing as npt


def get_batch(
        dataset: npt.NDArray,
        batch_size: int,
        context_length: int,
        device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 token 数组中随机采样一个批次。

    用 torch.randint 随机选取 batch_size 个起点，每个起点截取
    x = dataset[i : i+context_length]（前文）
    y = dataset[i+1 : i+context_length+1]（下一个 token，右移一位）

    max_id = dataset_len - context_length - 1 中减 1 是因为 y 最后一个元素
    的索引 = i + context_length，必须 ≤ dataset_len - 1，故 i ≤ max_id。

    Args:
        dataset:        整篇语料的 token ID 数组，形状 (total_tokens,)
        batch_size:     每批样本数
        context_length: 每个样本的 token 数（序列长度）
        device:         输出张量所在的设备，如 "cuda" 或 "cpu"

    Returns:
        x: [batch_size, context_length] — 输入 token 序列
        y: [batch_size, context_length] — 标签 token 序列（x 右移一位）
    """
    dataset_len = len(dataset)
    max_id = dataset_len - context_length - 1          # 最大合法起点
    ix = torch.randint(0, max_id + 1, (batch_size,))   # 随机选 batch_size 个起点
    x_stack = [dataset[i : i + context_length] for i in ix]
    y_stack = [dataset[i + 1 : i + context_length + 1] for i in ix]
    x = torch.from_numpy(np.array(x_stack)).to(device).long()
    y = torch.from_numpy(np.array(y_stack)).to(device).long()
    return x, y