# MicroLM 学习文档（第一阶段）

> **目标**：在阅读任何源码之前，建立 GPT/LLM 完整训练流程的全局认知。

## 整体流程

```text
文本
 ↓
Tokenizer
 ↓
Dataset
 ↓
Dataloader
 ↓
Embedding
 ↓
Transformer Block × N
 ↓
LM Head
 ↓
CrossEntropy Loss
 ↓
Optimizer
 ↓
Checkpoint
```

---

# 1. 文本（Raw Text）

训练数据最初是自然语言文本，例如：

```text
今天天气很好。
我想出去散步。
```

模型不能直接处理字符串，因此第一步需要将文本转换成数字。

---

# 2. Tokenizer

## 作用

Tokenizer 将文本编码成 Token ID。

例如：

```text
"今天天气很好"

↓

["今天", "天气", "很", "好"]

↓

[3145, 228, 511, 903]
```

训练过程中，模型始终处理的是 Token ID，而不是字符串。

### 输入

文本（String）

### 输出

Token ID 序列（List[int]）

---

# 3. Dataset

## 作用

Dataset 根据 block_size 将长序列切分成许多训练样本。

例如：

```text
Token:
1 2 3 4 5 6 7 8

block_size = 4
```

得到：

```text
输入：1 2 3 4
标签：2 3 4 5

输入：2 3 4 5
标签：3 4 5 6

输入：3 4 5 6
标签：4 5 6 7
```

Dataset 的职责：

- 保存所有样本
- 返回单个训练样本 `(x, y)`

---

# 4. DataLoader

## 作用

DataLoader 将多个样本组成 Batch，并负责随机打乱和多线程加载。

例如：

```text
batch_size = 4

Batch =
sample1
sample2
sample3
sample4
```

输入模型的数据形状通常为：

```text
(Batch, Sequence)
```

例如：

```text
(8, 128)
```

---

# 5. Embedding

## 为什么需要 Embedding？

Token ID 只是整数，不具备语义。

Embedding 将 Token ID 映射到高维连续向量。

例如：

```text
12

↓

[0.15, -0.33, ..., 0.42]
```

若 hidden_size=768，则：

```text
(B, T)

↓

(B, T, 768)
```

随后加入 Position Embedding，使模型感知顺序。

---

# 6. Transformer Block × N

Transformer Block 是 GPT 的核心计算单元。

一个 Block 包括：

```text
LayerNorm
↓
Self-Attention
↓
Residual
↓
LayerNorm
↓
MLP
↓
Residual
```

整个模型由多个 Block 堆叠而成：

```text
Block1
↓
Block2
↓
...
↓
BlockN
```

其主要作用是不断提取上下文语义信息。

---

# 7. LM Head

Transformer 输出隐藏状态：

```text
(B, T, Hidden)
```

LM Head 使用一个线性层将 Hidden 映射到词表大小：

```text
Hidden

↓

Vocabulary Size
```

例如：

```text
768

↓

50000
```

得到每个 Token 属于词表中所有词的预测分数（logits）。

---

# 8. CrossEntropy Loss

模型预测：

```text
今天 天气 很

↓

好
```

真实标签也是：

```text
好
```

CrossEntropy Loss 衡量预测结果与真实标签之间的差异。

Loss 越小，模型预测越准确。

---

# 9. Optimizer

反向传播得到梯度后：

```text
loss.backward()
```

Optimizer 根据梯度更新参数：

```text
optimizer.step()
```

常见优化器：

- AdamW
- SGD

其目标是不断降低 Loss，提高模型预测能力。

---

# 10. Checkpoint

训练结束或每隔若干步保存模型：

```text
checkpoint.pt
```

通常包括：

- 模型参数
- Optimizer 状态
- Epoch
- Global Step

Checkpoint 用于：

- 恢复训练
- 推理
- 微调

---

# train.py 的整体职责

```text
读取配置
    ↓
Tokenizer
    ↓
Dataset
    ↓
DataLoader
    ↓
Model
    ↓
Forward
    ↓
Loss
    ↓
Backward
    ↓
Optimizer.step()
    ↓
Save Checkpoint
```

`train.py` 更像整个训练流程的调度器，它负责把所有模块串联起来。

---

# 学习检查清单

完成本阶段后，应能回答：

- Tokenizer 的作用是什么？
- Dataset 与 DataLoader 有什么区别？
- block_size 的作用是什么？
- 为什么需要 Embedding？
- 为什么需要 Position Embedding？
- Transformer Block 的组成是什么？
- LM Head 为什么输出 vocab_size 维？
- CrossEntropy Loss 如何计算？
- backward() 与 optimizer.step() 有什么区别？
- train.py 为什么被称为训练流程调度器？
