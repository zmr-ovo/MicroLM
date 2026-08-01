"""MicroLM Transformer 语言模型 —— 从零实现的完整模型主干。

本文件自底向上实现了一个 31.7M 参数的微型 Transformer 语言模型，所有组件均使用纯
PyTorch 张量操作（einsum），不依赖 nn.Linear 等高级封装。层级结构如下：

  Linear（einsum + 截断正态初始化）
   ├── Embedding（非 nn.Embedding，权重索引查表）
   ├── RMSNorm（pre-norm，float32 中间计算）
   ├── RotaryPositionalEmbedding（RoPE，预计算 cos/sin 表，register_buffer）
   ├── SwiGLU（门控 FFN）+ SiLU_FFN（不含门控的变体）
   ├── softmax / scaled_dot_product_attention（自实现，数值稳定）
   ├── MultiHeadSelfAttention（8 heads, d_head=64, 支持 KV Cache）
   ├── TransformerBlock（pre-norm + Attention + FFN）× 8 层
   └── TransformerLM（顶层封装，含 forward 和自回归 generate）

核心设计理念：
  - 全部使用 einsum 替代标准 Linear，便于 LoRA 注入时透明操作权重
  - RoPE cos/sin 表通过 register_buffer 随模型自动迁移设备
  - pre-norm 架构 + KV Cache 支持增量推理（prefill / decode 两阶段）
  - generate() 内置 temperature 和 top-p 采样，开箱即用

关键超参数：vocab_size=6400 | context_length=512 | d_model=512 | num_layers=8 |
num_heads=8 | d_ff=1344 | rope_theta=1,000,000 | 总参数量 31,729,152
"""

import math

import torch
import torch.nn as nn
from einops import rearrange


# ═══════════════════════════════════════════════════════════════════
# 1. 基础层：Linear、Embedding、RMSNorm、Identity
# ═══════════════════════════════════════════════════════════════════

class Linear(nn.Module):
    """自定义线性变换层（替代 nn.Linear）。

    使用 torch.einsum 而非标准 F.linear 实现矩阵乘法。
    权重初始化采用截断正态分布（truncated normal），标准差 = sqrt(2/(in+out))，
    截断边界在 ±3*std。比 nn.Linear 更透明——权重就是一个裸 Parameter，
    后续 LoRA 注入时可以直接替换为 LoRALinear 而无需绕过任何内部封装。
    一个形状、行为与 nn.Linear 一致的线性层，但用截断正态初始化让训练更稳定，用裸 Parameter 让权重裸露在外——方便后续 LoRA 等微调技术直接替换权重，无需绕过 nn.Linear 的内部封装。
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # 截断正态初始化：std 结合了输入和输出维度，比默认 Kaiming 更保守
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # "... i, o i -> ... o"：输入最后一个维度 i 与权重的输入维度 i 点积，
        # 输出最后一个维度变成 o。支持任意前导批次维度。
        return torch.einsum("... i, o i -> ... o", x, self.weight)


class Embedding(nn.Module):
    """自定义嵌入层（替代 nn.Embedding）。

    不同于标准 nn.Embedding 使用 N(0,1) 初始化，这里使用截断正态分布 std=1.0。
    本质就是权重矩阵的索引查表：forward 时直接从权重中取出对应 token 的行向量。
    不包含任何额外的缩放或 norm 操作。
    Embedding 就是一个 (词表大小 × 向量维度) 的权重矩阵，forward 做索引查表——输入 token ID，输出对应的行向量。用截断正态 σ=1.0 初始化，砍掉 ±3σ 外的离群值，比标准 nn.Embedding 更稳定、更透明。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [batch, seq_len] → [batch, seq_len, embedding_dim]"""
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization（RMSNorm）。

    LLaMA/Qwen 系列标配的归一化层。公式：
        RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight

    与标准 LayerNorm 相比省去了减去均值的步骤，速度更快且实验效果相当。
    内部全部用 float32 计算（即使输入是 bfloat16），确保数值稳定，
    最后 cast 回输入的原始 dtype 保持和前续层一致。
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))  # 可学习的缩放因子

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype                    # 记住输入 dtype
        x_float = x.to(torch.float32)         # 转为 float32 做稳定计算
        rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x_float / rms) * self.weight.to(torch.float32)
        return out.to(in_dtype)               # cast 回原始 dtype


class Identity(nn.Module):
    """恒等映射层。当不需要某个归一化时用它作为占位符，保持代码路径一致。"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ═══════════════════════════════════════════════════════════════════
# 2. 激活函数与 FFN 变体：SiLU、SwiGLU、SiLU_FFN
# ═══════════════════════════════════════════════════════════════════

def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU 激活函数（也称 Swish）：silu(x) = x * sigmoid(x)。

    平滑的非线性函数，没有 ReLU 的硬截断问题。SwiGLU 的门控端会用到它。
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU 门控前馈网络 —— LLaMA 系列标配的 FFN 结构。

    公式：FFN(x) = W2( SiLU(W1·x) ⊙ W3·x )

    其中 ⊙ 是逐元素乘法。三层线性变换：
      - W1：d_model → d_ff（门控通路，通过 SiLU 激活）
      - W3：d_model → d_ff（线性通路，不激活，直接与门控结果相乘）
      - W2：d_ff → d_model（将门控后的结果投影回原维度）

    d_ff = 1344 ≈ 2.625 × d_model，使 SwiGLU（3 个权重矩阵）的总参数量
    与标准 FFN（d_ff=4*d_model，2 个权重矩阵）大致持平。
    SwiGLU = 两路并行：W1 经 SiLU 产出门控信号（可正可负，平滑无断点），W3 保留原始信息，两者逐元素相乘后经 W2 投影回原维度。门控不仅能"放行/阻挡"，还能翻转信息符号，比 ReLU 的 0/1 硬开关更灵活。
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # 门控投影
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # 输出投影
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # 线性投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(self.w1(x))   # 门控值（经过 SiLU 激活）
        linear = self.w3(x)       # 线性值（不激活）
        return self.w2(gate * linear)  # 逐元素门控 → 投影回 d_model


class SiLU_FFN(nn.Module):
    """不含门控的 SiLU FFN 变体：FFN(x) = W2(SiLU(W1·x))。

    d_ff 自动设为 4 * d_model，以匹配 SwiGLU 的参数量（2 个大矩阵 vs 3 个稍小的）。
    用于对照实验——验证 SwiGLU 门控机制本身带来的增益。
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


# ═══════════════════════════════════════════════════════════════════
# 3. KV Cache —— 增量推理的缓存数据结构
# ═══════════════════════════════════════════════════════════════════

class KVCache:
    """KV Cache 管理器，存储每层 Transformer 的 key/value 张量。

    结构简单：两个列表，各元素为每层的 K/V 张量（形状：..., head, seq, d_head）。
    初始全部为 None（空缓存），首次 prefill 时填入，后续 decode 逐 token 追加。

    使用方式：
      1. 创建 kv_cache = KVCache(num_layers=8)
      2. prefill 阶段：forward(token_ids, use_cache=True) → 每层缓存填满
      3. decode 阶段：forward(next_token, use_cache=True) → 追加到缓存末尾
      4. 调用 kv_cache.reset() 清空，开始新一轮生成
      KVCache 是两层列表（一层一个 [None]*N），存每层 Transformer 的历史 K/V 张量。自回归生成时，prefill 填缓存，decode 拼接追加大新 token 的 K/V。没有缓存每轮重算全部历史（O(N²)），有缓存每轮只算一个 token（O(N)）。
    """

    def __init__(self, num_layers: int):
        self.k = [None] * num_layers   # 每层一个 key 张量
        self.v = [None] * num_layers   # 每层一个 value 张量

    def reset(self):
        """清空所有层的缓存，开始新一轮对话时调用。"""
        for i in range(len(self.k)):
            self.k[i] = None
            self.v[i] = None



# ═══════════════════════════════════════════════════════════════════
# 4. RoPE 旋转位置编码
# ═══════════════════════════════════════════════════════════════════

class RotaryPositionalEmbedding(nn.Module):
    """RoPE（Rotary Position Embedding）—— 旋转位置编码。

    核心思想：不把位置信息加到 token 上，而是通过旋转矩阵编码到 attention 计算中。
    对每对相邻维度（第 2i 和 2i+1 维）施加位置相关的二维旋转。

    关键参数：
      - theta = 1,000,000：基频，控制不同频率分量的旋转速度
      - cos/sin 表在 __init__ 中预计算到 max_seq_len，通过 register_buffer 注册
        （非 persistent，不参与 checkpoint 保存），随模型自动迁移设备

    register_buffer 使这些张量享受与参数相同的设备/精度/序列化管理，
    但不会被优化器更新梯度——对预计算的常量张量是最佳选择。
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE")  # 必须偶数才能成对旋转

        # 计算每个维度对的旋转频率
        indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = theta ** (-indices / d_k)              # 高频→低频递减
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, inv_freq)          # [max_seq_len, d_k/2]

        # 预计算 cos/sin 表，persistent=False 表示 checkpoint 时不保存
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """对输入张量 x 在指定位置施加旋转。

        x 的形状：..., seq, head, d_head（其中 d_head = d_k 为偶数）
        token_positions：每个位置对应的绝对位置索引（用于查预计算的表）
        """
        # 根据位置索引从预计算表中取出对应的 cos/sin
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 如果输入是多头格式（有 head 维度），在 cos/sin 中插入 head 维度以对齐广播
        if x.ndim > cos.ndim and cos.ndim >= 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 将最后一维拆成 (d_k/2, 2) —— 每对相邻维度一组
        x_pairs = rearrange(x, "... seq (pair two) -> ... seq pair two", two=2)
        cos = cos.unsqueeze(-1)
        sin = sin.unsqueeze(-1)

        # 提取偶数维和奇数维（在每对内）
        x_even = x_pairs[..., 0:1]   # 第 2i 维
        x_odd = x_pairs[..., 1:2]    # 第 2i+1 维

        # 二维旋转公式：
        #   new_even = even*cos - odd*sin
        #   new_odd  = even*sin + odd*cos
        rotated = torch.cat(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
            dim=-1,
        )
        return rearrange(rotated, "... seq pair two -> ... seq (pair two)")


# ═══════════════════════════════════════════════════════════════════
# 5. 注意力计算：softmax + Scaled Dot-Product Attention
# ═══════════════════════════════════════════════════════════════════

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数值稳定的 softmax 实现。

    先减去每行的最大值（shifted），避免 exp() 对上溢。然后再 exp、
    求和、相除。等价于 torch.softmax 但代码自包含。
    """
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled Dot-Product Attention 的核心计算。

    公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

    - QK^T 在 float32 下计算（即使 q/k 是 bfloat16），确保数值稳定
    - 除以 sqrt(d_k) 防止大维度下点积值过大导致 softmax 梯度消失
    - mask 参数用于 causal masking（prefill 阶段）或 None（decode 阶段）
    """
    d_k = q.shape[-1]

    # 核心点积：QK^T / sqrt(d_k)，在 float32 精度下计算
    attn_dtype = v.dtype
    scores = torch.einsum(
        "... q d, ... k d -> ... q k",
        q.to(torch.float32),
        k.to(torch.float32),
    ) / math.sqrt(d_k)

    # causal mask：mask 中 False 的位置设为 -inf（softmax 后概率 = 0）
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    # softmax 得到注意力权重，cast 回 v 的 dtype
    probs = softmax(scores, dim=-1).to(attn_dtype)

    # 用注意力权重对 V 加权求和
    return torch.einsum("... q k, ... k d -> ... q d", probs, v.to(attn_dtype))


# ═══════════════════════════════════════════════════════════════════
# 6. 多头自注意力（Multi-Head Self-Attention）
# ═══════════════════════════════════════════════════════════════════

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力层。

    将输入投影到 Q/K/V 三个空间，按头拆分后：
      1. 对每头的 Q/K 施加 RoPE 旋转位置编码
      2. 计算 scaled dot-product attention
      3. 拼接所有头的结果，通过 output_proj 投影回 d_model

    支持两种模式：
      - 训练 / prefill（use_cache=False）：使用 causal mask 防止看到未来 token
      - decode（use_cache=True）：无 mask（只有 1 个 query），拼接历史 KV 缓存

      512 维投影后切成 (8 heads × 64 dim)，每头独立做 Q/K/V attention（Q/K 加 RoPE 旋转），8 头的 64 维输出拼回 512 维，经 output_proj 跨头混合。Decode 时拼接历史 KV 缓存省计算，Prefill 时用 causal mask 防偷看未来。
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None,
                 device=None, dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_head = d_model // num_heads                     # 每头的维度：512/8 = 64

        # 四个投影层：Q、K、V、输出
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 可选 RoPE：传入了 theta 和 max_seq_len 才启用
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, d_k=self.d_head,
                max_seq_len=max_seq_len, device=device,
            )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        seq_len = x.shape[-2]
        leading_shape = x.shape[:-2]

        # ---- 投影 + 多头拆分 ----
        # 输入: ... seq d_model → ... head seq d_head
        q = rearrange(self.q_proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)

        # ---- RoPE 旋转位置编码 ----
        if self.rope is not None:
            if token_positions is None:
                # 未传入位置时，自动生成从 0 开始的连续位置
                token_positions = torch.arange(seq_len, device=x.device)
                token_positions = token_positions.view(
                    *([1] * len(leading_shape)), seq_len
                ).expand(*leading_shape, seq_len)
            q = self.rope(q, token_positions)   # 对 Q 施加旋转
            k = self.rope(k, token_positions)   # 对 K 施加旋转（V 不旋转）

        # ---- 注意力计算（区分 prefill 和 decode） ----
        if use_cache:
            # Decode 模式：拼接历史 KV + 新 KV，不需要 causal mask
            if past_k is not None:
                k = torch.cat([past_k, k], dim=-2)   # 沿序列维度拼接
                v = torch.cat([past_v, v], dim=-2)
            attn_out = scaled_dot_product_attention(q, k, v, mask=None)
            new_k, new_v = k, v                       # 返回更新后的完整 KV
        else:
            # Prefill / 训练模式：使用下三角 causal mask 防止看到未来
            causal_mask = torch.tril(
                torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)
            )
            attn_out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
            new_k, new_v = None, None

        # ---- 拼接多头 + 输出投影 ----
        # ... head seq d_head → ... seq (head * d_head) = ... seq d_model
        attn_out = rearrange(attn_out, "... head seq d -> ... seq (head d)")
        out = self.output_proj(attn_out)

        if use_cache:
            return out, new_k, new_v
        return out


# ═══════════════════════════════════════════════════════════════════
# 7. Transformer Block —— 标准 pre-norm 残差块
# ═══════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """单个 Transformer 层 = pre-norm Attention + pre-norm FFN，均带残差连接。

    结构（pre-norm）：
        x = x + Attention( ln1(x) )
        x = x + FFN( ln2(x) )

    支持配置项：
      - use_rms_norm: True=RMSNorm, False=Identity（跳过归一化）
      - norm_mode: "pre"（默认）或 "post"（显式报错——post-norm 与 KV Cache 不兼容）
      - ffn_type: "swiglu"（默认门控）或 "silu"（纯 SiLU FFN）
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 use_rms_norm: bool = True, norm_mode: str = "pre", ffn_type: str = "swiglu",
                 device=None, dtype=None):
        super().__init__()
        self.norm_mode = norm_mode

        # 两个 pre-norm 层（Attention 前一个，FFN 前一个）
        norm_cls = lambda: RMSNorm(d_model, device=device, dtype=dtype) if use_rms_norm else Identity()
        self.ln1 = norm_cls()   # Attention 前的归一化
        self.ln2 = norm_cls()   # FFN 前的归一化

        self.attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads,
            max_seq_len=max_seq_len, theta=theta,
            device=device, dtype=dtype,
        )

        # 根据 ffn_type 选择 FFN 变体
        if ffn_type == "silu":
            self.ffn = SiLU_FFN(d_model=d_model, d_ff=4 * d_model, device=device, dtype=dtype)
        else:
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        past_k: torch.Tensor | None = None,
        past_v: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        # post-norm 与 KV Cache 的交互存在已知 bug，不做支持
        if self.norm_mode == "post":
            raise NotImplementedError("post-norm + kv cache 先别接，先跑通 pre-norm")

        # pre-norm: 先归一化，再 Attention，再残差相加
        h = self.ln1(x)

        if use_cache:
            # KV Cache 模式：传入历史 K/V，返回更新后的完整 K/V
            attn_out, new_k, new_v = self.attn(
                h, token_positions=token_positions,
                past_k=past_k, past_v=past_v, use_cache=True,
            )
            x = x + attn_out                           # 残差连接 1
            x = x + self.ffn(self.ln2(x))             # pre-norm FFN + 残差连接 2
            return x, new_k, new_v
        else:
            # 训练 / 普通推理模式
            x = x + self.attn(h, token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))
            return x


# ═══════════════════════════════════════════════════════════════════
# 8. TransformerLM —— 完整语言模型（顶层封装）
# ═══════════════════════════════════════════════════════════════════

class TransformerLM(nn.Module):
    """完整的 Transformer 语言模型。

    结构：
        Embedding → TransformerBlock × num_layers → RMSNorm(final) → lm_head → logits

    总参数量 = 31,729,152（~31.7M），超参数见文件头部 docstring。

    核心能力：
      - forward(): 前向传播，返回 logits。支持 KV Cache 增量推理（use_cache=True）
      - generate(): 自回归文本生成，内置 temperature + top-p 采样 + EOS 停止
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        use_rms_norm: bool = True,
        norm_mode: str = "pre",
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length

        # Embedding：token ID → d_model 维向量
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # 堆叠 num_layers 个 TransformerBlock
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                    max_seq_len=context_length, theta=rope_theta,
                    use_rms_norm=use_rms_norm, norm_mode=norm_mode,
                    ffn_type=ffn_type, device=device, dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # 最终归一化（输出前）
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if use_rms_norm else Identity()

        # lm_head：d_model → vocab_size，输出每个 token 的 logit
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        token_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ):
        """前向传播：token IDs → logits。

        Args:
            token_ids: 输入 token ID 张量，形状 [batch, seq_len]
            kv_cache: KV Cache 实例（use_cache=True 时使用）
            use_cache: 是否缓存 K/V（prefill + decode 推理模式）
            start_pos: 当前序列的起始绝对位置（decode 时每轮递增）

        Returns:
            无 cache：logits [batch, seq_len, vocab_size]
            有 cache：(logits, kv_cache)
        """
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError("input sequence length exceeds context length")

        # 计算每个 token 的绝对位置（用于 RoPE 查表）
        leading_shape = token_ids.shape[:-1]
        token_positions = torch.arange(start_pos, start_pos + seq_len, device=token_ids.device)
        token_positions = token_positions.view(
            *([1] * len(leading_shape)), seq_len
        ).expand(*leading_shape, seq_len)

        # ---- Embedding ----
        x = self.token_embeddings(token_ids)

        # 首次 prefill 时自动创建 KV Cache
        if use_cache and kv_cache is None:
            kv_cache = KVCache(len(self.layers))

        # ---- 逐层传播 ----
        for layer_idx, layer in enumerate(self.layers):
            if use_cache:
                x, new_k, new_v = layer(
                    x,
                    token_positions=token_positions,
                    past_k=kv_cache.k[layer_idx],
                    past_v=kv_cache.v[layer_idx],
                    use_cache=True,
                )
                kv_cache.k[layer_idx] = new_k   # 更新时间步的 K/V 缓存
                kv_cache.v[layer_idx] = new_v
            else:
                x = layer(x, token_positions=token_positions)

        # ---- 最终归一化 + 输出投影 ----
        x = self.ln_final(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        if use_cache:
            return logits, kv_cache
        return logits

    # ═══════════════════════════════════════════════════════════
    # 9. 自回归文本生成（generate）
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """自回归文本生成。

        分两阶段：
          1. Prefill：一次性送入完整 prompt，填充 KV Cache，取最后一个位置的 logit
          2. Decode：每轮只输入 1 个新 token，复用 KV Cache 避免重复计算

        采样策略（可叠加）：
          - temperature：缩放 logit 控制随机性（=1.0 不变，>1 更随机，<1 更确定）
          - top-p（nucleus sampling）：只保留累积概率不超过 p 的 token
          - EOS 停止：生成到 eos_token_id 自动终止
        """
        self.eval()

        # 总长度检查
        if prompt_ids.shape[1] + max_new_tokens > self.context_length:
            raise ValueError("当前 KV cache 版本暂不支持超过 context_length 的生成")

        generated = prompt_ids.clone()

        # 初始化空的 KV Cache
        kv_cache = KVCache(len(self.layers))

        # ---- Phase 1: Prefill ----
        # 将完整 prompt 一次性送入模型，填充每层的 K/V 缓存
        logits, kv_cache = self.forward(
            prompt_ids, kv_cache=kv_cache, use_cache=True, start_pos=0,
        )
        logits = logits[:, -1, :]  # 只取最后一个位置的 logit 用于预测下一个 token

        # ---- Phase 2: Decode（逐 token 生成） ----
        for _ in range(max_new_tokens):
            # Temperature 缩放
            if temperature != 1.0:
                logits = logits / (temperature + 1e-8)

            # Top-p（nucleus）过滤
            if top_p < 1.0:
                logits = self._top_p_filter(logits, top_p)

            # softmax → 按概率采样下一个 token
            probs = softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)

            # 遇到 EOS 停止
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # 当前 token 的绝对位置 = 已生成序列长度 - 1
            cur_pos = generated.shape[1] - 1

            # 只将新 token 送入模型（seq_len=1），复用 KV Cache
            logits, kv_cache = self.forward(
                next_token, kv_cache=kv_cache, use_cache=True, start_pos=cur_pos,
            )
            logits = logits[:, -1, :]  # 单 token 的 logit

        return generated

    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Top-p（nucleus sampling）过滤。

        算法：
          1. 将 logits 按降序排序
          2. 计算累积 softmax 概率
          3. 保留累积概率 ≤ p 的最小子集（至少保留最高概率的 token）
          4. 将过滤掉的 token logit 设为 -inf（softmax 后概率 = 0）
        """
        # 按 logit 值降序排列
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # 累积概率分布
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

        # 标记累积概率超过 p 的 token 为"待移除"
        sorted_indices_to_remove = cumulative_probs > p

        # 关键修正：将移除标记右移一位，确保至少保留第一个（最高概率）token
        # 且保留恰好让累积概率超过 p 的那个 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # 将排序后的掩码映射回原始词表索引位置
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

        return logits
