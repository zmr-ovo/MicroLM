"""KV Cache 教学示例 —— 最简自注意力 + 增量推理。

本文件以最小代码量演示自回归推理的两阶段机制：
  - Prefill：一次性送入全部 prompt token，并行算 K/V 并填充缓存
  - Decode：每轮只输入 1 个新 token，复用历史 K/V 避免重复计算

不同于项目中完整的 transformer.py（8 层 + RoPE + SwiGLU），本文件只实现
最基本的多头注意力 + KV Cache，用于直观理解核心概念。运行 demo() 即可看到
缓存从 prefill 到 decode 的 tensor 形状变化全过程。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleKVCache:
    """最简 KV Cache 管理器。

    仅维护两个张量 self.k 和 self.v，形状均为 [B, H, T, D]：
      - B：batch size
      - H：注意力头数
      - T：已缓存的序列长度（prefill 后 = prompt_len，decode 每步 +1）
      - D：每头的维度（head_dim）

    使用方式：
      1. cache = SimpleKVCache()
      2. Prefill：attn(prompt_tokens, cache=cache)  → cache 从 None 填满
      3. Decode： attn(new_token,     cache=cache)  → cache 沿 T 轴追加
      4. 新一轮： cache.reset()                     → cache 清空回 None
    """

    def __init__(self):
        self.k = None  # [B, H, T, D]，初始为空（无任何缓存）
        self.v = None  # [B, H, T, D]，同上

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """将新算出的 K/V 追加到缓存中。

        首次调用（prefill）时 self.k 为 None → 直接赋值。
        后续调用（decode）时沿 T 轴（dim=-2）拼接历史和新 K/V，
        返回更新后的完整 K/V 供 attention 计算使用。

        Args:
            k_new: [B, H, T_new, D] — 新 token 的 Key（prefill 时 T_new = prompt_len，
                   decode 时 T_new = 1）
            v_new: [B, H, T_new, D] — 新 token 的 Value，同上

        Returns:
            (完整 K 张量, 完整 V 张量)，形状均为 [B, H, T_old+T_new, D]
        """
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=-2)  # 沿序列维度拼接
            self.v = torch.cat([self.v, v_new], dim=-2)
        return self.k, self.v

    def reset(self):
        """清空缓存，开始新一轮对话或新的生成批次时调用。"""
        self.k = None
        self.v = None


class MiniAttention(nn.Module):
    """最简多头自注意力层（教学用途）。

    相比项目中完整的 MultiHeadSelfAttention，本类做了最大简化：
      - 无 RoPE 位置编码
      - 使用 nn.Linear 而非自定义 Linear（无截断正态初始化）
      - 使用 PyTorch 官方的 scaled_dot_product_attention（内置 causal mask）
      - 无 KV Cache 的 past_k / past_v 参数，改为直接接收 SimpleKVCache 对象

    结构：x → Q/K/V 投影 → 多头拆分 → Attention + Cache → 多头合并 → 输出投影
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads                     # 每头维度，如 512/8 = 64

        # 四个标准投影层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)   # Query 投影
        self.k_proj = nn.Linear(d_model, d_model, bias=False)   # Key 投影
        self.v_proj = nn.Linear(d_model, d_model, bias=False)   # Value 投影
        self.o_proj = nn.Linear(d_model, d_model, bias=False)   # 输出投影

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """多头拆分：把 d_model 切成 num_heads × head_dim，头轴前置。

        [B, T, D] → [B, H, T, Hd]
        例：(2, 5, 32) → (2, 4, 5, 8)，4 个头各 8 维独立做 attention
        """
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)   # 头轴移到位置轴之前

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """多头合并：将独立计算的多头输出拼回 d_model。

        [B, H, T, Hd] → [B, T, D]
        例：(2, 4, 5, 8) → (2, 5, 32)，4 头的输出拼接回 32 维
        """
        B, H, T, Hd = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # 头轴归位
        return x.view(B, T, H * Hd)

    def forward(self, x: torch.Tensor, cache: SimpleKVCache | None = None):
        """前向传播。

        根据输入序列长度自动区分 prefill 和 decode：
          - Prefill：x 形状 [B, T, D]，T > 1 → 并行处理所有 token，填入 cache
          - Decode： x 形状 [B, 1, D] → 只处理新 token，拼接历史 cache 中的 K/V

        内部使用 PyTorch 的 scaled_dot_product_attention，is_causal=True
        自动生成下三角 mask，等价于手动构建 causal mask。

        Args:
            x:     输入张量，prefill 时 [B, T, D]，decode 时 [B, 1, D]
            cache: SimpleKVCache 实例。传 None 表示训练模式（不使用缓存）

        Returns:
            注意力输出，形状与输入 x 相同 [B, T, D]
        """
        q = self._split_heads(self.q_proj(x))     # [B, H, T, Hd]
        k = self._split_heads(self.k_proj(x))     # [B, H, T, Hd]
        v = self._split_heads(self.v_proj(x))     # [B, H, T, Hd]

        if cache is not None:
            k_all, v_all = cache.update(k, v)     # 追加到 / 从缓存创建
        else:
            k_all, v_all = k, v                   # 训练模式：直接用当前 K/V

        # 官方 SDPA：flash attention 内核 + causal mask 自动处理
        out = F.scaled_dot_product_attention(
            q, k_all, v_all,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True       # 自动下三角 mask，防止看到未来 token
        )

        out = self._merge_heads(out)              # [B, H, T, Hd] → [B, T, D]
        return self.o_proj(out)


def demo():
    """KV Cache 完整流程演示。

    运行即输出 prefill 和 decode 两个阶段的 tensor 形状变化：
      - Prefill 后 cache.k 形状 = [B, H, T, Hd]，其中 T = prompt 长度
      - Decode 后 cache.k 形状 = [B, H, T+1, Hd]，序列长度自动增长

    关键观察：decode 时输入只有 [B, 1, D] 一个 token，
    但 attention 计算的 K/V 序列长度 = 历史全部 token。
    这就是 KV Cache 消除 O(N²) 重复计算的核心。
    """
    torch.manual_seed(0)

    B, T, D, H = 2, 5, 32, 4                     # batch=2, 序列=5, d_model=32, 8头
    attn = MiniAttention(d_model=D, num_heads=H)
    cache = SimpleKVCache()

    # 1) Prefill：一次性送入完整 prompt（5 个 token）
    x_prefill = torch.randn(B, T, D)              # [2, 5, 32]
    y_prefill = attn(x_prefill, cache=cache)
    print("prefill output:", y_prefill.shape)      # [2, 5, 32] — 每个位置都有输出
    print("cache k shape after prefill:", cache.k.shape)  # [2, 4, 5, 8]

    # 2) Decode：每次只输入一个新的 token（当前生成的第 6 个 token）
    x_next = torch.randn(B, 1, D)                 # [2, 1, 32] — 只一个！
    y_next = attn(x_next, cache=cache)
    print("decode output:", y_next.shape)          # [2, 1, 32] — 只一个位置的输出
    print("cache k shape after one decode step:", cache.k.shape)  # [2, 4, 6, 8]

if __name__ == "__main__":
    demo()