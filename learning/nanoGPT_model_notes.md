# nanoGPT model.py 完全解析笔记

> 这份笔记详细解析了 nanoGPT 的模型实现，适合深度学习初学者学习 Transformer/GPT 架构。

---

## 目录

1. [整体架构概览](#整体架构概览)
2. [LayerNorm 层归一化](#layernorm-层归一化)
3. [CausalSelfAttention 因果自注意力](#causalselfattention-因果自注意力)
4. [MLP 前馈网络](#mlp-前馈网络)
5. [Block Transformer块](#block-transformer块)
6. [GPT 主模型](#gpt-主模型)
7. [常见问题解答](#常见问题解答)

---

## 整体架构概览

```
GPT 模型结构
├── transformer (ModuleDict)
│   ├── wte: 词嵌入 Embedding(vocab_size, n_embd)
│   ├── wpe: 位置嵌入 Embedding(block_size, n_embd)
│   ├── drop: Dropout
│   ├── h: N 个 Block (Transformer 层)
│   │   └── Block
│   │       ├── ln_1: LayerNorm
│   │       ├── attn: CausalSelfAttention
│   │       ├── ln_2: LayerNorm
│   │       └── mlp: MLP
│   └── ln_f: 最终 LayerNorm
└── lm_head: 输出层 Linear(n_embd, vocab_size)
```

### 数据流向

```
输入 token IDs: (B, T)
       ↓
词嵌入 + 位置嵌入: (B, T, C)
       ↓
N 个 Transformer Block
       ↓
最终 LayerNorm
       ↓
输出层 → logits: (B, T, vocab_size)
```

---

## LayerNorm 层归一化

### 代码

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))   # 可学习的缩放参数
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # 可选偏置

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

### 原理

LayerNorm 对每个样本的特征维度做归一化：

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$y = \gamma \cdot \hat{x} + \beta$$

其中：
- $\mu$ = 均值
- $\sigma^2$ = 方差
- $\epsilon$ = 1e-5（防止除零）
- $\gamma$ = weight（可学习的缩放）
- $\beta$ = bias（可学习的偏移）

### 关键知识点

| 概念 | 解释 |
|------|------|
| `nn.Parameter` | 可学习参数，会被优化器更新 |
| `super().__init__()` | 必须调用，初始化 nn.Module |
| `forward()` | 定义前向传播，调用 `layer(x)` 时自动执行 |

### Q&A

**Q: `nn.Parameter(torch.ones(ndim))` 的 shape 是什么？如何创建不同维度的参数？**

```python
torch.ones(384)           # 一维: (384,)
torch.ones(1, 384)        # 二维: (1, 384)
torch.ones(384, 768)      # 二维: (384, 768)
torch.ones(32, 256, 384)  # 三维: (32, 256, 384)
```

---

## CausalSelfAttention 因果自注意力

### 核心代码结构

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Q、K、V 投影（合并为一个线性层提高效率）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # 因果掩码（只在不支持 Flash Attention 时使用）
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                    .view(1, 1, block_size, block_size))
```

### 注意力计算流程

```
输入 x: (B, T, C) = (32, 256, 384)
        │
        ▼
┌─────────────────────────────────────────────┐
│ c_attn: Linear(384 → 1152)                  │
│ 输出: (32, 256, 1152)                       │
│ split 成 q, k, v 各 (32, 256, 384)          │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ 重塑为多头格式:                              │
│ (32, 256, 384) → (32, 256, 6, 64)           │
│              → (32, 6, 256, 64)             │
│ 即 (B, n_head, T, head_size)                │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ 注意力计算:                                  │
│ att = softmax(Q @ K^T / √d_k) @ V           │
│ 输出: (32, 6, 256, 64)                      │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ 合并多头:                                    │
│ (32, 6, 256, 64) → (32, 256, 384)           │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│ c_proj: Linear(384 → 384)                   │
│ 输出投影，融合多头信息                       │
└─────────────────────────────────────────────┘
        │
        ▼
输出: (B, T, C) = (32, 256, 384)
```

### 因果掩码详解

```python
# 创建下三角掩码（假设 block_size=4）
torch.tril(torch.ones(4, 4))
# [[1, 0, 0, 0],    ← 位置0只能看位置0
#  [1, 1, 0, 0],    ← 位置1能看位置0,1
#  [1, 1, 1, 0],    ← 位置2能看位置0,1,2
#  [1, 1, 1, 1]]    ← 位置3能看位置0,1,2,3

# reshape 添加 batch 和 head 维度
.view(1, 1, block_size, block_size)  # shape: (1, 1, 4, 4)
```

**四个维度的含义：**

| 维度 | 名称 | 含义 | 为什么是1 |
|------|------|------|----------|
| dim0 | Batch | 批次 | 广播到所有样本 |
| dim1 | Head | 注意力头 | 广播到所有头 |
| dim2 | Query位置 | 当前位置（行） | block_size |
| dim3 | Key位置 | 被注意位置（列） | block_size |

**广播机制：**
```python
# att: (32, 6, 256, 256)
# bias: (1, 1, 256, 256)
# 广播后 bias "虚拟复制"成 (32, 6, 256, 256)
# 实际不占用额外内存！
```

### Buffer vs Parameter

| 类型 | 是否训练 | 是否保存 | 用途 |
|------|---------|---------|------|
| Parameter | ✅ | ✅ | 权重、偏置 |
| Buffer | ❌ | ✅ | 掩码、统计量 |
| 普通属性 | ❌ | ❌ | 配置值 |

```python
# Buffer 会随模型移动设备
model.to('cuda')  # bias 也会自动移到 GPU
```

### 关键代码解析

#### 1. Q、K、V 分割

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
# c_attn(x): (32, 256, 1152)
# split(384, dim=2): 在 dim=2 上每 384 切一刀
# q, k, v 各自: (32, 256, 384)
```

#### 2. 多头重塑

```python
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
# (32, 256, 384) 
# → view → (32, 256, 6, 64)
# → transpose(1,2) → (32, 6, 256, 64)
```

#### 3. 注意力计算

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
# q: (32, 6, 256, 64)
# k.transpose(-2,-1): (32, 6, 64, 256)
# q @ k^T: (32, 6, 256, 256)  ← 每个query与每个key的相似度
# 除以 √64 = 8 防止数值过大
```

**Q: 为什么 k.transpose(-2, -1) 只转置最后两维？**

`transpose(-2, -1)` 意思是交换倒数第2维和倒数第1维，保持 batch 和 head 维度不变。

#### 4. Softmax 在最后一维

```python
att = F.softmax(att, dim=-1)
# 对于每个 query 位置，计算它对所有 key 的注意力权重
# 权重之和 = 1
```

#### 5. contiguous() 的作用

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
```

- `transpose` 后内存可能不连续
- `view` 需要连续内存
- `contiguous()` 重新排列内存使其连续

### 输出投影的意义

```python
self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
```

**为什么需要输出投影：**

1. **多头融合**：6个头各自计算64维输出，拼接成384维，但只是简单拼接。c_proj 让模型学习如何组合不同头的信息
2. **残差连接要求**：输入输出维度必须相同才能 `x = x + attn(x)`
3. **增加表达能力**：多一层变换，模型更强

---

## MLP 前馈网络

### 代码

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)      # 384 → 1536（扩展4倍）
        x = self.gelu(x)      # 激活函数
        x = self.c_proj(x)    # 1536 → 384（压缩回来）
        x = self.dropout(x)
        return x
```

### 结构

```
输入: (B, T, 384)
    ↓
扩展: Linear(384 → 1536)
    ↓
激活: GELU
    ↓
压缩: Linear(1536 → 384)
    ↓
Dropout
    ↓
输出: (B, T, 384)
```

**为什么先扩展再压缩？** 给模型更大的"思考空间"，这是 Transformer 的标准设计。

---

## Block Transformer块

### 代码

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # 残差连接 1
        x = x + self.mlp(self.ln_2(x))    # 残差连接 2
        return x
```

### 残差连接

```
x = x + f(x)
```

**好处：**
- 解决深层网络梯度消失问题
- 让梯度可以"跳过"某些层反向传播
- 即使 f(x) 学不到东西，至少还有原始的 x

### Pre-Norm vs Post-Norm

- **nanoGPT (Pre-Norm)**：先 LayerNorm，再 Attention/MLP
- **原始 Transformer (Post-Norm)**：先 Attention/MLP，再 LayerNorm
- Pre-Norm 训练更稳定

---

## GPT 主模型

### 配置类

```python
@dataclass
class GPTConfig:
    block_size: int = 1024    # 最大序列长度
    vocab_size: int = 50304   # 词汇表大小
    n_layer: int = 12         # Transformer 层数
    n_head: int = 12          # 注意力头数
    n_embd: int = 768         # 嵌入维度
    dropout: float = 0.0      # Dropout 概率
    bias: bool = True         # 是否使用偏置
```

### 初始化

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),   # 词嵌入
    wpe = nn.Embedding(config.block_size, config.n_embd),   # 位置嵌入
    drop = nn.Dropout(config.dropout),
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
    ln_f = LayerNorm(config.n_embd, bias=config.bias),      # 最终 LN
))
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

# 权重共享
self.transformer.wte.weight = self.lm_head.weight
```

### 权重共享详解

**词嵌入 wte：** 字符 ID → 向量
- 权重矩阵: (vocab_size, n_embd) = (65, 384)

**输出层 lm_head：** 向量 → 字符概率
- 权重矩阵: (vocab_size, n_embd) = (65, 384)

**共享的直觉：**
- 嵌入：字符 "A" → 向量 v_A
- 输出：向量与 v_A 越相似，越可能预测为 "A"
- 两个操作本质上是"反过来"的

**好处：**
1. 减少参数量
2. 语义一致性

### 特殊初始化

```python
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

**为什么对残差路径用更小的标准差？**

每个 Block 有 2 个残差连接：
- attention 的 c_proj
- MLP 的 c_proj

6 层 × 2 = 12 个残差路径

如果不缩小初始化，输出方差会随深度线性增长，导致不稳定。

### forward 方法

```python
def forward(self, idx, targets=None):
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

    # 嵌入
    tok_emb = self.transformer.wte(idx)     # (B, T, C)
    pos_emb = self.transformer.wpe(pos)     # (T, C) → 广播到 (B, T, C)
    x = self.transformer.drop(tok_emb + pos_emb)

    # 通过 N 层 Transformer
    for block in self.transformer.h:
        x = block(x)
    x = self.transformer.ln_f(x)            # 最终 LayerNorm

    # 输出
    if targets is not None:
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    else:
        logits = self.lm_head(x[:, [-1], :])  # 只取最后一个位置
        loss = None

    return logits, loss
```

### 索引技巧

```python
# 用整数索引：维度会消失
x[:, -1, :]      # shape: (32, 384)

# 用列表索引：维度保留
x[:, [-1], :]    # shape: (32, 1, 384)
```

### crop_block_size 方法

```python
def crop_block_size(self, block_size):
    self.config.block_size = block_size
    self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
    for block in self.transformer.h:
        if hasattr(block.attn, 'bias'):
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
```

**作用：** 裁剪位置嵌入和因果掩码，使用更小的 block_size。

---

## 常见问题解答

### Q1: nn.ModuleDict vs 普通 dict？

`nn.ModuleDict` 会注册子模块，这样：
- `.parameters()` 能找到里面的参数
- `.to('cuda')` 能移动里面的模块
- `torch.save` 能保存里面的权重

### Q2: nn.Embedding 是什么？

本质是查找表：
```python
emb = nn.Embedding(65, 384)  # 65个词，每个384维
# 内部有权重矩阵 (65, 384)
# 输入整数 i，输出第 i 行
```

### Q3: 为什么位置嵌入和词嵌入相加而不是拼接？

- 相加更简洁，维度不变
- 实验证明效果不差
- 拼接会增加维度和参数量

### Q4: generate 方法的序列裁剪

```python
idx[:, -self.config.block_size:]
# :        第一维(batch)全选
# -256:    第二维(序列)取最后256个
```

模型上下文窗口有限，超过了只看最近的 block_size 个 token。

---

## 参数量计算

以 shakespeare_char 模型为例 (n_layer=6, n_head=6, n_embd=384, vocab_size=65)：

| 组件 | 参数量 |
|------|--------|
| wte (词嵌入) | 65 × 384 = 24,960 |
| wpe (位置嵌入) | 256 × 384 = 98,304 |
| 每个 Block | ~590,000 |
| ln_f | 384 × 2 = 768 |
| lm_head | 与 wte 共享 |
| **总计** | ~10.65M |

---

## 总结图

```
┌─────────────────────────────────────────────────────────────┐
│                         GPT 模型                            │
├─────────────────────────────────────────────────────────────┤
│  输入: token IDs (B, T)                                     │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  词嵌入 wte + 位置嵌入 wpe → (B, T, C)              │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Block × N                                          │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │  LayerNorm → CausalSelfAttention → 残差     │   │   │
│  │  │  LayerNorm → MLP → 残差                     │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  最终 LayerNorm ln_f                               │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  输出层 lm_head → logits (B, T, vocab_size)        │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  输出: 每个位置预测下一个 token 的概率分布                  │
└─────────────────────────────────────────────────────────────┘
```

---

*笔记整理于 2026-02-02*
*基于 nanoGPT by Andrej Karpathy*
