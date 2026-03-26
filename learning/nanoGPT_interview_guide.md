# nanoGPT 面试问答完全指南

> 这份指南包含面试可能遇到的所有问题、详细解答、必会手写代码和流程图。

---

## 📚 配套深度补充

**强烈推荐配合阅读**：[nanoGPT_interview_deep_dive.md](./nanoGPT_interview_deep_dive.md)

该补充文档包含：
- Q1-Q5 的深度展开（每个组件的详细原理）
- 相似度矩阵、K vs V 区别、缩放因子数学推导
- GPT/BERT 完整架构图（带维度标注）
- LLaMA 四大改进详解（RMSNorm、RoPE、SwiGLU、GQA）
- RoBERTa 训练优化详解
- Tokenizer 详解（BPE vs WordPiece vs 字节级 BPE）
- KV Cache、门控机制等高级概念
- Q6-Q13 深度补充：自回归生成、隐式集成、Pre-Norm vs Post-Norm、ALiBi、Weight Tying、h 向量详解

---

## 目录

1. [基础问题（必须秒答）](#一基础问题必须秒答)
2. [进阶问题（展示深度）](#二进阶问题展示深度)
3. [代码题（必须手写）](#三代码题必须手写)
4. [流程图（必须会画）](#四流程图必须会画)
5. [追问题（灵活应对）](#五追问题灵活应对)

---

# 一、基础问题（必须秒答）

## Q1: Transformer 的核心组件有哪些？

**答案**：

Transformer 由以下核心组件组成：

1. **Multi-Head Self-Attention**：让每个位置关注序列中的其他位置
2. **Feed-Forward Network (FFN)**：两层全连接网络，先扩展后压缩
3. **Layer Normalization**：归一化，稳定训练
4. **Residual Connection**：残差连接，缓解梯度消失
5. **Positional Encoding/Embedding**：注入位置信息

**记忆口诀**：注意力 + 前馈网 + 归一化 + 残差 + 位置

> 🔗 **深度补充**：[Q1 深度展开 - 五大组件详解](./nanoGPT_interview_deep_dive.md#q1-深度补充transformer-五大核心组件)

---

## Q2: Self-Attention 的计算过程是什么？

**答案**：

**公式**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**步骤**：

1. **线性投影**：输入 X 分别乘以 W_Q, W_K, W_V 得到 Q, K, V
2. **计算注意力分数**：Q 和 K^T 矩阵乘法，得到相似度矩阵
3. **缩放**：除以 √d_k，防止值过大
4. **Softmax**：归一化为概率分布
5. **加权求和**：用注意力权重对 V 加权求和

**直觉**：Q 是"查询"，K 是"键"，V 是"值"。Q 和 K 计算相关性，用这个相关性去加权 V。

> 🔗 **深度补充**：[Q2 深度展开 - 相似度矩阵、K vs V 区别、为什么乘 V](./nanoGPT_interview_deep_dive.md#q2-深度补充self-attention-计算细节)

---

## Q3: 为什么要除以 √d_k？

**答案**：

**原因**：防止点积值过大，导致 softmax 梯度消失。

**详细解释**：
- 假设 Q 和 K 的每个元素都是均值 0、方差 1 的随机变量
- 点积 q·k = Σ(q_i × k_i) 的方差是 d_k
- 当 d_k 很大（如 64）时，点积可能达到 ±几十
- softmax 对大数值的梯度接近 0，导致训练困难
- 除以 √d_k 后，方差变回 1，数值稳定

**数学推导**：
```
Var(q·k) = Var(Σq_i×k_i) = Σ Var(q_i×k_i) = d_k × 1 = d_k
Var(q·k / √d_k) = d_k / d_k = 1
```

> 🔗 **深度补充**：[Q3 深度展开 - d_k 是维度不是方差、完整数学推导](./nanoGPT_interview_deep_dive.md#q3-深度补充缩放因子-d_k-的数学原理)

---

## Q4: 什么是 Multi-Head Attention？为什么要用多头？

**答案**：

**定义**：把 Q、K、V 分成多个头，每个头独立计算注意力，最后拼接。

**为什么要用多头**：

1. **多种表示子空间**：不同头可以关注不同类型的信息（语法、语义、位置等）
2. **增加模型容量**：相当于多个注意力机制的集成
3. **稳定训练**：单头可能陷入某种模式，多头更鲁棒

**计算**：
```python
# 假设 n_head=8, d_model=512
# 每个头的维度 d_k = 512/8 = 64

head_i = Attention(Q_i, K_i, V_i)  # 每个头独立计算
output = Concat(head_1, ..., head_8) @ W_O  # 拼接后投影
```

> 🔗 **深度补充**：[Q4 深度展开 - 为什么拼接后要投影](./nanoGPT_interview_deep_dive.md#q4-深度补充多头注意力的投影)

---

## Q5: GPT 和 BERT 的区别是什么？

**答案**：

| 特性 | GPT | BERT |
|------|-----|------|
| **注意力方向** | 单向（只看左边） | 双向（看两边） |
| **预训练任务** | 语言建模（预测下一个词） | MLM + NSP |
| **掩码** | 因果掩码（下三角） | 随机掩码（[MASK]） |
| **适用任务** | 生成（对话、写作） | 理解（分类、问答） |
| **解码方式** | 自回归（逐个生成） | 非自回归 |

**一句话总结**：GPT 是"单向生成型"，BERT 是"双向理解型"。

> 🔗 **深度补充**：[Q5 深度展开 - GPT/BERT 完整架构图、Encoder-Decoder 关系、LLaMA/RoBERTa](./nanoGPT_interview_deep_dive.md#q5-深度补充gpt-vs-bert-完整对比)

---

## Q6: 什么是因果掩码（Causal Mask）？为什么需要？

**答案**：

**定义**：下三角矩阵，让位置 i 只能看到位置 0 到 i-1。

```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

**为什么需要**：

1. **语言建模的本质**：预测下一个词，不能"偷看"答案
2. **自回归生成**：生成时确实看不到未来
3. **训练-推理一致性**：训练时模拟推理的条件

**实现**：把掩码为 0 的位置填充 -∞，softmax 后变成 0。

> 🔗 **深度补充**：[Q6 深度展开 - 自回归生成详解](./nanoGPT_interview_deep_dive.md#q6-深度补充自回归生成是什么意思)

---

## Q7: LayerNorm 的作用是什么？和 BatchNorm 有什么区别？

**答案**：

**LayerNorm 作用**：
- 归一化每个样本的特征，稳定训练
- 减少内部协变量偏移
- 让不同层的输入分布一致

**区别**：

| 特性 | LayerNorm | BatchNorm |
|------|-----------|-----------|
| **归一化维度** | 特征维度 | Batch 维度 |
| **依赖 batch** | 不依赖 | 依赖 |
| **适用场景** | NLP、序列 | CV、图像 |
| **推理一致性** | 训练推理一致 | 需要 running mean/var |

**为什么 Transformer 用 LayerNorm**：
- 序列长度可变，batch 统计不稳定
- 训练和推理行为一致

---

## Q8: 残差连接（Residual Connection）的作用是什么？

**答案**：

**公式**：`output = x + f(x)`

**作用**：

1. **缓解梯度消失**：梯度可以通过"捷径"直接反向传播
2. **稳定深层网络**：即使 f(x) 学不好，至少有原始的 x
3. **隐式集成**：相当于不同深度网络的集成

**数学解释**：
```
∂L/∂x = ∂L/∂output × (1 + ∂f(x)/∂x)
                     ↑
                   梯度至少为 1，不会消失
```

> 🔗 **深度补充**：[Q8 深度展开 - 残差连接的"隐式集成"原理](./nanoGPT_interview_deep_dive.md#q8-深度补充残差连接的隐式集成)

---

## Q9: Pre-Norm 和 Post-Norm 的区别？为什么现在常用 Pre-Norm？

**答案**：

**区别**：

```python
# Pre-Norm（GPT、LLaMA 等）
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# Post-Norm（原始 Transformer）
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

**为什么用 Pre-Norm**：

1. **训练更稳定**：残差路径不经过 LayerNorm，梯度流动更顺畅
2. **不需要 warmup**：Post-Norm 通常需要更长的 warmup
3. **更容易训练深层网络**

**缺点**：Pre-Norm 最终性能可能略低于调好的 Post-Norm。

> 🔗 **深度补充**：[Q9 深度展开 - Pre-Norm 为什么性能可能略低？深度退化问题](./nanoGPT_interview_deep_dive.md#q9-深度补充pre-norm-为什么性能可能略低于-post-norm)

---

## Q10: GPT 的 Loss 函数是什么？

**答案**：

**交叉熵损失（Cross-Entropy Loss）**：

$$L = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t})$$

**直觉**：
- 模型预测每个位置下一个 token 的概率分布
- Loss = -log(正确 token 的预测概率)
- 预测越准确，概率越高，loss 越低

**代码**：
```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

---

## Q11: 什么是 Embedding？为什么需要？

**答案**：

**定义**：把离散的 token ID 映射到连续的向量空间。

**为什么需要**：
1. **神经网络处理连续数值**：不能直接处理离散 ID
2. **语义表示**：相似词的向量接近
3. **降维**：one-hot 太稀疏，embedding 更紧凑

**代码**：
```python
# vocab_size=65, n_embd=384
embedding = nn.Embedding(65, 384)
# 输入 token ID：[3, 7, 12]
# 输出向量：(3, 384)
```

---

## Q12: 什么是 Position Embedding？为什么需要？

**答案**：

**原因**：Attention 是排列不变的，不知道 token 的位置。

```python
# 交换位置，Attention 结果一样
Attention([A, B, C]) ≈ Attention([B, A, C])  # 对于 self-attention
```

**常见方式**：

| 方式 | 特点 |
|------|------|
| 学习的 | 简单，但有最大长度限制 |
| 正弦/余弦 | 可外推，固定的 |
| RoPE | 相对位置，可外推 |
| ALiBi | 注意力偏置，简单高效 |

**GPT 用的是学习的位置嵌入**：
```python
self.wpe = nn.Embedding(block_size, n_embd)  # 学习每个位置的向量
```

> 🔗 **深度补充**：[Q12 深度展开 - ALiBi 位置编码详解](./nanoGPT_interview_deep_dive.md#q12-深度补充alibi-位置编码详解)

---

## Q13: 什么是 Weight Tying（权重共享）？有什么好处？

**答案**：

**定义**：词嵌入层和输出层共享权重。

```python
self.transformer.wte.weight = self.lm_head.weight
```

**好处**：
1. **减少参数**：少了 vocab_size × n_embd 个参数
2. **语义一致性**：输入和输出用同一套表示
3. **正则化效果**：约束模型学习一致的表示

**直觉**：
- 嵌入：token "A" → 向量 v_A
- 输出：向量与 v_A 越相似，越可能是 "A"
- 两个操作是"反过来"的，共享权重合理

> 🔗 **深度补充**：[Q13 深度展开 - 为什么说"两个操作是反过来的"、h 向量详解](./nanoGPT_interview_deep_dive.md#q13-深度补充weight-tying-为什么说两个操作是反过来的)

---

## Q14: FFN（前馈网络）的结构是什么？为什么先扩展再压缩？

**答案**：

**结构**：
```python
FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
# W1: (n_embd, 4*n_embd)  扩展 4 倍
# W2: (4*n_embd, n_embd)  压缩回来
```

**为什么先扩展再压缩**：
1. **增加表达能力**：更大的中间维度 = 更多的"思考空间"
2. **非线性变换**：GELU 激活在高维空间更有效
3. **信息瓶颈**：压缩回来强制提取重要信息

**为什么是 4 倍**：实验经验值，效果好。

---

## Q15: 什么是 Dropout？为什么训练时用，推理时不用？

**答案**：

**定义**：训练时随机将一部分神经元输出置为 0。

**作用**：
1. **防止过拟合**：强制网络学习冗余表示
2. **隐式集成**：每次训练不同的子网络

**为什么推理时不用**：
- 训练时：随机丢弃，需要放大保留的（除以 1-p）
- 推理时：用完整网络，不丢弃

**代码**：
```python
model.train()  # dropout 生效
model.eval()   # dropout 关闭
```

---

# 二、进阶问题（展示深度）

## Q16: 梯度累积的原理？为什么 loss 要除以累积步数？

**答案**：

**场景**：显存不够用大 batch。

**原理**：
- 分多次小 batch 计算梯度
- 梯度累积后再更新参数
- 等效于大 batch 训练

**为什么要除**：
- PyTorch 的 backward() 是**累加**梯度，不是平均
- 不除的话，累积 N 次 = N 倍的梯度（太大）
- 除以 N 后，等效于大 batch 的平均梯度

```python
for i in range(gradient_accumulation_steps):
    loss = model(X, Y) / gradient_accumulation_steps  # 除以 N
    loss.backward()  # 梯度累加

optimizer.step()     # 累积后更新
optimizer.zero_grad()
```

---

## Q17: 学习率 Warmup 的作用是什么？

**答案**：

**问题**：训练初期，参数随机，梯度大且不稳定。大学习率可能导致发散。

**解决**：学习率从 0 慢慢增大到目标值。

**作用**：
1. **稳定初期训练**：避免一开始就"走歪"
2. **让 Adam 等优化器的动量估计更准**：初期估计不准
3. **适应性调整**：模型"热身"后再加速

**公式**：
```python
if it < warmup_iters:
    lr = max_lr * (it + 1) / (warmup_iters + 1)
```

---

## Q18: 余弦衰减（Cosine Decay）的原理和优点？

**答案**：

**公式**：
```python
decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
coeff = 0.5 * (1 + cos(π * decay_ratio))
lr = min_lr + coeff * (max_lr - min_lr)
```

**特点**：两头慢，中间快（S 形下降）

**优点**：
1. **开始时衰减慢**：还在学习重要特征
2. **结束时衰减慢**：精细调整，避免震荡
3. **比线性衰减效果好**：实验验证

---

## Q19: 混合精度训练的原理？GradScaler 的作用？

**答案**：

**混合精度**：
- 前向/反向传播：用 FP16（快，省显存）
- 参数存储和更新：用 FP32（精确）

**GradScaler 作用**：防止 FP16 梯度下溢

```python
# FP16 梯度可能太小 → 变成 0
# 解决：先放大 loss → 梯度也被放大 → 不会下溢
scaler.scale(loss).backward()  # loss × 1024
scaler.unscale_(optimizer)     # 梯度 ÷ 1024
scaler.step(optimizer)
```

**BF16 不需要 GradScaler**：数值范围和 FP32 一样大。

---

## Q20: DDP（分布式数据并行）的工作原理？

**答案**：

**步骤**：
1. 每个 GPU 复制一份完整模型
2. 每个 GPU 处理不同的数据
3. 反向传播后，AllReduce 同步梯度（求平均）
4. 每个 GPU 用相同梯度更新

**关键点**：
- 数据并行，模型不拆分
- AllReduce 保证参数同步
- 线性加速比（理想情况）

---

## Q21: 什么是 KV Cache？为什么能加速推理？

**答案**：

**问题**：自回归生成时，每生成一个 token 都要重新计算所有历史 token 的 K、V。

**解决**：把历史 token 的 K、V 缓存起来，每次只计算新 token 的。

**效果**：
- 没有 KV Cache：O(n²) 计算量
- 有 KV Cache：O(n) 计算量

```python
# 缓存 K、V
if past_key_value is not None:
    k = torch.cat([past_key_value[0], k], dim=2)
    v = torch.cat([past_key_value[1], v], dim=2)
present = (k, v)  # 返回新的缓存
```

---

## Q22: RoPE（旋转位置编码）的原理和优点？

**答案**：

**原理**：用旋转矩阵对 Q、K 编码相对位置

**优点**：
1. **相对位置**：只关心两个位置的距离
2. **可外推**：训练短序列，推理长序列
3. **高效**：只需要对 Q、K 做简单变换

**被 LLaMA、Mistral 等采用**。

---

## Q23: 为什么 Transformer 比 RNN 好？

**答案**：

| 特性 | Transformer | RNN |
|------|-------------|-----|
| **并行性** | 完全并行 | 顺序计算 |
| **长距离依赖** | O(1) 路径 | O(n) 路径 |
| **梯度问题** | 无 | 梯度消失/爆炸 |
| **训练速度** | 快 | 慢 |

**核心优势**：Attention 让任意两个位置直接交互。

---

## Q24: 如何减少推理延迟？

**答案**：

| 方法 | 原理 |
|------|------|
| KV Cache | 缓存历史 K、V |
| 量化 | FP16/INT8/INT4 |
| 剪枝 | 去掉不重要的权重 |
| 蒸馏 | 小模型学大模型 |
| Flash Attention | 减少显存访问 |
| Speculative Decoding | 小模型预测，大模型验证 |

---

## Q25: 什么是 Flash Attention？为什么快？

**答案**：

**问题**：标准 Attention 需要存储 N×N 的注意力矩阵，显存瓶颈。

**解决**：分块计算，不存储完整矩阵。

**原理**：
1. 把 Q、K、V 分成小块
2. 在 SRAM（快）中计算，减少 HBM（慢）访问
3. 利用 softmax 的数学性质，分块后仍能得到正确结果

**效果**：速度提升 2-4 倍，显存减少。

---

# 三、代码题（必须手写）

## 代码 1：Self-Attention Forward

```python
def self_attention(x, W_q, W_k, W_v, W_o, mask=None):
    """
    x: (B, T, C) 输入
    W_q, W_k, W_v: (C, C) 投影矩阵
    W_o: (C, C) 输出投影
    mask: (T, T) 因果掩码
    """
    B, T, C = x.shape
    
    # 1. 线性投影得到 Q, K, V
    Q = x @ W_q  # (B, T, C)
    K = x @ W_k  # (B, T, C)
    V = x @ W_v  # (B, T, C)
    
    # 2. 计算注意力分数
    d_k = C
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (B, T, T)
    
    # 3. 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 4. Softmax
    attn = F.softmax(scores, dim=-1)  # (B, T, T)
    
    # 5. 加权求和
    out = attn @ V  # (B, T, C)
    
    # 6. 输出投影
    out = out @ W_o  # (B, T, C)
    
    return out
```

---

## 代码 2：Multi-Head Attention

```python
def multi_head_attention(x, n_head, W_qkv, W_o, mask=None):
    """
    x: (B, T, C)
    n_head: 头数
    W_qkv: (C, 3*C) 合并的 QKV 投影
    W_o: (C, C) 输出投影
    """
    B, T, C = x.shape
    head_size = C // n_head
    
    # 1. 计算 Q, K, V 并分头
    qkv = x @ W_qkv  # (B, T, 3*C)
    q, k, v = qkv.split(C, dim=-1)  # 各 (B, T, C)
    
    # 2. 重塑为多头格式
    q = q.view(B, T, n_head, head_size).transpose(1, 2)  # (B, n_head, T, head_size)
    k = k.view(B, T, n_head, head_size).transpose(1, 2)
    v = v.view(B, T, n_head, head_size).transpose(1, 2)
    
    # 3. 计算注意力
    scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)  # (B, n_head, T, T)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    out = attn @ v  # (B, n_head, T, head_size)
    
    # 4. 合并多头
    out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
    
    # 5. 输出投影
    out = out @ W_o
    
    return out
```

---

## 代码 3：Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, mask=None):
        # Pre-Norm + 残差
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x
```

---

## 代码 3.5：GPT 完整模型结构（串联所有组件）

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # ====== Transformer 主体 ======
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # Token Embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Position Embedding
            drop = nn.Dropout(config.dropout),                      # Dropout
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),  # N 个 Block
            ln_f = nn.LayerNorm(config.n_embd),                    # 最后的 LayerNorm
        ))
        
        # ====== 输出层 ======
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # ====== 权重共享（Weight Tying）======
        self.transformer.wte.weight = self.lm_head.weight  # Embedding 和输出层共享权重
        
        self.block_size = config.block_size
    
    def forward(self, idx, targets=None):
        """
        idx: (B, T) 输入 token IDs
        targets: (B, T) 目标 token IDs（训练时提供）
        """
        B, T = idx.shape
        device = idx.device
        
        # ====== 第1步：Embedding ======
        tok_emb = self.transformer.wte(idx)                              # (B, T, n_embd)
        pos_emb = self.transformer.wpe(torch.arange(T, device=device))   # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)                     # (B, T, n_embd)
        
        # ====== 第2步：经过 N 个 Transformer Block ======
        for block in self.transformer.h:
            x = block(x)                                                 # (B, T, n_embd)
        
        # ====== 第3步：最后的 LayerNorm ======
        x = self.transformer.ln_f(x)                                     # (B, T, n_embd)
        
        # ====== 第4步：输出层 ======
        logits = self.lm_head(x)                                         # (B, T, vocab_size)
        
        # ====== 第5步：计算 Loss（如果有 targets）======
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1)                    # (B*T,)
            )
        
        return logits, loss
```

---

## 代码 4：训练循环（带梯度累积）

```python
def train_step(model, optimizer, scaler, dataloader, 
               gradient_accumulation_steps, grad_clip):
    
    model.train()           # 开启训练模式（Dropout 生效）
    optimizer.zero_grad()   # 清零梯度
    
    # ====== 梯度累积循环 ======
    for micro_step in range(gradient_accumulation_steps):  # 循环取小 batch
        X, Y = next(dataloader)                            # 获取一个 micro batch
        
        with torch.cuda.amp.autocast():                    # 开启混合精度（FP16）
            logits, loss = model(X, Y)                     # 前向传播
            loss = loss / gradient_accumulation_steps      # 缩放 loss（为了正确累积）
        
        scaler.scale(loss).backward()  # 放大 loss 防止 FP16 下溢 + 反向传播（梯度累加）
    
    # ====== 梯度处理 ======
    scaler.unscale_(optimizer)                                    # 恢复真实梯度（除以放大倍数）
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # 梯度裁剪防爆炸
    
    # ====== 参数更新 ======
    scaler.step(optimizer)  # 用累积的梯度更新参数
    scaler.update()         # 动态调整放大倍数（根据是否出现 inf/nan）
    optimizer.zero_grad()   # 清零梯度，准备下一轮
```

---

## 代码 5：学习率调度

```python
def get_lr(it, warmup_iters, lr_decay_iters, max_lr, min_lr):
    # 1. 预热阶段
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    
    # 2. 衰减结束后
    if it > lr_decay_iters:
        return min_lr
    
    # 3. 余弦衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

---

## 代码 5.5：外层训练循环（串联代码 4 和代码 5）

```python
def train(config):
    """
    完整的训练流程，串联所有组件
    """
    # ====== 初始化模型 ======
    model = GPT(config)
    model = model.to(config.device)
    
    # ====== 初始化优化器 ======
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.max_lr,           # 初始学习率（会被动态调整）
        betas=(0.9, 0.95),          # Adam 的动量参数
        weight_decay=0.1            # 权重衰减（L2 正则化）
    )
    
    # ====== 初始化混合精度 Scaler ======
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))
    
    # ====== 训练循环 ======
    for iter in range(config.max_iters):
        
        # 1. 动态调整学习率（代码5）
        lr = get_lr(iter, config.warmup_iters, config.lr_decay_iters, 
                    config.max_lr, config.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 2. 执行一步训练（代码4）
        train_step(model, optimizer, scaler, train_dataloader,
                   config.gradient_accumulation_steps, config.grad_clip)
        
        # 3. 定期评估
        if iter % config.eval_interval == 0:
            model.eval()                                    # 切换到评估模式
            with torch.no_grad():
                val_loss = estimate_loss(model, val_dataloader)
            model.train()                                   # 切换回训练模式
            print(f"iter {iter}: lr {lr:.6f}, val_loss {val_loss:.4f}")
        
        # 4. 定期保存 checkpoint
        if iter % config.save_interval == 0 and iter > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': iter,
                'config': config,
            }
            torch.save(checkpoint, f'ckpt_{iter}.pt')
            print(f"saved checkpoint to ckpt_{iter}.pt")
    
    return model


def estimate_loss(model, dataloader, eval_iters=50):
    """在验证集上评估 loss"""
    losses = []
    for _ in range(eval_iters):
        X, Y = next(dataloader)
        with torch.cuda.amp.autocast():
            _, loss = model(X, Y)
        losses.append(loss.item())
    return sum(losses) / len(losses)
```

---

## 代码 6：生成（自回归采样）

```python
@torch.no_grad()  # 关闭梯度计算，节省显存，加速推理
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    model: 训练好的 GPT 模型
    idx: (B, T) 起始 token 序列
    max_new_tokens: 要生成的新 token 数量
    temperature: 温度参数，越大越随机，越小越确定
    top_k: 只从概率最高的 k 个词中采样
    """
    for _ in range(max_new_tokens):  # 循环生成 max_new_tokens 个 token
        
        # ====== 第1步：裁剪序列 ======
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]  # 超过最大长度则只取最后 block_size 个
        
        # ====== 第2步：前向传播 ======
        logits, _ = model(idx_cond)        # 输入序列，得到预测 logits: (B, T, vocab_size)
        logits = logits[:, -1, :]          # 只取最后一个位置的预测: (B, vocab_size)
        logits = logits / temperature      # 温度缩放：T<1 更确定，T>1 更随机
        
        # ====== 第3步：Top-K 过滤（可选）======
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # 找出最大的 k 个值
            logits[logits < v[:, [-1]]] = float('-inf')             # 小于第 k 大的值设为 -inf，softmax 后变 0
        
        # ====== 第4步：采样 ======
        probs = F.softmax(logits, dim=-1)              # 转换为概率分布（和为 1）
        idx_next = torch.multinomial(probs, num_samples=1)  # 按概率随机采样一个 token: (B, 1)
        
        # ====== 第5步：拼接 ======
        idx = torch.cat([idx, idx_next], dim=1)  # 新 token 拼接到序列末尾: (B, T+1)
    
    return idx  # 返回完整序列（原始输入 + 生成的所有 token）
```

---

# 四、流程图（必须会画）

## 图 1：Transformer Block 结构

```
输入 x ─────────────────────┐
    │                       │
    ▼                       │ 残差1
┌─────────────┐             │
│  LayerNorm  │             │
└─────────────┘             │
    │                       │
    ▼                       │
┌─────────────┐             │
│  Self-Attn  │◄── 因果掩码 │
└─────────────┘             │
    │                       │
    ├───────────────────────┘
    │ (+)
    ▼
    x ──────────────────────┐
    │                       │
    ▼                       │ 残差2
┌─────────────┐             │
│  LayerNorm  │             │
└─────────────┘             │
    │                       │
    ▼                       │
┌─────────────┐             │
│    FFN      │             │
│ (扩展4倍→   │             │
│  GELU→压缩) │             │
└─────────────┘             │
    │                       │
    ├───────────────────────┘
    │ (+)
    ▼
输出
```

---

## 图 2：Self-Attention 计算

```
输入 X
   │
   ├─────────┬─────────┐
   ▼         ▼         ▼
┌─────┐  ┌─────┐  ┌─────┐
│ W_Q │  │ W_K │  │ W_V │
└─────┘  └─────┘  └─────┘
   │         │         │
   ▼         ▼         ▼
   Q         K         V
   │         │         │
   └────┬────┘         │
        ▼              │
   Q × K^T             │
        │              │
        ▼              │
   ÷ √d_k              │
        │              │
        ▼              │
     Mask              │
        │              │
        ▼              │
    Softmax            │
        │              │
        └──────┬───────┘
               ▼
          Attn × V
               │
               ▼
             W_O
               │
               ▼
            输出
```

---

## 图 3：GPT 整体架构

```
Token IDs: [3, 7, 12, ...]
         │
         ▼
┌─────────────────────────────┐
│  Token Embedding (wte)      │
│  + Position Embedding (wpe) │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    Transformer Block 1      │
│    (LN → Attn → LN → FFN)   │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    Transformer Block 2      │
└─────────────────────────────┘
         │
        ...
         │
         ▼
┌─────────────────────────────┐
│    Transformer Block N      │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│      Final LayerNorm        │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│    Output Layer (lm_head)   │
│    → logits (vocab_size)    │
└─────────────────────────────┘
         │
         ▼
预测下一个 token 的概率分布
```

---

## 图 4：学习率调度曲线

```
学习率
  ^
max│              ●  ← warmup结束
   │             ╱╲
   │            ╱  ╲
   │           ╱    ╲    余弦衰减
   │          ╱      ╲
   │         ╱        ╲
   │        ╱          ╲
   │       ╱            ╲____________
min│______╱              ↑
   +─────────────────────────────────→ 步数
   0   warmup       lr_decay_iters
```

---

## 图 5：训练流程

```
训练循环 for iter in range(max_iters):
    │
    ▼
  计算学习率 lr = get_lr(iter)（图四）
  写入 optimizer: param_group['lr'] = lr
    │
    ▼
  ┌─ 梯度累积循环 ──────────────────┐
  │  获取 batch → 前向传播（图三）   │
  │  → loss/N → 反向传播            │
  │  （梯度自动累积）               │
  └─────────────────────────────────┘
    │
    ▼
  梯度裁剪
    │
    ▼
  optimizer.step()（用上面设好的 lr 更新参数）
    │
    ▼
  optimizer.zero_grad()（清零梯度）
    │
    ▼
  定期评估 & 保存检查点
```

---

# 五、追问题（灵活应对）

## "如果把 xxx 改成 yyy 会怎样？"

| 问题 | 回答 |
|------|------|
| 如果去掉 LayerNorm？ | 训练不稳定，loss 发散 |
| 如果去掉残差连接？ | 深层梯度消失，难以训练 |
| 如果去掉位置编码？ | 模型不知道词序，效果差 |
| 如果用 ReLU 替代 GELU？ | 可以，但 GELU 更平滑 |
| 如果 head 数变多？ | 每个 head 维度变小，可能影响表达能力 |
| 如果 FFN 扩展倍数改成 2？ | 参数减少，容量下降 |

---

## "这个参数怎么选？"

| 参数 | 经验值 | 说明 |
|------|--------|------|
| n_layer | 6-96 | 越大越强，但越慢 |
| n_head | 8-96 | 通常让 d_head = 64 或 128 |
| d_model | 512-8192 | 越大表示能力越强 |
| batch_size | 尽量大 | 受显存限制 |
| learning_rate | 1e-4 ~ 6e-4 | 大模型用小学习率 |
| warmup | 2000 | 通常占总步数的 1-5% |
| dropout | 0 ~ 0.2 | 预训练用 0，微调可加 |

---

## "你做这个项目学到了什么？"

**模板回答**：

"通过这个项目，我深入理解了：

1. **Transformer 架构**：从代码层面理解了 Attention、FFN、LayerNorm 等组件的实现和作用

2. **训练技巧**：学习了梯度累积、混合精度、学习率调度等工程实践

3. **调试能力**：遇到 loss 不下降等问题时，如何定位和解决

4. **实验意识**：尝试了不同的超参数和温度，对模型行为有了直觉"

---

*面试准备指南 - 2026-02-02*
