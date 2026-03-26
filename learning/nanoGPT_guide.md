# nanoGPT 学习指南

> 作者：Andrej Karpathy（OpenAI 联合创始人、前特斯拉 AI 总监）
> 
> 这是一个最简化的 GPT 训练/微调框架，核心代码只有约 600 行，非常适合大模型初学者学习。

---

## 目录

1. [项目总览](#一项目总览)
2. [文件结构详解](#二文件结构详解)
3. [文件之间的关系](#三文件之间的关系)
4. [核心概念解析](#四核心概念解析)
5. [预训练 vs 微调](#五预训练-vs-微调)
6. [7天学习计划](#六7天学习计划)
7. [代码精读指南](#七代码精读指南)
8. [实践命令速查](#八实践命令速查)
9. [学习检验清单](#九学习检验清单)
10. [求职准备](#十求职准备)

---

## 一、项目总览

### 1.1 项目简介

**nanoGPT** 是一个最简化的 GPT 训练/微调框架：
- `model.py`：~300 行，定义 GPT 模型结构
- `train.py`：~300 行，训练循环

### 1.2 项目功能

1. 从零训练一个新的 GPT 模型（预训练）
2. 微调 OpenAI 预训练的 GPT-2 权重
3. 从训练好的模型生成文本

### 1.3 推荐学习资源

- **必看视频**：[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- **完整课程**：[Zero to Hero 系列](https://karpathy.ai/zero-to-hero.html)
- **升级版项目**：[nanochat](https://github.com/karpathy/nanochat)

---

## 二、文件结构详解

### 2.1 项目目录结构

```
nanoGPT/
├── model.py                 # ⭐⭐⭐⭐⭐ GPT 模型定义
├── train.py                 # ⭐⭐⭐⭐⭐ 训练主脚本
├── sample.py                # ⭐⭐⭐⭐ 文本生成脚本
├── configurator.py          # 配置系统
├── bench.py                 # 性能基准测试
├── config/                  # 配置文件夹
│   ├── train_shakespeare_char.py   # 字符级训练配置（入门首选）
│   ├── train_gpt2.py              # GPT-2 预训练配置
│   ├── finetune_shakespeare.py    # 微调配置
│   └── eval_gpt2*.py              # 评估配置
├── data/                    # 数据文件夹
│   ├── shakespeare_char/    # 字符级莎士比亚（入门）
│   ├── shakespeare/         # BPE token 级莎士比亚（微调用）
│   └── openwebtext/         # 大规模网页数据（预训练用）
├── transformer_sizing.ipynb # Transformer 计算分析
├── scaling_laws.ipynb       # Scaling Laws 分析
└── README.md
```

### 2.2 核心文件说明

| 文件 | 作用 | 重要程度 | 代码行数 |
|------|------|----------|----------|
| `model.py` | GPT 模型的完整实现，包含 Transformer 所有组件 | ⭐⭐⭐⭐⭐ | ~300 行 |
| `train.py` | 训练主脚本，包含完整的训练循环、DDP 分布式训练支持 | ⭐⭐⭐⭐⭐ | ~300 行 |
| `sample.py` | 文本生成脚本，从训练好的模型采样生成文本 | ⭐⭐⭐⭐ | ~90 行 |
| `configurator.py` | 配置系统，支持命令行参数覆盖 | ⭐⭐⭐ | ~50 行 |
| `bench.py` | 性能基准测试脚本 | ⭐⭐ | ~120 行 |

### 2.3 配置文件说明

| 配置文件 | 用途 | 模式 |
|----------|------|------|
| `train_shakespeare_char.py` | 字符级莎士比亚模型训练（入门首选） | 预训练 |
| `train_gpt2.py` | 完整 GPT-2 124M 训练（需要 8×A100） | 预训练 |
| `finetune_shakespeare.py` | 微调预训练 GPT-2 在莎士比亚数据上 | 微调 |
| `eval_gpt2*.py` | 评估不同大小 GPT-2 模型 | 评估 |

### 2.4 数据集说明

| 数据集 | 主要用途 | Tokenization | 数据量 |
|--------|----------|--------------|--------|
| `shakespeare_char/` | 从零训练小模型 | 字符级 | ~100万字符 |
| `shakespeare/` | 微调预训练模型 | BPE token | ~30万 tokens |
| `openwebtext/` | 大规模预训练 | BPE token | ~90亿 tokens |

---

## 三、文件之间的关系

### 3.1 整体架构图

```
                    ┌─────────────────────────────────────┐
                    │           配置文件 config/          │
                    │  (train_shakespeare_char.py 等)     │
                    └──────────────┬──────────────────────┘
                                   │ 覆盖默认参数
                                   ▼
┌──────────────┐    ┌─────────────────────────────────────┐
│ 数据准备脚本 │───▶│            train.py                  │
│ data/*/      │    │  • 加载数据 (train.bin, val.bin)    │
│ prepare.py   │    │  • 创建模型 (调用 model.py)         │
└──────────────┘    │  • 训练循环 (前向/反向/优化)        │
       │            │  • 保存 checkpoint                   │
       │            └──────────────┬──────────────────────┘
       │                           │ 导入 GPT 类
       ▼                           ▼
┌──────────────┐    ┌─────────────────────────────────────┐
│ train.bin    │    │            model.py                  │
│ val.bin      │    │  • LayerNorm                        │
│ meta.pkl     │    │  • CausalSelfAttention (自注意力)   │
└──────────────┘    │  • MLP (前馈网络)                   │
                    │  • Block (Transformer 块)           │
                    │  • GPT (完整模型)                   │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │           sample.py                  │
                    │  • 加载 checkpoint                   │
                    │  • 文本生成 (自回归采样)            │
                    └─────────────────────────────────────┘
```

### 3.2 完整工作流程（含 model.py 调用关系）

**注意**：`model.py` 不是直接运行的脚本，而是被 `train.py` 和 `sample.py` 作为模块导入使用。

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据准备阶段                              │
│  python data/xxx/prepare.py  →  生成 train.bin, val.bin         │
│  （不需要 model.py）                                             │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     train.py 训练阶段                            │
│                                                                  │
│   from model import GPTConfig, GPT  ←─────┐                      │
│                                           │                      │
│   # 创建模型                              │   ┌──────────────┐   │
│   gptconf = GPTConfig(**model_args)       │   │  model.py    │   │
│   model = GPT(gptconf)                    │   │  (模块)      │   │
│                                           ├───│              │   │
│   # 训练循环                              │   │  • GPTConfig │   │
│   logits, loss = model(X, Y)              │   │  • GPT 类    │   │
│   loss.backward()                         │   │  • Attention │   │
│   optimizer.step()                        │   │  • MLP       │   │
│                                           │   └──────────────┘   │
│   方式 A：预训练 init_from='scratch'      │                      │
│   方式 B：微调   init_from='gpt2-xl'      │                      │
│                                           │                      │
│   输出：out-xxx/ckpt.pt                   │                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     sample.py 生成阶段                           │
│                                                                  │
│   from model import GPTConfig, GPT  ←─────┐                      │
│                                           │                      │
│   # 加载模型                              │   ┌──────────────┐   │
│   gptconf = GPTConfig(**args)             │   │  model.py    │   │
│   model = GPT(gptconf)                    ├───│  (模块)      │   │
│   model.load_state_dict(checkpoint)       │   └──────────────┘   │
│                                           │                      │
│   # 生成文本                              │                      │
│   y = model.generate(x, max_new_tokens)   │                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 文件角色总结

| 文件 | 角色 | 如何使用 |
|------|------|----------|
| `model.py` | **模块/库** | 不直接运行，被其他脚本 import |
| `train.py` | **主程序** | 直接运行，import model.py 创建并训练模型 |
| `sample.py` | **主程序** | 直接运行，import model.py 加载模型并生成 |
| `prepare.py` | **主程序** | 直接运行，准备数据，不需要 model.py |

简单理解：
```
model.py  = 定义"什么是 GPT"（蓝图/图纸）
train.py  = 根据蓝图建造 GPT 并训练它
sample.py = 使用训练好的 GPT 生成文本
```

---

## 四、核心概念解析

### 4.1 model.py 中的核心组件

| 组件 | 代码位置 | 功能说明 |
|------|----------|----------|
| `LayerNorm` | 18-27 行 | 层归一化，稳定训练 |
| `CausalSelfAttention` | 29-76 行 | **最核心**，因果自注意力机制 |
| `MLP` | 78-92 行 | 前馈神经网络（FFN） |
| `Block` | 94-106 行 | 一个 Transformer 块 = Attention + MLP |
| `GPTConfig` | 108-116 行 | 模型配置类 |
| `GPT` | 118-330 行 | 完整 GPT 模型 |

### 4.2 重点理解的概念

| 概念 | 在代码中的位置 | 说明 |
|------|----------------|------|
| 位置编码 | `model.py` - `wpe` embedding | 让模型知道 token 的位置信息 |
| 自注意力 | `model.py` - `CausalSelfAttention` | Q、K、V 计算注意力分数 |
| 因果 Mask | `model.py` - `self.bias` 下三角矩阵 | 防止看到未来的 token |
| 权重共享 | `model.py` - `wte.weight = lm_head.weight` | embedding 和输出层共享权重 |
| 梯度累积 | `train.py` - `gradient_accumulation_steps` | 模拟更大的 batch size |
| 学习率调度 | `train.py` - `get_lr()` | cosine decay with warmup |
| 混合精度 | `train.py` - `GradScaler` | float16/bfloat16 加速训练 |

### 4.3 CausalSelfAttention 核心代码解析

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        # Q, K, V 投影层（合并为一个线性层提高效率）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 因果 mask（下三角矩阵）
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # 1. 计算 Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 2. 多头注意力：reshape 为 (B, num_heads, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # 3. 计算注意力分数：Q @ K^T / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 4. 应用因果 mask（关键！）
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # 5. Softmax 归一化
        att = F.softmax(att, dim=-1)
        
        # 6. 加权求和：Attention @ V
        y = att @ v
        
        # 7. 合并多头，输出投影
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
```

### 4.4 训练循环核心代码解析

```python
# train.py 核心训练循环
while True:
    # 学习率调度
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # 评估和保存 checkpoint
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        if losses['val'] < best_val_loss:
            # 保存最佳模型
            torch.save(checkpoint, 'ckpt.pt')
    
    # 梯度累积
    for micro_step in range(gradient_accumulation_steps):
        logits, loss = model(X, Y)            # 前向传播
        loss = loss / gradient_accumulation_steps  # 缩放 loss
        loss.backward()                        # 反向传播
        X, Y = get_batch('train')             # 预取下一批数据
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # 更新参数
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    iter_num += 1
```

---

## 五、预训练 vs 微调

### 5.1 核心区别

| 对比项 | 预训练 (Pretrain) | 微调 (Finetune) |
|--------|-------------------|-----------------|
| 目的 | 学习通用语言知识 | 适应特定任务/领域 |
| 数据量 | 大（数十亿 tokens） | 小（数万~数百万 tokens） |
| 训练时间 | 长（数天~数周） | 短（数分钟~数小时） |
| 初始权重 | 随机初始化 | 加载预训练权重 |
| 学习率 | 较高 (6e-4) | 较低 (3e-5) |

### 5.2 代码中的区别

```python
# train.py 中通过 init_from 参数控制

# 预训练：从零开始
init_from = 'scratch'

# 继续训练：从 checkpoint 恢复
init_from = 'resume'

# 微调：加载预训练权重
init_from = 'gpt2'       # 124M
init_from = 'gpt2-medium'  # 350M
init_from = 'gpt2-large'   # 774M
init_from = 'gpt2-xl'      # 1558M
```

### 5.3 推荐学习路径

```
第一步：shakespeare_char/ + train_shakespeare_char.py 
       → 从零训练小模型（理解预训练）

第二步：shakespeare/ + finetune_shakespeare.py 
       → 微调 GPT-2（理解微调）

第三步：openwebtext/ + train_gpt2.py 
       → 大规模预训练（需要强大 GPU）
```

---

## 六、7天学习计划

### Day 1：跑通代码 + 建立全局观

| 时间 | 任务 | 目标 |
|------|------|------|
| 上午 | 环境配置，跑通 shakespeare_char 训练 | 成功启动训练 |
| 下午 | 用 sample.py 生成文本，观察效果 | 看到生成结果 |
| 晚上 | 通读 README.md，理解项目结构 | 知道每个文件的作用 |

**需要理解的知识点**：
- 项目整体结构
- 数据准备流程
- 训练和生成的基本流程

---

### Day 2：精读 model.py（上半部分）

| 时间 | 任务 | 代码位置 |
|------|------|----------|
| 上午 | 理解 `GPTConfig`、`LayerNorm` | 108-116行, 18-27行 |
| 下午 | **重点精读 `CausalSelfAttention`** | 29-76行 |
| 晚上 | 用纸笔画出 attention 的计算过程 | - |

**需要理解的知识点**：
- LayerNorm 的作用和实现
- 多头自注意力机制（Multi-Head Attention）
- Q、K、V 的含义和计算
- 因果 Mask 的原理（为什么需要下三角矩阵）
- 缩放点积注意力（Scaled Dot-Product Attention）

---

### Day 3：精读 model.py（下半部分）

| 时间 | 任务 | 代码位置 |
|------|------|----------|
| 上午 | 理解 `MLP`、`Block` | 78-106行 |
| 下午 | 精读 `GPT` 类的 `__init__` 和 `forward` | 118-193行 |
| 晚上 | 理解 `generate` 函数（自回归生成） | 305-330行 |

**需要理解的知识点**：
- MLP（前馈神经网络）的结构
- GELU 激活函数
- Pre-LN vs Post-LN 结构
- 残差连接（Residual Connection）
- Token Embedding 和 Position Embedding
- 权重共享（Weight Tying）
- 自回归生成的原理
- Temperature 和 Top-K 采样

---

### Day 4：精读 train.py

| 时间 | 任务 | 代码位置 |
|------|------|----------|
| 上午 | 理解配置参数、DDP 初始化 | 1-100行 |
| 下午 | 精读 `get_batch`、训练循环 | 116-131行, 249-333行 |
| 晚上 | 理解学习率调度 `get_lr`、梯度累积 | 231-242行 |

**需要理解的知识点**：
- 分布式训练（DDP）基础
- 数据加载和 batch 构造
- 梯度累积（Gradient Accumulation）的原理
- 学习率调度（Warmup + Cosine Decay）
- 混合精度训练（AMP）
- 梯度裁剪（Gradient Clipping）
- Checkpoint 保存和加载

---

### Day 5：数据处理 + sample.py + 动手实验

| 时间 | 任务 |
|------|------|
| 上午 | 理解三个 `prepare.py` 的区别 |
| 下午 | 精读 `sample.py`，理解生成过程 |
| 晚上 | 动手实验：修改超参数，观察效果 |

**需要理解的知识点**：
- 字符级 Tokenization vs BPE Tokenization
- tiktoken 库的使用
- 数据的二进制存储格式（memmap）
- 训练集/验证集划分
- Temperature 对生成的影响
- Top-K 采样的原理

**实验建议**：
```bash
# 实验1：改变模型大小
python train.py config/train_shakespeare_char.py --n_layer=4 --n_head=4

# 实验2：改变学习率
python train.py config/train_shakespeare_char.py --learning_rate=1e-4

# 实验3：改变 temperature 生成
python sample.py --out_dir=out-shakespeare-char --temperature=0.5
python sample.py --out_dir=out-shakespeare-char --temperature=1.5
```

---

### Day 6：微调实践 + 深入理解

| 时间 | 任务 |
|------|------|
| 上午 | 跑通微调流程（shakespeare + finetune） |
| 下午 | 学习 `transformer_sizing.ipynb` |
| 晚上 | 理解混合精度训练、Flash Attention |

**需要理解的知识点**：
- 微调和预训练的区别
- 预训练权重的加载（from_pretrained）
- 参数量计算
- FLOPs 计算
- MFU（Model FLOPs Utilization）
- Flash Attention 的原理和优势

---

### Day 7：总结 + Bug-Free 检验

| 时间 | 任务 |
|------|------|
| 上午 | 从头到尾梳理整个流程，画架构图 |
| 下午 | 闭卷默写核心代码（检验理解程度） |
| 晚上 | 准备面试讲解：用自己的话描述整个项目 |

**检验方法**：
1. 不看代码，画出 Transformer Block 结构图
2. 手写 CausalSelfAttention 的 forward 函数
3. 解释训练循环的每一步
4. 说明预训练和微调在代码上的区别

---

## 七、代码精读指南

### 7.1 model.py 阅读顺序

```
1. GPTConfig (108-116行) - 理解所有超参数
   ↓
2. LayerNorm (18-27行) - 层归一化
   ↓
3. CausalSelfAttention (29-76行) - ⭐最重要
   ↓
4. MLP (78-92行) - 前馈网络
   ↓
5. Block (94-106行) - Transformer 块
   ↓
6. GPT.__init__ (120-148行) - 模型构建
   ↓
7. GPT.forward (170-193行) - 前向传播
   ↓
8. GPT.generate (305-330行) - 文本生成
```

### 7.2 train.py 阅读顺序

```
1. 配置参数 (32-78行) - 所有可调参数
   ↓
2. DDP 初始化 (82-101行) - 分布式训练设置
   ↓
3. get_batch (116-131行) - 数据加载
   ↓
4. 模型初始化 (146-193行) - 创建/加载模型
   ↓
5. estimate_loss (214-228行) - 评估函数
   ↓
6. get_lr (231-242行) - 学习率调度
   ↓
7. 训练循环 (249-333行) - 核心训练逻辑
```

---

## 八、实践命令速查

### 8.1 环境配置

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### 8.2 预训练（从零训练）

```bash
# 准备数据
python data/shakespeare_char/prepare.py

# GPU 训练
python train.py config/train_shakespeare_char.py

# CPU 训练（Mac 无 GPU）
python train.py config/train_shakespeare_char.py --device=cpu --compile=False

# Mac MPS 加速
python train.py config/train_shakespeare_char.py --device=mps --compile=False
```

### 8.3 微调

```bash
# 准备数据
python data/shakespeare/prepare.py

# 微调 GPT-2
python train.py config/finetune_shakespeare.py --device=mps --compile=False
```

### 8.4 生成文本

```bash
# 从训练好的模型生成
python sample.py --out_dir=out-shakespeare-char --device=mps

# 直接从 GPT-2 生成
python sample.py --init_from=gpt2-xl --device=mps

# 调整生成参数
python sample.py --out_dir=out-shakespeare-char --temperature=0.8 --top_k=200
```

### 8.5 评估预训练模型

```bash
python train.py config/eval_gpt2.py
python train.py config/eval_gpt2_medium.py
python train.py config/eval_gpt2_large.py
python train.py config/eval_gpt2_xl.py
```

---

## 九、学习检验清单

### 9.1 基础理解 ✅

- [ ] 能画出 Transformer Block 的结构图
- [ ] 能解释 Self-Attention 的计算过程
- [ ] 理解因果 Mask 的作用
- [ ] 理解 Position Embedding 的必要性
- [ ] 理解残差连接的作用

### 9.2 代码理解 ✅

- [ ] 能解释 `CausalSelfAttention` 每一行代码的作用
- [ ] 能说清楚 `init_from` 三种模式的区别
- [ ] 能解释梯度累积的原理和作用
- [ ] 能说明预训练和微调在代码上的区别
- [ ] 能从头写出 `generate` 函数的核心逻辑

### 9.3 实践能力 ✅

- [ ] 能独立跑通训练流程
- [ ] 能修改超参数进行实验
- [ ] 能分析训练 loss 曲线
- [ ] 能用自己的数据训练模型

### 9.4 进阶理解 ✅

- [ ] 理解 DDP 分布式训练的基本原理
- [ ] 理解混合精度训练的优势
- [ ] 理解 Flash Attention 的原理
- [ ] 能计算模型的参数量和 FLOPs

---

## 十、求职准备

### 10.1 学透后能获得的技能

1. **Transformer 模型实现**：能手写 Attention 机制
2. **PyTorch 分布式训练**：理解 DDP 原理
3. **混合精度训练**：理解 AMP、GradScaler
4. **大模型训练优化**：梯度累积、学习率调度、梯度裁剪

### 10.2 简历写法建议

> 深入研究 nanoGPT 项目，掌握 Transformer 模型实现细节，包括多头自注意力机制、位置编码、残差连接等核心组件。具备从零训练和微调 GPT 模型的实践经验，熟悉 PyTorch 分布式训练（DDP）和混合精度训练技术。

### 10.3 面试常见问题

1. **请描述 Transformer 的结构**
2. **Self-Attention 是如何计算的？**
3. **为什么需要因果 Mask？**
4. **预训练和微调有什么区别？**
5. **梯度累积是什么？有什么作用？**
6. **学习率 warmup 有什么好处？**
7. **什么是混合精度训练？**

---

## 附录：GPT-2 模型规格

| 模型 | 参数量 | 层数 | 头数 | 嵌入维度 |
|------|--------|------|------|----------|
| gpt2 | 124M | 12 | 12 | 768 |
| gpt2-medium | 350M | 24 | 16 | 1024 |
| gpt2-large | 774M | 36 | 20 | 1280 |
| gpt2-xl | 1558M | 48 | 25 | 1600 |

---

*文档最后更新：2026年1月*
