# nanoGPT train.py 完全解析笔记

> 这份笔记详细解析了 nanoGPT 的训练代码，包含所有核心概念的深入讲解。

---

## 目录

1. [整体架构概览](#整体架构概览)
2. [导入模块](#导入模块)
3. [配置参数详解](#配置参数详解)
4. [分布式训练 DDP](#分布式训练-ddp)
5. [设备和精度设置](#设备和精度设置)
6. [数据加载](#数据加载)
7. [模型初始化](#模型初始化)
8. [优化器和编译](#优化器和编译)
9. [评估函数](#评估函数)
10. [学习率调度](#学习率调度)
11. [训练循环核心](#训练循环核心)
12. [梯度累积详解](#梯度累积详解)
13. [混合精度训练](#混合精度训练)

---

## 整体架构概览

```
train.py 执行流程
├── 1. 加载配置（命令行参数覆盖默认值）
├── 2. 设置设备和精度（CPU/GPU，FP16/BF16/FP32）
├── 3. 初始化分布式训练（如果多GPU）
├── 4. 创建数据加载器
├── 5. 初始化模型（从头/继续/预训练）
├── 6. 配置优化器
├── 7. 训练循环
│   ├── 评估（定期）
│   ├── 前向传播
│   ├── 反向传播（梯度累积）
│   ├── 梯度裁剪
│   └── 参数更新
└── 8. 保存检查点
```

---

## 导入模块

```python
import os                    # 操作系统接口（文件路径、环境变量）
import time                  # 时间测量
import math                  # 数学函数（cos、sqrt）
import pickle                # Python 对象序列化
from contextlib import nullcontext  # 空上下文管理器

import numpy as np           # 数值计算
import torch                 # PyTorch 框架
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式训练
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT   # 导入模型定义
```

---

## 配置参数详解

### I/O 配置

```python
out_dir = 'out'              # 输出目录，保存检查点
eval_interval = 2000         # 每多少步评估一次
log_interval = 1             # 每多少步打印日志
eval_iters = 200             # 评估时用多少个 batch 计算平均 loss
eval_only = False            # 是否只评估不训练
always_save_checkpoint = True # 每次评估都保存 vs 只在 val loss 下降时保存
init_from = 'scratch'        # 初始化方式：'scratch'/'resume'/'gpt2*'
```

### wandb 配置

```python
wandb_log = False            # 是否用 Weights & Biases 记录
wandb_project = 'owt'        # 项目名
wandb_run_name = 'gpt2'      # 运行名
```

### 数据配置

```python
dataset = 'openwebtext'              # 数据集名称
gradient_accumulation_steps = 5 * 8  # 梯度累积步数（见详解）
batch_size = 12                      # micro-batch 大小
block_size = 1024                    # 序列长度（上下文窗口）
```

### 模型配置

```python
n_layer = 12                 # Transformer 层数
n_head = 12                  # 注意力头数
n_embd = 768                 # 嵌入维度
dropout = 0.0                # Dropout（预训练通常为0）
bias = False                 # 是否使用偏置
```

### 优化器配置

```python
learning_rate = 6e-4         # 最大学习率
max_iters = 600000           # 总训练步数
weight_decay = 1e-1          # 权重衰减系数
beta1 = 0.9                  # Adam 一阶矩衰减率
beta2 = 0.95                 # Adam 二阶矩衰减率
grad_clip = 1.0              # 梯度裁剪阈值
```

### 学习率调度配置

```python
decay_lr = True              # 是否衰减学习率
warmup_iters = 2000          # 预热步数
lr_decay_iters = 600000      # 衰减到 min_lr 的步数
min_lr = 6e-5                # 最小学习率
```

### 系统配置

```python
device = 'cuda'              # 设备
dtype = 'bfloat16'           # 数据类型
compile = True               # 是否用 torch.compile
```

### 配置覆盖机制

```python
config_keys = [k for k,v in globals().items() 
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # 执行配置器，解析命令行参数
config = {k: globals()[k] for k in config_keys}  # 保存最终配置
```

**工作原理**：
- `globals()` 返回当前模块所有全局变量
- `configurator.py` 解析命令行参数（如 `--device=cpu`）并覆盖变量
- 命令行的 `--device=cpu` 被转换为 `device = 'cpu'` 并执行

---

## 分布式训练 DDP

### 什么是 DDP？

**DDP（Distributed Data Parallel）** 是 PyTorch 的多 GPU 训练方案。

```
┌─────────────────────────────────────────────────────────────────┐
│                         DDP 工作流程                             │
├─────────────────────────────────────────────────────────────────┤
│    GPU 0                GPU 1                GPU 2               │
│    ┌────┐               ┌────┐               ┌────┐              │
│    │模型│               │模型│               │模型│              │
│    │副本│               │副本│               │副本│              │
│    └────┘               └────┘               └────┘              │
│      ↓                    ↓                    ↓                 │
│   数据 A                数据 B                数据 C              │
│      ↓                    ↓                    ↓                 │
│   梯度 A                梯度 B                梯度 C              │
│      └────────────────────┼────────────────────┘                 │
│                           ↓                                      │
│                    AllReduce（求平均）                            │
│                           ↓                                      │
│                   所有 GPU 用相同梯度更新                         │
└─────────────────────────────────────────────────────────────────┘
```

**关键点**：
1. 每个 GPU 有完整的模型副本
2. 每个 GPU 处理不同的数据
3. 反向传播后，AllReduce 同步梯度
4. 所有 GPU 用相同梯度更新，保持同步

### DDP 初始化代码

```python
ddp = int(os.environ.get('RANK', -1)) != -1  # 检查是否分布式环境

if ddp:
    init_process_group(backend=backend)       # 初始化进程组通信
    ddp_rank = int(os.environ['RANK'])        # 全局进程编号（0,1,2...）
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 本机内进程编号
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # 总进程数
    device = f'cuda:{ddp_local_rank}'          # 每个进程用不同 GPU
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0             # rank 0 负责日志和保存
    seed_offset = ddp_rank                      # 不同进程用不同随机种子
    gradient_accumulation_steps //= ddp_world_size  # 梯度累积步数分摊
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
```

### DDP 包装模型

```python
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```

**为什么要包装？**
- DDP 注册梯度钩子，每次 backward 后自动同步
- 管理跨 GPU 通信
- 确保所有 GPU 参数一致

**获取原始模型**：
```python
raw_model = model.module if ddp else model
# DDP 包装后，原始模型在 model.module 里
# 保存权重时要用 raw_model.state_dict()
```

---

## 设备和精度设置

```python
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
```

**Token 总数计算**：
```
每次参数更新处理的 token 数 = 
    梯度累积步数 × GPU数量 × batch大小 × 序列长度

例如：40 × 1 × 12 × 1024 = 491,520 tokens
```

```python
torch.manual_seed(1337 + seed_offset)          # 设置随机种子（可复现）
torch.backends.cuda.matmul.allow_tf32 = True   # 允许 TF32 加速
torch.backends.cudnn.allow_tf32 = True
```

```python
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```

**ctx 上下文管理器**：
- CPU：`nullcontext()` 什么都不做
- GPU：`autocast()` 自动混合精度

---

## 数据加载

```python
data_dir = os.path.join('data', dataset)

def get_batch(split):
    # 使用 memmap 避免内存泄漏
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # 随机选择起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # 构造输入和目标（目标是输入右移一位）
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # 移动到设备
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
```

**x 和 y 的关系**：
```
假设 data = [H, E, L, L, O, W, O, R, L, D]
如果 i=0, block_size=5:
x = [H, E, L, L, O]      # 输入
y = [E, L, L, O, W]      # 目标（右移一位）

模型任务：给定 x[0]，预测 y[0]；给定 x[0:2]，预测 y[1]...
```

**np.memmap**：
- 内存映射文件，不需要加载整个文件到内存
- 按需读取，适合大文件

**pin_memory() + non_blocking**：
- 把数据固定在 CPU 内存，加速 CPU→GPU 传输
- 异步传输，不阻塞 CPU

---

## 模型初始化

### 准备工作

```python
iter_num = 0                # 当前迭代数
best_val_loss = 1e9         # 最佳验证损失（初始化为很大）
```

**为什么初始化为很大的值（1e9）？**
```python
if losses['val'] < best_val_loss:  # 当前 loss 比历史最佳更低
    best_val_loss = losses['val']  # 更新最佳
    # 保存检查点
```
第一次评估时，无论 loss 多少都比 1e9 小，确保第一次会保存。

### 读取词汇表大小

```python
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size}")
```

### 三种初始化方式

#### 1. 从头训练（scratch）

```python
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)    # 创建配置对象
    model = GPT(gptconf)                  # 创建模型
```

**`GPTConfig(**model_args)` 是什么？**

`**model_args` 把字典展开成关键字参数：
```python
model_args = {'n_layer': 6, 'n_head': 6, 'n_embd': 384}

# 这两行等价：
gptconf = GPTConfig(**model_args)
gptconf = GPTConfig(n_layer=6, n_head=6, n_embd=384)
```

#### 2. 继续训练（resume）

```python
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 从检查点恢复配置
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # 加载权重
    state_dict = checkpoint['model']
    # 处理 torch.compile 添加的前缀（工程细节，可跳过）
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    # 恢复训练状态
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
```

#### 3. 从预训练模型微调（gpt2*）

```python
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)  # 覆盖 dropout 参数
    model = GPT.from_pretrained(init_from, override_args)
```

**`override_args` 的作用**：
- 加载预训练模型时，大部分配置用预训练的
- 但某些参数想自定义（如 dropout）
- `override_args` 告诉函数："其他用预训练的，但 dropout 用我指定的"

### 移动模型到设备

```python
model.to(device)
```

---

## 优化器和编译

### GradScaler（梯度缩放器）

```python
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```

**为什么需要 GradScaler？**

FP16 的数值范围小（±65504），梯度可能"下溢"（太小变成0）。

**解决方案**：
```python
# 1. 放大 loss
scaled_loss = scaler.scale(loss)  # loss × 1024
scaled_loss.backward()            # 梯度也被放大

# 2. 缩小梯度并更新
scaler.unscale_(optimizer)        # 梯度 ÷ 1024
scaler.step(optimizer)            # 更新参数
scaler.update()                   # 调整缩放因子
```

**BF16 不需要 GradScaler**：因为 BF16 数值范围和 FP32 一样大。

### 浮点精度对比

| 类型 | 位数 | 数值范围 | 精度 | 需要 GradScaler |
|------|------|---------|------|-----------------|
| FP32 | 32位 | ±3.4×10³⁸ | 高 | 不需要 |
| FP16 | 16位 | ±65504 | 低 | **需要** |
| BF16 | 16位 | ±3.4×10³⁸ | 中 | 不需要 |

### 配置优化器

```python
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # 释放内存
```

### 模型编译

```python
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # PyTorch 2.0 编译优化
```

**torch.compile** 把 Python 代码编译成高效机器码，可加速 1.5-2 倍。

---

## 评估函数

```python
@torch.no_grad()              # 装饰器：函数内不计算梯度
def estimate_loss():
    out = {}                   # 存放结果 {'train': xxx, 'val': xxx}
    model.eval()               # 评估模式：关闭 dropout
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)  # 存放每次的 loss
        
        for k in range(eval_iters):       # 跑 eval_iters 次
            X, Y = get_batch(split)       # 获取 batch
            with ctx:                      # 混合精度上下文
                logits, loss = model(X, Y) # 前向传播
            losses[k] = loss.item()       # .item() 张量→Python数字
        
        out[split] = losses.mean()        # 取平均
    
    model.train()              # 切回训练模式
    return out
```

**为什么跑多次取平均？**
- 每次 get_batch 随机采样不同数据
- 单次 loss 有随机性
- 多次平均更稳定、更准确

**model.eval() vs model.train()**：
- eval()：关闭 Dropout，BatchNorm 用全局统计量
- train()：Dropout 正常工作

---

## 学习率调度

### 完整调度函数

```python
def get_lr(it):
    # 1) 预热阶段：线性增长
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    
    # 2) 衰减结束后：保持最小值
    if it > lr_decay_iters:
        return min_lr
    
    # 3) 中间阶段：余弦衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
```

### 学习率曲线图

```
学习率
  ^
max│              ●  ← warmup结束，达到峰值
   │             ╱╲
   │            ╱  ╲
   │           ╱    ╲         余弦衰减
   │          ╱      ╲        （曲线下降）
   │         ╱        ╲
   │        ╱          ╲
   │       ╱            ╲
   │      ╱              ╲_______________  ← 保持 min_lr
   │     ╱                ↑
min│____╱              lr_decay_iters
   +─────────────────────────────────────→ 训练步数
   0        ↑
         warmup_iters
```

### 三个阶段详解

| 阶段 | 范围 | 公式 | 曲线形状 |
|------|------|------|---------|
| 预热 | 0 → warmup_iters | `lr × (it+1) / (warmup+1)` | 线性上升 📈 |
| 余弦衰减 | warmup → lr_decay | 余弦公式 | 曲线下降 🎢 |
| 保持最小 | lr_decay → 结束 | `min_lr` | 水平 ➡️ |

### 预热公式解析

```python
return learning_rate * (it + 1) / (warmup_iters + 1)
```

**为什么用 `(it + 1)` 而不是 `it`？**
- 如果用 `it`：当 it=0 时，lr=0（完全不更新！）
- 用 `(it + 1)`：当 it=0 时，lr = learning_rate / (warmup+1)（很小但不是0）

**例子**（learning_rate=1e-3, warmup_iters=100）：
```
it=0:   lr = 1e-3 × 1/101 ≈ 9.9e-6
it=50:  lr = 1e-3 × 51/101 ≈ 5.0e-4
it=100: 进入余弦衰减阶段
```

### 余弦衰减详解

```python
decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
return min_lr + coeff * (learning_rate - min_lr)
```

**decay_ratio（衰减进度）**：
- decay_ratio = 0：衰减刚开始
- decay_ratio = 0.5：衰减到一半
- decay_ratio = 1：衰减结束

**coeff（系数）**：
- coeff = 1：学习率 = max
- coeff = 0.5：学习率 = (max + min) / 2
- coeff = 0：学习率 = min

**余弦曲线特点**：两头慢，中间快（S形下降）

| 进度 | cos值 | coeff | 说明 |
|------|-------|-------|------|
| 0.0 | 1.00 | 1.00 | 开始，下降慢 |
| 0.2 | 0.81 | 0.90 | |
| 0.5 | 0.00 | 0.50 | 中间，下降快 |
| 0.8 | -0.81 | 0.10 | |
| 1.0 | -1.00 | 0.00 | 结束，下降慢 |

---

## 训练循环核心

### 初始化

```python
X, Y = get_batch('train')     # 获取第一个 batch
t0 = time.time()              # 记录开始时间
local_iter_num = 0            # 本进程迭代计数
raw_model = model.module if ddp else model  # 获取原始模型
running_mfu = -1.0            # 运行中的 MFU
```

**MFU（Model FLOPs Utilization）**：
- 定义：实际计算速度 / 硬件理论峰值速度
- 例如 MFU=50%：达到硬件一半性能
- 用于衡量代码效率

### 主循环

```python
while True:
    # 1. 设置学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # 2. 定期评估和保存
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    # 3. 前向和反向传播（见下一节详解）
    
    # 4. 终止条件
    iter_num += 1
    if iter_num > max_iters:
        break
```

---

## 梯度累积详解

### 为什么需要梯度累积？

**问题**：想用 batch_size=480 训练，但显存只够 batch_size=12。

**解决**：分 40 次，每次 12 个样本，累积梯度后一起更新。

### 核心代码

```python
for micro_step in range(gradient_accumulation_steps):
    if ddp:
        # 只在最后一步同步梯度
        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
    
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps  # 缩放 loss
    
    X, Y = get_batch('train')          # 预取下一个 batch
    scaler.scale(loss).backward()       # 反向传播，累积梯度

# 梯度裁剪
if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

# 更新参数
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

### 为什么 loss 要除以 gradient_accumulation_steps？

**PyTorch 的 backward() 会累加梯度，不是取平均！**

```python
# 不除（错误）：
loss1.backward()  # grad = g1
loss2.backward()  # grad = g1 + g2
...
loss40.backward() # grad = g1 + g2 + ... + g40 = 40 × 平均梯度（太大！）

# 除以40（正确）：
(loss1/40).backward()  # grad = g1/40
(loss2/40).backward()  # grad = g1/40 + g2/40
...
(loss40/40).backward() # grad = (g1+...+g40)/40 = 平均梯度 ✓
```

### DDP 梯度同步优化

**问题**：DDP 默认每次 backward 都跨 GPU 同步，40 次就要通信 40 次。

**优化**：前 39 次不同步，只在第 40 次同步。

```python
model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
```

```
┌─────────────────────────────────────────────────────────────────┐
│            默认（每次同步，40次通信）                             │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: backward → AllReduce ← 通信                            │
│  Step 2: backward → AllReduce ← 通信                            │
│  ...                                                            │
│  Step 40: backward → AllReduce ← 通信                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│            优化后（只同步一次，1次通信）                          │
├─────────────────────────────────────────────────────────────────┤
│  Step 1-39: backward（不同步，各GPU自己累积）                    │
│  Step 40: backward → AllReduce ← 只这一次通信                   │
└─────────────────────────────────────────────────────────────────┘
```

**为什么结果一样？** 加法满足交换律：先加再平均 = 每次平均再加

### 梯度裁剪

```python
if grad_clip != 0.0:
    scaler.unscale_(optimizer)  # 先还原梯度（如果用了 GradScaler）
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

**作用**：
- 计算所有参数梯度的 L2 范数
- 如果超过阈值（如 1.0），按比例缩小
- 防止梯度爆炸

### 参数更新

```python
scaler.step(optimizer)              # 更新参数
scaler.update()                     # 调整 scaler 缩放因子
optimizer.zero_grad(set_to_none=True)  # 清零梯度
```

**set_to_none=True**：把梯度设为 None 而不是 0 张量，节省内存。

---

## 混合精度训练

### 什么是混合精度？

- **前向/反向传播**：用 FP16/BF16（快）
- **参数存储和更新**：用 FP32（精确）

```
┌─────────────────────────────────────────────────────────────────┐
│                      混合精度训练流程                            │
├─────────────────────────────────────────────────────────────────┤
│  FP32 参数 ─────────────────────────────────┐                   │
│      ↓ (复制)                               ↑                   │
│  FP16 参数                                  │                   │
│      ↓                                      │                   │
│  前向传播 (FP16) → 计算 loss               │                   │
│      ↓                                      │                   │
│  反向传播 (FP16) → FP16 梯度               │                   │
│      ↓                                      │                   │
│  转换为 FP32 梯度 → 更新 FP32 参数 ─────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### 代码中的实现

```python
# 上下文管理器
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)

with ctx:
    logits, loss = model(X, Y)  # 自动用 FP16 计算
```

---

## 完整训练流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    一次完整的参数更新                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ for micro_step in range(gradient_accumulation_steps):   │    │
│  │                                                          │    │
│  │   X, Y = get_batch()                                     │    │
│  │   loss = model(X, Y) / gradient_accumulation_steps       │    │
│  │   loss.backward()      ← 梯度累积                        │    │
│  │                                                          │    │
│  │   [DDP: 只在最后一步同步梯度]                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 梯度裁剪（防止梯度爆炸）                                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ optimizer.step()  ← 更新参数                             │    │
│  │ optimizer.zero_grad()  ← 清零梯度                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心概念速查表

| 概念 | 作用 | 关键点 |
|------|------|--------|
| 梯度累积 | 小显存模拟大batch | loss 要除以步数 |
| DDP | 多GPU训练 | AllReduce 同步梯度 |
| 混合精度 | 加速训练 | FP16计算，FP32存储 |
| GradScaler | 防止FP16下溢 | 放大loss，缩小梯度 |
| 学习率预热 | 稳定初期训练 | 从小到大线性增长 |
| 余弦衰减 | 平滑降低学习率 | 两头慢中间快 |
| 梯度裁剪 | 防止梯度爆炸 | 限制梯度范数 |

---

*笔记整理于 2026-02-02*
*基于 nanoGPT by Andrej Karpathy*
