# SFT + LoRA 深度解析

> 从预训练到微调，理解指令微调（SFT）和参数高效微调（LoRA）的原理、数学和实现。
> 路线 A 第 ② 步，是 LLM 面试最高频考点。

---

## 目录

1. [预训练 vs 微调：全局视角](#一预训练-vs-微调全局视角)
2. [SFT（监督式微调）](#二sft监督式微调)
3. [全参数微调的问题](#三全参数微调的问题)
4. [LoRA 原理详解](#四lora-原理详解)
5. [LoRA 数学推导](#五lora-数学推导)
6. [LoRA 完整代码实现](#六lora-完整代码实现)
7. [QLoRA（量化 + LoRA）](#七qlora量化--lora)
8. [实战：微调流程](#八实战微调流程)
9. [面试题精选](#九面试题精选)

---

# 一、预训练 vs 微调：全局视角

## 1.1 LLM 训练的三个阶段

```
阶段 1：预训练（Pre-training）
  ├── 你在 nanoGPT 里做的就是这个！
  ├── 数据：海量无标注文本（几 TB）
  ├── 任务：预测下一个 token
  ├── 目标：学习语言知识和世界知识
  ├── 成本：几百万美元，几千张 GPU，几个月
  └── 结果：Base Model（会续写，但不会"对话"）

       │
       ▼

阶段 2：监督式微调（SFT = Supervised Fine-Tuning）  ← 本章重点
  ├── 数据：高质量指令-回答对（几万~几十万条）
  ├── 任务：学习如何遵循指令、格式化回答
  ├── 目标：从"续写机器"变成"对话助手"
  ├── 成本：几千~几万美元，几张 GPU，几天
  └── 结果：Chat Model（能对话，但可能不安全）

       │
       ▼

阶段 3：对齐（RLHF / DPO）                        ← 下一章
  ├── 数据：人类偏好数据（哪个回答更好）
  ├── 任务：学习人类价值观和偏好
  ├── 目标：安全、有用、诚实
  └── 结果：最终部署的模型
```

## 1.2 直觉类比

```
预训练 = 上大学：
  读了几万本书，掌握了广泛知识
  但不知道怎么回答具体问题
  
  Base Model 的行为：
  用户："什么是光合作用？"
  模型："什么是细胞分裂？什么是蒸馏？什么是..."  ← 续写，不是回答！

SFT = 岗前培训：
  学了几千个"问题→回答"的例子
  知道了：收到问题 → 给出有条理的回答
  
  SFT 后的行为：
  用户："什么是光合作用？"
  模型："光合作用是植物利用太阳能将二氧化碳和水..."  ← 正确回答！

RLHF = 老师指导：
  人类反馈"这个回答好/不好"
  学会了更有用、更安全的回答方式
```

---

# 二、SFT（监督式微调）

## 2.1 SFT 的数据格式

### 最简单格式：指令-回答对

```json
{
  "instruction": "解释什么是光合作用",
  "output": "光合作用是绿色植物利用太阳光能，将二氧化碳和水转化为有机物（主要是葡萄糖）并释放氧气的过程。其化学方程式为：6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂。"
}
```

### Alpaca 格式（有输入的版本）

```json
{
  "instruction": "把以下句子翻译成英文",
  "input": "今天天气真好",
  "output": "The weather is really nice today."
}
```

### 多轮对话格式（ShareGPT / ChatML）

```json
{
  "conversations": [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "什么是 Transformer？"},
    {"role": "assistant", "content": "Transformer 是一种基于注意力机制的..."},
    {"role": "user", "content": "它和 RNN 有什么区别？"},
    {"role": "assistant", "content": "主要有三点区别：1. 并行性..."}
  ]
}
```

## 2.2 SFT 的训练过程

### 核心：和预训练几乎一模一样！

```python
# 预训练（你在 nanoGPT 里做的）：
输入: "The cat sat on the"
标签: "cat sat on the mat"
Loss: 每个 token 都计算

# SFT（微调）：
输入: "[INST] 什么是光合作用？ [/INST] 光合作用是..."
标签:                                  "光合作用是..."
Loss: 只在回答部分计算！（instruction 部分不算 loss）
```

### 为什么只在回答部分计算 Loss？

```
指令部分（"什么是光合作用？"）：
  → 这是人写的，不需要模型学习"生成"
  → 计算 loss 没有意义，反而可能干扰训练

回答部分（"光合作用是植物..."）：
  → 这是模型需要学会生成的
  → 在这里计算 loss，引导模型学习如何回答
```

### 代码实现

```python
def sft_loss(model, input_ids, labels, ignore_index=-100):
    """
    SFT 训练的 Loss 计算
    
    input_ids: 完整序列（instruction + response）
    labels: 和 input_ids 一样，但 instruction 部分填 -100（忽略）
    """
    # 例：
    # input_ids = [INST, 什么, 是, 光合, 作用, /INST, 光合, 作用, 是, ...]
    # labels    = [-100, -100, -100, -100, -100, -100, 光合, 作用, 是, ...]
    #              ↑ instruction 部分被忽略           ↑ 只在回答部分计算 loss
    
    logits = model(input_ids).logits
    
    # shift: 用位置 t 的输出预测位置 t+1 的 token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index  # -100 的位置不计算 loss
    )
    return loss
```

## 2.3 Chat Template（对话模板）

不同模型用不同的特殊 token 标记对话结构：

```
LLaMA-2 Chat 格式：
<s>[INST] <<SYS>>
你是一个有用的助手。
<</SYS>>

什么是 Transformer？ [/INST] Transformer 是一种... </s>
<s>[INST] 它和 RNN 的区别？ [/INST] 主要区别... </s>

ChatML 格式（GPT / Qwen）：
<|im_start|>system
你是一个有用的助手。<|im_end|>
<|im_start|>user
什么是 Transformer？<|im_end|>
<|im_start|>assistant
Transformer 是一种...<|im_end|>
```

### 为什么 Chat Template 很重要？

```
如果训练和推理用不同的模板 → 模型困惑 → 效果差

训练时：用 LLaMA 格式 [INST]...[/INST]
推理时：用 ChatML 格式 <|im_start|>...

模型："这是什么？我没见过这种格式..."
→ 效果大幅下降

必须保持训练和推理的模板一致！
```

---

# 三、全参数微调的问题

## 3.1 显存分析

```
微调 LLaMA-7B（全参数）需要多少显存？

① 模型参数（FP16）：7B × 2 bytes = 14 GB
② 梯度（FP16）：      7B × 2 bytes = 14 GB
③ 优化器状态（Adam）：
   - 一阶动量 m（FP32）：7B × 4 bytes = 28 GB
   - 二阶动量 v（FP32）：7B × 4 bytes = 28 GB
④ 激活值：              约 10-20 GB（取决于 batch_size）

总计：14 + 14 + 28 + 28 + 15 ≈ 99 GB

需要至少 2 张 A100-80GB！（或 4 张 A100-40GB）
```

## 3.2 全参数微调的三个问题

```
问题 1：显存太大
  7B 模型需要 ~100GB 显存
  70B 模型需要 ~1TB 显存
  → 绝大多数人用不起

问题 2：灾难性遗忘
  微调所有参数 → 模型可能"忘记"预训练学到的知识
  特别是微调数据量小时

问题 3：每个任务一份完整模型
  微调 3 个不同任务 → 存 3 份 7B 模型（42GB 磁盘）
  → 部署和切换成本高
```

## 3.3 LoRA 的动机

```
核心观察（Aghajanyan et al., 2020）：

预训练模型在微调时，参数变化量 ΔW 是"低秩"的。

什么意思？
  原始参数矩阵 W: 4096 × 4096（1677 万个参数）
  微调后的变化 ΔW = W_new - W_old
  
  对 ΔW 做 SVD 分解，发现：
  前 8 个奇异值占了总能量的 90%+
  → ΔW 可以用一个秩为 8 的矩阵近似！
  → 不需要更新全部 1677 万个参数

直觉：
  预训练已经学好了通用知识（占满了 4096 维）
  微调只需要"微调"一点点方向（只要 8 个维度就够了）
  
  类比：GPS 导航已经知道全国所有道路（预训练）
       → 只需要告诉它"走这条小路更近"（微调 = 低秩修正）
```

---

# 四、LoRA 原理详解

## 4.1 核心公式

$$W' = W + \Delta W = W + BA$$

其中：
- $W \in \mathbb{R}^{d \times d}$：原始预训练权重（**冻结，不更新**）
- $B \in \mathbb{R}^{d \times r}$：LoRA 矩阵 B（可训练）
- $A \in \mathbb{R}^{r \times d}$：LoRA 矩阵 A（可训练）
- $r$：秩（rank），远小于 $d$（如 $r = 8$，$d = 4096$）

## 4.2 图解

```
                        原始路径（冻结）
输入 x ────────────────── W ──────────────────→ Wx
  │                                               │
  │        LoRA 旁路（可训练）                      │
  │     ┌─────────────────────────┐               │
  └────→│  A (d→r)  →  B (r→d)   │──→ BAx        │
        │  降维          升维      │    │          │
        │  4096→8       8→4096    │    │          │
        └─────────────────────────┘    │          │
                                       ▼          ▼
                                    输出 = Wx + BAx
                                         = (W + BA)x
```

## 4.3 为什么这样做有效？

### 参数量对比

```
原始 W:  4096 × 4096 = 16,777,216 个参数

LoRA (r=8):
  A: 4096 × 8 = 32,768
  B: 8 × 4096 = 32,768
  总计: 65,536 个参数

压缩比: 65,536 / 16,777,216 = 0.39%
→ 只训练 0.39% 的参数！
```

### LLaMA-7B 全模型对比

```
全参数微调：
  可训练参数 = 7B = 7,000,000,000
  
LoRA (r=8, 应用于 Q,V 投影)：
  每层：2 × (4096×8 + 8×4096) = 131,072
  32层：32 × 131,072 = 4,194,304
  总计 ≈ 4.2M = 4,200,000
  
  占比：4.2M / 7B = 0.06%
  
显存对比：
  全参数：~100 GB
  LoRA：  ~16 GB（仅需 1 张消费级 GPU！）
```

## 4.4 LoRA 的关键设计

### 初始化策略

```python
# A 用高斯随机初始化
A = nn.Parameter(torch.randn(r, d) * 0.01)

# B 初始化为零！
B = nn.Parameter(torch.zeros(d, r))

# 为什么 B = 0？
# 因为 BA = 0 × A = 0
# → 训练开始时 ΔW = 0
# → 模型行为和预训练完全一样
# → 从预训练的"好状态"开始微调
```

### Scaling Factor α

```python
output = x @ W + (x @ A.T @ B.T) * (alpha / r)
#                                    ↑ 缩放因子

# alpha：控制 LoRA 的"贡献强度"
# 除以 r：让不同 r 值的效果可比较

# 常见设置：
# alpha = r   → 缩放 = 1（标准）
# alpha = 2r  → 缩放 = 2（LoRA 贡献更大）
# alpha = 16, r = 8 → 缩放 = 2
```

### 应用在哪些层？

```
Transformer Block 中有多个线性层：
  ├── Attention:
  │   ├── W_q  ← 常用 ✓
  │   ├── W_k  ← 有时用
  │   ├── W_v  ← 常用 ✓
  │   └── W_o  ← 有时用
  └── FFN:
      ├── W_up     ← 有时用
      ├── W_gate   ← 有时用
      └── W_down   ← 有时用

最常见配置：只对 Q 和 V 加 LoRA
  原因：论文实验发现 Q, V 效果最好

更激进配置：对所有线性层加 LoRA
  效果通常更好，但参数量翻倍
```

---

# 五、LoRA 数学推导

## 5.1 低秩分解的直觉

### 什么是"秩"？

```
矩阵的秩 = 矩阵中"独立方向"的数量

例：
M = [1 2 3]     秩 = 1
    [2 4 6]     第二行 = 2 × 第一行，不独立
    [3 6 9]     第三行 = 3 × 第一行，不独立

可以分解为：
M = [1] × [1, 2, 3] = 外积（rank-1）
    [2]
    [3]
```

### 低秩近似

```
任何矩阵都可以用 SVD 分解：
M = UΣV^T

其中 Σ 是对角矩阵，对角元素（奇异值）从大到小排列

低秩近似 = 只保留前 r 个奇异值：
M ≈ U_r Σ_r V_r^T

                    原始矩阵           低秩近似
                   ┌─────────┐       ┌───┐ ┌─┐ ┌───────┐
                   │         │       │   │ │ │ │       │
    (d × d)    =   │    M    │   ≈   │U_r│×│Σ│×│ V_r^T │
                   │         │       │   │ │ │ │       │
                   └─────────┘       └───┘ └─┘ └───────┘
                   d² 个参数         (d×r) (r) (r×d) 个参数

如果 r << d：参数量从 d² 降到 2dr + r ≈ 2dr
```

## 5.2 LoRA 的数学等价性

```
前向传播：
  h = xW + xBA × (α/r)
  
  = x(W + BA × α/r)
  
  = xW'   其中 W' = W + (α/r)BA

梯度计算（关键：W 冻结，只有 A 和 B 有梯度）：
  ∂L/∂A = (α/r) × B^T × (∂L/∂h)^T × x
  ∂L/∂B = (α/r) × (∂L/∂h)^T × x × A^T

  注意：
  - W 的梯度不计算（冻结）→ 省了最大块的梯度显存
  - A, B 很小（d×r）→ 梯度也很小
  - 优化器状态也只针对 A, B → 大幅节省显存
```

## 5.3 为什么 r 可以很小？

```
论文中的实验（LLaMA-7B 在不同 r 值下的效果）：

r=1:   效果已经不错（70-80% of 全参数微调）
r=4:   效果很好（90%+ of 全参数微调）
r=8:   效果几乎等同全参数微调 ✓（最常用）
r=16:  效果 ≈ 全参数微调
r=64:  效果 = 全参数微调（但参数量已经不少了）
r=256: 过参数化，可能过拟合

直觉：
  预训练模型的参数空间是 4096 维的
  但微调需要调整的"方向"只有几个
  r=8 就意味着"8 个调整方向就够了"

类比：
  你已经会开车了（预训练）
  现在要学习在冰面上开（微调）
  你不需要重新学所有驾驶技能
  只需要调整 "刹车力度" 和 "转向角度" 两个维度（r=2）
```

---

# 六、LoRA 完整代码实现

## 6.1 LoRA 层实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """
    LoRA: Low-Rank Adaptation
    
    给任意线性层添加低秩旁路
    """
    def __init__(
        self,
        original_layer: nn.Linear,  # 原始的线性层
        r: int = 8,                 # 秩
        alpha: float = 16,          # 缩放因子
        dropout: float = 0.05,      # LoRA dropout
    ):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # ===== 冻结原始权重 =====
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
        
        # ===== LoRA 矩阵 =====
        # A: 降维 (in_features → r)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        # B: 升维 (r → out_features)
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        
        # ===== Dropout =====
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # ===== 初始化 =====
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # A: 随机初始化
        nn.init.zeros_(self.lora_B)                             # B: 零初始化
        # → BA = 0 → 初始时 ΔW = 0 → 和原模型行为一致
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始路径（冻结，不计算梯度）
        original_output = self.original_layer(x)
        
        # LoRA 旁路（可训练）
        # x → dropout → A^T（降维）→ B^T（升维）→ × scaling
        lora_output = self.lora_dropout(x)
        lora_output = lora_output @ self.lora_A.T   # (B, T, in) → (B, T, r)
        lora_output = lora_output @ self.lora_B.T    # (B, T, r) → (B, T, out)
        lora_output = lora_output * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """
        推理时：把 LoRA 权重合并到原始权重中
        → 推理速度和原模型完全一样！零额外开销！
        """
        with torch.no_grad():
            self.original_layer.weight += (self.lora_B @ self.lora_A) * self.scaling
        # 合并后可以删除 LoRA 参数，模型大小不变
```

## 6.2 给模型添加 LoRA

```python
def add_lora_to_model(model, r=8, alpha=16, target_modules=None):
    """
    给模型的指定层添加 LoRA
    
    target_modules: 要添加 LoRA 的层名（如 ['q_proj', 'v_proj']）
    """
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']  # 默认只对 Q, V
    
    # ===== 第1步：冻结所有原始参数 =====
    for param in model.parameters():
        param.requires_grad = False
    
    # ===== 第2步：替换目标层为 LoRA 层 =====
    lora_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查是否是目标层
            if any(target in name for target in target_modules):
                # 找到父模块和属性名
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)
                
                # 创建 LoRA 层替换原始线性层
                lora_layer = LoRALayer(module, r=r, alpha=alpha)
                setattr(parent, attr_name, lora_layer)
                lora_layers.append(name)
    
    # ===== 第3步：统计可训练参数 =====
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"添加 LoRA 到: {lora_layers}")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params:.4%})")
    
    return model
```

## 6.3 LoRA 权重的保存和加载

```python
def save_lora_weights(model, path):
    """只保存 LoRA 参数（很小！）"""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:  # 只保存可训练的（即 LoRA 参数）
            lora_state_dict[name] = param.data
    
    torch.save(lora_state_dict, path)
    
    size_mb = sum(p.numel() * p.element_size() for p in lora_state_dict.values()) / 1e6
    print(f"LoRA 权重已保存: {path} ({size_mb:.1f} MB)")
    # 7B 模型的 LoRA 权重通常只有 ~10-30 MB！


def load_lora_weights(model, path):
    """加载 LoRA 参数"""
    lora_state_dict = torch.load(path)
    
    # 只加载 LoRA 参数，原始权重不变
    model_dict = model.state_dict()
    model_dict.update(lora_state_dict)
    model.load_state_dict(model_dict)
    
    print(f"LoRA 权重已加载: {path}")
```

## 6.4 LoRA 推理时的合并（零开销）

```
训练时：
  x → [原始 W] → h₁ ──┐
  x → [A] → [B] → h₂ ──┼──→ h₁ + h₂  （两条路径，有额外计算）
  
推理时（合并后）：
  W' = W + (α/r) × BA
  x → [W'] → h                         （一条路径，和原模型一样快！）

这是 LoRA 的核心优势之一：
  训练时：两条路径（有开销但可接受）
  推理时：合并为一条路径（零额外开销！）
  
  → 部署时模型大小、速度和原模型完全一样
```

---

# 七、QLoRA（量化 + LoRA）

## 7.1 核心思想

```
LoRA 已经很省了，但加载 7B 模型本身还需要 14GB（FP16）
QLoRA 进一步：把原始模型量化到 4-bit，再加 LoRA

显存对比（LLaMA-7B）：
  全参数微调 FP16:  ~100 GB
  LoRA FP16:        ~16 GB
  QLoRA 4-bit:      ~6 GB   ← 单张消费级 GPU（RTX 3090）
```

## 7.2 QLoRA 的三个技术

```
① NF4 量化（NormalFloat4）
  把 FP16 权重量化到 4 bit
  基于正态分布设计量化区间 → 对预训练权重更友好
  7B × 4 bit = 3.5 GB（原来 14 GB）

② 双重量化（Double Quantization）
  量化参数本身也量化 → 进一步省显存
  额外节省 ~0.4 GB

③ 分页优化器（Paged Optimizers）
  用 CPU 内存做 GPU 显存的"交换区"
  当 GPU 显存不够时，自动转移到 CPU
```

## 7.3 QLoRA 的计算流程

```
训练时：
  1. 从显存读取 4-bit 权重
  2. 反量化为 FP16/BF16（临时，不存储）
  3. 做正常的矩阵乘法（FP16 精度）
  4. LoRA 的 A, B 始终保持 FP16/BF16
  5. 梯度只回传到 LoRA 的 A, B

  原始 4-bit 权重 ──→ 反量化 ──→ xW（FP16）──→ ┐
                                                ├──→ 输出
  LoRA A, B (FP16) ──→ xBA（FP16）──→ ──────────┘
                                 ↑
                           只更新这部分
```

---

# 八、实战：微调流程

## 8.1 使用 HuggingFace PEFT 库（工业标准）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ===== 第1步：加载基础模型 =====
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",         # 自动分配 GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ===== 第2步：配置 LoRA =====
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 因果语言模型
    r=8,                           # 秩
    lora_alpha=16,                 # 缩放因子
    lora_dropout=0.05,             # LoRA dropout
    target_modules=[               # 对哪些层加 LoRA
        "q_proj", "v_proj",        # 基础配置
        # "k_proj", "o_proj",      # 可选
        # "gate_proj", "up_proj",  # FFN 层（可选）
    ],
)

# ===== 第3步：应用 LoRA =====
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出：trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622%

# ===== 第4步：训练（和正常训练一样！）=====
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # 有效 batch_size = 4 × 4 = 16
    learning_rate=2e-4,              # LoRA 通常用较大学习率
    warmup_steps=100,
    logging_steps=10,
    save_steps=200,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,    # 准备好的 SFT 数据集
    data_collator=data_collator,
)

trainer.train()

# ===== 第5步：保存 LoRA 权重 =====
model.save_pretrained("./lora-output")
# 只保存 LoRA 参数，约 10-30 MB
```

## 8.2 使用 QLoRA 微调（消费级 GPU）

```python
from transformers import BitsAndBytesConfig

# ===== 4-bit 量化配置 =====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                # 加载 4-bit 模型
    bnb_4bit_quant_type="nf4",        # NormalFloat4 量化
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算用 BF16
    bnb_4bit_use_double_quant=True,   # 双重量化
)

# ===== 加载量化模型 =====
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# 后续步骤和标准 LoRA 一样！
model = get_peft_model(model, lora_config)
# ... 训练 ...
```

## 8.3 微调超参数经验值

| 参数 | SFT 推荐值 | 说明 |
|------|-----------|------|
| r | 8-64 | 小数据集用 8，大数据集用 32-64 |
| alpha | 16-32 | 通常 alpha = 2r |
| lr | 1e-4 ~ 3e-4 | 比预训练大（因为只训练小部分参数） |
| epochs | 1-3 | SFT 数据量小，别训太多轮避免过拟合 |
| batch_size | 16-128 | 越大越稳定 |
| warmup | 5-10% 总步数 | |
| dropout | 0.05 | LoRA 的 dropout |
| target_modules | q,v 或 all | all 效果更好但参数量翻倍 |

---

# 九、面试题精选

## Q1："什么是 SFT？和预训练有什么区别？"

> SFT 是在预训练好的 base model 上，用指令-回答对进行监督微调。和预训练的区别：①数据不同——预训练用海量无标注文本，SFT 用高质量指令数据②loss 计算不同——预训练对所有 token 算 loss，SFT 只对回答部分算③目标不同——预训练学语言知识，SFT 学遵循指令的能力④成本不同——预训练需要几千张 GPU 几个月，SFT 用几张 GPU 几天。

## Q2："LoRA 的原理是什么？"

> LoRA 基于一个关键观察：微调时参数的变化量 ΔW 是低秩的。所以用两个小矩阵 B（d×r）和 A（r×d）的乘积来近似 ΔW，其中 r 远小于 d（如 r=8, d=4096）。训练时冻结原始权重 W，只训练 A 和 B。这样可训练参数从 d² 降到 2dr，减少了几百倍。B 初始化为零保证训练开始时模型行为不变。推理时可以把 BA 合并到 W 中，没有额外计算开销。

## Q3："LoRA 的 r 怎么选？"

> r 控制 LoRA 的"容量"。r=8 在大多数场景下效果就很好（达到全参数微调 95%+ 的效果）。任务越复杂、训练数据越多，可以适当增大 r（如 16-64）。实践中 r=8 到 r=32 是最常用的范围。过大的 r（如 256）反而可能过拟合。另外 r 通常和 alpha 配合，常见做法是 alpha = 2r。

## Q4："LoRA 为什么 B 初始化为零？"

> 为了保证训练开始时 ΔW = BA = 0，即模型行为和预训练完全一致。如果 B 随机初始化，ΔW ≠ 0，等于在预训练权重上加了随机噪声，会破坏预训练学到的知识，导致训练不稳定。零初始化保证了从一个好的起点开始微调。

## Q5："LoRA 一般加在哪些层？为什么？"

> 最常见是加在 Attention 的 Q 和 V 投影上。原论文实验发现 Q、V 的效果最好，K 的效果相对差一些（可能因为 K 的作用更多是"被查询的索引"，变化需求小）。更激进的做法是对所有线性层（Q、K、V、O、gate、up、down）都加 LoRA，通常效果更好但参数量更多。实践中可以先试 Q+V，不够再扩展到所有层。

## Q6："QLoRA 和 LoRA 有什么区别？"

> QLoRA 在 LoRA 基础上增加了三个技术：①NF4 量化——把原始模型量化到 4-bit，大幅减少模型加载的显存②双重量化——量化参数本身也量化③分页优化器——用 CPU 内存扩展 GPU 显存。核心思想：4-bit 存储原始权重，计算时反量化到 FP16，LoRA 参数始终保持 FP16。效果：7B 模型只需 ~6GB 显存即可微调，单张消费级 GPU 就能用。效果和全精度 LoRA 接近。

## Q7："LoRA 和全参数微调效果一样吗？"

> 在大多数任务上，LoRA（r=8-16）能达到全参数微调 95%+ 的效果。某些非常复杂或与预训练分布差异很大的任务上，全参数微调可能略好。但 LoRA 有三个独特优势：①显存大幅减少②可以快速切换不同的 LoRA 权重（同一个 base model + 不同 LoRA = 不同任务）③推理时合并后零额外开销。性价比极高。

## Q8："SFT 有什么常见问题？"

> ①灾难性遗忘：微调后模型可能忘记预训练知识，特别是微调数据和预训练分布差异大时②对数据质量敏感：SFT 数据中的错误模式会被模型学会③过拟合：SFT 数据通常只有几万条，容易过拟合④格式依赖：模型会严重依赖训练时的 Chat Template，推理时格式不一致会导致效果大幅下降。

---

## 知识串联：从 nanoGPT 到 LoRA

```
你在 nanoGPT 中学到的                    LoRA 中的对应
─────────────────────────────────────────────────────────
train.py 中的训练循环                →    SFT 的训练循环（几乎一样！）
cross_entropy loss                  →    SFT 的 loss（只在回答部分算）
model.parameters()                  →    只对 LoRA 参数 requires_grad=True
nn.Linear(d, d)                     →    W 冻结 + BA 旁路
optimizer = AdamW(params)           →    optimizer 只优化 LoRA 参数
混合精度训练                         →    QLoRA 的 4-bit + FP16 计算
model.eval() + generate()           →    merge_weights() 后一样生成
```

---

*SFT + LoRA 深度解析 - 路线 A 第②步 - 2026-02-07*
*上一步：Tokenization | 下一步：RLHF / DPO*
