# RLHF + DPO 深度解析

> 从 SFT 到对齐：理解人类反馈强化学习（RLHF）和直接偏好优化（DPO）的原理、数学和实现。
> 路线 A 第 ③ 步，LLM 对齐（Alignment）是面试和工业界的核心话题。

---

## 目录

1. [为什么需要对齐？](#一为什么需要对齐)
2. [RLHF 全流程](#二rlhf-全流程)
3. [奖励模型（Reward Model）](#三奖励模型reward-model)
4. [PPO 强化学习微调](#四ppo-强化学习微调)
5. [DPO：去掉奖励模型](#五dpo去掉奖励模型)
6. [DPO 数学推导](#六dpo-数学推导)
7. [DPO 完整代码实现](#七dpo-完整代码实现)
8. [RLHF vs DPO 对比](#八rlhf-vs-dpo-对比)
9. [其他对齐方法速览](#九其他对齐方法速览)
10. [面试题精选](#十面试题精选)

---

# 一、为什么需要对齐？

## 1.1 SFT 之后模型的问题

```
SFT 后的模型已经能"对话"了，但还有严重问题：

问题 1：有害内容
  用户："教我怎么制作炸弹"
  SFT 模型："好的，首先你需要准备以下材料..."  ← 危险！

问题 2：编造事实（幻觉）
  用户："爱因斯坦什么时候获得诺贝尔文学奖？"
  SFT 模型："爱因斯坦于 1925 年获得诺贝尔文学奖..."  ← 胡说！

问题 3：阿谀奉承（Sycophancy）
  用户："我觉得地球是平的，对吧？"
  SFT 模型："是的，您说得对，地球确实是平的..."  ← 迎合用户的错误观点

问题 4：缺乏细腻的判断力
  SFT 只教了"怎么回答"，没教"什么样的回答更好"
  面对复杂问题，模型不知道如何权衡
```

## 1.2 对齐的目标："3H"原则

```
OpenAI 提出的对齐三原则：

Helpful（有用）：
  能真正帮到用户，回答准确、详尽
  
Harmless（无害）：
  拒绝危险请求，不产生有害内容
  
Honest（诚实）：
  不编造事实，不确定时说"我不知道"

对齐 = 让模型的行为符合人类的价值观和偏好
```

## 1.3 全局视角：三个阶段的目标

```
阶段 1：预训练
  目标：学习语言知识（"知道什么"）
  数据：海量文本
  ↓
阶段 2：SFT
  目标：学习对话格式（"怎么说"）
  数据：指令-回答对
  ↓
阶段 3：对齐（RLHF / DPO）  ← 本章重点
  目标：学习人类偏好（"什么样的回答更好"）
  数据：人类偏好数据（对比数据）

直觉类比：
  预训练 = 上大学，学知识
  SFT    = 岗前培训，学流程
  对齐   = 老师指导，学什么是"好"的回答
```

---

# 二、RLHF 全流程

## 2.1 RLHF 的三个步骤

```
RLHF = Reinforcement Learning from Human Feedback
     = 从人类反馈中进行强化学习

┌─────────────────────────────────────────────────────────┐
│                    RLHF 三步走                           │
│                                                         │
│  Step 1: SFT（已完成，上一章）                            │
│    预训练模型 → SFT → 得到 SFT Model                     │
│                                                         │
│  Step 2: 训练奖励模型（Reward Model）                     │
│    收集人类偏好数据 → 训练 RM → 得到"评分器"              │
│                                                         │
│  Step 3: PPO 强化学习微调                                │
│    用 RM 的评分作为奖励 → PPO 优化 SFT Model              │
│    → 得到最终对齐模型                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 2.2 人类偏好数据

```
核心数据格式：对同一个 prompt，给出两个回答，人类标注哪个更好

{
  "prompt": "解释量子纠缠",
  "chosen":   "量子纠缠是量子力学中的一种现象，两个粒子...",  ← 人类觉得这个好 (y_w)
  "rejected": "量子纠缠就是两个东西连在一起。"              ← 人类觉得这个差 (y_l)
}

标注过程：
  1. 给标注员看一个 prompt
  2. 模型生成 2-4 个不同回答
  3. 标注员排序：回答 A > 回答 B > 回答 C
  4. 转化为两两对比：(A > B), (A > C), (B > C)

数据量：
  InstructGPT：~33,000 个对比对
  LLaMA-2：   ~1,000,000+ 个对比对
  
成本：
  标注一个对比对 ~$1-5（需要专业标注员）
  整个过程可能花费 $50K - $500K
```

---

# 三、奖励模型（Reward Model）

## 3.1 奖励模型是什么？

```
奖励模型 = 一个"评分员"
  输入：(prompt, response) 对
  输出：一个标量分数（这个回答有多好）

本质上：把人类的偏好"蒸馏"成一个可自动评分的模型

类比：
  人类标注员  = 米其林评审员（贵、慢、但准确）
  奖励模型    = 训练出来的美食 AI 评分系统（便宜、快、近似准确）
```

## 3.2 奖励模型的架构

```
通常：把 SFT 模型去掉 lm_head，换成一个输出标量的线性层

SFT 模型架构：
  Input → Transformer Blocks → Hidden States → lm_head → Vocab Logits
                                                ↑ 去掉这个

奖励模型架构：
  Input → Transformer Blocks → Hidden States → reward_head → 标量 Score
                                                ↑ 换成这个（Linear(d, 1)）

实现：
  hidden_states = transformer(input_ids)       # (B, T, d)
  last_token_hidden = hidden_states[:, -1, :]  # (B, d) 取最后一个 token
  reward = reward_head(last_token_hidden)       # (B, 1) 一个分数
```

## 3.3 奖励模型的训练

### Bradley-Terry 模型

```
核心假设：人类偏好可以用 Bradley-Terry 模型描述

给定 prompt x，chosen 回答 y_w，rejected 回答 y_l：

P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

其中：
  r(x, y) = 奖励模型对 (x, y) 的评分
  σ = sigmoid 函数
  
直觉：
  如果 r(x, y_w) 比 r(x, y_l) 高很多
  → σ(大正数) → 接近 1 → 模型认为 y_w 确实更好
  → 符合人类标注 ✓
```

### 训练 Loss

```
Loss = -E[log σ(r(x, y_w) - r(x, y_l))]

解读：
  让 chosen 回答的分数尽可能高于 rejected 回答的分数
  
  如果模型正确判断（y_w 分高于 y_l）：
    r(x, y_w) - r(x, y_l) > 0 → σ > 0.5 → log > -0.69 → loss 小 ✓
    
  如果模型判断错误（y_l 分反而更高）：
    r(x, y_w) - r(x, y_l) < 0 → σ < 0.5 → log < -0.69 → loss 大 ✗
    → 梯度推动模型修正
```

### 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """奖励模型：给 (prompt, response) 打分"""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model  # SFT 模型的 Transformer 部分
        hidden_size = base_model.config.hidden_size
        
        # 把 lm_head 替换为 reward_head
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        # 通过 Transformer 得到 hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # 最后一层
        
        # 取最后一个 token 的 hidden state
        if attention_mask is not None:
            # 找到每个序列的最后一个非 padding token
            last_idx = attention_mask.sum(dim=1) - 1  # (B,)
            last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_idx]
        else:
            last_hidden = hidden_states[:, -1, :]
        
        # 输出标量奖励分数
        reward = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return reward


def reward_model_loss(reward_model, chosen_ids, rejected_ids, 
                      chosen_mask=None, rejected_mask=None):
    """
    奖励模型的训练 Loss
    
    让 chosen 的分数高于 rejected 的分数
    """
    # 计算两个回答的奖励分数
    r_chosen = reward_model(chosen_ids, chosen_mask)     # (B,)
    r_rejected = reward_model(rejected_ids, rejected_mask)  # (B,)
    
    # Bradley-Terry Loss
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    
    # 准确率（chosen 分数 > rejected 分数的比例）
    accuracy = (r_chosen > r_rejected).float().mean()
    
    return loss, accuracy
```

## 3.4 奖励模型的问题

```
问题 1：Reward Hacking（奖励欺骗）
  模型学会"钻空子"：产生高分但实际质量差的回答
  例：奖励模型喜欢长回答 → 模型疯狂灌水凑长度
  
问题 2：分布外泛化差
  RM 只在标注数据的分布上准确
  当 policy 模型生成 RM 没见过的回答时 → 评分不可靠

问题 3：标注一致性
  不同标注员可能有不同偏好
  同一标注员在不同时间可能给出不同判断
  标注员间一致率通常只有 70-80%
```

---

# 四、PPO 强化学习微调

## 4.1 RL 的基本框架

```
在 RLHF 中，把语言模型生成看作一个 RL 问题：

Policy（策略）：    语言模型 π_θ（要优化的）
State（状态）：     prompt + 已生成的 token
Action（动作）：    生成下一个 token
Reward（奖励）：    奖励模型的评分（只在最后一个 token 给）
Environment（环境）：用户的 prompt

目标：最大化期望奖励
  max_θ E_{x~D, y~π_θ(·|x)} [r(x, y)]

但直接最大化奖励会导致 Reward Hacking！
→ 需要加约束
```

## 4.2 KL 散度约束

```
关键问题：
  如果只最大化奖励，模型会"跑偏"
  → 生成的文本越来越奇怪（但能骗过奖励模型）
  
解决：加 KL 散度惩罚，让模型不要偏离 SFT 模型太远

完整目标函数：
  max_θ E[r(x, y)] - β · KL(π_θ || π_ref)
  
  第一项：最大化奖励（回答要好）
  第二项：不要偏离参考模型太远（别跑偏）
  β：平衡系数（通常 0.01 - 0.2）
  π_ref：参考模型（通常就是 SFT 后的模型）

直觉：
  就像放风筝 🪁
  奖励信号 = 风（推动模型往"好"的方向走）
  KL 惩罚 = 风筝线（不让模型飞太远失控）
  β = 线的长度（越小越自由，越大越保守）

展开写出 KL 散度：
  KL(π_θ || π_ref) = E_{y~π_θ} [log π_θ(y|x) - log π_ref(y|x)]
  
  直觉：如果 π_θ 和 π_ref 对某个 y 的概率差别很大 → KL 惩罚大
  → 强迫模型保持在 SFT 模型附近
```

## 4.3 PPO 算法概述

```
PPO = Proximal Policy Optimization（近端策略优化）

为什么用 PPO？
  ① 稳定性好：限制每次更新的步幅，不会剧烈震荡
  ② 在 RL 领域经过大量验证
  ③ OpenAI 本身就是 PPO 的提出者

PPO-RLHF 的训练循环（每个 batch）：

┌─────────────────────────────────────────────────────────┐
│  1. 采样 prompt x ~ Dataset                             │
│                                                         │
│  2. 用当前 policy π_θ 生成回答 y ~ π_θ(·|x)             │
│                                                         │
│  3. 用 RM 评分 r = RM(x, y)                             │
│                                                         │
│  4. 计算 KL 惩罚 kl = log π_θ(y|x) - log π_ref(y|x)    │
│                                                         │
│  5. 计算最终奖励 reward = r - β × kl                     │
│                                                         │
│  6. 用 PPO 算法更新 π_θ                                  │
│     (包括 GAE 优势估计 + clip 目标函数)                   │
└─────────────────────────────────────────────────────────┘

注意：需要同时加载 4 个模型！
  ① π_θ（当前 policy，要更新的）
  ② π_ref（参考 policy，冻结的 SFT 模型）
  ③ RM（奖励模型，冻结的）
  ④ Critic / Value Model（价值网络，PPO 需要的）
  
  → 显存爆炸！7B 模型 × 4 = 28B 参数需要加载
```

## 4.4 PPO 的核心公式

### 优势估计（GAE）

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)  （TD 误差）

V(s) = Critic 网络估计的状态价值
γ = 折扣因子（通常 1.0，因为只在最后给奖励）
λ = GAE 参数（通常 0.95）

在 RLHF 中：
  奖励只在序列最后一个 token 给出
  中间 token 的奖励 = -β × (log π_θ(token) - log π_ref(token))
  → 每个 token 都有 KL 惩罚作为"即时奖励"
```

### PPO Clip 目标函数

```
L_PPO = E[min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)]

其中：
  ratio = π_θ_new(a|s) / π_θ_old(a|s)  （新旧策略的概率比）
  A = 优势值（这个 action 比平均好多少）
  ε = clip 范围（通常 0.2）

直觉：
  如果 A > 0（好动作）→ 鼓励增大概率，但不超过 (1+ε) 倍
  如果 A < 0（差动作）→ 鼓励减小概率，但不低于 (1-ε) 倍
  → 保证每步更新不会太激进
```

## 4.5 PPO-RLHF 简化代码

```python
def ppo_rlhf_step(
    policy_model,       # π_θ：要训练的模型
    ref_model,          # π_ref：SFT 参考模型（冻结）
    reward_model,       # RM：奖励模型（冻结）
    value_model,        # Critic：价值网络
    prompts,            # 一批 prompt
    beta=0.1,           # KL 惩罚系数
    clip_eps=0.2,       # PPO clip 参数
):
    # ===== Step 1: 生成回答 =====
    with torch.no_grad():
        responses = policy_model.generate(prompts, max_new_tokens=256)
        # responses: 完整的 (prompt + generated response)
    
    # ===== Step 2: 计算各种 log prob =====
    # 当前 policy 的 log prob
    logprobs = get_logprobs(policy_model, prompts, responses)
    
    # 参考模型的 log prob（用于 KL 惩罚）
    with torch.no_grad():
        ref_logprobs = get_logprobs(ref_model, prompts, responses)
    
    # ===== Step 3: 计算奖励 =====
    with torch.no_grad():
        # RM 评分（只在最后一个 token）
        scores = reward_model(responses)  # (B,)
        
        # 每个 token 的 KL 惩罚
        kl_per_token = logprobs - ref_logprobs  # (B, T)
        
        # 最终奖励 = RM 分数 - KL 惩罚
        rewards = -beta * kl_per_token  # (B, T) 每个 token
        rewards[:, -1] += scores        # 最后一个 token 加上 RM 分数
    
    # ===== Step 4: 计算优势值（GAE）=====
    with torch.no_grad():
        values = value_model(responses)  # (B, T)
        advantages = compute_gae(rewards, values, gamma=1.0, lam=0.95)
    
    # ===== Step 5: PPO 更新 =====
    # 旧的 log prob（用于计算 ratio）
    old_logprobs = logprobs.detach()
    
    # 重新计算当前 log prob（因为模型可能更新了）
    new_logprobs = get_logprobs(policy_model, prompts, responses)
    
    # 概率比
    ratio = torch.exp(new_logprobs - old_logprobs)  # (B, T)
    
    # PPO Clip Loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value Loss（Critic 更新）
    new_values = value_model(responses)
    value_loss = F.mse_loss(new_values, rewards + values)  # 简化版
    
    # 总 Loss
    total_loss = policy_loss + 0.5 * value_loss
    
    return total_loss


def get_logprobs(model, prompts, responses):
    """计算模型在每个 token 位置的 log probability"""
    logits = model(responses).logits  # (B, T, V)
    logprobs = F.log_softmax(logits, dim=-1)  # (B, T, V)
    
    # 取出实际生成的 token 的 log prob
    token_logprobs = logprobs[:, :-1, :].gather(
        dim=-1, 
        index=responses[:, 1:].unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)
    
    return token_logprobs
```

## 4.6 RLHF 的问题

```
问题 1：训练极不稳定
  4 个模型交互 → 训练 dynamics 复杂
  奖励模型不准 → 错误信号 → 模型崩溃
  超参数敏感（β, lr, clip_eps 都需要仔细调）

问题 2：计算成本极高
  需要同时加载 4 个模型
  7B 模型：至少需要 4-8 张 A100
  70B 模型：需要整个 GPU 集群

问题 3：Reward Hacking
  模型学会欺骗 RM 而不是真正变好
  例：用复杂华丽的措辞包装空洞内容

问题 4：实现复杂
  需要实现 PPO（本身就很复杂）
  需要管理多个模型的训练/推理模式切换
  需要处理生成和训练的 pipeline

→ 有没有更简单的方法？
→ DPO！
```

---

# 五、DPO：去掉奖励模型

## 5.1 DPO 的核心思想

```
DPO = Direct Preference Optimization（直接偏好优化）

RLHF 的路径：
  偏好数据 → 训练 RM → 用 RM + PPO 优化 policy
  （两步，复杂）

DPO 的路径：
  偏好数据 → 直接优化 policy
  （一步，简单！）

核心洞察（Rafailov et al., 2023）：
  奖励模型 r(x,y) 可以用 policy 本身来表示！
  不需要单独训练一个 RM！
  
  r(x,y) = β × log[π_θ(y|x) / π_ref(y|x)] + f(x)
  
  这意味着：
  policy 自己就隐含了一个"奖励模型"
  policy 和 RM 之间存在封闭解（closed-form solution）
```

## 5.2 DPO 的 Loss 函数

```
L_DPO = -E[log σ(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

简化记号：
  L = -E[log σ(β × (Δ_w - Δ_l))]

其中：
  Δ_w = log π_θ(y_w|x) - log π_ref(y_w|x)   （chosen 的 log ratio）
  Δ_l = log π_θ(y_l|x) - log π_ref(y_l|x)   （rejected 的 log ratio）

直觉解读：
  Δ_w = policy 相对于 ref 有多"偏向" chosen 回答
  Δ_l = policy 相对于 ref 有多"偏向" rejected 回答
  
  DPO 要求：Δ_w > Δ_l
  即：policy 应该比 ref 更偏向 chosen（而不是 rejected）

  如果 Δ_w - Δ_l >> 0 → σ 接近 1 → loss 接近 0（好）
  如果 Δ_w - Δ_l << 0 → σ 接近 0 → loss 很大（坏）
```

## 5.3 DPO 的直觉

```
DPO 做的事情可以这样理解：

对于每一对 (chosen, rejected) 数据：
  ① 增大 policy 生成 chosen 回答的概率   ↑
  ② 减小 policy 生成 rejected 回答的概率 ↓
  ③ 但不要偏离 ref 模型太远               ← β 控制

对比 SFT：
  SFT 只有 chosen 数据，只做 ①
  DPO 同时做 ① 和 ②，而且有 ref 约束 ③
  
  → 这就是为什么 DPO 效果更好的原因之一
  → 不仅知道什么是好的，还知道什么是坏的

类比：
  SFT  = 老师说"这道题的正确答案是 A"
  DPO  = 老师说"A 比 B 好，因为 B 有这些问题..."
  → 后者学到的更深刻
```

## 5.4 对比 RLHF 和 DPO 的数据流

```
RLHF 数据流：
  偏好数据 ──→ 训练 RM ──→ RM 评分 ──→ PPO 更新 policy
               (额外模型)   (在线生成)   (复杂 RL)

DPO 数据流：
  偏好数据 ──→ 直接计算 Loss ──→ 更新 policy
               (离线，一步到位)    (简单梯度下降)

DPO 的简洁性：
  ① 不需要训练 RM（省掉一个阶段）
  ② 不需要在线生成回答（离线训练）
  ③ 不需要 PPO（普通梯度下降就行）
  ④ 不需要 Critic / Value 网络
  ⑤ 只需要 2 个模型：π_θ 和 π_ref
```

---

# 六、DPO 数学推导

## 6.1 从 RLHF 目标推导 DPO

### 第一步：写出 RLHF 的目标函数

```
RLHF 要优化的目标：
  max_π E_{x~D, y~π(·|x)} [r(x,y)] - β · KL(π || π_ref)

展开 KL 散度：
  max_π E_x [E_{y~π} [r(x,y) - β · log(π(y|x)/π_ref(y|x))]]
  
  = max_π E_x [E_{y~π} [r(x,y) - β·log π(y|x) + β·log π_ref(y|x)]]
```

### 第二步：求最优解（封闭解）

```
对于给定的 x，最优 policy 是：

π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)

其中 Z(x) = Σ_y π_ref(y|x) · exp(r(x,y)/β) 是归一化常数

证明思路（可跳过）：
  这是一个 KL 正则化的优化问题
  用拉格朗日乘数法，令导数为 0
  解出 π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)

直觉：
  最优 policy = 参考模型 × exp(奖励/温度)
  奖励高的回答概率指数级增大
  β 越小，奖励的影响越大（更"贪心"）
  β 越大，越接近参考模型（更保守）
```

### 第三步：反解出奖励函数

```
从 π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)

两边取 log：
  log π*(y|x) = -log Z(x) + log π_ref(y|x) + r(x,y)/β

解出 r：
  r(x,y) = β · log[π*(y|x) / π_ref(y|x)] + β · log Z(x)
  
  = β · log[π*(y|x) / π_ref(y|x)] + f(x)
  
  其中 f(x) = β·log Z(x) 只和 x 有关，和 y 无关

关键发现：
  奖励 r(x,y) 可以完全用 policy 和 ref 的 log ratio 表示！
  → 不需要单独的奖励模型！
```

### 第四步：代入 Bradley-Terry 模型

```
回忆 RM 的训练目标：
  P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

把第三步的 r 代入：
  r(x, y_w) - r(x, y_l) 
  = β·log[π*(y_w|x)/π_ref(y_w|x)] + f(x) - β·log[π*(y_l|x)/π_ref(y_l|x)] - f(x)
  = β·(log[π*(y_w|x)/π_ref(y_w|x)] - log[π*(y_l|x)/π_ref(y_l|x)])
  
  注意 f(x) 消掉了！（因为两个回答对应同一个 prompt）

所以 DPO Loss = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

### 推导总结

```
RLHF 目标 → 最优 policy 有封闭解 → 反解出 r → 代入 BT 模型 → DPO Loss

本质：
  DPO 把 "训练 RM + RL" 的两步过程
  压缩成了 "直接在 policy 上优化偏好" 的一步过程
  
  数学上等价（在 BT 模型假设下）
  实践上更简单、更稳定
```

## 6.2 DPO 梯度分析

```
∇L_DPO = -β · E[(1 - σ(β·Δ)) · (∇log π_θ(y_w|x) - ∇log π_θ(y_l|x))]

其中 Δ = log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)

分析这个梯度：

权重项 (1 - σ(β·Δ))：
  如果模型已经正确偏好 chosen（Δ >> 0）→ σ 接近 1 → 权重接近 0
  → 已经学好的样本贡献小（不过拟合）
  
  如果模型还没学好（Δ ≈ 0 或 < 0）→ σ 接近 0.5 或更小 → 权重大
  → 困难样本贡献大（聚焦学习）

方向项 (∇log π_θ(y_w|x) - ∇log π_θ(y_l|x))：
  增大 chosen 的概率 + 减小 rejected 的概率
  
这是一个自然的"课程学习"：
  → 模型自动聚焦于还没学好的偏好对
  → 已经学好的自动降权
```

---

# 七、DPO 完整代码实现

## 7.1 DPO Loss 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def dpo_loss(
    policy_model,         # π_θ：要训练的模型
    ref_model,            # π_ref：参考模型（冻结）
    chosen_input_ids,     # y_w：chosen 回答的 token ids
    rejected_input_ids,   # y_l：rejected 回答的 token ids
    chosen_labels,        # chosen 的 labels（instruction 部分为 -100）
    rejected_labels,      # rejected 的 labels
    beta=0.1,             # 温度参数
):
    """
    DPO Loss 计算
    
    核心：让 policy 比 ref 更偏向 chosen，更远离 rejected
    """
    
    # ===== Step 1: 计算 policy 的 log prob =====
    chosen_logprobs = get_sequence_logprobs(policy_model, chosen_input_ids, chosen_labels)
    rejected_logprobs = get_sequence_logprobs(policy_model, rejected_input_ids, rejected_labels)
    
    # ===== Step 2: 计算 ref 的 log prob（不需要梯度）=====
    with torch.no_grad():
        ref_chosen_logprobs = get_sequence_logprobs(ref_model, chosen_input_ids, chosen_labels)
        ref_rejected_logprobs = get_sequence_logprobs(ref_model, rejected_input_ids, rejected_labels)
    
    # ===== Step 3: 计算 log ratios =====
    # Δ_w = log π_θ(y_w|x) - log π_ref(y_w|x)
    chosen_log_ratio = chosen_logprobs - ref_chosen_logprobs
    # Δ_l = log π_θ(y_l|x) - log π_ref(y_l|x)
    rejected_log_ratio = rejected_logprobs - ref_rejected_logprobs
    
    # ===== Step 4: DPO Loss =====
    # L = -log σ(β × (Δ_w - Δ_l))
    logits = beta * (chosen_log_ratio - rejected_log_ratio)  # (B,)
    loss = -F.logsigmoid(logits).mean()
    
    # ===== 额外指标 =====
    with torch.no_grad():
        # chosen 被偏好的概率
        chosen_reward = beta * chosen_log_ratio
        rejected_reward = beta * rejected_log_ratio
        reward_accuracy = (chosen_reward > rejected_reward).float().mean()
        reward_margin = (chosen_reward - rejected_reward).mean()
    
    return loss, {
        'reward_accuracy': reward_accuracy.item(),
        'reward_margin': reward_margin.item(),
        'chosen_reward': chosen_reward.mean().item(),
        'rejected_reward': rejected_reward.mean().item(),
    }


def get_sequence_logprobs(model, input_ids, labels):
    """
    计算序列的 log probability（只在 labels != -100 的位置计算）
    
    返回：每个样本的总 log prob (B,)
    """
    logits = model(input_ids).logits  # (B, T, V)
    
    # Shift: 用位置 t 的 logits 预测位置 t+1 的 token
    shift_logits = logits[:, :-1, :]     # (B, T-1, V)
    shift_labels = labels[:, 1:]          # (B, T-1)
    
    # 计算每个位置的 log prob
    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)
    
    # 取出实际 token 的 log prob
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)
    
    # 只在 labels != -100 的位置累加（忽略 instruction 部分）
    mask = (shift_labels != -100).float()  # (B, T-1)
    sequence_log_probs = (token_log_probs * mask).sum(dim=-1)  # (B,)
    
    return sequence_log_probs
```

## 7.2 完整训练循环

```python
def train_dpo(
    policy_model,
    ref_model,
    train_dataloader,
    optimizer,
    beta=0.1,
    num_epochs=1,
    max_grad_norm=1.0,
):
    """DPO 训练循环"""
    
    # 冻结参考模型
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    policy_model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        
        for step, batch in enumerate(train_dataloader):
            # batch 包含：chosen_ids, rejected_ids, chosen_labels, rejected_labels
            
            loss, metrics = dpo_loss(
                policy_model=policy_model,
                ref_model=ref_model,
                chosen_input_ids=batch['chosen_input_ids'],
                rejected_input_ids=batch['rejected_input_ids'],
                chosen_labels=batch['chosen_labels'],
                rejected_labels=batch['rejected_labels'],
                beta=beta,
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            total_accuracy += metrics['reward_accuracy']
            
            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Reward Accuracy: {metrics['reward_accuracy']:.4f}")
                print(f"  Reward Margin: {metrics['reward_margin']:.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        avg_acc = total_accuracy / len(train_dataloader)
        print(f"\nEpoch {epoch}: Avg Loss = {avg_loss:.4f}, Avg Accuracy = {avg_acc:.4f}")
```

## 7.3 DPO + LoRA（最实用的组合）

```python
from peft import LoraConfig, get_peft_model

# ===== 加载模型 =====
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# ===== 创建参考模型（冻结的 SFT 模型）=====
ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
ref_model.eval()

# ===== 给 policy 模型加 LoRA =====
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
policy_model = get_peft_model(base_model, lora_config)

# ===== 训练 =====
# 只需要训练 LoRA 参数（~4.2M），而不是全部 7B！
# ref_model 和 policy_model 可以共享 base 权重（PEFT 自动处理）
# → 显存大幅减少
```

---

# 八、RLHF vs DPO 对比

## 8.1 全面对比

```
                    RLHF (PPO)              DPO
─────────────────────────────────────────────────────────
训练步骤          3 步（SFT→RM→PPO）        2 步（SFT→DPO）
需要的模型数      4 个                      2 个（policy + ref）
是否需要 RM       是                        否
是否需要在线生成   是（每步都要 generate）    否（离线训练）
训练算法          PPO（RL）                 梯度下降（SL 风格）
实现复杂度        非常高                    低（和 SFT 差不多）
训练稳定性        不稳定                    稳定
计算成本          高（4 个模型 + 生成）      低（2 个模型，无生成）
Reward Hacking    容易出现                  不太会出现
理论等价性        ─                        在 BT 模型下与 RLHF 等价
超参数            多且敏感                  主要就是 β

实际效果：
  在大多数任务上，DPO ≈ RLHF 甚至更好
  但在某些需要"探索"的任务上，RLHF 可能略优
  （因为 PPO 有在线生成 → 探索能力更强）
```

## 8.2 工业界趋势

```
2023 年：
  OpenAI: RLHF（PPO）— ChatGPT / GPT-4
  Anthropic: RLHF + Constitutional AI

2024-2025 年：
  Meta (LLaMA-3): DPO 为主
  Google (Gemma): DPO 为主
  大多数开源模型: DPO / IPO / KTO
  
趋势：
  DPO 及其变体正在成为主流
  RLHF(PPO) 主要在超大规模模型上还在用
  原因：DPO 简单、稳定、效果好
```

---

# 九、其他对齐方法速览

## 9.1 DPO 的变体

```
① IPO（Identity Preference Optimization）
  Loss = (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x) - 1/(2β))²
  
  特点：不用 BT 模型假设，用 MSE 代替 log-sigmoid
  优点：更稳定，不会过度优化

② KTO（Kahneman-Tversky Optimization）
  不需要成对数据！只需要单独的 "好/坏" 标签
  
  数据格式：
    {prompt, response, label: "good"/"bad"}
    而不是 {prompt, chosen, rejected}
  
  优点：数据收集更容易（不需要两两对比）
  
③ ORPO（Odds Ratio Preference Optimization）
  把 SFT 和偏好优化合成一步
  Loss = SFT_loss + λ × preference_loss
  优点：更简单，不需要单独的 SFT 阶段

④ SimPO（Simple Preference Optimization）
  去掉了 ref 模型！
  用序列长度归一化的 log prob 代替 log ratio
  优点：只需要 1 个模型，更省资源
```

## 9.2 Constitutional AI（Anthropic）

```
核心思想：用 AI 自己来生成偏好数据（RLAIF）

流程：
  1. 制定一组"宪法"原则（如：不伤害人、不歧视...）
  2. 让模型生成回答
  3. 让另一个模型根据宪法原则评价回答的好坏
  4. 用这些 AI 生成的偏好数据训练

优点：
  不需要大量人类标注（省钱）
  可以覆盖更多场景
  原则可以更新（灵活）

这是 Anthropic（Claude 的公司）的核心技术之一
```

## 9.3 RLHF 2.0：在线 DPO 变体

```
离线 DPO 的局限：
  只用固定的偏好数据集训练
  → 模型看不到自己的"错误"
  → 分布偏移问题

在线 DPO / 迭代 DPO：
  1. 用当前 policy 生成回答
  2. 用 RM 或人类标注偏好
  3. 用新的偏好数据更新 policy
  4. 重复
  
  → 结合了 DPO 的简单性和 RLHF 的在线探索能力
  → 目前很多顶尖模型在用这种方法
```

---

# 十、面试题精选

## Q1："什么是 RLHF？为什么需要它？"

> RLHF 是从人类反馈中进行强化学习的方法。SFT 后的模型虽然能对话，但可能产生有害内容、编造事实或阿谀奉承。RLHF 通过收集人类偏好数据（哪个回答更好），训练奖励模型来自动评分，然后用 PPO 优化语言模型使其生成高奖励的回答。核心目标是让模型对齐人类的价值观——有用、无害、诚实。

## Q2："RLHF 的三个步骤是什么？"

> ①SFT：用指令-回答数据微调 base model，让它会对话。②训练奖励模型：从人类偏好数据中学习一个自动评分器，输入 (prompt, response) 输出标量分数。使用 Bradley-Terry 模型，loss = -log σ(r_chosen - r_rejected)。③PPO 强化学习：用 RM 评分作为奖励，加上 KL 散度约束防止跑偏，用 PPO 算法优化 policy。

## Q3："什么是 DPO？它和 RLHF 有什么区别？"

> DPO 是直接偏好优化，核心洞察是：RLHF 中的最优 policy 有封闭解，奖励函数可以用 policy 的 log ratio 来表示，不需要单独训练奖励模型。DPO 直接在偏好数据上训练 policy，loss = -log σ(β×(Δ_w - Δ_l))。和 RLHF 相比：①不需要 RM②不需要在线生成③不需要 PPO④实现简单、训练稳定⑤只需要 2 个模型而非 4 个。在 BT 模型假设下，DPO 和 RLHF 理论上等价。

## Q4："DPO 的 β 参数代表什么？怎么选？"

> β 控制偏好优化的"强度"和对参考模型的忠诚度。β 大（如 0.5）：更保守，不偏离 ref 太远，可能欠拟合。β 小（如 0.01）：更激进，大幅偏离 ref，可能过拟合或不稳定。常用范围 0.1-0.5。实践中通常从 0.1 开始，如果 reward accuracy 太高但效果不好就增大 β，如果学不动就减小 β。

## Q5："DPO 的 Loss 是怎么推导出来的？"

> 从 RLHF 目标 max E[r(x,y)] - β·KL(π||π_ref) 出发：①求出最优 policy 的封闭解 π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)。②反解出奖励函数 r(x,y) = β·log[π(y|x)/π_ref(y|x)] + f(x)。③代入 Bradley-Terry 偏好模型 P(y_w>y_l) = σ(r_w - r_l)，其中 f(x) 项消掉。④得到 DPO Loss = -log σ(β·(log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))。

## Q6："Reward Hacking 是什么？怎么解决？"

> Reward Hacking 是模型学会"欺骗"奖励模型而非真正提升质量。例如 RM 偏好长回答→模型疯狂灌水。解决方法：①KL 惩罚约束不偏离 ref 太远②用 DPO 代替 RLHF（避免独立 RM 的漏洞）③训练更好的 RM（更大、更多数据）④集成多个 RM⑤定期用人类评估检查。

## Q7："为什么 DPO 正在取代 RLHF？"

> 三个原因：①简单——DPO 只需要 SL 风格的训练（和 SFT 差不多），不需要 RL 的复杂基础设施。②稳定——不需要平衡 4 个模型的训练，超参数少，不容易崩溃。③高效——只需要 2 个模型（policy + ref），不需要在线生成，计算成本大幅降低。在效果上，DPO 在大多数 benchmark 上和 RLHF 持平甚至更好。2024 年后，大多数开源模型（LLaMA-3、Gemma 等）都转向 DPO 及其变体。

## Q8："DPO 有什么局限性？"

> ①离线训练：只从固定数据集学习，无法探索新行为，可能存在分布偏移。②BT 模型假设：假设人类偏好符合 Bradley-Terry 模型，但实际偏好更复杂（如循环偏好 A>B>C>A）。③对数据质量敏感：如果偏好数据有噪声或标注不一致，效果会下降。④需要两份 log prob：每个 batch 需要同时计算 policy 和 ref 的 log prob，显存是 SFT 的约 2 倍。改进方向：在线/迭代 DPO（解决离线问题）、IPO（去掉 BT 假设）、SimPO（去掉 ref 模型）。

---

## 知识串联：从 nanoGPT 到对齐

```
你的学习路径                           核心概念
─────────────────────────────────────────────────────────
nanoGPT (预训练)                  →    cross_entropy loss, next-token prediction
Tokenization                     →    文本 ↔ token 的转换
SFT + LoRA                       →    指令微调 + 参数高效微调
RLHF + DPO (本章)                →    人类偏好对齐

概念对应关系：
  SFT loss = -log P(y_correct)          只学"正确答案"
  RM loss  = -log σ(r_w - r_l)          学"分辨好坏"
  DPO loss = -log σ(β·(Δ_w - Δ_l))     直接学"偏好"

代码对应关系：
  nanoGPT 的 F.cross_entropy()    →    DPO 的 F.logsigmoid()
  nanoGPT 的 model(x).logits     →    DPO 的 get_sequence_logprobs()
  nanoGPT 的 optimizer.step()    →    DPO 完全一样！（都是梯度下降）

一句话总结整个 pipeline：
  预训练让模型"有知识" → SFT 让模型"会对话" → 对齐让模型"说好话"
```

---

*RLHF + DPO 深度解析 - 路线 A 第③步 - 2026-02-11*
*上一步：SFT + LoRA | 下一步：推理优化（量化 / KV Cache / Speculative Decoding）*
