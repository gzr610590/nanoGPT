# Tokenization 深度解析

> 从字符级到子词级，理解 BPE 算法的原理、实现和面试要点。
> 路线 A 第 ① 步，衔接 nanoGPT（字符级）→ 真实 LLM（子词级）。

---

## 目录

1. [为什么需要 Tokenization](#一为什么需要-tokenization)
2. [三种粒度对比](#二三种粒度对比字符-vs-单词-vs-子词)
3. [BPE 算法详解](#三bpe-算法详解)
4. [Byte-level BPE（GPT-2/3/4 使用）](#四byte-level-bpegpt-234-使用)
5. [WordPiece（BERT 使用）](#五wordpiecebert-使用)
6. [SentencePiece 与 tiktoken](#六sentencepiece-与-tiktoken)
7. [词表大小的权衡](#七词表大小的权衡)
8. [从零实现 BPE](#八从零实现-bpe)
9. [面试题精选](#九面试题精选)

---

# 一、为什么需要 Tokenization

## 1.1 回顾：nanoGPT 的字符级 Tokenization

你在训练 Shakespeare 模型时用的就是最简单的 tokenization：

```python
# data/shakespeare_char/prepare.py 中的核心逻辑
chars = sorted(list(set(data)))  # 所有唯一字符
stoi = {ch: i for i, ch in enumerate(chars)}  # 字符 → 整数

# "hello" → [46, 43, 50, 50, 53]
# 每个字符 = 1 个 token
```

```
词表大小 = 65（26小写 + 26大写 + 标点空格换行）
"hello" = 5 个 token
```

这在小实验中没问题，但真实 LLM 不这样做。**为什么？**

## 1.2 字符级的问题

```
问题 1：序列太长
  "The transformer architecture" = 30 个字符 = 30 个 token
  但如果用子词："The" "trans" "former" "architecture" = 4 个 token
  
  序列长 → Attention 计算 O(n²) → 太慢太贵！

问题 2：每个 token 信息量太少
  字符 "t" 本身几乎没有语义
  模型需要"好几层"才能把字符组合成有意义的单位
  浪费了模型容量

问题 3：上下文窗口被浪费
  block_size = 1024 时：
  字符级 → 只能看 ~200 个单词
  子词级 → 能看 ~700 个单词
```

---

# 二、三种粒度对比：字符 vs 单词 vs 子词

## 2.1 对比表

| 粒度 | 词表大小 | "unhappiness" 的表示 | 优点 | 缺点 |
|------|---------|---------------------|------|------|
| **字符级** | ~100 | u-n-h-a-p-p-i-n-e-s-s (11 token) | 词表小，无 OOV | 序列太长 |
| **单词级** | ~100,000+ | unhappiness (1 token) | 语义清晰 | 词表爆炸，新词怎么办？ |
| **子词级** | ~30,000-100,000 | un-happi-ness (3 token) | **平衡！** | 需要训练分词器 |

## 2.2 子词的核心思想

```
"unhappiness" → "un" + "happi" + "ness"

高频词保持完整：  "the", "is", "and"  → 1 个 token
低频词拆成片段：  "unhappiness"       → 3 个 token
未见过的词也能处理："transformerify"   → "transform" + "er" + "ify"

精妙之处：
- 常见词 = 1 个 token → 序列短，高效
- 罕见词 = 多个 token → 不需要巨大词表
- 新词也能分 → 永远不会遇到 "未知词"
```

---

# 三、BPE 算法详解

## 3.1 BPE 的直觉

**BPE = Byte Pair Encoding = 字节对编码**

核心思想极其简单：**反复合并最高频的相邻对**。

```
类比：文本压缩

原始：a a b a a b a a b
观察：a a 出现最多（5次）
合并：把 "a a" 替换为 "Z" → Z b Z b Z b
观察：Z b 出现最多（3次）
合并：把 "Z b" 替换为 "Y" → Y Y Y
完成！

BPE 就是这个过程，只不过应用在文本 token 上。
```

## 3.2 训练阶段（学习合并规则）

### 完整示例

训练语料：`"low low low low low lowest lowest newer newer newer wider wider wider"`

**第 0 步：初始化 — 所有单字符作为初始词表**

```
初始词表 = {l, o, w, e, s, t, n, r, i, d, ' '}（+ 词尾标记 _）

把每个单词拆成字符：
  "low"    → [l, o, w, _]      出现 5 次
  "lowest" → [l, o, w, e, s, t, _]  出现 2 次
  "newer"  → [n, e, w, e, r, _]     出现 3 次
  "wider"  → [w, i, d, e, r, _]     出现 3 次
```

**第 1 步：统计所有相邻对的频率**

```
(l, o): 5+2 = 7 次    ← 最高频！
(o, w): 5+2 = 7 次
(w, _): 5 次
(w, e): 2+3 = 5 次
(e, r): 3+3 = 6 次
(e, s): 2 次
(s, t): 2 次
(n, e): 3 次
(r, _): 3+3 = 6 次
(i, d): 3 次
(d, e): 3 次
...
```

**合并最高频对 (l, o) → "lo"**

```
新词表 = {..., "lo"}

更新序列：
  "low"    → [lo, w, _]
  "lowest" → [lo, w, e, s, t, _]
  "newer"  → [n, e, w, e, r, _]    （不含 lo，不变）
  "wider"  → [w, i, d, e, r, _]    （不含 lo，不变）
```

**第 2 步：再统计，再合并**

```
(lo, w): 5+2 = 7 次   ← 最高频！
(e, r): 6 次
(r, _): 6 次
...

合并 (lo, w) → "low"

更新序列：
  "low"    → [low, _]
  "lowest" → [low, e, s, t, _]
```

**第 3 步：继续...**

```
(e, r): 6 次   ← 最高频
合并 → "er"

  "newer" → [n, e, w, er, _]
  "wider" → [w, i, d, er, _]
```

**第 4 步：**

```
(er, _): 6 次   ← 最高频
合并 → "er_"

  "newer" → [n, e, w, er_]
  "wider" → [w, i, d, er_]
```

**...重复直到达到目标词表大小**

### 训练过程总结

```
循环：
  1. 统计所有相邻 token 对的频率
  2. 找到最高频的对
  3. 合并这个对，得到新 token
  4. 更新所有序列
  5. 词表大小 +1
  6. 重复，直到达到目标词表大小

合并规则（merge rules）= 最终输出
  [("l", "o") → "lo",
   ("lo", "w") → "low",
   ("e", "r") → "er",
   ("er", "_") → "er_",
   ...]
```

## 3.3 推理阶段（应用合并规则）

```
输入："lowest"

第 0 步：拆成字符
  [l, o, w, e, s, t]

第 1 步：应用规则 1 (l,o)→lo
  [lo, w, e, s, t]

第 2 步：应用规则 2 (lo,w)→low
  [low, e, s, t]

第 3 步：应用规则 3 (e,s)→es（如果有这条规则）
  [low, es, t]

第 4 步：没有更多可合并的了
  最终 tokens = [low, es, t]
  对应 IDs = [256, 312, 87]（查词表）
```

---

# 四、Byte-level BPE（GPT-2/3/4 使用）

## 4.1 和标准 BPE 的区别

```
标准 BPE：
  初始词表 = 所有 Unicode 字符（几万个！）
  问题：初始词表就很大

Byte-level BPE：
  初始词表 = 256 个字节（0x00-0xFF）
  任何文本先转成 UTF-8 字节序列，再做 BPE

  "你好" → UTF-8 字节: [228, 189, 160, 229, 165, 189]  （6 个字节）
         → 初始 tokens: [228, 189, 160, 229, 165, 189]  （6 个 token）
         → BPE 逐步合并:
            (228,189)→256  → [256, 160, 229, 165, 189]   （5 个 token）
            (256,160)→312  → [312, 229, 165, 189]         （4 个 token）
            (229,165)→285  → [312, 285, 189]              （3 个 token）
            (285,189)→450  → [312, 450]                   （2 个 token）
                              ↑     ↑
                  vocab[312]的字节=[228,189,160] → 解码为"你"
                  vocab[450]的字节=[229,165,189] → 解码为"好"

  最终 token 仍然是整数 ID，只不过它们对应的字节序列恰好能解码为完整汉字
```

## 4.2 优势

```
① 永远不会有 [UNK]（未知 token）
   任何数据都是字节 → 任何输入都能编码
   
② 初始词表极小
   只有 256 个 → 后续合并更灵活

③ 天然多语言
   中文、日文、emoji... 都是字节 → 统一处理

④ 代码/数学/特殊字符全能处理
   "∫e^x dx" → 字节序列 → BPE → 没问题
```

## 4.3 GPT-2 的实际 Tokenization

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")  # GPT-2 的 tokenizer

# 英文（高效，常见词 = 1 token）
enc.encode("hello world")
# → [31373, 995]  只要 2 个 token！

# "Transformer"
enc.encode("Transformer")
# → [8291, 16354]  "Trans" + "former" = 2 个 token

# 中文（每个字大约 2-3 个 token）
enc.encode("你好")
# → [19526, 254, 25001, 121]  4 个 token（不高效）

# GPT-4 的 cl100k_base 对中文更友好
enc4 = tiktoken.get_encoding("cl100k_base")
enc4.encode("你好")
# → [57668, 53901]  只要 2 个 token！
```

## 4.4 不同模型的词表大小

> 注意：以下模型（除 BERT 外）使用的**算法都是 Byte-level BPE**，
> 区别在于：①训练语料不同 → 合并规则不同 ②词表大小不同 ③实现库不同。

| 模型 | 算法 | 实现库 | 词表名称 | 词表大小 | 特点 |
|------|------|--------|---------|---------|------|
| GPT-2 | Byte-level BPE | tiktoken | gpt2 | 50,257 | 英文语料为主 |
| GPT-3.5/4 | Byte-level BPE | tiktoken | cl100k_base | 100,256 | 更多多语言语料 |
| LLaMA 1/2 | Byte-level BPE | SentencePiece | - | 32,000 | 较小词表 |
| LLaMA-3 | Byte-level BPE | tiktoken | - | 128,256 | 大词表，多语言 |
| BERT | **WordPiece** | HF Tokenizers | - | 30,522 | **不同算法！** |

---

# 五、WordPiece（BERT 使用）

## 5.1 和 BPE 的区别

```
BPE：合并"出现次数最多"的对
WordPiece：合并"使语料库似然提升最大"的对

具体区别：
  BPE 选择标准：count(A,B)
  WordPiece 选择标准：count(AB) / (count(A) × count(B))
  
  WordPiece 倾向于合并那些"单独出现少、组合出现多"的对
  → 更好地捕捉"组合后有新含义"的子词
```

## 5.2 WordPiece 的标记方式

```
BERT 用 ## 表示"续接"：
  "unhappiness" → ["un", "##happi", "##ness"]
  
  "un" = 词的开头
  "##happi" = 接在前面的
  "##ness" = 接在前面的

BPE 用相反的方式 — 空格前缀（Ġ）：
  "I love cats" → ["I", "Ġlove", "Ġcats"]
  
  Ġ = 空格字符的可视化表示
  "Ġlove" 的实际内容是 " love"（前面有空格）
  含义：Ġ 开头 = 新词的开始

两者逻辑相反：
  WordPiece：标记"续接部分"（加 ##）→ 有 ## 的不是词首
  BPE：      标记"新词开头"（带空格）→ 有 Ġ 的是词首
```

## 5.3 面试对比

| 特性 | BPE | WordPiece |
|------|-----|-----------|
| 合并标准 | 频率最高 | 似然提升最大 |
| 代表模型 | GPT 系列 | BERT 系列 |
| 词边界标记 | Ġ = 空格前缀（新词带空格）| ## = 续接前缀（非词首加 ##）|
| 实现 | tiktoken / SentencePiece | HuggingFace Tokenizers |

---

# 六、SentencePiece 与 tiktoken

## 6.1 SentencePiece（LLaMA 1/2 使用）

```
特点：
① 直接在原始文本上训练（不需要预分词）
② 把空格当作普通字符（用 ▁ 表示空格）
③ 支持 BPE 和 Unigram 两种算法
④ 语言无关 → 多语言友好

"I love NLP" → ["▁I", "▁love", "▁N", "LP"]
                  ↑ 空格被编码为 ▁
```

## 6.2 tiktoken（GPT-3.5/4, LLaMA-3 使用）

### tiktoken 是什么？

```
tiktoken = OpenAI 开发的分词工具库

就是一个 Python 库，帮你把文字变成 token ID：
  import tiktoken
  enc = tiktoken.get_encoding("gpt2")
  enc.encode("hello world")  → [31373, 995]
  enc.decode([31373, 995])    → "hello world"

它的底层用 Rust 语言写的（所以速度极快），
但你只需要用 Python 接口调用，不需要懂 Rust。
```

### tiktoken 的特别之处：预分词

```
问题：BPE 在合并时，可能把不同单词的字母错误地合并

  "I'll go" 的字节里，"l" 和 " "(空格) 是相邻的
  → BPE 可能把 "l " 合并成一个 token → 不合理！

tiktoken 的解决方案：先把文本切成"块"，再在每个块内做 BPE

  "I'll go"
     │
     ▼  第1步：按规则切分（把英语缩写拆开）
  ["I", "'ll", " go"]      ← 三个独立的块
     │     │      │
     ▼     ▼      ▼         第2步：每个块内部分别做 BPE
  [40]  [1183]  [467]       ← 互不干扰，不会跨块合并

切分规则举例：
  I'll   → "I" + "'ll"     （缩写拆开）
  don't  → "don" + "'t"
  he's   → "he" + "'s"
  Hello World → "Hello" + " World"  （按空格+字母切）
```

### SentencePiece 不需要预分词

```
SentencePiece 的做法不同：
  它直接在原始文本上做 BPE，把空格当作普通字符（用 ▁ 表示）
  不需要先切分 → 实现更简单，天然支持任何语言
```

## 6.3 对比

```
               SentencePiece              tiktoken
是什么         Google 的分词工具库         OpenAI 的分词工具库
算法           Byte-level BPE             Byte-level BPE（一样的算法）
预分词         不需要（直接处理原文）       需要（先按规则切块）
空格处理       用 ▁ 代替空格              空格作为字节处理
速度           快                          极快（底层用 Rust 写的）
使用模型       LLaMA 1/2, T5              GPT-3.5/4, LLaMA-3
```

---

# 七、词表大小的权衡

## 7.1 核心矛盾

```
词表大 → 每个 token 信息量多 → 序列短 → Attention 快
         但 Embedding 矩阵大 → 参数多 → 显存高

词表小 → Embedding 矩阵小 → 参数少
         但序列更长 → Attention 计算量大

     词表大小
     ────────────────────────────────────────→
     256      32K      50K     100K    128K

Embedding 参数量：
     少 ─────────────────────────────────→ 多

序列长度（同样文本）：
     长 ─────────────────────────────────→ 短

Attention 计算量：
     大 ─────────────────────────────────→ 小
```

## 7.2 实际影响

```
假设 d_model = 4096

词表 32K:  Embedding 参数 = 32000 × 4096 = 131M
词表 128K: Embedding 参数 = 128000 × 4096 = 524M  ← 多了 393M！

但同一段文本：
词表 32K:  "Hello world" = 3 tokens
词表 128K: "Hello world" = 2 tokens → 序列短 25%

Attention 是 O(n²)，序列短 25% → 计算量减少 ~44%
```

## 7.3 趋势

```
早期（2018-2020）：词表较小（30K-50K）
  BERT: 30522, GPT-2: 50257

现在（2023-2026）：词表越来越大
  GPT-4: 100K, LLaMA-3: 128K, Gemma: 256K

原因：
① 多语言需要更多 token（中文、日文等）
② 现代硬件显存更大，Embedding 矩阵不是瓶颈
③ 缩短序列对推理加速意义重大
```

---

# 八、从零实现 BPE

## 8.1 完整 Python 实现

```python
"""
从零实现 BPE（Byte Pair Encoding）

参考 Andrej Karpathy 的 minbpe 项目简化版
"""

def get_stats(ids: list[int]) -> dict:
    """
    统计所有相邻 token 对的出现频率
    
    输入: [1, 2, 3, 1, 2]
    输出: {(1,2): 2, (2,3): 1, (3,1): 1}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple, new_id: int) -> list[int]:
    """
    把序列中所有出现的 pair 替换为 new_id
    
    输入: ids=[1,2,3,1,2], pair=(1,2), new_id=4
    输出: [4, 3, 4]
    """
    new_ids = []
    i = 0
    while i < len(ids):
        # 如果当前位置匹配 pair 的第一个，且下一个匹配第二个
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(new_id)
            i += 2  # 跳过两个，合并为一个
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BasicBPE:
    """最小 BPE Tokenizer 实现"""
    
    def __init__(self):
        self.merges = {}      # (int, int) → int 合并规则
        self.vocab = {}       # int → bytes 词表
    
    def train(self, text: str, vocab_size: int):
        """
        训练 BPE Tokenizer
        
        text: 训练语料
        vocab_size: 目标词表大小（必须 >= 256）
        """
        assert vocab_size >= 256, "词表至少包含 256 个字节"
        num_merges = vocab_size - 256  # 需要合并的次数
        
        # ===== 第1步：文本 → UTF-8 字节序列 =====
        tokens = list(text.encode("utf-8"))
        # "hello" → [104, 101, 108, 108, 111]
        
        # ===== 第2步：初始化词表（256 个字节）=====
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # ===== 第3步：反复合并 =====
        self.merges = {}
        
        for i in range(num_merges):
            # 统计相邻对频率
            stats = get_stats(tokens)
            if not stats:
                break
            
            # 找到最高频的对
            best_pair = max(stats, key=stats.get)
            
            # 分配新 ID
            new_id = 256 + i
            
            # 执行合并
            tokens = merge(tokens, best_pair, new_id)
            
            # 记录合并规则
            self.merges[best_pair] = new_id
            
            # 更新词表
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # 打印进度
            if (i + 1) % 100 == 0 or i < 10:
                freq = stats[best_pair]
                token_str = self.vocab[new_id]
                print(f"merge {i+1}/{num_merges}: {best_pair} → {new_id} "
                      f"(freq={freq}, token='{token_str}')")
        
        print(f"\n训练完成！词表大小: {len(self.vocab)}")
        print(f"压缩率: {len(text.encode('utf-8'))} → {len(tokens)} "
              f"({len(tokens)/len(text.encode('utf-8')):.2%})")
    
    def encode(self, text: str) -> list[int]:
        """
        编码：文本 → token IDs
        
        按照训练时学到的合并规则，依次应用
        """
        tokens = list(text.encode("utf-8"))
        
        # 按照合并规则的学习顺序，依次尝试合并
        while len(tokens) >= 2:
            # 统计当前所有相邻对
            stats = get_stats(tokens)
            
            # 找到最早学到的（优先级最高的）可合并的对
            # 合并规则的顺序 = 优先级
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            
            # 如果这个对不在合并规则中，停止
            if pair not in self.merges:
                break
            
            # 执行合并
            new_id = self.merges[pair]
            tokens = merge(tokens, pair, new_id)
        
        return tokens
    
    def decode(self, ids: list[int]) -> str:
        """
        解码：token IDs → 文本
        """
        tokens = b"".join(self.vocab[id] for id in ids)
        return tokens.decode("utf-8", errors="replace")


# ===== 使用示例 =====
if __name__ == "__main__":
    # 训练语料
    text = "the cat sat on the mat. the cat ate the rat."
    
    # 训练
    tokenizer = BasicBPE()
    tokenizer.train(text, vocab_size=276)  # 256 + 20 次合并
    
    # 编码
    encoded = tokenizer.encode("the cat")
    print(f"\n'the cat' → {encoded}")
    
    # 解码
    decoded = tokenizer.decode(encoded)
    print(f"{encoded} → '{decoded}'")
    
    # 验证往返一致性
    assert decoded == "the cat"
    print("✓ 编解码一致!")
```

## 8.2 代码核心逻辑图解

```
训练阶段：
  "hello world hello" 
      │
      ▼ encode UTF-8
  [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 32, ...]
      │
      ▼ get_stats → (108, 108): 2次, (104, 101): 2次, ...
      │
      ▼ 合并最高频 (108, 108) → 256
  [104, 101, 256, 111, 32, 119, 111, 114, 108, 100, 32, ...]
      │
      ▼ get_stats → ...
      │
      ▼ 重复 num_merges 次
      │
      ▼ 输出：merges 规则表 + vocab 词表


推理阶段：
  "hello"
      │
      ▼ encode UTF-8
  [104, 101, 108, 108, 111]
      │
      ▼ 按 merges 规则顺序依次合并
  [104, 101, 256, 111]    ← (108,108)→256
      │
      ▼ 继续尝试合并...
      │
      ▼ 没有更多可合并的了
  [104, 101, 256, 111]    ← 最终 token IDs
```

## 8.3 关键理解：训练和推理的合并顺序

```
训练时：贪心算法 → 每次合并全局最高频的对
推理时：按训练时的学习顺序 → 先合并早期学到的规则

为什么推理时不能贪心？
  因为不同文本的频率分布不同
  推理时没有全局统计信息
  必须用固定的规则顺序保证确定性

例：规则顺序 [r1, r2, r3, ...]
  先尝试 r1 能不能在当前序列中匹配
  再尝试 r2
  ...
  直到没有规则可以匹配
```

---

# 九、面试题精选

## Q1："BPE 的训练过程是什么？"

> BPE 从字节（256 个）开始，反复统计所有相邻 token 对的频率，每次合并出现最多的对，产生一个新 token 加入词表。重复这个过程直到达到目标词表大小。核心是贪心算法——每一步都选局部最优的合并。

## Q2："BPE 和 WordPiece 有什么区别？"

> 核心区别在合并标准：BPE 合并出现次数最多的对，WordPiece 合并使语料似然提升最大的对（count(AB) / count(A)×count(B)）。WordPiece 倾向于合并"单独出现少但组合出现多"的对，更适合捕捉词素级的语义组合。BPE 用于 GPT 系列，WordPiece 用于 BERT 系列。

## Q3："什么是 Byte-level BPE？和标准 BPE 有什么区别？"

> 标准 BPE 的初始词表是所有 Unicode 字符（可能上万个），Byte-level BPE 的初始词表是 256 个字节。任何文本先转成 UTF-8 字节序列再做 BPE。优点：①初始词表极小（256）②永远不会有 UNK③天然支持任何语言和特殊字符。GPT-2/3/4 都用 Byte-level BPE。

## Q4："词表大小怎么选？"

> 需要权衡：词表大 → 序列短（Attention 更快）但 Embedding 参数多；词表小 → 参数少但序列长。趋势是越来越大：BERT 30K、GPT-2 50K、GPT-4 100K、LLaMA-3 128K。原因是多语言需求和序列缩短对推理加速的价值。通常 32K-128K 是合理范围。

## Q5："Tokenization 对模型效果有什么影响？"

> 影响非常大：①词表不适合目标语言会导致效率低下（如 GPT-2 处理中文，每个汉字需要 2-3 个 token）②Tokenizer 的训练数据分布影响分词质量③分词粒度影响模型能"理解"的最小单位④数字和代码的 tokenization 特别关键（"123456" 可能被分成奇怪的片段，影响算术能力）。

## Q6："为什么 LLM 做数学不好？和 Tokenization 有关系吗？"

> 有很大关系。BPE 可能把 "12345" 分成 "123" 和 "45"，模型看不到单独的数位。或者 "380" 是一个 token 但 "381" 被分成 "38" + "1"——同样的数字，表示方式完全不同。这让模型很难学习数值的规律性。解决方案包括：①数字逐位分词②使用专门的数学 tokenizer③CoT 推理分步计算。

---

*Tokenization 深度解析 - 路线 A 第①步 - 2026-02-07*
*下一步：SFT + LoRA 微调*
