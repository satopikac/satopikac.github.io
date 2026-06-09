---
title: "Lecture 7-Sequence Modeling and Transformers"
publishDate: 2026-06-09
description: '序列建模、Transformer、VLM 与机器人动作 tokenization'
tags:
  - AI
  - Robot Learning
language: '中文'
---
## 课程说明

课程资源

- 课程网站: [Robot learning](http://cvg.ethz.ch/lectures/Robot-Learning/)
- 课程作业网站: [mees-robot-learning-course/ethz-course-2026](https://github.com/mees-robot-learning-course/ethz-course-2026)

本文整理 Lecture 7: Sequence Modeling & Transformers。序列建模、Transformer、VLM 和机器人动作 tokenization 的主线。

## 从 Reactive Policy 到 Sequence Modeling

前面几讲讨论的大多数策略都是 reactive policy：输入当前单帧观测 `o_t`，输出下一步动作 `a_t`。这种形式在理论上简单，但在真实机器人中很容易失效，因为一帧图像通常不足以恢复完整状态。

主要问题有两个：

- 缺少记忆。机器人不知道几秒前做了什么，也不知道物体、行人或自身状态的运动趋势。
- 动作不平滑。每个时间步独立预测一个动作，容易产生抖动、振荡和不连续轨迹。

真实机器人更接近 partially observable MDP。单帧观测只是部分信息，历史观测和历史动作才能共同形成对状态的 belief。因此机器人策略需要从“单步映射”转成“序列建模”。

## Autoregressive Modeling

序列建模的通用工具是概率链式法则。任意联合分布都可以分解为条件分布的乘积：

$$
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t|x_{<t})
$$

autoregressive model 的目标就是学习这些条件分布。对机器人来说，序列中可以包含语言 tokens、图像 tokens、观测历史、动作 tokens 和 proprioception。

关键问题是：如何高效建模很长、很高维的条件分布？这正是 RNN 和 Transformer 试图解决的问题。

## RNN 的局限

RNN 用 hidden state 总结过去信息，每来一个新 token 就更新一次 hidden state。它能处理可变长度输入，但有三类主要问题：

- 长程依赖困难。第 1 步的信息要传到第 100 步，需要穿过 99 个中间状态，梯度容易消失或爆炸。LSTM 缓解但没有彻底解决。
- 难以并行。RNN 的计算必须按时间顺序展开，训练效率低。
- 固定瓶颈。无论序列多长，历史都压缩到固定维度 hidden state 中，信息容量受限。

机器人任务中，长时域历史、视觉变化和多步动作计划都很重要，因此 RNN 的这些限制会成为扩展瓶颈。

## Attention 与 Transformer

Transformer 用 attention 替代 recurrence。attention 的关键性质是：序列中任意两个位置之间只有一跳连接。最后一个 token 可以直接关注第一个 token，不需要穿过所有中间 token。

这同时解决了 RNN 的三类问题：

- 长程依赖通过直接 attention 建立。
- 训练可以并行，因为没有递归依赖。
- 没有单个固定 hidden state 瓶颈，每个 token 都能访问其他 token。

### Attention 机制

每个 token 会生成 query、key、value：

- query 表示“我在找什么”。
- key 表示“我能被怎样匹配”。
- value 表示“如果被关注，我提供什么信息”。

attention score 来自 query 与 key 的点积，经过 softmax 归一化后，对 value 做加权平均：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

可以把 attention 看成可微的软字典查找。模型不只学习 token 表征，也学习在当前上下文中应该检索哪些信息。

### Positional Encoding

self-attention 本身是 set operation，不知道顺序。如果没有位置信息，“机器人拿起勺子”和“勺子拿起机器人”会被当作同一组 token。

常见位置编码有两类：

- absolute positional encoding：给每个位置加固定或可学习向量。简单有效，但对训练时没见过的更长序列泛化差。
- relative positional encoding：编码 token 之间的相对距离，能更好泛化到不同长度序列。

现代大模型常用 RoPE（rotary positional embedding）：不是把位置向量加到 embedding 上，而是按位置旋转 query/key，使点积依赖相对距离。

### Encoder-Decoder、Causal Mask 与 Cross-Attention

原始 Transformer 用于机器翻译，有 encoder-decoder 结构：

- encoder 使用 bidirectional self-attention，输入句子所有 token 互相可见。
- decoder 使用 causal self-attention，每个 token 只能看自己和之前生成的 token。
- cross-attention 让 decoder 查询 encoder 输出，决定当前生成步骤需要关注源句子的哪部分。

causal mask 会把未来 token 的 attention score 设为负无穷，使 softmax 后概率为 0。这是 autoregressive generation 的基础。

### Teacher Forcing

训练 autoregressive model 时常用 teacher forcing：即使模型前一步预测错了，下一步仍喂 ground-truth token，而不是模型自己的错误输出。这样一次 forward pass 就能计算整条序列上所有 next-token loss。

对 Transformer 来说，这比 RNN 更高效，因为所有时间步可并行计算。

## Tokenization

Transformer 只能处理 tokens，所以关键问题是：语言、图像和动作如何 token 化？

### Language Tokenization

字符级 tokenization 词表小，但序列长；单词级 tokenization 序列短，但难处理未登录词。subword tokenization 在二者之间折中。

BPE（Byte Pair Encoding）的流程：

1. 从字符或 byte 作为基础词表开始。
2. 统计最常见的相邻 token pair。
3. 把最常见 pair 合并成新 token。
4. 重复直到达到目标词表大小。

常见子词会被合并，序列长度缩短；未见过的新词仍可拆成已知 subwords。更大的语料能学到更好的压缩模式。现代 LLM 常用 byte-level BPE，使模型能覆盖任意输入，包括罕见字符和 emoji。

### 从 Transformer 到 LLM

现代 GPT 类 LLM 相比原始 Transformer 有几项关键变化：

- 使用 decoder-only architecture，只做 next-token prediction。
- 用 byte-level BPE 处理任意文本。
- 用 RoPE 等相对位置编码增强长上下文泛化。
- 用 FlashAttention 降低 attention 的内存瓶颈，不显式存储完整 `N x N` attention matrix。

LLM 的扩展展示了 scaling 的力量：GPT-2 出现 zero-shot generalization，GPT-3 出现 in-context learning。课程把这与 Sutton 的 bitter lesson 联系起来：最终胜出的往往是能随数据和算力扩展的通用方法，而不是高度手工设计的领域特化方法。

### Scaling Laws 与 Chinchilla

Chinchilla scaling law 的核心结论是：给定固定训练算力，应当同时按比例增加模型大小和训练 token 数，而不是只盲目增大模型。

一个常用经验是 compute-optimal 训练大约需要每个参数 20 个 tokens。GPT-3 被认为相对 undertrained：参数很多，但训练 tokens 不够。后来的 Llama 系列则更接近或有意超过 compute-optimal frontier，因为实际部署时推理成本也重要：对服务大量用户的模型来说，训练时多喂数据、部署时用更小模型可能更划算。

## Vision Tokenization 与 VLM

### Vision Transformer

ViT 把图像切成 patch，并把每个 patch 当作 token：

1. 将图像切成固定大小 patch，例如 `16 x 16`。
2. 展平成向量。
3. 线性投影到 token embedding。
4. 加位置编码。
5. 输入 Transformer。

这与语言 BPE 的思想类似：找到合适的基本单位，把它变成向量，再交给 Transformer 处理。

### CLIP：对齐图像和语言

CLIP 使用双塔结构：

- image encoder 编码图像。
- text encoder 编码文本。
- 用 contrastive learning 让匹配的 image-text pair 在 embedding space 中接近，不匹配的远离。

训练后，图像中的 apple 和文本 token “apple”会在语义空间中对齐。这为机器人理解语言指令和视觉目标提供基础。

### LLaVA：Early Fusion

LLaVA 的问题是：如何把视觉信息注入预训练 LLM，让它能像处理文本一样处理图像？

做法是：

1. 用冻结的 CLIP vision encoder 提取 image patch embeddings。
2. 用一个可训练 MLP connector 把视觉 embedding 投影到 LLM token space。
3. 把视觉 tokens prepend 到文本 token 前。
4. 让 LLM 对统一序列做 self-attention。

这是 early fusion：图像和文本在同一个 self-attention 流中自由交互。

### Flamingo：Late Fusion

Flamingo 保留 LLM 的文本 self-attention，在每层插入 gated cross-attention，让文本 token 查询视觉 features。

关键点：

- vision encoder 冻结。
- perceiver resampler 把视觉输出压缩成固定数量 keys/values。
- LLM 每层通过 cross-attention 获取视觉信息。
- tanh gate 初始化为 0，训练开始时模型等价于原 LLM，之后逐渐打开视觉通道。

late fusion 更好地保护预训练语言能力，但视觉和语言交互受 cross-attention 通道限制。

### Native Multimodal Models

第三条路线是从头训练统一多模态 Transformer，把图像、文本、音频、视频等都作为 token 序列处理。它没有主模态和附加模态之分，灵活性最高，但训练成本也最大。

三类 VLM 路线的取舍：

- early fusion：交互最充分，但可能影响 LLM 原有能力。
- late fusion：更保守地保留 LLM，但模态交互受限。
- native multimodal：最统一、最灵活，但需要巨量数据和算力。

## 把机器人动作也变成 Token

机器人可以被看作多模态序列预测问题：

$$
\text{language tokens} + \text{image tokens} + \text{action tokens}
$$

与 VLM 的区别是，模型输出不再是文本答案，而是控制机器人电机或末端执行器的动作。

### Naive Action Discretization

早期 VLA，如 RT-2 和 OpenVLA，常用 per-dimension, per-timestep binning：

1. 对每个动作维度计算训练集的 1st 和 99th percentile。
2. clip 到这个范围并归一化到 `[-1, 1]`。
3. 离散成 256 个 bins。
4. 把这些 bin 注入 LLM vocabulary。

相比 min-max normalization，quantile normalization 更鲁棒，因为 outlier 不会把大部分正常动作压到少数 bins 中。

把 action bin 变成 LLM token 后，就可以继续用 next-token prediction 和 cross entropy 训练。

## Action Chunking

单步动作预测容易抖动。Action chunking 改成一次预测未来 `K` 步动作：

$$
a_{t:t+K}
$$

这样一段动作由同一次 forward pass 共同规划，轨迹更平滑。执行完 chunk 后再重新查询模型。

如果 chunk 很长，机器人在执行期间会“闭眼”太久，不能及时响应新观测。temporal ensembling 解决这个问题：每个时间步都重新查询模型，维护多个重叠 action chunk，并用指数衰减权重融合当前时刻的多个预测。这样既保留 chunking 的平滑性，又保持每步响应新观测。

ACT 是 action chunking 的代表。它在 bimanual Aloha 上以 50Hz 控制双臂，14 个自由度，一次预测 100 步 action chunk，相当于每次 forward pass 输出 1400 个连续动作值。对电池插入等高精度双臂任务，chunking 显著减少抖动。

## 高频控制中的 Action Token 问题

把 ACT 式长 action chunk 直接变成 autoregressive tokens 会遇到问题：高频控制下，相邻动作几乎一样，token 之间高度相关。autoregressive model 很容易通过复制前一个 action token 降低训练误差，而不是学习真正的灵巧技能。

控制频率越高，相邻 token 的新增信息越少，naive binning 在 10Hz、20Hz 甚至更高频率下会明显退化。

解决方向是动作压缩：不要把每个时间步每个维度都当作独立 token，而是寻找类似语言 subword、图像 patch 的“动作基本单位”。

## FAST Tokenization

FAST 使用来自信号处理的 DCT（Discrete Cosine Transform）压缩动作块。直觉是：机器人 action chunk 通常是平滑轨迹，平滑信号的大部分能量集中在低频，很多高频系数接近 0。

流程大致是：

1. 对每个动作维度的 chunk 做 DCT，从时间域转到频域。
2. 缩放并四舍五入 DCT coefficients。
3. 按低频优先顺序 flatten 成整数序列。
4. 对整数序列运行 BPE。
5. 得到紧凑的 action tokens，可直接接入 LLM vocabulary。

BPE 会把大量 0 和常见低频系数组合压缩掉，从而大幅降低 token 数，同时保留控制中真正重要的低频运动结构。

结果上，FAST 让 autoregressive VLA 在更高控制频率下不再崩溃，并能在一些设置中比 flow-matching action head 更快达到相同性能。它推动了 generality 和 dexterity 的结合：既能做开放语言任务，也能执行高频、长时域、双臂灵巧操作。

## 本讲主线

Lecture 7 的核心是把机器人学习放进 sequence modeling 框架。单帧 reactive policy 无法处理部分可观测性和动作平滑问题；Transformer 通过 attention 提供可扩展的长程记忆；LLM/VLM 说明统一 token 序列和规模化训练能带来强泛化能力。

把这套框架迁移到机器人时，关键难点是动作 tokenization。低频、短 horizon 下 naive discretization 可以工作；高频灵巧控制下必须引入 action chunking、temporal ensembling 和 FAST 这类压缩 tokenization。整体思路仍然是 bitter lesson 在机器人中的版本：找到合适表示，使用通用可扩展架构，再让数据和算力发挥作用。
