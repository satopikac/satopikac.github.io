---
title: "Lecture 9-Generalist Robot Policies"
publishDate: 2026-06-09
description: '通用机器人策略：Open X-Embodiment、Octo、CrossFormer、VLA 和可扩展评估'
tags:
  - AI
  - Robot Learning
language: '中文'
---
## 课程说明

课程资源

- 课程网站: [Robot learning](http://cvg.ethz.ch/lectures/Robot-Learning/)
- 课程作业网站: [mees-robot-learning-course/ethz-course-2026](https://github.com/mees-robot-learning-course/ethz-course-2026)

本文整理 Lecture 9: Generalist Robot Policies。通用机器人策略的数据、模型和评估主线。

## 通用机器人策略的目标

ACT、Diffusion Policy 等方法在单任务、单机器人、灵巧操作中表现很强，但它们本质上更像 specialist policies：换一个机器人、换一个任务或换一个环境，往往需要重新训练或大量 fine-tuning。

generalist robot policy 的目标更接近机器人 foundation model：一个模型能控制多种机器人，在多种环境中完成多种任务，并能和人类共同工作。理想情况是，用户可以下载一个“机器人脑”，装到自己的机器人上，就能得到合理且有用的初始行为。

这个目标受 LLM/VLM 的 foundation model 范式启发：大模型在大量多任务数据上训练，往往超过为单一任务手工设计的 specialist model。但机器人领域不能直接复制 NLP 的 recipe，因为机器人有几个额外困难：

- 数据少且昂贵。机器人数据需要真实硬件或遥操作采集，远不如文本和图像易得。
- 数据高度异质。不同机器人有不同相机、传感器、自由度、控制频率和动作空间。
- 控制有实时性要求。机器人不能等一个大模型慢慢生成动作。
- 评估成本高。真实 rollout 慢、容易受硬件故障和相机位置变化影响。

因此，第 9 讲把通用机器人策略分成三件事：数据从哪里来，如何建模异质数据，以及如何可扩展地评估。

## 数据：Open X-Embodiment

机器人数据虽然不像互联网文本那样集中，但世界各地实验室都积累了很多小规模数据集。问题是这些数据通常格式不同、机器人不同、任务不同，很难联合训练。

Open X-Embodiment 的关键思想是：把来自不同实验室、不同机器人 embodiment 的数据整理到统一格式中，让模型可以 co-training。

这个数据集包含：

- 超过一百万个真实机器人 episodes。
- 超过 20 种机器人 embodiment。
- 来自多个研究机构的操作、导航、抓取等数据。

它的重要性不只是数据量，而是第一次让社区能在多机器人、多任务数据上系统训练 generalist policies。课程中强调，Open X-Embodiment 让后续 Octo、CrossFormer、OpenVLA、Pi 系列等工作有了共同数据基础。

## 把机器人策略看作多模态序列建模

常规机器人策略输入当前图像和任务描述，输出末端执行器 pose 或动作序列。如果把图像换成互联网图像，把任务描述换成 caption，输出换成文本答案，这就很像 vision-language model。

因此，很多 generalist robot policy 把机器人建模成：

$$
\text{image tokens} + \text{language tokens} \rightarrow \text{action tokens}
$$

这有一个重要好处：可以借用 NLP 和视觉社区已经成熟的 transformer、tokenization、attention、VLM pretraining 和 sequence modeling 技术。困难在于 action 不是普通文本，它受 embodiment、控制频率和动作空间约束。

## Octo

Octo 是较早的开源 generalist robot policy。它在 Open X-Embodiment 的约 80 万条机器人轨迹上训练，是一个 transformer-based policy。

### 架构

Octo 包含几部分：

- language encoder：用预训练 T5 编码自然语言任务。
- observation tokenizer：把图像 patchify，并用浅层 CNN/投影得到图像 tokens。
- transformer backbone：处理任务 tokens 和观测 tokens。
- readout tokens：作为占位 tokens，从 transformer 中汇聚信息，交给后续 action heads。
- diffusion action head：生成多模态动作序列。

Octo 使用 goal-conditioned behavior cloning。目标可以用自然语言指定，也可以用 goal image 指定。这种形式让同一个模型能处理不同任务条件。

### Block-Wise Causal Structure

Octo 的 transformer 有 block-wise causal 结构：观测 tokens 不能看未来时间步的信息，但可以看任务信息和当前/过去观测。这让模型能处理序列历史，同时避免信息泄漏。

### 设计取舍

课程强调了几个 Octo 的设计经验：

- 尽量把参数放在 transformer backbone 中，而不是给每个相机或输入单独设计很大的视觉 encoder。
- 没有手工对齐所有数据集的坐标系，这是为了扩展性。
- 但 gripper dimension 必须对齐，因为早期大量失败与 gripper 开合语义不一致有关。例如不同数据集可能用 1 表示打开，也可能用 1 表示关闭，需要统一。

### Zero-Shot 与 Fine-Tuning

Octo 在多种机器人上做了 zero-shot 和 fine-tuning 评估。它能 out-of-the-box 控制多个 embodiment，并在一些设置中超过 RT-1-X，接近更大规模闭源 VLA RT-2-X。

更重要的是适应性：即使新机器人有新传感器、新相机数、force-torque 输入或不同动作空间，也可以插入新的 input tokens/head，再 fine-tune。实验发现，从 generalist robot policy fine-tune 通常比从普通视觉 backbone 或单任务模型开始更有效。

## CrossFormer：跨 Embodiment 策略

Octo 主要面向单臂 manipulation。更通用的策略需要跨越更大 embodiment 差异：双臂、导航、四足、室内外移动、灵巧操作等。

CrossFormer 的目标是训练一个单一 checkpoint 控制多种 embodiment。它在 Octo 数据基础上加入双臂操作、导航、locomotion 等，总计约 90 万条轨迹，并使用 shared transformer backbone 加 embodiment-specific heads。

### 为什么需要 Embodiment-Specific Heads

不同机器人差异很大：

- 动作维度不同，例如 7 自由度单臂 vs 14 自由度双臂。
- 控制频率不同，例如 5Hz 单臂 vs 50Hz 双臂 dexterous manipulation。
- action chunk 长度不同。
- sensor 输入数量和相机视角不同。

CrossFormer 通过共享尽可能多的参数来促进迁移，例如共享相同类型相机的 image encoder，同时为不同 embodiment 使用不同 action head。这避免了强行动作空间对齐。

课程中特别强调：CrossFormer 的不同机器人 rollout 来自同一个模型 checkpoint，而不是每个机器人单独选一个最优 checkpoint。这说明单一神经网络权重可以跨 embodiment 工作。

## 多任务预训练是否帮助 Post-Training

机器人社区普遍相信多任务预训练能帮助下游 fine-tuning，但要严格验证并不容易。TRI 的相关研究用 diffusion transformer policies，在约 1700 小时预训练数据上训练，再 fine-tune 到双臂 Franka 任务，并做了严谨的真实世界 blind A/B testing。

主要结论：

- 预训练确实能让下游单任务 post-training 更样本高效，提升可达 3-5 倍。
- 低数据 fine-tuning regime 中收益最大。
- 数据归一化等工程细节有时比架构改动影响更大。

这说明通用机器人策略不仅是“模型更大”，还很依赖数据管线、normalization、action representation 和评估协议。

## 从 VLM 到 VLA

Open X-Embodiment 和 Octo 这类方法更多是从机器人数据从头训练 generalist policy。但机器人数据相比 LLM/VLM 训练数据仍然很小，因此另一个方向是利用互联网规模预训练 priors，把 VLM fine-tune 成 VLA。

基本思路类似 LLaVA 把图像 tokens 接入 LLM：

1. 从预训练 VLM/LLM 开始。
2. 把机器人动作离散成 action tokens。
3. 把 action tokens 注入 LLM vocabulary，常见做法是覆盖最少使用的 tokens。
4. 用 next-token prediction 训练模型输出动作 tokens。

早期 RT-2、OpenVLA 等使用 per-dimension, per-timestep binning 的 action discretization。第 7 讲已经讨论过，这种 naive tokenization 对高频灵巧控制有局限。

## 训练 VLA 的实践细节

### 收敛指标

对 autoregressive VLA，一个实用经验是跟踪 action token accuracy。课程给出的经验阈值是约 95%：在 teacher forcing 下，动作 token 预测准确率没有到这个水平前，真实 rollout 通常不会好。

但 action token accuracy 不等于真实任务成功率。还应该同时看：

- L1 action error。
- L2 action error。
- detokenize 和 unnormalize 后的连续动作误差。

### 异质输入 batching

不同机器人相机数不同，不能直接 stack 成 batch。常见做法有两种：

1. padding + attention mask：补齐到 batch 中最大输入数，简单但会浪费大量 FLOPs。
2. sequence packing：把多个样本 token 拼成一个长序列，用 block-diagonal attention mask 防止不同样本互相 attend，并在样本边界重置 positional encoding。

sequence packing 更高效，但实现复杂，尤其要保证 attention mask 和位置编码正确。

### 数据加载和 Shuffle Buffer

大规模机器人预训练中，data loading 会显著影响结果。

一种常见方案是 sequential reads + shuffle buffer：从 shards 顺序读数据进 buffer，再从 buffer 中采样 batch。如果 shuffle buffer 太小，batch 会高度相关，影响泛化。

Octo 训练中曾发现，20,000 样本级 shuffle buffer 不够；通过在解码图像前跨轨迹打散 frames，将 buffer 提升到 500,000 样本后，性能显著改善。

另一种方案是真随机读取：维护全数据集索引，随机 seek 取样。它更接近 unbiased randomness，但需要很强 IO，否则会成为训练瓶颈。

### 不同 Action Spaces

处理不同 embodiment 的 action space 有两类常见策略：

- CrossFormer style：共享 backbone，不同 embodiment 使用不同 action head。优点是输出维度和 chunk 长度灵活；缺点是新增 embodiment 需要新增 head。
- Pi-Zero style：共享 backbone 加单一 action expert，输出统一 padded action space。优点是架构简单；缺点是固定 action chunk/维度，模型需要从输入观测、proprioception 和语言中隐式推断当前 embodiment。

在 CrossFormer 中，用户显式指定使用哪个 head；在 Pi-Zero style 中，模型通常靠相机视角、proprioception 和任务语言自动识别要控制的机器人。

## Pi-0.5 风格 VLA Recipe

课程总结了当前很多 state-of-the-art VLA 采用的训练思路：

- VLM backbone 用 next-token prediction 在 web data 和 FAST-tokenized robot actions 上共同训练。
- web data 用来保持 VLM 的语义能力，减少 catastrophic forgetting。
- action expert 使用 flow matching head 生成连续动作。
- action expert 读取 VLM backbone，但通过 stop-gradient 阻止 action loss 更新 backbone，从而保护预训练知识。

这种“知识隔离”让 VLM 语义能力和机器人动作生成更稳定地结合。

## 评估：为什么需要 Simpler

真实机器人评估非常昂贵且脆弱：

- 相机被碰一下就可能改变视觉分布。
- 电机或硬件维修会改变 proprioception。
- rollout 时间长，低频动作慢。
- 不同实验室难以复现同一套硬件环境。

因此，通用机器人策略需要更可扩展的评估。Simpler 的目标是用 simulation 评估真实世界训练出来的 policy，并让模拟中的相对排名与真实世界相关。

### Simpler 的关键步骤

1. 缩小 control gap：通过少量真实样本做 system identification，让仿真和真实低层控制器的末端执行效果尽量一致。
2. 缩小 visual distribution shift：用 green screen、真实背景 overlay、真实物体纹理投影等方法，让仿真图像更接近真实环境。

Simpler 不保证绝对成功率一致。它不能说“仿真中 55% 成功，真实也会 55%”。它提供的是 pairwise ranking correlation：如果 policy A 在 Simpler 中比 policy B 好，那么真实世界里大概率也更好。

后续工作如 Polaris、Robot Arena Infinity、RoboLab 等都在继续探索 real-to-sim correlated evaluation。

## 本讲主线

Lecture 9 讲的是从 specialist robot policies 走向 generalist robot policies 所需的三件事。

第一是数据：Open X-Embodiment 把分散在各实验室的机器人数据整理成统一格式，让跨数据集 co-training 成为可能。第二是建模：Octo、CrossFormer 和 VLA/Pi 系列展示了如何把异质观测、语言和动作放进 transformer 框架，并处理多 embodiment、不同 action space 和高频控制。第三是评估：真实机器人评估太昂贵，Simpler 这类 correlated simulation evaluation 让通用策略研究更可复现、更可扩展。

这讲的核心结论是，通用机器人策略不是单靠“更大模型”就能得到。数据格式、动作表示、输入 batching、shuffle 策略、normalization、跨 embodiment head 设计和评估协议，都会直接决定模型是否真正能泛化。
