---
title: "Lecture 10-Embodied Reasoning and Test-time Scaling"
publishDate: 2026-06-09
description: '具身推理与测试时扩展：从 Embodied Chain-of-Thought 到 GRPO、视觉推理和机器人规划'
tags:
  - AI
  - Robot Learning
language: '中文'
---
## 课程说明

课程资源

- 课程网站: [Robot learning](http://cvg.ethz.ch/lectures/Robot-Learning/)
- 课程作业网站: [mees-robot-learning-course/ethz-course-2026](https://github.com/mees-robot-learning-course/ethz-course-2026)

本文整理 Lecture 10: Embodied Reasoning and Test-time Scaling。具身推理、测试时计算扩展和机器人策略训练相关内容。

## 为什么只扩机器人数据还不够

过去几讲讨论了 VLA、generalist robot policies 和大规模机器人数据。一个自然想法是：只要继续收集 teleoperation data，像 LLM 一样扩大数据和模型，机器人就会越来越强。

课程强调，这不是完整答案。数据当然重要，它提供了行为先验，把演示分布压缩进模型权重。但数据本身不等于智能推理。即使最强的 VLA 遇到训练分布外场景，通常也缺少“想办法解决”的机制，只会按已有模式失败。

LLM 的经验也类似：即使在互联网文本上训练，模型仍不能自动解决所有复杂问题。近年的突破不是只靠更多预训练数据，还包括专门训练 reasoning，让模型能组合已有知识、分解问题、检查中间步骤，并在测试时投入更多计算。

机器人也需要类似能力：不是只复制看过的动作，而是能在未见过的物理情境中重新组合已有技能和知识。

## Embodied Chain-of-Thought

Embodied Chain-of-Thought（ECoT）的核心思想是：不要让 VLA 直接从图像和语言指令跳到动作，而是在中间加入具身推理步骤。

普通 VLA：

$$
\text{image} + \text{instruction} \rightarrow \text{actions}
$$

ECoT：

$$
\text{image} + \text{instruction}
\rightarrow
\text{grounded reasoning}
\rightarrow
\text{actions}
$$

这些中间 reasoning 可以包括：

- 子任务计划，例如先接近目标物体，再抓取，再移动。
- 目标物体的 bounding box。
- 末端执行器未来移动轨迹。
- 与图像中物体、空间关系和任务目标相关的 grounded annotations。

它相当于把语言模型中的 chain-of-thought 放进物理世界：让模型在行动前先“看清楚”和“想清楚”。

## Reasoning 数据从哪里来

大规模机器人数据不可能都由人手工标注 reasoning trace。ECoT 的做法是用互联网规模 foundation models 自动标注已有机器人数据：

- 用 VLM 识别与任务相关的物体。
- 生成 bounding boxes 或 grounding。
- 推测接下来要交互的目标。
- 标注 gripper 未来运动轨迹或子任务描述。

然后训练 autoregressive VLA，让它先预测这些 grounded reasoning tokens，再预测 action tokens。动作 token 因此条件化在中间推理上，模型被迫在输出动作前显式处理任务结构。

## ECoT 的收益

课程中提到的结果：在 distractor objects、spatial relations、unseen objects、unseen instructions 等 out-of-distribution 场景中，ECoT 相比同架构无 reasoning 的 OpenVLA 有约 30% 性能提升，并且能超过更大的 RT-2-X。

关键点是：没有收集额外机器人 teleop 数据，只是用 reasoning supervision 改变训练方式，就提高了泛化。

ECoT 还有两个额外好处：

- interpretability：人能看到模型预测的子任务、框、轨迹，从而理解失败原因。
- interactiveness：如果测试时模型推理链错了，人可以在线编辑 reasoning tokens，因为后续 action tokens 条件化在这些 tokens 上，动作会随之改变。

一个有趣现象是，ECoT 的 reasoning 能泛化到未见过的机器人。虽然训练 reasoning 只来自某个 WidowX 数据集，模型在 Open X-Embodiment 的其他机器人、其他视角上也能生成合理 reasoning。这说明很多 embodied reasoning 子任务与 VLM 预训练任务相似，VLM 已有部分视觉和空间知识，ECoT 更像是在解锁这些能力。

## Reasoning 的代价

显式 reasoning 很慢。原始 OpenVLA 约每秒输出 4 个动作；加入完整 ECoT 后，由于要生成大量 reasoning tokens，可能变成几秒一个动作。对机器人控制来说，这很难接受。

因此出现一个更基本的问题：ECoT 为什么有效？如果知道原因，就能保留收益、减少推理延迟。

课程列出三种可能机制：

### 1. 更好的表征学习

reasoning traces 可能提供额外监督，让模型内部表征更关注目标物体、空间关系和运动意图。如果主要收益来自训练时学到的表征，那么测试时不一定需要真的生成 reasoning。

### 2. 更好的学习课程

让 VLM 直接从图像预测连续机器人动作很困难，因为动作预测与 VLM 预训练任务差异很大。reasoning 可能提供一种 curriculum：先学习物体位置、运动描述、子任务计划等更接近 VLM 原能力的任务，再逐渐对齐到动作预测。

如果这个解释成立，reasoning 可以作为 training scaffold，部署时移除。

### 3. 更多 token 带来的表达能力

Transformer 可能把中间 tokens 当作 scratch space 使用。即使 token 语义不强，只要提供了更多 autoregressive 计算步骤，就可能提升表达能力。这在 LLM 中也有研究支持。

问题是：ECoT 的收益到底来自语义 grounding，还是来自更多 test-time compute？

## ECoT-Light

ECoT-Light 通过实验区分这些机制，并试图降低测试时成本。

### Reasoning Pretraining

两阶段训练：

1. 先用 VLM/VLA 只预测 reasoning，不训练 action loss。
2. 再用这个模型初始化，fine-tune 到动作预测，但测试时不生成 reasoning。

这样 reasoning 的好处被内化到表示中，部署时延迟接近普通 VLA。

### Reasoning Dropout

在单一联合训练中同时预测 reasoning 和 actions，但训练时以一定概率 dropout reasoning，使模型学会在有 reasoning 和无 reasoning 两种情况下都能预测动作。

测试时可以按 latency budget 选择：

- 打开 reasoning：更强泛化，更慢。
- 关闭 reasoning：接近普通 VLA 的速度，但保留一部分训练收益。

### 实验结论

在 LIBERO-90 和真实机器人实验中，reasoning pretraining 和 reasoning dropout 在不生成 reasoning 的情况下接近完整 ECoT 的性能，同时保留标准 VLA 的推理频率。

选择取决于场景：

- 完整 ECoT：适合不严格实时、需要最高泛化或可解释性的场景。
- reasoning pretraining：适合 reasoning 数据和 action 数据不完全配对的情况，例如从人类视频或仿真中迁移 reasoning。
- reasoning dropout：最灵活，一个模型可按需要开启或关闭 reasoning。

## Test-Time Compute Scaling

前半讲讨论的是固定数量 reasoning tokens。后半讲的问题是：能不能让模型在难题上想更久？也就是 test-time compute scaling。

这与经典规划有相似之处，但对象从显式状态空间逐步转向神经模型的隐式分布：

- A*：结构化离散搜索空间和手写 heuristic。
- MCTS：可模拟环境和价值估计。
- LLM/VLM test-time scaling：在模型输出分布中搜索、采样、验证和重试。

每一步都减少显式结构，增加开放性和通用性。

### AlphaGo 的启示

AlphaGo 是经典例子。raw policy network 已经很强，但加入 Monte Carlo Tree Search 后 Elo 大幅提升。课程强调的直觉是：测试时搜索可以替代极大规模的模型扩展。经验上，提升约 120 Elo 可以来自模型规模翻倍，也可以来自测试时搜索预算翻倍。

在围棋这类任务中，至今强系统仍依赖 test-time search，而不是只靠更大网络一次前向传播。

### 语言中的 Test-Time Scaling

语言任务没有围棋那样完美 verifier，但 test-time compute 仍有效。相关研究表明，在 FLOPs 匹配下，训练较小模型并让它在推理时多采样、多检查，可能超过一个约 14 倍大的模型。

但这有边界：test-time compute 放大的是模型已经知道或能偶尔生成的正确答案。如果正确答案完全不在模型分布中，搜索再多也找不到。对更难问题，单纯 test-time search 可能不如更强 base model。

## 为什么 Reasoning Tokens 有计算意义

理论上，固定层数 Transformer 的一次 forward pass 只有固定的顺序计算深度。生成 autoregressive reasoning tokens 相当于把计算展开到多个步骤。

一个直觉结论是：任何需要 `T` 个顺序计算步骤的问题，可以由固定大小 Transformer 通过生成 `O(T)` 个 reasoning tokens 来解决。没有中间 token 时，则可能需要更大模型在一次 forward pass 中“压缩”这些计算。

这解释了为什么“想更久”不是纯粹比喻，而是真正增加了模型可用的顺序计算预算。

## Chain-of-Thought Prompting 到训练 Reasoning

原始 chain-of-thought prompting 很简单：在 prompt 中写几个带推理步骤的示例，让模型在新问题上模仿这种格式。

但它有几个限制：

- 对 prompt 很敏感。
- 需要人工写示例。
- 小模型上不一定有效，甚至可能伤害性能。
- 通常要模型足够大、足够强，reasoning 才会涌现。

因此社区转向“训练模型推理”，而不是只靠 prompt 诱导。

## 用 RL 训练 Reasoning

DeepSeek-R1 类工作展示了 RL 可以让模型自己发现 reasoning 行为。模型不是模仿人类写好的推理格式，而是在可验证任务上通过奖励学习：如果更仔细推理能提高正确率，RL 就会强化这种行为。

### GRPO

GRPO（Group Relative Policy Optimization）和 PPO 有关系，但去掉了 critic/value function。它适合语言模型，因为训练一个 value model 评估 token sequence 很昂贵。

GRPO 的做法：

1. 对同一个问题采样一组回答。
2. 用 verifier 给每个回答打分。
3. 计算 group-relative advantage：

$$
A_i = \frac{r_i-\text{mean}(r)}{\text{std}(r)}
$$

4. 用类似 PPO 的 clipped objective 更新策略。

它不需要人类标签、不需要 reward model，也不需要 value function。只要有可验证奖励，例如数学答案对错，模型就能通过比较自己的一组输出学习。

关键前提是 base model 必须够强，至少偶尔能产生正确答案。否则没有成功样本可强化，RL 无法启动。

### Reasoning 如何涌现

DeepSeek-R1 中的一个现象是：训练早期模型几乎不会说“wait”之类的自我反思词；到某个训练阶段后，模型发现停下来重新检查能提高奖励，于是这类反思语言频率突然上升。

这说明 RL 可以发现监督学习没有显式指定的策略。不是人告诉模型“你应该反思”，而是反思作为提高奖励的有效策略被强化出来。

## 从语言推理到视觉推理

GRPO 不关心 verifier 是什么，只要奖励可验证。因此它也能用于视觉任务。

Visual RFT 是一个例子：模型需要预测 bounding box，verifier 使用 IoU（intersection over union）作为奖励。模型在 thinking block 中进行视觉推理，再输出框；如果更好的视觉推理带来更高 IoU，就会被强化。

但视觉和多模态任务有一个重要失败模式：只奖励最终答案可能允许 hallucinated reasoning。模型可以在 reasoning block 中胡乱描述场景，但最终 box 恰好正确，仍然得到奖励。数学中“错推理得到对答案”相对少见，视觉中这种情况更容易发生。

## 验证中间推理：Argos

Argos 的思路是：不只奖励最终答案，也奖励中间 reasoning 是否 grounded。

它会为不同样本选择合适 verifier，例如 foundation models 或几何指标，同时检查：

- 最终答案是否正确。
- 相关物体是否被正确定位。
- reasoning 中的空间关系是否与图像/视频一致。
- 视频中的时序 grounding 是否正确。

这样模型不能靠 hallucinated reasoning 获得高奖励，因为推理链本身也会被验证。

这对 multimodal agents 很关键：只验证 outcome 容易得到表面正确但过程不可信的模型；验证 reasoning trace 才能让模型学会真正 grounded 的推理。

## 回到机器人：两层架构

在 embodied AI 和机器人中，test-time reasoning 对高层规划很有帮助，例如任务分解、导航、ALFRED 类模拟任务等。RL-trained reasoning 在困难高层任务上能显著超过 supervised chain-of-thought。

但低层连续控制不能每个控制步都生成完整推理链。机器人需要实时、平滑、高频动作。因此课程给出的趋势是两层结构：

- 高层：RL-trained reasoning，用于任务分解、规划、验证和长时域决策。
- 低层：ECoT-Light 式 reasoning internalization，用 reasoning 训练表征，但部署时关闭推理链，实现快速 reactive control。

也就是说，机器人需要会“想”，但不应该在每个电机控制步都慢慢写一篇推理。

## 本讲主线

Lecture 10 的核心是把 scaling 从训练时扩展到测试时。预训练数据和模型规模提供行为先验，但复杂机器人任务还需要推理、分解、搜索和验证。

本讲有四个关键结论：

- test-time compute 是新的扩展轴，在很多任务上比单纯增大模型更有效。
- 连续控制不必在部署时支付完整 reasoning 成本，可以训练时 internalize reasoning，测试时关闭。
- RL 可以让 reasoning 涌现，因为推理本身可能是最大化奖励的有效策略。
- 多模态和机器人系统不能只验证最终结果，也要验证中间 reasoning 是否 grounded。

真正开放的问题是：如何把高层可验证推理、测试时搜索、低层快速控制和真实世界反馈整合进同一个机器人策略，使机器人不仅能模仿和泛化，还能在未见过的物理情境中思考、修正和改进。
