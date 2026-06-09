---
title: "Lecture 8-World Models"
publishDate: 2026-06-09
description: '世界模型：从 action-conditioned prediction 到 Dreamer、video world model 与 JEPA'
tags:
  - AI
  - Robot Learning
language: '中文'
---
## 课程说明

课程资源

- 课程网站: [Robot learning](http://cvg.ethz.ch/lectures/Robot-Learning/)
- 课程作业网站: [mees-robot-learning-course/ethz-course-2026](https://github.com/mees-robot-learning-course/ethz-course-2026)

本文整理 Lecture 8: World Models。world models 的定义、算法脉络和机器人应用。

## 什么是 World Model

课程用棒球击球作为直觉例子：快速球从投手到本垒只需要约 400ms，而视觉信号从眼睛到大脑就要约 100ms。击球手不可能完全依赖实时感知反应，而是要根据内部模型预测球将到达哪里，并提前发起动作。

world model 的目标就是让 agent 具备类似能力：根据当前状态、动作和历史，预测接下来会发生什么。

策略和 world model 问的问题不同：

- policy 问：给定当前状态和目标，我应该采取什么动作？
- world model 问：给定当前状态和动作，下一状态会是什么？

形式上，world model 建模的是：

$$
p(s_{t+1}|s_t,a_t,\text{history})
$$

其中 history 很重要，因为单帧只能告诉我们物体在哪里，历史才能告诉我们物体正在往哪里走。

## 为什么机器人需要 World Model

如果机器人有可靠的 dynamics model，就可以在真实执行前先“想象”多个未来：

1. 在 world model 中 rollout 多个候选动作序列。
2. 预测每条轨迹的未来状态和奖励。
3. 选择最好的计划。
4. 只在真实世界中执行被选中的动作。

这比真实试错便宜得多，也更安全。

world model 可以看作 learned simulator。传统物理仿真器由专家手写刚体动力学、摩擦、碰撞几何等规则，速度快但只对被显式建模的现象准确；world model 直接从交互数据、机器人轨迹或视频中学习 transition distribution，因此理论上能覆盖难以手写的软物体、接触、变形和复杂视觉动态。

但它也受训练数据限制：如果数据中没有某类物理现象，模型不会凭空学会。

严格说，一个有用的机器人 world model 应该是 action-conditioned 的。纯文本到视频模型可以生成视频，但如果不能回答“我在当前状态执行这个动作会怎样”，它就不能直接用于规划和控制。

## Pixel-Space Action-Conditioned World Models

早期例子是 Chelsea Finn 等人的 action-conditioned video prediction。直接预测下一帧每个像素很难，因此模型不从零生成整张图，而是预测 flow field：给定当前图像和动作，预测每个像素会移动到哪里，再 warp 当前图像得到下一帧。

这样做的好处是把问题从“生成所有像素”变成“建模运动”，更结构化，也更容易泛化到未见过的物体。

### Visual Foresight 与 Visual MPC

Visual Foresight 把这种 pixel-space world model 用于控制。用户可以指定视觉目标，例如点击图像中某个像素，表示希望把物体从当前位置移到目标位置。控制器使用 visual MPC：

1. 用 CEM 采样多个动作序列。
2. 用 world model 预测每个动作序列导致的视频。
3. 根据目标像素或图像 classifier 给候选轨迹打分。
4. 执行最优序列的第一部分，再重新规划。

该方法的训练数据可以来自机器人自主随机交互，不需要人工标注。因为模型学习的是 motion/flow，也更容易对新物体泛化。

### Pixel-Space 的局限

pixel-space world model 有明显问题：

- L2 loss 会导致模糊预测，因为模型对不确定未来取平均。
- 长时域 rollout 会累积误差并逐渐漂移。
- 每个候选动作序列都要预测完整图像序列，测试时计算昂贵。
- 很多背景像素对控制没有用，但模型仍要花容量重建它们。

这推动了 latent-space world model：把观测压缩到控制相关 latent，再在 latent 中预测和规划。

## Latent World Models

latent world model 的基本结构是：

1. encoder 把图像观测压缩成 latent state。
2. dynamics model 在 latent space 中预测下一 latent。
3. recurrent hidden state 维护历史记忆。
4. reward model 在 latent space 中预测 imagined reward。
5. policy 在 imagined latent rollouts 中训练。

只要 world model 足够好，策略就可以在 imagination 中训练，减少真实机器人交互。

### World Models: VAE + RNN + Controller

早期 World Models 架构包含三个模块：

- vision model：VAE，把图像压缩成 latent `z`。
- memory model：RNN/MDN，根据历史 latent 预测未来 latent distribution。
- controller：一个小的全连接网络，根据 `z` 和 RNN hidden state 输出动作。

VAE 的重建不必像真实图像一样完美，只要 latent 包含足够控制信息即可。memory model 使用 mixture density network，是因为未来可能多模态，单高斯会平均掉多个可能未来。

一个重要设计思想是：world model 做表示和预测的重活，controller 可以很小。原始工作中 controller 甚至用类似 CEM 的黑箱优化训练，而不是端到端反向传播。

### 分布偏移问题

这种早期结构有一个核心问题。训练时，RNN 输入来自 VAE 对真实观测编码出的 latent；想象时，没有真实观测，RNN 只能把自己预测出的 hallucinated latent 再喂给自己。这造成 distribution mismatch，长 rollout 容易漂移。

问题本质是：VAE 独立编码真实图像，但没有学习“给定当前记忆，当前 latent 应该落在哪里”的 prior。

## RSSM 与 Dreamer

### Recurrent State-Space Model

RSSM（Recurrent State-Space Model）通过把 latent state 拆成 deterministic 和 stochastic 两部分解决这个问题：

- `h_t`：deterministic hidden state，通过 RNN 传递历史记忆，不被采样污染。
- `z_t`：stochastic latent，表示当前状态的不确定性。

训练时，从真实观测推断 posterior：

$$
q(z_t|h_t,o_t)
$$

想象时，从 learned prior 采样：

$$
p(z_t|h_t)
$$

用 KL loss 约束 prior 接近 posterior，使 imagined latents 落在模型理解的区域内。这样在 closed-loop imagination 中，不需要真实观测也能稳定 rollout。

### PlaNet 到 Dreamer

PlaNet 使用 RSSM，但仍在每个真实控制步骤用 CEM 做在线规划：采样大量动作序列，rollout，refit Gaussian，再执行。这样测试时很慢。

Dreamer 的想法是把这种规划 amortize 成一个 learned policy：

1. 用真实数据训练 world model。
2. 冻结 world model。
3. 在 latent imagination 中 rollout。
4. 通过可微 RSSM dynamics 训练 actor-critic。
5. 部署时直接运行 actor，不再每一步做 CEM 搜索。

这使得 planning 的计算成本从测试时转移到训练时。

### DayDreamer

DayDreamer 把 Dreamer 用到真实机器人上。同一套算法和超参数可以在真实硬件上训练，例如四足机器人从零学习行走。课程中特别强调了一个结果：约 1 小时真实交互就能学会四足 locomotion，因为 world model 让策略能在想象中训练相当于很多天的经验。

真实硬件上的工程关键包括：

- policy 以实时控制频率运行。
- world model 在后台异步训练。
- 数据采集和模型更新并行进行。

### Dreamer 系列演进

Dreamer 系列大致脉络：

- World Models：VAE + RNN + controller，证明 imagination training 可行。
- PlaNet/RSSM：引入 recurrent state-space model，修正 imagined latent drift。
- Dreamer V1：用 actor-critic 替代测试时 CEM。
- Dreamer V2/V3：改进 categorical latents、KL balancing 等训练细节。
- Dreamer V3：能从 pixels 在 Minecraft 中获得钻石，任务时域超过两万步。
- Dreamer V4：进一步支持 offline setting，并引入无动作视频预训练和 transformer KV cache。

Dreamer V4 的两个关键变化：

- dynamics model 可先在无动作视频上预训练，再用 action data fine-tune 成 action-conditioned model。
- 用 transformer KV cache 替代固定 RNN hidden state，获得更长程、更强的记忆能力。

## 从 Domain-Specific Latent 到 Video World Model

Dreamer 类 latent 通常是 domain-specific：在 Minecraft、DM Control 或某个机器人环境中学到的 latent 不一定能迁移到其他世界。

如果想要一个更通用的 world model，就需要从更大规模的视频数据中学习视觉世界动态。这引出 generative video models。

### Video Tokenization

视频 tokenization 的困难是 token 数爆炸。一个 `256 x 256` 视频，如果每帧按 `16 x 16` patch tokenization，一帧约 256 tokens；10 秒 20 FPS 就超过 50,000 tokens。

视频高度冗余，因此压缩有三条轴：

- spatial compression：用更深 encoder、更大 stride 或 VQ-style tokenizer 压缩空间维度。
- temporal compression：把多帧合并成一个 token 或一个 3D patch。控制场景常需要 causal temporal compression，不能偷看未来。
- adaptive compression：简单静态片段用少量 tokens，复杂动态片段分配更多 tokens。

Cosmos 等模型会组合空间和时间压缩，例如同时做 8 倍空间压缩和 8 倍时间压缩，把原本 5 万级 tokens 的短视频压到约百级 tokens。

## Actions 在 Video Model 中的位置

互联网视频通常没有机器人动作标签。要把视频模型用于机器人控制，关键问题是：动作在哪里？

### World Action Models

WAM（World Action Model）把动作作为输出。模型联合预测未来视频和产生这些变化的动作。和 Dreamer 不同，Dreamer 的动作是输入条件；WAM 中动作是模型要生成的一部分。

Dream Zero 是代表之一：它用一个 causal DiT 接收视频 latents、noisy action tokens、proprioception 和语言，使用 flow matching 联合 denoise 视频和动作。推理时 autoregressively 生成未来视频 chunk 和对应动作。由于是闭环控制，执行动作后可以用真实新观测替换 KV cache 中生成的帧，从而减少纯视频生成中的 compounding error。

### Video Action Models

VAM（Video Action Model）通常冻结一个大规模互联网视频预训练 backbone，再在其特征上训练轻量 inverse dynamics model，把视觉动态特征映射到机器人动作。

Mimic Video 是这个思路的例子。视频 backbone 承担语义和动态理解，action head 只需要学习低维机器人控制映射，因此需要的机器人数据更少。

这种路线的动机是：传统 VLA 多从静态 image-text VLM 出发，VLM 有语义知识，但没有视觉动态和物理因果。机器人 fine-tuning 需要用昂贵 teleop 数据补上这些动态知识。视频 backbone 则从一开始就有运动和因果线索，fine-tuning 更像是学习“把视觉计划翻译成机器人动作”。

Mimic Video 的 oracle 实验表明，如果 action decoder 能看到真实未来帧的 video latents，动作预测几乎完美。这说明瓶颈更多在 video prediction quality，而不是 action decoding。随着视频模型变好，下游机器人表现也会随之提升。

## WAM 与 VAM 的取舍

不同 world model 路线的取舍可以从几个维度看：

- action-conditioned world model 可以使用成功、失败、play data 等更宽泛数据，并能在 imagination 中做 RL，但跨 embodiment 和大规模预训练较难。
- WAM/VAM 继承互联网视频 backbone，跨 embodiment 更自然，因为视频模型本身与机器人形态无关。
- VAM 保留预训练 video backbone 能力较好，但控制和 world model 分离。
- WAM 把视频和动作放入一个模型，耦合更紧，但训练和实时推理更复杂。

## JEPA：不重建像素的 World Model

前面的方法大多某处有 decoder，需要预测或重建像素。JEPA（Joint Embedding Predictive Architecture）问的是：能不能只预测未来 embedding，而不预测未来像素？

基本做法：

1. encoder 编码当前帧。
2. predictor 根据当前 embedding 和动作预测下一帧 embedding。
3. 同一个或 target encoder 编码真实下一帧。
4. 最小化 predicted embedding 和 target embedding 的距离。

没有 decoder，没有 pixel loss，也不浪费容量建模无关像素。

核心风险是 representation collapse：如果 encoder 把所有帧都映射到同一个向量，预测误差可以 trivially 为零。

不同 JEPA 系列工作用不同方法避免 collapse：

- V-JEPA 2：使用 EMA target encoder，让 predictor 追逐一个缓慢移动的目标。
- DINO world model：使用预训练 encoder，但牺牲部分端到端学习能力。
- 更新的 world model 方法：用 Gaussian regularizer 迫使 latent distribution 展开，避免常量解。

## 本讲主线

Lecture 8 的核心问题是：机器人能否在行动前先预测行动后果？

课程从 pixel-space action-conditioned video prediction 开始，说明 world model 如何用于 visual MPC；再转向 latent world model，介绍 RSSM 和 Dreamer 如何让策略在 imagination 中训练；随后扩展到 internet-scale video world models，讨论 WAM、VAM 和 video backbone 对机器人控制的价值；最后引出 JEPA，说明不重建像素、只预测 embedding 的路线。

这些方法都在回答同一个问题：如何表示状态、动作和未来，使机器人能在真实执行前先在内部模型中“试一遍”。world model 的价值不只是生成视频，而是为规划、样本效率和泛化提供一个可查询的未来。
