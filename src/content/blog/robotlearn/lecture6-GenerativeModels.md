---
title: "Lecture 6-Generative Models"
publishDate: 2026-06-09
description: '生成模型在机器人学习中的作用：从 VAE、VQ-VAE 到 diffusion policy 与 flow matching'
tags:
  - AI
  - Robot Learning
language: '中文'
---
## 课程说明

课程资源

- 课程网站: [Robot learning](http://cvg.ethz.ch/lectures/Robot-Learning/)
- 课程作业网站: [mees-robot-learning-course/ethz-course-2026](https://github.com/mees-robot-learning-course/ethz-course-2026)

本文整理 Lecture 6: Generative Models。

## 为什么机器人学习需要生成模型

机器人策略中的观测到动作映射通常不是一个单值函数，而是一个分布。原因有两类：

- 机器人本身有冗余。一个 7 自由度机械臂可以用很多不同关节构型到达同一个末端位姿。
- 专家演示天然不一致。不同人、不同时间、不同习惯都可能给出有效但不同的轨迹，例如同样是抓取物体，有人从侧面抓，有人从上方抓。

如果用 deterministic policy 加 MSE 做 behavior cloning，模型会被迫预测所有演示的平均动作。对多模态任务来说，这个平均值可能根本不是有效动作：左绕树和右绕树的平均可能正好撞树。生成模型的作用就是表示这种高维、多峰的动作分布，而不是只输出一个均值。

从统一视角看，生成模型学习的是一个从简单分布到复杂数据分布的变换。简单分布通常是标准高斯噪声，复杂数据分布可以是图片、视频，也可以是机器人动作序列。

## Autoencoder 与 VAE

### Autoencoder

autoencoder 由 encoder 和 decoder 组成：

- encoder 把输入 `x` 压缩成 latent code `z`。
- decoder 从 `z` 重建原始输入。
- 训练目标通常是重建误差，例如 L2 loss。

瓶颈结构会迫使 encoder 保留最重要的信息，因此 autoencoder 可以做自监督压缩。但普通 autoencoder 有三个关键限制：

- latent space 没有结构，encoder 可以把不同样本随意放在空间中。
- 唯一约束来自 bottleneck 的维度，不告诉模型 latent 应该如何组织。
- 它是 deterministic 的，每个输入只映射到一个点，测试时不能自然地从 latent prior 中采样新数据。

### Variational Autoencoder

VAE 把 deterministic latent point 改成 latent distribution。encoder 不再输出单个 `z`，而是输出一个高斯分布的参数，例如 `mu` 和 `sigma`，然后从中采样：

$$
z \sim q_\phi(z|x)
$$

训练目标不是直接最大化不可 tractable 的 `log p_theta(x)`，而是最大化 evidence lower bound（ELBO）：

$$
\log p_\theta(x) \ge
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
-D_{KL}(q_\phi(z|x)||p(z))
$$

其中：

- reconstruction term 要求 decoder 能从 sampled `z` 重建输入。
- KL regularizer 要求 approximate posterior 接近高斯 prior。

这样做的好处是，测试时可以直接从 prior `p(z)` 采样，再用 decoder 生成有意义的输出。

### Reparameterization Trick

VAE 中的采样操作本身不可微。解决方法是把随机性移到外部噪声里：

$$
z = \mu + \sigma \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
$$

这样从梯度视角看，`mu` 和 `sigma` 仍处在确定性计算图中，梯度可以从 decoder 传回 encoder。与 REINFORCE/score function estimator 相比，reparameterization trick 的方差更低，因为梯度能直接看到 `z` 如何随 encoder 参数变化。

### VAE 的问题

VAE 也有两个常见失败模式：

- posterior collapse：decoder 太强时，可以直接建模 `p(x)`，忽略 latent `z`。
- prior mismatch：训练数据只覆盖了 latent 空间中的某些区域，但测试时从高斯 prior 采样可能落在演示之间的空白区域，decoder 会在没见过的 latent 上生成不可靠输出。

对机器人来说，prior mismatch 特别危险，因为 hallucinated action 可能意味着不安全的硬件动作。

## VQ-VAE

VQ-VAE 用离散 codebook 替代连续高斯 latent。它维护 `K` 个可学习 code vectors。encoder 先输出连续向量，再通过 nearest neighbor lookup 选取最近的 codebook vector，decoder 只看到这个离散 code。

直觉上，VQ-VAE 把连续信号转成离散 token：

1. encoder 输出连续 latent。
2. quantization 选择最近的 codebook entry。
3. decoder 根据离散 code 重建输入。

这种方式缓解了 posterior collapse 和 prior mismatch，因为模型被迫选择 codebook 中的离散模式，而不是从连续高斯空间中采样任意点。

### Straight-Through Estimator

nearest neighbor lookup 是 `argmin` 操作，几乎处处不可导。VQ-VAE 用 straight-through estimator：前向传播照常量化，反向传播时假装量化没有发生，把 decoder 输入处的梯度直接传回 encoder 输出。

这里会用到 `stop_gradient`：

- forward pass 中像 identity。
- backward pass 中梯度为 0。

这让 codebook、encoder 和 decoder 可以共同训练。

### VQ-VAE Loss

VQ-VAE 的训练目标通常有三项：

- reconstruction loss：让 decoder 重建输入。
- codebook loss：把 codebook vectors 拉向 encoder 输出，学习数据的离散词表。
- commitment loss：让 encoder 输出贴近选中的 codebook vector，避免 encoder 飘到离 codebook 很远的位置。

### 如何生成新样本

训练好 VQ-VAE 后，还需要第二阶段训练 prior model。具体做法是：

1. 用 encoder 把数据集转换成 code index sequences。
2. 训练一个 prior model，例如 autoregressive transformer，建模这些离散 code 序列。
3. 生成时先从 prior model 采样 code indices，再查表得到 code vectors，最后用 decoder 生成图像、视频或动作。

VQ-VAE 的核心意义是把连续观测、视频或动作转成 token，使 transformer 可以在统一 token 空间中建模机器人行为。

### 局限

VQ-VAE 仍有局限：

- codebook collapse：很多 code 在训练中从未被使用。
- codebook size 需要预先设定。
- 如果重建损失仍用 MSE，输出端仍可能有 mean-seeking 行为，例如图像模糊。

机器人学习中，VQ-VAE 常用于图像/视频 tokenization、latent action learning、latent plan quantization，以及把连续机器人动作离散成 VLA 可预测的 tokens。

## Diffusion Models

diffusion model 不从压缩 latent 空间生成，而是直接建模完整连续数据分布。它包含两个过程：

- forward process：从干净样本开始，逐步加入高斯噪声，直到信息几乎完全丢失。
- reverse process：训练网络逐步去噪，从噪声恢复数据。

关键思想是：不要一次性生成复杂样本，而是把生成拆成许多小的局部修正。

### Forward Process

每一步把当前样本缩小一点，再加入噪声：

$$
x_t=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t
$$

其中 `beta_t` 是 noise schedule，控制每一步注入多少噪声。通过高斯可加性，可以推导出从 `x_0` 直接采样任意噪声等级 `x_t` 的 closed form，不必真的迭代 500 或 1000 步：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
$$

这让训练可以高效地随机采样时间步。

### Reverse Process 与 Noise Prediction

反向分布原则上不可 tractable，因为同一个 noisy sample 可能来自无数干净样本。DDPM 的关键简化是：不直接预测 `x_{t-1}`，而是预测 forward process 中加入的噪声 `epsilon`。

网络输入 noisy sample 和 timestep，输出 predicted noise：

$$
\epsilon_\theta(x_t,t)
$$

训练目标就是简单的 MSE：

$$
\mathcal{L}=\|\epsilon-\epsilon_\theta(x_t,t)\|^2
$$

这比直接预测完整样本更容易，因为每一步只需要做一个小的局部去噪修正。

### DDPM Sampling 与 DDIM

DDPM 生成时从高斯噪声开始，从 `T` 逐步去噪到 0。每一步包含：

- deterministic denoising：根据预测噪声往 clean sample 方向移动。
- stochastic noise injection：再加一点噪声以保持样本多样性。

DDPM 通常需要很多步，例如 1000 步，推理很慢。DDIM 把训练时的时间步和推理时的时间步解耦，只选较少的子集时间步跳跃式去噪。DDIM 通常是 deterministic 的，因此能用更少步骤生成样本，更适合需要实时控制的机器人场景。

### Conditional Generation 与 Classifier-Free Guidance

机器人策略通常不是无条件生成动作，而是要根据当前观测、任务语言或目标图像生成动作。最简单做法是把 condition `c` 输入 noise predictor。

问题是模型可能学会忽略 condition，退化成 unconditional denoising。classifier-free guidance 的做法是：

1. 训练时以一定概率丢弃 condition，让同一个网络同时学习 conditional 和 unconditional denoising。
2. 推理时混合两种预测：

$$
\epsilon = \epsilon_{uncond} + w(\epsilon_{cond}-\epsilon_{uncond})
$$

`w` 是 guidance scale。它越大，生成越符合条件，但多样性可能下降。

## Diffusion Policy

Diffusion Policy 把 diffusion 用到机器人动作生成中。核心替换很直接：

- 图像生成里 denoise pixels。
- 机器人策略里 denoise action sequences。

训练时使用 DDPM loss，推理时常用 DDIM 加速。当前相机图像、机器人状态或任务信息作为 condition 输入 noise prediction network。模型直接输出未来一段动作序列，不需要预测未来图像。

一个重要效率细节是：observation embedding 通常只计算一次，然后在多步去噪中复用。原始 Diffusion Policy 使用 1D temporal U-Net，把动作序列看成时间上的一维信号；也有 transformer 版本，用 cross-attention 条件化动作 denoising。

Diffusion Policy 的优势是能自然表达多模态动作分布，在真实机器人任务中表现很强。限制是很多 diffusion policy 仍是 specialist model：对单任务或单数据分布拟合很好，但多任务、跨机器人、通用策略扩展更难。

后续工作包括：

- Octo：用大 transformer 处理多任务机器人数据，再接 conditional diffusion action head。
- Diffusion Transformer/DiT block policies：用 AdaLN-Zero 等方式替代 cross-attention，使条件注入更稳定，能扩展到更长时域、更灵巧的任务。

## Flow Matching

diffusion 通过复杂 noise schedule 逐步破坏数据并学习反向去噪。flow matching 换了一个视角：直接学习一个 velocity field，把噪声沿连续路径运输到数据。

最简单路径是 noise 和 data 之间的直线：

$$
x_t=(1-t)x_0+t x_1
$$

其中 `x_0` 是噪声，`x_1` 是数据。ground-truth velocity 是指向数据的向量。模型学习在任意时间 `t` 和位置 `x_t` 上预测这个速度。

生成时不再做概率意义上的 denoising，而是从噪声出发，用 ODE solver 沿 learned velocity field 积分到数据分布。

### 与 Diffusion 的关系

在某些线性 schedule 下，flow matching 和 diffusion 可以通过重参数化联系起来：

- diffusion 预测 noise。
- flow matching 预测 velocity。

flow matching 的优点是路径更直、更容易学、推理步数更少，并且少了复杂 noise schedule 的调参负担。因此它正在图像生成和机器人动作生成中快速取代 diffusion 成为常用框架。

### Rectified Flow

随机配对 noise sample 和 data sample 时，不同直线路径可能交叉。路径交叉会让同一点需要对应多个速度方向，velocity field 被迫弯曲，推理变慢。

rectified flow 通过迭代重配对解决这个问题：

1. 先训练一个 flow model。
2. 从噪声出发，用当前 flow 生成数据点，得到自然耦合的 noise-data pair。
3. 用这些耦合 pair 重新训练 flow。
4. 路径变得更直，交叉减少，积分步数也可以更少。

Pi-Zero 是机器人 VLA 中较早采用 flow matching 生成动作的代表之一。它把 VLM fine-tune 成能通过 flow matching 输出动作的模型，也让 flow matching 成为后续 VLA action head 的重要默认选择。一个实现细节是训练时可以偏向采样更 noisy、更困难的 flow time，让模型在最难预测的阶段学得更充分。

## 本讲主线

机器人动作预测天然是多模态分布建模问题。MSE 回归和 deterministic policy 会把多个有效行为平均掉，导致无效甚至危险动作。Lecture 6 介绍的三类生成模型给出了不同工具：

- VAE/VQ-VAE：学习 latent representation 或离散 token，把复杂连续信号压缩成可建模的结构。
- Diffusion：通过逐步去噪直接建模连续动作分布，适合多模态 action sequence generation。
- Flow matching：用 velocity field 连接噪声和数据，路径更直、采样更快，正在成为新一代 VLA action generation 的核心方法。

这些模型的共同目标不是生成漂亮图像，而是让机器人策略能够表示真实行为分布中的多个有效解，并在物理世界里选择可执行、稳定且符合任务条件的动作。
