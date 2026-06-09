---
title: "Lecture 4&5 — Reinforcement Learning"
publishDate: 2026-06-09
description: '强化学习 I/II：从 Bellman 方程到 DQN、PPO 与 SAC'
heroImage: { src: './image/lecture4-ReinforcementLearning/1780995526472.png', color: '#48B29E' }
tags:
  - AI
  - Robot Learning
language: '中文'
---

## 课程说明

本文合并整理 Robot Learning 2026 的 Lecture 4 和 Lecture 5。原始录音稿中的课堂通知、麦克风调试、掌声、论文讨论和与课程组织相关的内容没有纳入，只保留强化学习算法主线。

![1780995526472](image/lecture4-ReinforcementLearning/1780995526472.png)

## Part 1: 从模仿学习到强化学习

### 1.1 为什么需要强化学习

模仿学习从专家轨迹中学习策略，目标是让机器人复制专家在相同状态下会采取的动作。它的优势是直接、稳定，但根本限制也很明显：策略的上限被专家数据决定。如果数据质量不高、覆盖不全，或者任务需要发现比人类演示更优的行为，模仿学习本身没有机制突破这个上限。

强化学习（Reinforcement Learning, RL）把问题改成 agent 与环境交互：机器人自己试错，观察状态、执行动作、收到奖励，并通过最大化长期回报来学习策略。它不要求每一步都有专家告诉机器人该怎么做，而是通过奖励信号定义“什么是好行为”。

### 1.2 MDP 回顾

强化学习建立在 Markov Decision Process（MDP）之上。一个 MDP 通常包括：

| 组成                   | 含义                             |
| ---------------------- | -------------------------------- |
| 状态空间 `S`           | 环境可能处于的状态               |
| 动作空间 `A`           | agent 可执行的动作               |
| 转移函数 `P`           | 执行动作后环境如何进入下一状态   |
| 奖励函数 `r`           | 对当前行为的即时评价             |
| 折扣因子 `gamma`       | 平衡近期奖励与远期奖励           |

策略 `pi_theta(a | s)` 给出在状态 `s` 下选择动作 `a` 的概率。策略和环境交互会产生轨迹：

$$
\tau = (s_0, a_0, s_1, a_1, \ldots)
$$

轨迹概率由初始状态分布、策略概率和环境转移概率共同决定：

$$
p_\theta(\tau)=p(s_0)\prod_t \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)
$$

强化学习的目标是最大化策略诱导出的期望累计回报：

$$
J(\theta)=\mathbb{E}_{\tau \sim p_\theta(\tau)}\left[\sum_t \gamma^t r_t\right]
$$

### 1.3 不连续奖励为什么仍然可优化

机器人任务的奖励经常是不连续的。例如车留在山路上给 `+1`，掉下悬崖给 `-1`。这个奖励函数像阶跃函数，本身不可导，不能直接用梯度下降优化。

强化学习优化的不是单个奖励函数值，而是奖励在策略分布下的期望。随机策略会把“硬边界”的奖励函数平滑成关于策略参数的期望目标。即使底层奖励不连续，期望回报仍可能是平滑的，因此可以用梯度方法优化策略参数。

## Part 2: Value Function 与 Bellman 方程

### 2.1 为什么需要价值函数

直接评估一条完整轨迹很慢，甚至危险。机器人可能要等车真的掉下悬崖，才知道前面的动作不好。价值函数的作用是把长期未来压缩成局部状态上的一个数。

状态价值函数：

$$
V^\pi(s)=\mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s\right]
$$

它表示：从状态 `s` 出发并遵循策略 `pi`，预期能拿到多少长期回报。

强化学习总目标也可以看成初始状态分布下价值函数的平均：

$$
J(\pi)=\mathbb{E}_{s_0 \sim p(s_0)}[V^\pi(s_0)]
$$

### 2.2 Bellman 方程

Bellman 方程的核心是把无限时域问题拆成一步递归：

$$
V^\pi(s)=\mathbb{E}_{a \sim \pi, s' \sim P}\left[r(s,a)+\gamma V^\pi(s')\right]
$$

意思是：当前状态的价值等于当前一步奖励，加上折扣后的下一状态价值。后半段未来不需要完整展开，而由当前的价值函数估计接管。这种“用当前估计更新当前估计”的思想叫 bootstrapping。

最优价值函数定义为所有策略中能达到的最高价值：

$$
V^*(s)=\max_\pi V^\pi(s)
$$

一旦有了 `V*`，长期规划就可以变成一步贪心决策：在当前状态枚举动作，选择能带来最高即时奖励加下一状态价值的动作。最优价值函数把无限时域规划压缩成局部计算。

### 2.3 Value Iteration

Value iteration 用 Bellman optimality update 直接计算最优价值函数：

$$
V_{k+1}(s)=\max_a \sum_{s'}P(s'|s,a)\left[r(s,a,s')+\gamma V_k(s')\right]
$$

算法流程：

1. 初始化所有状态价值为 0。
2. 对每个状态重复应用 Bellman 最优更新。
3. 收敛后，通过一步 greedy 提取最优策略。

它有清晰的收敛保证，因为 Bellman operator 是 contraction mapping：反复应用会让不同价值函数之间的距离变小。

限制也很强：

- 需要已知环境动力学 `P(s' | s, a)`。
- 需要遍历所有状态和动作。
- 通常只适合离散、小规模、tabular 状态空间。
- 对高维连续机器人状态和动作空间不现实。

### 2.4 Policy Evaluation 与 Policy Iteration

Policy evaluation 问的是：给定一个固定策略 `pi`，它到底有多好？它和 value iteration 的形式很像，但把 `max` 换成策略下的期望：

$$
V^\pi(s)=\sum_a \pi(a|s)\sum_{s'}P(s'|s,a)\left[r(s,a,s')+\gamma V^\pi(s')\right]
$$

Policy evaluation 本身不会改进策略，但它是改进策略的前提。Policy iteration 把两个步骤交替执行：

1. Policy evaluation：评估当前策略。
2. Policy improvement：基于价值函数做 greedy 改进。

在有限 MDP 和确定性策略空间中，policy iteration 也有收敛保证，并且通常收敛很快。

## Part 3: Q Function 与 Model-Free 学习

### 3.1 为什么引入 Q Function

价值函数 `V(s)` 只告诉我们某个状态有多好，但选择动作时仍要通过动力学模型看下一状态会怎样。在真实机器人上，动力学模型难以准确获得，即使有模型，在部署时为每个候选动作模拟未来也很慢。

Q function 把动作也放进价值函数：

$$
Q^\pi(s,a)=\mathbb{E}_\pi\left[r(s,a)+\gamma V^\pi(s')\right]
$$

它表示：在状态 `s` 先执行动作 `a`，之后遵循策略 `pi`，预期能拿到多少回报。最优 Q function 满足：

$$
Q^*(s,a)=\mathbb{E}_{s' \sim P}\left[r(s,a,s')+\gamma \max_{a'}Q^*(s',a')\right]
$$

好处是，部署时只需计算：

$$
a^*=\arg\max_a Q^*(s,a)
$$

不必再显式查询动力学模型。代价是函数维度更高，因为它依赖 `state-action` 对。

### 3.2 Q-Learning

Q-value iteration 仍然需要在训练时知道转移模型。Q-learning 更进一步：用真实交互采样替代对动力学模型的期望。执行动作、观察一个样本转移 `(s, a, r, s')`，构造 TD target：

$$
y = r + \gamma \max_{a'} Q(s',a')
$$

更新规则：

$$
Q(s,a) \leftarrow Q(s,a)+\alpha\left[y-Q(s,a)\right]
$$

这让环境本身成为“采样器”。不需要显式模型，但样本效率比动态规划低，因为 agent 必须花时间探索环境。

### 3.3 Off-Policy 与 SARSA

Q-learning 是 off-policy 的。原因在于 TD target 里使用的是 `max`：它评估的是下一状态的最优动作，而不关心当前这条数据是由什么行为策略采集的。数据可以来自随机策略、旧策略、人类演示，或者混合数据。

SARSA 是对应的 on-policy 版本。它不问“下一步最优动作是什么”，而问“当前策略下一步实际会做什么”：

$$
y = r + \gamma Q(s',a')
$$

其中 `a'` 是当前策略在下一状态实际采样出来的动作。SARSA 会把探索错误也纳入价值评估，因此可能学出更保守、更安全的策略。经典 cliff walking 例子中，Q-learning 会贴着悬崖边走最短路径；SARSA 会因为考虑探索时跌落风险，学到离悬崖更远的路径。

## Part 4: Deep Q-Learning

### 4.1 DQN

tabular Q-learning 无法处理图像观测。Atari 游戏中，即使只用 `84 x 84` 像素并堆叠多帧，状态空间也大到无法建表。DQN（Deep Q-Network）的关键做法是：用卷积神经网络替代表格。

DQN 输入一组原始图像帧，输出每个离散动作的 Q 值：

$$
Q_\theta(s,\cdot) = [Q_\theta(s,a_1), Q_\theta(s,a_2), \ldots]
$$

它的重要性不只是分数高，而是同一套模型结构和超参数可以直接从 raw pixels 学会多个 Atari 游戏，减少了任务特定的手工工程。

### 4.2 Experience Replay

DQN 的第一个关键技巧是 replay buffer。agent 与环境交互得到 transition 后，不立刻用完就丢弃，而是存入经验池，训练时随机采样 mini-batch。

Replay buffer 的作用：

- 打破连续轨迹中的时间相关性，使样本更接近 IID。
- 重复利用昂贵经验，提高样本效率。
- 保留罕见但重要的失败或成功案例。

### 4.3 Target Network

DQN 的第二个关键技巧是 target network。Q-learning 的 target 依赖下一状态的 Q 值，而这些 Q 值也由同一个正在更新的网络预测。如果当前网络一边预测 target、一边被 target 更新，训练很容易震荡或发散。

DQN 维护一个延迟更新的目标网络 `Q_{\theta^-}`：

$$
y = r+\gamma \max_{a'}Q_{\theta^-}(s',a')
$$

目标网络参数会周期性复制或缓慢移动平均到在线网络，从而稳定训练。

### 4.4 Double DQN

DQN 有系统性 overestimation bias。原因是 `max` 会放大噪声：一组带噪 Q 估计的最大值通常会高估真实最大值。

Double DQN 的修正很简单：动作选择和动作评估分离。

$$
a^* = \arg\max_{a'}Q_{\theta}(s',a')
$$

$$
y = r+\gamma Q_{\theta^-}(s',a^*)
$$

在线网络负责选择动作，目标网络负责评估该动作，减少同一个网络同时“挑最大”和“给分”带来的偏差。

## Part 5: 连续动作空间中的 Q-Learning

### 5.1 机器人为什么难用 DQN

DQN 适合离散动作空间，但机器人动作通常是连续的，例如机械臂关节角、末端速度或力矩。此时：

$$
\arg\max_a Q(s,a)
$$

不能靠枚举所有动作解决。连续动作空间中的 Q-learning 核心难点就是如何找到 Q 函数峰值。

### 5.2 三类处理方式

| 方法 | 思路 | 优点 | 问题 |
| ---- | ---- | ---- | ---- |
| Discretization | 把连续动作切成离散 bins | 可以直接套 DQN | 维度一高组合爆炸 |
| Sampling / CEM | 采样多个候选动作，取 Q 值最高者 | 不要求动作离散 | 测试时计算昂贵，结果依赖采样 |
| Learn the argmax | 用 actor 网络直接预测高 Q 动作 | 测试时一次前向传播 | 训练更脆弱 |

### 5.3 空间离散化与 Push-Grasping

某些机器人任务有可利用的几何结构。例如桌面抓取可以把动作映射到 top-down RGB-D heightmap 上，让网络输出每个像素、每个离散朝向的 Q 值。这样 `argmax` 就变成在图像 Q map 上找最高点。

push-grasping 工作展示了一个有意思的现象：机器人没有被显式编程“先推开杂物再抓”，但 Q-learning 通过 Bellman backup 学到了这种策略。拥挤场景下直接抓取价值低，推动物体能带来更高的未来抓取成功率，因此“先整理再抓取”的行为从奖励最大化中自然涌现。

### 5.4 Monte Carlo Sampling 与 CEM

最简单的连续动作选择方法是 Monte Carlo sampling：从动作空间采样 `N` 个候选动作，分别计算 `Q(s,a)`，选择最高者。它利用神经网络 Q 函数的平滑性，但需要每个控制步做 `N` 次前向传播，且没有保证找到真正最大值。

Cross-Entropy Method（CEM）是更聪明的采样方法：

1. 从一个高斯分布中采样候选动作。
2. 保留 Q 值最高的一批 elite actions。
3. 用 elite actions 重新拟合高斯分布。
4. 重复若干轮后输出最优候选动作。

CEM 会逐步把采样集中到高价值区域。局限是它偏向单峰分布，并且测试时要付出多轮采样和前向传播的代价。

QT-Opt 是这类思路在真实机器人抓取中的代表：用 Q-learning、CEM 和大 replay buffer 训练抓取策略。大量真实抓取数据能被反复用于梯度更新，提高昂贵真实样本的利用率。

### 5.5 DDPG

DDPG（Deep Deterministic Policy Gradient）选择学习 `argmax` 本身。它引入 actor-critic 结构：

- critic：Q 网络，估计 `Q(s,a)`，用 TD loss 训练。
- actor：确定性策略网络，输入状态，直接输出连续动作。

actor 的目标是输出 critic 认为价值高的动作。测试时不再需要 CEM 或大量采样，只需一次 actor 前向传播。

训练 actor 时，通过链式法则把 Q 对动作的梯度传回 actor 参数：

$$
\nabla_\theta J \approx \nabla_a Q_\phi(s,a)|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s)
$$

DDPG 的探索通常是在 actor 输出上加高斯噪声，而不是在连续动作空间里随便采样随机动作。这样探索更平滑，不容易给真实机器人发出极端危险的动作。

DDPG 能处理连续动作和高维观测，但它出了名地脆弱：对超参数、噪声规模和环境细节敏感，收敛不稳定。

## Part 6: Policy Gradient

### 6.1 从 Value-Based 到 Policy-Based

前面的方法都属于 value-based：先学习每个动作有多好，再通过 `argmax` 得到策略。第 5 讲转向 policy gradient：直接参数化策略 `pi_theta`，通过调整神经网络参数，让策略更倾向于产生高回报行为。

这类方法尤其适合连续动作空间，因为策略可以直接输出动作分布，不需要在每一步解复杂的 `argmax`。

### 6.2 Monte Carlo 估计策略表现

我们无法解析计算真实世界中所有可能轨迹的期望回报，因此用 Monte Carlo sampling：

1. 用当前策略 `pi_theta` rollout 多条轨迹。
2. 计算每条轨迹的累计回报。
3. 用平均回报估计 `J(theta)`。

这里有一个重要约束：数据必须来自当前策略。因此 vanilla policy gradient 是 on-policy 的，旧数据通常不能直接复用。

### 6.3 Log-Derivative Trick

策略梯度的困难在于轨迹是采样出来的，不能直接对真实世界动力学反向传播。log-derivative trick 让我们把梯度改写成可采样形式：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau \sim p_\theta(\tau)}
\left[
\nabla_\theta \log p_\theta(\tau) R(\tau)
\right]
$$

把轨迹概率展开后，初始状态分布和环境转移概率都不依赖 `theta`，梯度为零，最后只剩策略项：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau}
\left[
\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau)
\right]
$$

这就是 REINFORCE，也叫 vanilla policy gradient。

### 6.4 REINFORCE 与 Behavior Cloning 的关系

Behavior cloning 最大化专家数据中动作的 likelihood，不管这些动作最终导致什么回报。REINFORCE 也在提高某些动作的 log probability，但会用轨迹回报加权：

- 高回报轨迹中的动作概率被提高。
- 低回报轨迹中的动作概率被降低。

因此，REINFORCE 可以理解为“带回报权重的动作似然最大化”。

## Part 7: 降低 Policy Gradient 方差

### 7.1 Reward-to-Go

vanilla REINFORCE 会用整条轨迹总回报惩罚或奖励每一个动作。这违反因果直觉：时间 `t` 的动作只能影响当前和未来奖励，不可能影响过去已经发生的奖励。

Reward-to-go 把总回报换成从当前时刻开始的未来回报：

$$
G_t = \sum_{t'=t}^{T}\gamma^{t'-t}r_{t'}
$$

策略梯度变为：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)G_t
\right]
$$

这不改变梯度期望，但显著降低方差，并让信用分配更符合因果关系。

### 7.2 Baseline

REINFORCE 的另一个问题是高方差。假设两条轨迹回报分别是 99 和 101，真正有信息量的差异只有 2，但梯度会被 99 和 101 这样的绝对数值放大，优化器看到的大部分信号都是噪声。

可以从 reward-to-go 中减去 baseline：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)(G_t-b)
\right]
$$

只要 baseline 不依赖当前动作，这不会引入偏差，却能降低方差。例如把平均回报 100 作为 baseline 后，99 变成 `-1`，101 变成 `+1`，梯度方向更清楚。

### 7.3 Advantage Function

Reward-to-go 本质上是 `Q^\pi(s,a)` 的 Monte Carlo 估计。更自然的 baseline 是状态价值 `V^\pi(s)`，于是得到 advantage function：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

它回答的问题是：在状态 `s` 下采取动作 `a`，比当前策略平均会做的动作好多少？

- `A > 0`：这个动作比平均更好，提高概率。
- `A < 0`：这个动作比平均更差，降低概率。

Advantage 把绝对回报变成相对回报，是现代 policy gradient 和 actor-critic 方法的核心。

## Part 8: 从 On-Policy 到可复用数据

### 8.1 On-Policy Wall

vanilla policy gradient 要求每次更新都使用当前策略采集的数据。对真实机器人来说，这非常浪费：采集一次数据昂贵、慢，还可能损坏硬件。如果一次梯度更新后就丢掉所有数据，样本效率很低。

目标是让策略梯度尽量复用旧数据，也就是从完全 on-policy 走向近似 off-policy。

### 8.2 Importance Sampling

Importance sampling 用旧分布的数据估计新分布下的期望。假设旧策略是 `pi_old`，新策略是 `pi_theta`，可以给每个样本乘上概率比：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
$$

直觉是：如果某个动作在新策略下比旧策略更可能出现，就给它更大权重；反之给更小权重。

完整轨迹的重要性权重会包含所有时间步的概率比乘积，长时域下容易指数级爆炸或衰减。因此实践中常用每个时间步的 action ratio，并把难以估计的 state distribution ratio 近似为 1。这要求新旧策略不能差太远。

### 8.3 Entropy Bonus

Importance sampling 还要求旧策略对新策略可能采样的动作有非零概率。如果旧策略太确定性，某些动作概率接近 0，概率比就会不稳定。

Entropy bonus 鼓励策略保持一定随机性：

$$
J_{entropy} = J + \alpha H(\pi)
$$

`alpha` 控制探索强度。较大时策略更分散，探索更多；较小时策略更贪心。离散动作可用 Shannon entropy，连续高斯策略可用 differential entropy。

## Part 9: TRPO 与 PPO

### 9.1 为什么要限制策略更新幅度

如果用同一批旧数据做多次梯度更新，新策略可能离旧策略越来越远。此时 importance ratio 会变得很大或很小，梯度估计失真甚至爆炸。

解决思路是限制每次策略更新的幅度，让新旧策略保持接近。

### 9.2 TRPO

TRPO（Trust Region Policy Optimization）通过 KL divergence 约束新旧策略距离：

$$
\max_\theta \; \mathbb{E}\left[\rho_t(\theta)A_t\right]
\quad
\text{s.t.}
\quad
D_{KL}(\pi_{old}||\pi_\theta) \le \delta
$$

它的优点是有理论上的 monotonic improvement 保证：在近似条件下，每次更新不会让真实性能变差。

缺点是计算代价高。KL 约束涉及 Fisher information matrix 和二阶优化，内存和计算开销都很重。TRPO 在机器人 locomotion、sim-to-real 等任务中有成功案例，但现在使用频率不如 PPO。

### 9.3 PPO

PPO（Proximal Policy Optimization）用更简单的一阶方法近似 TRPO 的稳定性。核心思想是直接裁剪 importance ratio：

$$
L^{CLIP}(\theta)=
\mathbb{E}
\left[
\min
\left(
\rho_t(\theta)A_t,
\text{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

当新策略试图把某个动作概率改得太多时，clip 会限制收益，避免一次更新走太远。

PPO 的特点：

- 实现简单，兼容 Adam 等一阶优化器。
- 不需要 TRPO 那样昂贵的二阶计算。
- 稳定性强，跨任务表现好。
- 仍然是 near on-policy，样本效率不如完全 off-policy 方法。

OpenAI 的 Shadow Hand 解 Rubik's Cube 是 PPO 在机器人中的著名案例。它通过大量仿真和 domain randomization 实现 sim-to-real，但策略输入并不是原始相机图像，而是通过感知模块得到的魔方位置和姿态等状态信息。

### 9.4 PPO 的完整 Actor-Critic 目标

PPO 常见实现包含三项：

| 项 | 作用 |
| ---- | ---- |
| clipped surrogate objective | actor loss，更新策略并限制策略漂移 |
| value function loss | critic loss，训练价值函数估计 advantage |
| entropy bonus | 保持探索，避免策略过早坍缩 |

这就是 actor-critic 框架：

- actor：策略 `pi_theta`，决定行为。
- critic：价值函数 `V_phi` 或 Q 函数，评价当前行为。

critic 通常用 TD update 训练，通过 bootstrapping 估计价值，不必等完整 episode 结束。它会引入少量 bias，但显著降低 policy gradient 的方差，提高样本效率。

对机器人尤其重要的一点是：critic 可以在训练时使用 privileged information。比如仿真中可以把完整状态给 critic，让它更准确估计 advantage；部署时只保留 actor，因此 actor 仍然可以只依赖真实机器人可获得的观测。

## Part 10: Soft Actor-Critic

### 10.1 从 PPO 到 SAC

PPO 稳定，但仍接近 on-policy。Soft Actor-Critic（SAC）进一步走向 fully off-policy actor-critic：像 DQN 一样使用 replay buffer，可以反复利用旧数据，因此样本效率更高。

SAC 的核心是 maximum entropy RL。它不仅最大化奖励，还最大化策略熵：

$$
J(\pi)=\mathbb{E}\left[\sum_t \gamma^t(r_t+\alpha H(\pi(\cdot|s_t)))\right]
$$

这意味着策略偏好两类状态：

- 能获得高奖励的状态。
- 有多种好动作可选、保留灵活性的状态。

### 10.2 SAC 的 Actor 与 Critic

SAC 的 actor 在 exploitation 和 exploration 之间权衡：

- exploitation：选择 Q 值高的动作。
- exploration：保持较高熵，不要过早变成确定性策略。

常见写法中 actor 最小化：

$$
\mathbb{E}_{a \sim \pi_\theta}\left[\alpha \log \pi_\theta(a|s)-Q_\phi(s,a)\right]
$$

critic 用 replay buffer 中的旧 transition 做 TD 学习。关键点是：虽然状态来自旧数据，但 actor 会在这些旧状态上询问“如果当前策略今天在这里，会选择什么动作”。critic 像一个学习出来的局部评估器，估计新动作在旧状态中可能得到的价值。

连续动作策略涉及采样，为了能把梯度传回 actor，SAC 通常使用 reparameterization trick。例如高斯动作可以写成：

$$
a = \mu_\theta(s)+\sigma_\theta(s)\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
$$

这样随机性由 `epsilon` 承担，网络参数仍可被反向传播优化。

### 10.3 为什么 SAC 适合机器人

SAC 在机器人中常用，原因是：

- fully off-policy，可复用 replay buffer，样本效率高。
- 支持连续动作空间。
- 最大熵目标带来稳定探索。
- actor-critic 结构兼顾直接策略优化和低方差价值估计。

真实机器人实验中，SAC 被用于积木堆叠、四足控制、灵巧手操作等任务。相比 vanilla policy gradient，它更接近机器人需要的算法特性：少浪费真实交互数据，并且能在连续高维动作空间中稳定学习。

## Part 11: 方法谱系总结

| 方法族 | 代表算法 | 优点 | 局限 |
| ------ | -------- | ---- | ---- |
| Exact dynamic programming | Value iteration, Policy iteration | 收敛保证强，概念清晰 | 需要已知动力学，适合小型 tabular MDP |
| Tabular model-free RL | Q-learning, SARSA | 不需要模型，可从经验学习 | 状态/动作空间必须小，探索成本高 |
| Deep value-based RL | DQN, Double DQN | 可处理图像观测，经验回放提高效率 | 主要适合离散动作，无强收敛保证 |
| Continuous-action Q methods | QT-Opt, CEM, DDPG | 可进入机器人连续控制 | CEM 测试时昂贵，DDPG 训练脆弱 |
| Policy gradient | REINFORCE | 直接优化策略，适合连续动作 | 高方差，on-policy，样本效率低 |
| Trust-region / clipped PG | TRPO, PPO | 稳定，工程上好用 | 仍接近 on-policy，样本需求大 |
| Off-policy actor-critic | SAC | 样本效率高，适合连续机器人控制 | 实现复杂，需要调节熵系数和 critic 训练 |

## 本两讲主线

Lecture 4 从 Bellman 方程出发，解释了如何用价值函数把长期未来压缩成局部递归，并逐步从 value iteration、policy iteration 走到 Q-learning、DQN 和连续动作控制。核心转变是：从已知模型的动态规划，走向不依赖动力学模型、能处理高维观测的 model-free deep RL。

Lecture 5 则从另一个方向切入：既然最终要的是策略，能不能直接优化策略？REINFORCE 给出最朴素的 policy gradient；reward-to-go、baseline 和 advantage 解决方差问题；importance sampling、entropy、TRPO 和 PPO 让策略更新更稳定且能适度复用数据；SAC 最后把 actor-critic、replay buffer 和最大熵目标结合起来，成为机器人连续控制中非常实用的一类算法。

把两讲合在一起看，强化学习在机器人中的核心矛盾始终是：真实交互昂贵、动作空间连续、环境动力学复杂、奖励稀疏且长期。不同算法的差别，本质上是在稳定性、样本效率、动作空间表达能力和工程复杂度之间做取舍。
