---
title: "Lecture 11-Frontier and Open Problems"
publishDate: 2026-06-09
description: '机器人学习前沿与开放问题：泛化、数据、具身智能、终身学习与研究方法'
tags:
  - AI
  - Robot Learning
language: '中文'
---

## 课程说明

课程资源

- 课程网站: [Robot learning](http://cvg.ethz.ch/lectures/Robot-Learning/)
- 课程作业网站: [mees-robot-learning-course/ethz-course-2026](https://github.com/mees-robot-learning-course/ethz-course-2026)

本文整理 Lecture 11: Frontier & Open Problems。录音稿中的项目设备通知、课程结束致谢、个人经历细节和课堂掌声没有纳入，只保留关于机器人学习前沿问题与研究方法的内容。

## 不要被 Demo 误导

机器人领域经常出现“机器人已经解决”“这是 robotics 的 ChatGPT moment”之类的宣传。很多 demo 看起来很强，但需要谨慎理解：一个机器人在特定桌面、特定光照、特定相机位置、特定物体集合下表现好，并不等于它真正具备开放世界泛化能力。

课程给了一个“做出漂亮机器人 demo”的不那么神秘的 recipe：

1. 在目标平台和目标环境中尽可能多采集训练数据。
2. 训练 behavior cloning policy，并继续采集数据，直到对策略来说 extrapolation 变成 interpolation。
3. 尽量当天训练、当天测试，避免光照、相机、桌高、物体位置等分布偏移。

这类策略可以做出很有说服力的视频，但通常不能说明机器人真正解决了泛化问题。把同一个机器人搬到隔壁房间，换一个相机角度或换一批物体，系统可能马上失败。

## 真正的问题是世界级泛化

可以把泛化范围看成一个层级：

- 一个物体。
- 一张桌子。
- 一个房间。
- 一栋楼。
- 真实世界。

很多 impressive demos 仍停留在很窄的范围内。真正目标是让机器人像人一样在世界中泛化：不需要见过每一个厨房才能做饭，不需要抓过每一种杯子才能理解新杯子的抓取方式。

这种能力不是简单 interpolation。它需要把已有知识组合起来，对新情境进行推理，并把已学技能迁移到未见过的对象、场景和任务中。

## 当前两类能力的缺口

当前系统大致分成两类：

- 通用 foundation models：VLM、LLM、视频生成模型等有丰富语义知识和广泛互联网先验，但缺少物理 grounding。它们能描述桌上的玻璃杯，却不一定理解把杯子推下桌会破碎。
- 具身 specialist policies：能在特定平台上完成灵巧任务，能使用相机、proprioception 等更接近真实控制的信号，但通常泛化范围窄，离开训练环境就容易失效。

机器人学习的关键问题是结合二者：既要有 foundation model 的开放世界语义和推理能力，也要有真实硬件上的物理交互、接触、力觉和高频控制能力。

这要求模型不只从 passive web data 中学习，还要被 richer embodied data grounding。这类数据应当是多模态、空间化、时间化的，并且来自真实物理交互。

## 开放问题 1：机器人 Backbone 应该是什么

课程把当前社区的 backbone 路线分成三个阵营。

### VLM Backbone

第一种路线使用 vision-language model。优势是 VLM 已经有丰富语义知识，能识别物体、理解语言指令、做一定视觉推理。把它 fine-tune 到机器人数据上，可能直接获得 object-level semantic generalization。

问题是，VLM 多来自静态 image-text 任务，缺少动作导致状态变化的时序和物理因果知识。

### Generative Video Backbone

第二种路线使用生成式视频模型。视频模型更接近机器人需要的能力，因为它建模世界如何随时间演化，包含更多动态和物理交互先验。相比静态 VLM，视频 backbone 可能更适合预测动作后果。

问题是，视频模型计算昂贵，tokenization 和长时域建模困难，而且视频中的动作通常没有机器人控制标签。

### From-Scratch Robot Backbone

第三种路线更激进：如果能收集足够多真实机器人数据，就从头训练机器人 foundation model，不依赖互联网预训练。

优势是所有容量都服务于机器人控制，没有 web pretraining 和 robot control 之间的任务错配。问题是现实中真实机器人数据极其昂贵，目前还远达不到互联网数据规模。

目前没有定论。backbone 选择和数据 recipe 紧密相关。

## 开放问题 2：机器人数据 Recipe

不同路线对应不同数据配方。

### Web + Simulation + Real

一种三层配方是：

1. 用 web data 学世界知识、语言 grounding 和语义先验。
2. 用 simulation 提供便宜、可扩展的交互经验。
3. 用少量真实机器人数据弥合 sim-to-real gap。

这个路线的难点是模拟整个真实世界非常困难，尤其是接触、软物体、摩擦、变形、长尾物体等。

### Web Video + Human Egocentric Video + Robot Data

另一种配方把 video 当成第一等数据源：

1. 先在互联网视频上学视觉世界动态。
2. 再用人类第一视角视频做 embodied mid-training。
3. 最后 post-train 到机器人动作预测。

直觉是，视频比静态 VQA 更接近机器人任务，因为它包含运动、交互和时间结构。

### Real Robot Data Only

还有一种路线是尽量减少 simulation 和 web mismatch，直接大规模收集真实机器人数据。它避免了模拟误差和互联网任务错配，但数据成本最高。

课程没有给出唯一答案，而是强调：数据 recipe 本身就是机器人学习最重要的开放问题之一。

## 开放问题 3：如何规模化采集真实数据

真实机器人数据始终绕不过去。不同采集方式在三个维度上取舍：

- scalability：能采多少数据。
- hardware alignment：数据和目标机器人硬件是否一致。
- task complexity：能演示多复杂的任务。

常见数据来源：

- teleoperation：与目标机器人最对齐，但慢、贵、硬件易损。
- puppeteering / leader-follower arms：数据质量高，但设备复杂、成本高。
- VR teleop：相对灵活，但动作映射和精度有限。
- wearable gloves / data wearables：介于人类动作和机器人动作之间，规模较好但存在 embodiment gap。
- egocentric human videos：规模最大，可以达到千万小时级别，但与机器人 embodiment 差距也最大。

一个重要方向是缩小 embodiment gap。例如，人形机器人可能更容易利用人类 egocentric video，因为形态更接近人类；但这仍然远未解决。

## 开放问题 4：Dextrous Generalist Policies

机器人真正有用时，需要同时具备 dexterity 和 generalization。

### Dexterity 需要什么

灵巧操作通常需要：

- 高频控制。
- 精确接触。
- force/tactile feedback。
- 多传感器融合。
- 对物体形状、材质和接触状态的敏感性。

这些要求和大 foundation model 的慢推理、高延迟、主要依赖视觉输入存在冲突。

### Generalization 需要什么

泛化通常需要：

- 大规模 foundation model 先验。
- 跨环境经验。
- 移动能力，让机器人能接触更广泛场景。
- 对新物体、新任务和新空间关系的推理。

只固定在一张桌子前训练机械臂，很难获得开放世界泛化。

目标是二者交集：一个既能做精细接触操作，又能在未见过环境中可靠工作的 dextrous generalist policy。

## 开放问题 5：视觉之外的具身感知

很多机器人策略仍然主要依赖视觉。但灵巧操作中，接触发生时视觉常常不够。课程举了一个直观例子：机器人即使能从图像估计杯子形状和姿态，如果没有触觉或力反馈，抓取时仍可能用力过猛，把杯子捏坏。

当前 VLA 很可能仍在“盲抓”：它们看到了接触前的世界，却没有足够理解接触后的力和材质变化。

引入 touch、force、audio、proprioception 等模态并不简单，因为会遇到 long tail of robot sensing：

- 数据稀缺：触觉、力觉硬件昂贵、脆弱、不标准。
- 配对数据稀缺：同时有视觉、语言、触觉、力觉、音频、动作的数据几乎不存在。
- 传感器异质：不同机器人传感器规格差异很大，迁移困难。

一个有意思的方向是用语言作为 bridge，把分散模态连接起来。例如模型可以学习“摸起来像菠萝但颜色是棕色的物体”这类跨模态查询，即使没有大规模完全配对的多模态机器人数据。

## 开放问题 6：机器人什么时候应该推理

第 10 讲讨论了 embodied reasoning 和 test-time scaling。第 11 讲进一步提出：机器人不应该每一步都慢慢推理，也不应该永远不推理。

可以把推理策略看成一个 spectrum：

- in-distribution：已知对象、已知任务、已知场景。直接运行高频 policy，不浪费计算。
- uncertain：模型不确定但可能通过思考解决。增加 test-time compute，做 reasoning、分解或搜索。
- out-of-distribution 且无法解决：停止，向人类请求帮助。

这个框架的前提是模型知道自己什么时候不知道，也就是 introspection。

## 开放问题 7：不确定性与自省

当前大模型并不擅长 calibration。它们可能在完全不懂的情况下生成看似合理的答案。模型越大，pattern completion 能力越强，陌生场景也可能表面看起来熟悉，OOD detection 反而更难。

机器人中的不确定性来源更多：

- 新物体。
- 新环境。
- 新 embodiment。
- 新任务。
- 传感器噪声。
- 光照、视角、硬件状态变化。

策略可能因为完全不同的原因失败，却表现出相似的置信度。要做 adaptive reasoning、human intervention 或 safe stop，就必须先解决“模型如何知道自己不知道”的问题。

## 开放问题 8：突破 Imitation 的上限

当前许多机器人策略依赖 imitation learning，因此被演示数据限制。如果数据中只演示了从状态 1 到 2、从 2 到 3 的路径，behavior cloning 没有机制发现也许存在更好的 1 到 3 路径。

强化学习可能突破这个限制。尤其是 offline RL 可以从次优数据中 stitching 出更优轨迹，发现人类没有显式演示过的行为。RL 已经在语言 reasoning、agent、知识发现等领域带来新能力。

但真实机器人 RL 仍然很难：

- 真实交互昂贵。
- 探索有安全风险。
- 奖励设计困难。
- 长时域任务信用分配困难。
- 硬件故障和环境变化会破坏训练稳定性。

课程认为，大规模机器人 RL 是当前最有影响力的开放方向之一，不是简单堆算力就能解决，需要新的算法。

## 开放问题 9：Data Flywheel

机器人学习的理想 flywheel 是：

1. 更多数据训练出更好模型。
2. 更好模型让机器人部署更广。
3. 更多部署产生更多真实交互数据。
4. 新数据继续改进模型。

问题是，部署数据和专家数据分布差异很大：

- 专家数据通常更平滑、更高质量、更短路径。
- 自主 rollout 可能更慢、更长、更抖、更次优，甚至失败。
- 控制频率和行为质量都不同。

如果直接用 imitation learning 混合专家数据和自主数据，模型可能被低质量自主数据误导。真正需要的是能从所有数据中提取有用信号的方法：成功、失败、次优、探索数据都不应被浪费。

这也是 RL 和 offline RL 可能发挥作用的地方。真实数据太贵，目标应该是 never throw away data。

## 开放问题 10：Lifelong Learning

当前主流范式是训练一个模型，冻结权重，部署后希望它泛化。但真正的机器人应该随着真实交互持续变好。

lifelong learning 需要解决：

- 如何从部署数据中持续学习。
- 如何避免 catastrophic forgetting。
- 如何混合不同质量、不同时期、不同环境的数据。
- 如何保证持续更新不会引入危险行为。

这比一次性 pretraining/fine-tuning 更接近智能体在真实世界中成长的方式。

## 开放问题 11：快速适应与 In-Context Learning

除了长期学习，还有部署中的即时失败恢复。比如家用机器人第一次打开用户冰箱失败时，不一定要重新训练模型：

- 如果只是位置偏差，用户一句语言反馈“往左一点抓”可能足够。
- 如果完全不会开冰箱，用户演示一次，机器人应能把这段示范放进 context 并立刻使用。

这就是 robotics 中的 in-context learning 愿景：面对新情境时，机器人能从少量语言、视频、演示、纠错信号中快速适应，而不是每次都重训。

相关方向包括 non-parametric adaptation、few-shot demonstration retrieval、context-based policy adaptation 等。课程提到 PAL 这类方法可以用很少示范达到接近大量 fine-tuning 的效果。

## 开放问题 12：移动操作与环境表示

当前 mobile manipulation 常常是分裂的：

- 导航部分依赖传统 SLAM、地图、定位和路径规划。
- 到达目标附近后，才切换到 learned policy 或 VLA 做操作。

这是一种硬切换。更理想的是统一模型同时处理 navigation 和 manipulation，也就是 whole-body control。

关键问题包括：

- 长距离导航需要什么环境表示？
- 需要显式 3D map、BEV、topological map、scene graph，还是 walkthrough video 就够？
- 状态估计应该显式做，还是由模型隐式完成？
- 环境持续变化时，地图或记忆如何更新？
- 是否能在局部无地图情况下可靠导航？

移动能力也和泛化密切相关。能在房间、楼宇和真实环境中移动的机器人，能接触更多数据，也更可能形成广泛经验。

## 开放问题 13：长期记忆

如果机器人长期部署，它会产生连续的多模态经验流：

- 视频。
- 深度。
- 语言。
- 触觉。
- 音频。
- bounding boxes。
- 行为轨迹。
- 成功与失败记录。

问题是：什么该存，存多久，以什么粒度存？

可能存在不同时间尺度：

- 短期记忆：密集视频、最近动作、当前任务状态。
- 中期记忆：压缩成语义事件，例如“昨天把杯子放进左边柜子”。
- 长期记忆：跨天、跨月、跨任务的用户偏好、环境变化和技能经验。

还需要解决：

- 何时写入记忆。
- 何时遗忘。
- 如何检索相关经验。
- 如何跨模态对齐。
- 如何避免记忆污染和错误累积。

## 一些已有探索

课程提到几个相关方向：

- VLMaps：把 VLM embedding 融合进 3D spatial map，实现 open-vocabulary zero-shot navigation。
- LeLaN：从 YouTube 视频学习 mapless navigation，训练视频越多，开放词汇导航性能越好。
- autonomous improvement：利用 foundation models 根据环境 affordance 提出和评分任务，让机器人在少人工干预下自我改进。
- PAL：用非参数适应方法，让 generalist policy 用少量示范适应新的长时域任务。

这些工作都还不是最终答案，但指向同一目标：让机器人能在真实世界中感知、行动、记忆、适应和持续改进。

## 四个支柱

课程最后把真正的 embodied intelligence 总结为几个支柱：

- embodied reasoning：能在物理世界中分解任务、推理空间关系、判断不确定性。
- dexterous mobile manipulation：既能移动到不同环境，又能完成精细接触操作。
- lifelong learning：长期部署中持续从经验中学习。
- rapid adaptation：遇到新任务、新环境、新用户时能快速适应。

这些问题解决后，机器人学习才更接近真实可用，而不是只在 demo 中好看。

## 研究方法建议

这一讲最后也给了做研究的建议，值得保留。

### 研究的几个现实

- 大多数研究是 incremental 的，这不是坏事，科学通常就是这样积累。
- 大多数想法不会变成论文。很多实验失败是正常过程。
- 大多数论文不会长期重要，领域变化很快。
- 在 scale 很重要的时代，简单想法往往更有影响力，因为更容易扩展。

如果一个方法需要 17 个组件才能工作，应该先问：有没有更简单的版本能达到 80% 效果？

### 好问题需要问题和计划

一个好的研究项目需要两个条件：

- 问题重要。
- 有可执行的切入计划。

只有重要问题但没有切入点，会过于空泛；只有一个小技巧但不解决重要问题，影响有限。应该寻找“重要问题 + 清晰 attack angle”的交集。

还要提前问：如果这个方向成功，它会解锁什么？社区因此能做以前不能做的什么事？

### 兴奋感很重要

研究很慢，实验经常失败，deadline 和 rejection 都会发生。如果自己对问题不兴奋，困难阶段很难坚持。选择问题时，技术重要性之外，也要考虑自己是否真的愿意长期投入。

### 做自己的 Reviewer 2

在投入几个月前，先诚实问自己：这个想法最可能为什么失败？

如果回答了这个问题后仍然觉得值得做，才更适合继续。这样做是在保护研究中最稀缺的资源：时间。

### Problem-Driven vs Method-Driven

两种研究方式都可能成功：

- method-driven：从一个新技术出发，寻找适合应用的问题。
- problem-driven：从一个具体重要问题出发，寻找或发明能解决它的方法。

对长期研究尤其是 PhD 来说，problem-driven 通常更稳。如果某个方法失败，仍然可以换方法继续解决同一个重要问题；如果一开始没有重要问题，方法失败后就没有落点。

### 实验建议

- 从应该绝对能工作的最小实验开始，例如 overfitting 一个小数据集。
- 再逐步增加难度，不要一开始就全复杂度。
- 永远可视化输入和输出，很多 bug 只有看数据才会发现。
- 多和别人讨论，包括同事、相关论文作者、领域内研究者。
- 有结果后尽量分享：开源代码、数据、模型，写清楚结论。
- 认真打磨图表。好的图应该让读者不读正文也能看懂核心结论。
- 传播研究很重要。没人知道的好论文几乎没有影响，但传播时也要避免夸大成“解决机器人学”。

## 本讲主线

Lecture 11 的核心是给前面所有技术画出边界：今天的机器人策略已经能做出很强 demo，但离世界级泛化、长期自主学习和真实物理智能还很远。

开放问题集中在几条线：选择什么 backbone，用什么数据 recipe，如何规模化真实数据，如何同时获得 dexterity 和 generalization，如何引入触觉/力觉等长尾感知，如何让模型知道自己不知道，如何用 RL 突破 imitation 上限，如何形成 data flywheel、lifelong learning、rapid adaptation、mobile manipulation 和长期记忆。

这讲的判断很明确：机器人学习的“end game”不是更漂亮的单场景视频，而是能在真实世界中推理、行动、学习、记忆和适应的具身系统。
