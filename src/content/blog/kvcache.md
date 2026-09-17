---
title: "Transformer 推理中的 Q、K、V 与 KV Cache"
publishDate: "2026-09-17"
description: "所以为什么要有kvcache和GQA MQA"
tags:
  - "llm"
language: "中文"
draft: false
comment: true
---

在当前主流的大语言模型中，Transformer 几乎已经成为默认的基础架构，而理解 Transformer 在推理阶段如何计算 Query、Key 和 Value，是进一步理解 KV Cache、长上下文推理、PagedAttention、GQA、MQA 以及大模型推理性能瓶颈的基础。很多初学者第一次接触 QKV 时，会把它们理解为三个简单的矩阵投影，但真正到了自回归推理阶段，问题会变得更有工程意味：模型为什么不在每次生成新 token 时重新计算整个上下文？为什么只缓存 K 和 V，而不缓存 Q？为什么上下文越长，单 token 解码速度往往越慢？为什么现代大模型大量采用 GQA？这些问题其实都可以从同一条计算链路中得到解释。

## 一、Q、K、V 从哪里来

设某一层 Transformer 接收到的隐藏状态为

$$
X\in\mathbb{R}^{L\times d_{\text{model}}},
$$

其中 $L$ 表示当前序列长度，$d_{\text{model}}$ 表示模型的隐藏维度。注意力机制首先会通过三个不同的线性映射得到 Query、Key 和 Value：

$$
Q=XW_Q,
$$

$$
K=XW_K,
$$

$$
V=XW_V.
$$

其中 $W_Q$、$W_K$、$W_V$ 是模型训练得到的参数矩阵。随后进行缩放点积注意力计算：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
$$

对于 GPT、LLaMA、Qwen 等 Decoder-only 自回归语言模型，还需要加入 causal mask，使得位置 $i$ 的 token 只能关注位置 $1$ 到 $i$，不能看到未来位置。因此实际形式更接近

$$
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V,
$$

其中 $M$ 是因果掩码矩阵。

从直觉上看，可以把 Q 理解为“当前 token 想查询什么”，K 理解为“每个历史 token 能以什么方式被匹配”，V 则表示“真正需要被取出的信息内容”。于是 $QK^\top$ 计算的是查询与各个键之间的匹配程度，softmax 将这些匹配程度转化为权重，再用这些权重对 V 进行加权求和。

## 二、训练阶段为什么可以一次性计算完整序列

在训练阶段，模型通常一次拿到整段 token 序列。假设输入是四个 token，那么可以把隐藏状态表示成

$$
X=
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
x_4
\end{bmatrix}.
$$

随后一次性得到

$$
Q=
\begin{bmatrix}
q_1\\
q_2\\
q_3\\
q_4
\end{bmatrix},
\qquad
K=
\begin{bmatrix}
k_1\\
k_2\\
k_3\\
k_4
\end{bmatrix},
\qquad
V=
\begin{bmatrix}
v_1\\
v_2\\
v_3\\
v_4
\end{bmatrix}.
$$

此时 $QK^\top$ 会形成一个 $4\times 4$ 的注意力分数矩阵。由于模型训练时已经知道整段目标序列，因此可以使用 causal mask 屏蔽未来 token，从而并行计算所有位置的注意力结果。也就是说，虽然语言模型在语义上是自回归的，但训练时并不需要真的一个 token 一个 token 地串行运行，GPU 可以一次执行大规模矩阵运算。

这也是 Transformer 相比传统循环神经网络的一项关键优势：训练阶段具有很强的并行性。

## 三、推理阶段为什么和训练阶段不同

真正生成文本时，情况发生了根本变化。假设用户输入了一个 prompt，模型只能先基于这个 prompt 预测下一个 token，得到这个 token 后，再把它加入上下文继续预测下一个 token。整个生成过程因此是严格自回归的。

现代 Decoder-only 大模型的推理过程通常被划分为两个阶段：

$$
\text{Prefill}+\text{Decode}.
$$

Prefill 是对用户已经提供的完整 prompt 进行一次整体前向传播；Decode 则是之后每次只生成一个新 token 的阶段。理解这两个阶段的区别，是理解 KV Cache 的关键。

## 四、Prefill 阶段在做什么

假设 prompt 一共有 $L$ 个 token。Prefill 阶段会像训练时一样，把所有 token 一次送入模型。对于某一层 Transformer，模型计算

$$
Q_{1:L}=X_{1:L}W_Q,
$$

$$
K_{1:L}=X_{1:L}W_K,
$$

$$
V_{1:L}=X_{1:L}W_V.
$$

随后执行完整的 causal self-attention。由于整个 prompt 是已知的，这一阶段仍然能够使用大规模矩阵乘法，因此通常具有较高的 GPU 计算利用率。

与此同时，一个非常重要的事情发生了：模型会把每一层中 prompt 对应的 K 和 V 保存下来。也就是说，对于这一层，我们会保留

$$
K_1,K_2,\dots,K_L
$$

和

$$
V_1,V_2,\dots,V_L.
$$

这些被保存的数据，就是后续 Decode 阶段使用的 KV Cache。

## 五、如果没有 KV Cache，会发生什么

假设 prompt 已经处理完毕，模型生成了一个新 token，记为第 $L+1$ 个 token。现在模型需要基于包含这个新 token 的完整上下文继续预测第 $L+2$ 个 token。

如果没有 KV Cache，最直接的做法是把整个序列重新送进模型。也就是说，模型会再次对前面所有 token 计算 Q、K、V，然后重新执行每一层 Transformer 的前向传播。这样做虽然在数学上是正确的，但计算效率极低。

原因在于，历史 token 并没有发生变化。对于固定模型参数，只要历史 token 的隐藏状态和位置编码保持不变，那么它们对应的 K 和 V 也是确定的。重新计算

$$
K_1,K_2,\dots,K_L
$$

和

$$
V_1,V_2,\dots,V_L
$$

实际上是在重复此前已经完成过的工作。

随着生成长度增加，这种重复会越来越严重。生成第一个 token 时处理一次 prompt，生成第二个 token 时又处理更长的一段序列，生成第三个 token 时再处理更长的一段序列。大量算力被浪费在已经计算过的历史 token 上。

## 六、KV Cache 如何消除重复计算

KV Cache 的思路非常直接：既然历史 token 的 K 和 V 不会因为未来 token 的出现而改变，就把它们缓存起来，以后直接复用。

假设在 Prefill 阶段已经保存了

$$
K_{\text{cache}}
=
[K_1,K_2,\dots,K_L],
$$

$$
V_{\text{cache}}
=
[V_1,V_2,\dots,V_L].
$$

现在新生成 token $x_{L+1}$。进入下一次解码时，只需要针对这个新 token 计算

$$
q_{L+1}=x_{L+1}W_Q,
$$

$$
k_{L+1}=x_{L+1}W_K,
$$

$$
v_{L+1}=x_{L+1}W_V.
$$

然后把新的 K 和 V 追加到缓存中：

$$
K_{\text{cache}}
\leftarrow
[K_1,K_2,\dots,K_L,k_{L+1}],
$$

$$
V_{\text{cache}}
\leftarrow
[V_1,V_2,\dots,V_L,v_{L+1}].
$$

当前 token 的注意力只需要执行

$$
\operatorname{softmax}
\left(
\frac{
q_{L+1}
K_{\text{cache}}^\top
}{
\sqrt{d_k}
}
\right)
V_{\text{cache}}.
$$

这样，模型就避免了对所有历史 token 重新做 QKV 投影和整层 Transformer 计算。

从工程角度看，KV Cache 的本质就是：保存所有未来 token 还会重复读取的中间结果。

## 七、为什么只缓存 K 和 V，而不缓存 Q

这是理解 KV Cache 时最常见的问题之一。

在生成第 $t$ 个 token 时，当前 Query $q_t$ 的任务是查询历史的 K，并利用历史 V 得到注意力输出。等这一轮注意力计算结束之后，$q_t$ 的使命也就结束了。未来生成第 $t+1$ 个 token 时，会产生新的 $q_{t+1}$，这时并不需要再次使用 $q_t$。

因此历史 Query 没有复用价值。

而历史 K 和 V 恰恰相反。生成第 $t$ 个 token 时要访问它们，生成第 $t+1$ 个 token 时还要访问，生成第 $t+2$ 个 token 时仍然要访问。只要这些历史 token 还处在上下文窗口中，它们对应的 K 和 V 就持续具有价值。

因此实际需要保存的是

$$
K_1,K_2,\dots,K_t
$$

以及

$$
V_1,V_2,\dots,V_t,
$$

而不是历史 Query。这也是它被称为 KV Cache 而不是 QKV Cache 的原因。

## 八、从张量形状看 Prefill 与 Decode 的区别

设 batch size 为 $B$，注意力头数量为 $H$，每个 head 的维度为 $d_h$，当前 prompt 长度为 $L$。

在传统 Multi-Head Attention 中，Prefill 阶段 Q、K、V 的典型形状为

$$
[B,H,L,d_h].
$$

此时注意力分数矩阵的形状为

$$
[B,H,L,L].
$$

这意味着 Prefill 阶段在执行一类大规模的矩阵乘矩阵运算。

进入 Decode 之后，每一步只有一个新 token。因此当前 Query 的形状变成

$$
[B,H,1,d_h],
$$

而缓存中的 K 和 V 仍然覆盖整个历史上下文：

$$
K_{\text{cache}},V_{\text{cache}}
\in
\mathbb{R}^{B\times H\times L\times d_h}.
$$

于是注意力分数的形状是

$$
[B,H,1,L].
$$

这时不再是完整的 $L\times L$ 注意力矩阵，而是一个新的 Query 与所有历史 Key 做匹配。

这一变化非常重要，因为它意味着 Prefill 和 Decode 的硬件特征完全不同。Prefill 通常包含大量大矩阵乘法，更容易把 GPU 的计算单元充分利用起来；Decode 每一步只有一个或少量 Query，却需要不断读取庞大的模型权重和 KV Cache，因此更容易受到显存带宽限制。

## 九、KV Cache 改变了什么复杂度

KV Cache 能显著提高解码速度，但它并没有让自注意力的全部复杂度从 $O(N^2)$ 变成 $O(N)$。

假设已经生成到第 $t$ 个 token。即使使用 KV Cache，当前 Query 仍然要和全部历史 Key 做点积：

$$
q_tK_{1:t}^\top.
$$

因此单步 attention 的计算量仍然与当前上下文长度 $t$ 成正比，即约为

$$
O(t).
$$

如果连续生成 $N$ 个 token，那么注意力相关计算累计仍然近似为

$$
1+2+\cdots+N
=
O(N^2).
$$

KV Cache 真正避免的是另一部分更昂贵的重复：历史 token 不再需要反复经过所有 Transformer 层，也不需要反复重新计算 QKV 投影、MLP、归一化等操作。

所以更准确地说，KV Cache 显著减少了“对历史上下文进行重复前向传播”的成本，而不是彻底消除注意力随序列长度增长带来的二次复杂度。

## 十、为什么上下文越长，Decode 往往越慢

使用 KV Cache 后，每一步虽然只计算一个新 token，但这个新 token 的 Query 仍然必须读取全部历史 Key 和 Value。因此上下文长度越大，需要访问的 KV Cache 就越多。

当上下文只有几百 token 时，读取 KV Cache 的成本相对较小；当上下文达到几万甚至十几万 token 时，每生成一个 token 都需要扫描大量历史 K/V。此时解码延迟会明显增加。

这也是为什么长上下文模型不仅对显存容量提出要求，也对显存带宽提出极高要求。Decode 阶段往往不是算术运算不足，而是大量数据需要从 HBM 中不断搬运到计算单元。

因此可以把现代大模型推理的一个重要特征概括为：

$$
\text{Prefill 更偏 compute-bound，Decode 更偏 memory-bandwidth-bound}.
$$

## 十一、KV Cache 为什么非常占显存

对于传统 Multi-Head Attention，设模型有 $N_L$ 层，KV head 数量为 $H_{kv}$，每个 head 维度为 $d_h$，batch size 为 $B$，上下文长度为 $L$，每个数占用 $b$ 字节。

由于每一层都要同时保存 K 和 V，所以 KV Cache 的近似显存占用为

$$
\text{KV Cache}
=
2BN_LH_{kv}Ld_hb.
$$

其中最前面的 2 来自 K 和 V 两份缓存。

可以看到，KV Cache 的大小与 batch size、层数、KV head 数、上下文长度以及 head dimension 都线性相关。尤其是上下文长度 $L$ 增大时，KV Cache 会同步线性增长。

因此在长上下文、高并发推理服务中，KV Cache 往往会占据大量 GPU 显存，有时甚至成为限制 batch size 和并发请求数量的主要因素。

## 十二、为什么现代大模型大量使用 GQA 和 MQA

传统 Multi-Head Attention 通常为每个 Query head 配置独立的 Key head 和 Value head。例如，如果模型有 32 个 attention heads，就可能有 32 个 Q heads、32 个 K heads 和 32 个 V heads。

这样做表达能力强，但 KV Cache 很大。

Multi-Query Attention，也就是 MQA，采用多个 Query head 共享同一组 K 和 V。Grouped-Query Attention，也就是 GQA，则介于 MHA 与 MQA 之间，让若干 Query heads 共享一个 KV head。

例如，模型可以有 32 个 Query heads，但只有 8 个 KV heads。这样 KV Cache 大小理论上就能缩小到传统 MHA 的约四分之一。

因此 GQA 并不仅仅是一个注意力结构上的变化，它也是非常重要的推理优化设计。它能够明显降低 KV Cache 的显存占用和带宽压力，所以近年来在大模型中非常常见。

## 十三、RoPE 与 KV Cache 的关系

现代 Decoder-only 大模型常常使用 RoPE，也就是 Rotary Position Embedding。

对于某个位置 $t$，模型先通过线性映射获得

$$
q_t=W_Qx_t,
$$

$$
k_t=W_Kx_t.
$$

然后根据当前位置对 Q 和 K 应用旋转位置编码：

$$
\tilde q_t=R_tq_t,
$$

$$
\tilde k_t=R_tk_t.
$$

实际注意力使用的是旋转后的 Query 和 Key：

$$
\tilde q_t\tilde k_j^\top.
$$

因此在常见实现中，KV Cache 中保存的是已经处理好位置编码后的 Key，也就是

$$
\tilde k_1,\tilde k_2,\dots,\tilde k_t,
$$

而 V 不需要经过 RoPE。

进入下一步解码时，只需要对新 token 生成新的 $q_t$、$k_t$、$v_t$，再根据当前位置对 Q 和 K 应用 RoPE，并把新的旋转后 Key 和 Value 追加进缓存即可。

从这个角度看，KV Cache 并不是简单保存线性层输出，而是保存后续 attention 真正会直接使用的中间表示。

## 十四、为什么 vLLM 特别关注 KV Cache

在单个请求的简单推理里，KV Cache 只是一段不断增长的张量。但在真实在线服务中，同时可能有几十、几百甚至更多请求，每个请求的上下文长度和生成长度都不同。如果简单为每个请求连续分配大块显存，很容易产生显存碎片，也难以高效复用空闲空间。

vLLM 的一个重要设计就是围绕 KV Cache 管理展开。PagedAttention 借鉴操作系统分页思想，把 KV Cache 划分为许多固定大小的 block，不要求一个请求的 KV 数据在物理显存中连续存放。这样可以降低碎片，提高 KV Cache 的利用率，并支持 continuous batching。

因此，PagedAttention 本质上不是改变 Attention 的数学定义，而是改变 KV Cache 在 GPU 内存中的组织和管理方式。

## 十五、用一个完整流程理解大模型推理

如果把整个过程连起来，可以把 Decoder-only LLM 推理理解为以下逻辑。

首先，用户输入 prompt。模型进入 Prefill 阶段，对所有 prompt token 一次性进行 Transformer 前向传播，每一层计算完整的 Q、K、V，并保存 K 和 V。随后，最后一个位置的 hidden state 用于预测第一个新 token。

接下来模型进入 Decode 阶段。每次只输入最新生成的 token。对于每一层，只计算该 token 对应的 Q、K、V，把新的 K 和 V 追加到缓存中，然后让当前 Query 与此前缓存下来的全部 Key 做匹配，再根据 attention 权重读取全部历史 Value。最终经过所有 Transformer 层和输出层之后得到下一个 token。

下一个 token 再重复同样过程。

因此整个推理过程可以概括为：

$$
\text{Prompt}
\rightarrow
\text{Prefill}
\rightarrow
\text{建立 KV Cache}
\rightarrow
\text{逐 token Decode}
\rightarrow
\text{不断扩展 KV Cache}.
$$

这条链路几乎贯穿了现代大模型推理优化的全部核心问题。

## 十六、从系统角度重新理解 KV Cache

如果只从算法角度看，KV Cache 似乎只是“避免重复计算”的一个小技巧；但从现代大模型服务系统角度看，它实际上是核心基础设施之一。

模型参数本身通常是固定的，而每个用户请求都会动态产生自己的 KV Cache。模型越大、上下文越长、并发越高，KV Cache 管理就越关键。服务系统需要解决缓存分配、释放、复用、分页、压缩、量化以及跨设备调度等问题。

这也是为什么很多高性能推理框架都会围绕 KV Cache 做大量优化。例如 vLLM 的 PagedAttention、TensorRT-LLM 的 KV Cache 管理、不同形式的 KV quantization，以及长上下文模型中的 cache eviction 或稀疏注意力方案，本质上都在试图降低 KV Cache 带来的计算、显存和带宽压力。

## 结语

理解 Transformer 推理时最值得建立的一个核心认识是：训练阶段和推理阶段虽然使用的是同一套模型参数，但其计算模式截然不同。

训练阶段通常把整个序列并行处理，Q、K、V 都可以一次计算完成；推理中的 Prefill 阶段与训练较为相似，而进入 Decode 之后，每一步只有一个新的 Query，但必须持续访问完整的历史 Key 和 Value。

因此，历史 Q 没有必要保留，而历史 K 和 V 会在未来每一个 token 的生成中被重复使用，这正是 KV Cache 存在的根本原因。

进一步地，随着上下文增长，KV Cache 会不断变大，单步 Decode 需要读取的数据也越来越多。于是现代 LLM Serving 的瓶颈逐渐从单纯的矩阵计算扩展到显存容量、显存带宽和缓存管理。GQA、MQA、PagedAttention、KV Cache 量化等技术，都是围绕这一现实展开的。

如果把这一点真正理解透彻，那么之后再学习 vLLM、FlashAttention、PagedAttention、Speculative Decoding 以及长上下文推理优化时，会发现它们并不是彼此孤立的技术，而是在共同解决同一条推理链路中的不同瓶颈。
