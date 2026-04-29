# DeepSeek-V4 技术报告中文翻译（严格逐段版）

> 原文：**DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence**  
> 作者：DeepSeek-AI  
> 原始链接：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf>  
> 版本说明：本文为**严格逐段中文翻译版**，以忠实传达原文信息为首要目标。除特别标注的“编者注”外，正文不主动加入总结、评论或二次改写。

---

## 标题

**DeepSeek-V4：迈向高效百万 Token 上下文智能**

---

## 摘要

我们发布了 DeepSeek-V4 系列的预览版本，其中包括两个强大的混合专家（Mixture-of-Experts, MoE）语言模型，分别是参数量为 1.6T（激活参数 49B）的 DeepSeek-V4-Pro，以及参数量为 284B（激活参数 13B）的 DeepSeek-V4-Flash，二者都支持 100 万 token 的上下文长度。DeepSeek-V4 系列在架构和优化方面纳入了若干关键升级：（1）一种混合注意力架构，将 Compressed Sparse Attention（CSA）与 Heavily Compressed Attention（HCA）结合起来，以提升长上下文效率；（2）Manifold-Constrained Hyper-Connections（mHC），用于增强传统残差连接；（3）Muon 优化器，用于实现更快收敛和更强训练稳定性。我们在超过 32T 的多样化高质量 token 上对两个模型进行预训练，随后又通过一条完整的后训练流水线，解锁并进一步增强其能力。DeepSeek-V4-Pro-Max 作为 DeepSeek-V4-Pro 的最大推理强度模式，为开源模型重新定义了当前最优水平，在核心任务上超越了其前代模型。与此同时，DeepSeek-V4 系列在长上下文场景下具有很高效率。在 100 万 token 上下文设定下，相比 DeepSeek-V3.2，DeepSeek-V4-Pro 仅需其单 token 推理 FLOPs 的 27% 和其 KV cache 的 10%。这使我们能够常态化支持 100 万 token 上下文，从而让长时程任务以及进一步的 test-time scaling 更具可行性。模型权重可在 <https://huggingface.co/collections/deepseek-ai/deepseek-v4> 获取。

---

## 图 1

**左图**：DeepSeek-V4-Pro-Max 及其对应模型在基准测试上的表现。  
**右图**：DeepSeek-V4 系列与 DeepSeek-V3.2 的推理 FLOPs 和 KV cache 大小。

---

## 目录

1. 引言  
2. 架构  
2.1 继承自 DeepSeek-V3 的设计  
2.2 流形约束超连接  
2.3 结合 CSA 与 HCA 的混合注意力  
2.3.1 压缩稀疏注意力  
2.3.2 强压缩注意力  
2.3.3 其他细节  
2.3.4 效率讨论  
2.4 Muon 优化器  
3. 通用基础设施

---

## 1. 引言

推理模型（DeepSeek-AI, 2025；OpenAI, 2024c）的出现，确立了一种新的 test-time scaling 范式，并推动大语言模型（LLMs）的性能显著提升。然而，这一扩展范式从根本上受限于标准注意力机制（vanilla attention mechanism）的二次计算复杂度（Vaswani et al., 2017），这在超长上下文和推理过程中造成了难以承受的瓶颈。与此同时，从复杂 agent 工作流到大规模跨文档分析，长时程场景与任务的出现，也使得对超长上下文的高效支持成为未来进展的关键。尽管近来的开源工作（Bai et al., 2025a；DeepSeek-AI, 2024；MiniMax, 2025；Qwen, 2025）已经提升了通用能力，但这种在处理超长序列时的核心架构低效性仍然是一个关键障碍，它限制了 test-time scaling 的进一步收益，也阻碍了对长时程场景与任务的进一步探索。

为了突破超长上下文中的效率壁垒，我们开发了 DeepSeek-V4 系列，其中包括参数量为 1.6T（激活参数 49B）的 DeepSeek-V4-Pro 预览版，以及参数量为 284B（激活参数 13B）的 DeepSeek-V4-Flash 预览版。通过架构创新，DeepSeek-V4 系列在处理超长序列时实现了计算效率上的显著跃升。这一突破使得对 100 万 token 上下文长度的高效支持成为可能，开启了下一代大语言模型的百万长度上下文新时代。我们相信，高效处理超长序列的能力，将开启 test-time scaling 的下一前沿，为更深入研究长时程任务铺平道路，并为探索在线学习等未来范式奠定必要基础。

与 DeepSeek-V3 架构（DeepSeek-AI, 2024）相比，DeepSeek-V4 系列保留了 DeepSeekMoE 框架（Dai et al., 2024）和 Multi-Token Prediction（MTP）策略，同时在架构和优化方面引入了若干关键创新。为了提升长上下文效率，我们设计了一种混合注意力机制，将 Compressed Sparse Attention（CSA）与 Heavily Compressed Attention（HCA）结合起来。CSA 沿序列维度压缩 KV cache，随后执行 DeepSeek Sparse Attention（DSA）（DeepSeek-AI, 2025）；而 HCA 则对 KV cache 施加更激进的压缩，但保持 dense attention。为了增强建模能力，我们引入了 Manifold-Constrained Hyper-Connections（mHC）（Xie et al., 2026），用于升级传统残差连接。此外，我们还将 Muon（Jordan et al., 2024；Liu et al., 2025）优化器引入 DeepSeek-V4 系列的训练中，从而实现更快的收敛和更好的训练稳定性。

为了实现 DeepSeek-V4 系列的高效训练与推理，并同时提升研发效率，我们引入了若干基础设施优化。首先，我们为 MoE 模块设计并实现了一个单一融合 kernel，使计算、通信和内存访问可以完全重叠。其次，我们采用 TileLang（Wang et al., 2026），这是一种领域专用语言（DSL），用于在开发效率与运行时效率之间取得平衡。第三，我们提供了高效的、batch-invariant 且 deterministic 的 kernel 库，以确保训练和推理过程中的 bitwise reproducibility。第四，我们将 FP4 quantization-aware training 引入到 MoE expert 权重和 indexer QK path 中，以降低内存和计算开销。第五，在训练框架方面，我们通过 tensor-level checkpointing 扩展了 autograd 框架，以实现细粒度的重计算控制；并通过面向 Muon 优化器的 hybrid ZeRO 策略、借助重计算和 fused kernels 实现的高性价比 mHC，以及用于管理压缩注意力的两阶段 contextual parallelism，提升训练效率。最后，在推理框架方面，我们设计了一种异构 KV cache 结构，并配合磁盘存储策略，以实现高效的 shared-prefix reuse。

通过采用混合 CSA 和 HCA，并在计算与存储上做精度优化，DeepSeek-V4 系列相比 DeepSeek-V3.2 实现了显著更低的推理 FLOPs 和大幅缩小的 KV cache，尤其是在长上下文设定下。图 1 右侧展示了 DeepSeek-V3.2 与 DeepSeek-V4 系列在单 token 推理 FLOPs 估算值和累计 KV cache 大小方面的对比。在 100 万 token 上下文场景下，即使是激活参数量更大的 DeepSeek-V4-Pro，其单 token FLOPs（按等效 FP8 FLOPs 计）也仅为 DeepSeek-V3.2 的 27%，而 KV cache 大小仅为其 10%。此外，激活参数量更小的 DeepSeek-V4-Flash 进一步推进了效率边界：在 100 万 token 上下文设定下，其单 token FLOPs 仅为 DeepSeek-V3.2 的 10%，KV cache 大小仅为其 7%。此外，对于 DeepSeek-V4 系列，路由专家参数采用 FP4 精度。尽管在现有硬件上，FP4 × FP8 运算的峰值 FLOPs 目前与 FP8 × FP8 相同，但从理论上讲，在未来硬件上它们可以实现额外 1/3 的效率提升，这将进一步增强 DeepSeek-V4 系列的效率。

在预训练期间，我们分别用 32T token 训练 DeepSeek-V4-Flash，用 33T token 训练 DeepSeek-V4-Pro。完成预训练后，这两个模型都能够原生且高效地支持 100 万长度上下文。在我们的内部评测中，DeepSeek-V4-Flash-Base 凭借更高参数效率的设计，已经在大多数 benchmark 上超过了 DeepSeek-V3.2-Base。DeepSeek-V4-Pro-Base 则进一步扩大了这一优势，在推理、编码、长上下文和世界知识任务上实现了全面领先，树立了 DeepSeek 基础模型中的新性能标准。

DeepSeek-V4 系列的后训练流水线采用两阶段范式：先独立培养领域专用专家，再通过 on-policy distillation 完成统一模型整合（Lu and Lab, 2025）。最初，对于每一个目标领域，例如数学、编码、agent 和 instruction following，都会分别独立训练一个专家模型。基础模型先在高质量的领域特定数据上进行监督微调（Supervised Fine-Tuning, SFT），以建立基础能力。随后，再使用 Group Relative Policy Optimization（GRPO）（DeepSeek-AI, 2025）进行强化学习（RL），并借助针对特定成功标准量身定制的奖励模型，进一步将模型优化到更符合该领域行为目标的状态。这一阶段会得到一组多样化的专用专家，每个专家都在各自领域中表现出色。最后，为了整合这些不同的专长，会通过 on-policy distillation 训练一个统一模型，其中统一模型作为 student，在 teacher 模型的指导下学习优化 reverse KL loss。

### 核心评测结果摘要

- **知识（Knowledge）**：在广泛世界知识评测中，DeepSeek-V4-Pro-Max，也就是 DeepSeek-V4-Pro 的最大推理强度模式，在 SimpleQA（OpenAI, 2024d）和 Chinese-SimpleQA（He et al., 2024）benchmark 上显著超过领先的开源模型。在教育知识方面，即通过 MMLU-Pro（Wang et al., 2024b）、HLE（Phan et al., 2025）和 GPQA（Rein et al., 2023）评测时，DeepSeek-V4-Pro-Max 相比其开源对手仅有微弱领先。尽管在这些知识类评测中，DeepSeek-V4-Pro-Max 仍落后于领先的闭源模型 Gemini-3.1-Pro，但它已经显著缩小了差距。
- **推理（Reasoning）**：通过扩展推理 token，DeepSeek-V4-Pro-Max 在标准推理 benchmark 上相对于 GPT-5.2 和 Gemini-3.0-Pro 展现出更优性能。不过，它的表现仍略逊于 GPT-5.4 和 Gemini-3.1-Pro，这表明其发展轨迹大约落后最前沿模型 3 到 6 个月。此外，DeepSeek-V4-Flash-Max 在复杂推理任务上实现了与 GPT-5.2 和 Gemini-3.0-Pro 相当的表现，确立了其作为一种高性价比架构的地位。
- **Agent**：在公开 benchmark 上，DeepSeek-V4-Pro-Max 与领先开源模型，如 Kimi-K2.6 和 GLM-5.1，大致处于同一水平，但略逊于前沿闭源模型。在我们的内部评测中，DeepSeek-V4-Pro-Max 超过了 Claude Sonnet 4.5，并接近 Opus 4.5 的水平。
- **长上下文（Long-Context）**：DeepSeek-V4-Pro-Max 在 100 万 token 上下文窗口下，在合成任务和真实用例上都表现强劲，甚至在学术 benchmark 上超过了 Gemini-3.1-Pro。
- **DeepSeek-V4-Pro 与 DeepSeek-V4-Flash 对比**：由于参数规模更小，DeepSeek-V4-Flash-Max 在知识类评测中的表现较弱。然而，当给予更大的 thinking budget 时，它在推理任务上取得了可比结果。在 agent 评测中，虽然 DeepSeek-V4-Flash-Max 在若干 benchmark 上与 DeepSeek-V4-Pro-Max 表现相当，但在更复杂、更高难度的任务上仍落后于后者。

---

## 2. 架构

总体而言，DeepSeek-V4 系列保留了 Transformer（Vaswani et al., 2017）架构和 Multi-Token Prediction（MTP）模块（DeepSeek-AI, 2024；Gloeckle et al., 2024），同时相较于 DeepSeek-V3 引入了若干关键升级：（1）首先，我们引入 Manifold-Constrained Hyper-Connections（mHC）（Xie et al., 2026），以增强传统残差连接；（2）其次，我们设计了一种混合注意力架构，通过 Compressed Sparse Attention 和 Heavily Compressed Attention 大幅提升长上下文效率；（3）第三，我们采用 Muon（Jordan et al., 2024；Liu et al., 2025）作为优化器。对于混合专家（Mixture-of-Experts, MoE）部分，我们仍然采用 DeepSeekMoE（Dai et al., 2024）架构，仅对 DeepSeek-V3 做了少量调整。Multi-Token Prediction（MTP）（DeepSeek-AI, 2024；Gloeckle et al., 2024；Li et al., 2024；Qi et al., 2020）的配置则与 DeepSeek-V3 完全一致。其他未特别说明的细节，均沿用 DeepSeek-V3（DeepSeek-AI, 2024）中已经建立的设置。图 2 展示了 DeepSeek-V4 的整体架构，具体细节如下所述。

### 2.1 继承自 DeepSeek-V3 的设计

**混合专家（Mixture-of-Experts）**。与此前的 DeepSeek 系列模型（DeepSeek-AI, 2024；DeepSeek-AI, 2024）一样，DeepSeek-V4 系列也在前馈网络（Feed-Forward Networks, FFNs）中采用 DeepSeekMoE 范式（Dai et al., 2024），即设置细粒度路由专家和共享专家。不同于 DeepSeek-V3 的地方在于，我们将用于计算 affinity scores 的激活函数从 `Sigmoid(·)` 改为 `Sqrt(Softplus(·))`。在负载均衡方面，我们仍采用无辅助损失（auxiliary-loss-free）策略（DeepSeek-AI, 2024；Wang et al., 2024a），并额外加入一个轻量的 sequence-wise balance loss，以防止单个序列内部出现极端不平衡。对于 DeepSeek-V4，我们移除了对路由目标节点数量的约束，并仔细重新设计了并行策略，以保持训练效率。此外，与 DeepSeek-V3 相比，我们将最前面的若干个 Transformer block 中的 dense FFN 层替换为使用 Hash routing（Roller et al., 2021）的 MoE 层。Hash routing 策略会根据输入 token ID 上的预定义哈希函数，为每个 token 确定目标专家。

**Multi-Token Prediction**。与 DeepSeek-V3 一样，DeepSeek-V4 系列也设置了 MTP 模块和目标函数。鉴于 MTP 策略已经在 DeepSeek-V3 中得到验证，我们在 DeepSeek-V4 系列中不做修改，直接采用同样的策略。

### 2.2 流形约束超连接

如图 2 所示，DeepSeek-V4 系列引入了 Manifold-Constrained Hyper-Connections（mHC）（Xie et al., 2026），以增强相邻 Transformer block 之间的传统残差连接。与朴素的 Hyper-Connections（HC）（Zhu et al., 2025）相比，mHC 的核心思想是将残差映射约束在一个特定流形上，从而在保留模型表达能力的同时，增强跨层信号传播的稳定性。本小节先简要介绍标准 HC，再说明我们如何设计 mHC 以实现稳定训练。

**标准 Hyper-Connections**。标准 HC 会将残差流的宽度扩展为原来的 `n_hc` 倍。具体来说，残差流的形状从 `R^d` 扩展到 `R^(n_hc × d)`，其中 `d` 是实际层输入的隐藏维度。记第 `l` 层之前的残差状态为 `X_l = [x_(l,1); ...; x_(l,n_hc)]^T ∈ R^(n_hc × d)`。HC 引入三个线性映射：输入映射 `A_l ∈ R^(1 × n_hc)`、残差变换 `B_l ∈ R^(n_hc × n_hc)` 和输出映射 `C_l ∈ R^(n_hc × 1)`。此时残差状态更新可写为：

`X_(l+1) = B_l X_l + C_l F_l(A_l X_l)`  (1)

其中，`F_l` 表示第 `l` 层（例如一个 MoE 层），其输入和输出形状均为 `R^d`。注意，实际层输入 `A_l X_l ∈ R^d` 仍然是 `d` 维，因此残差流宽度的扩展并不会影响内部层的设计。HC 将残差宽度与实际隐藏维度解耦出来，提供了一个额外的扩展维度，同时计算开销很小，因为 `n_hc` 通常远小于隐藏维度 `d`。然而，尽管 HC 已经展现出提升模型性能的潜力，我们发现，当堆叠多层时，训练过程会频繁出现数值不稳定，这阻碍了 HC 的进一步扩展。

**流形约束残差映射**。mHC 的核心创新，在于将残差映射矩阵 `B_l` 约束到双随机矩阵（doubly stochastic matrices）流形，也就是 Birkhoff polytope `M` 上，从而增强跨层信号传播的稳定性：

`B_l ∈ M := { M ∈ R^(n×n) | M 1_n = 1_n, 1_n^T M = 1_n^T, M ≥ 0 }`  (2)

这一约束保证映射矩阵的谱范数 `||B_l||_2` 被限制在 1 以内，因此残差变换是 non-expansive 的，这会在前向传播和反向传播过程中提高数值稳定性。此外，集合 `M` 在乘法下是封闭的，这保证了在深层堆叠 mHC 时的稳定性。同时，输入变换 `A_l` 和输出变换 `C_l` 也通过 Sigmoid 函数被约束为非负且有界，以避免信号抵消的风险。

**动态参数化**。三个线性映射的参数是动态生成的，它们被分解为动态成分（依赖输入）和静态成分（不依赖输入）。给定输入 `X_l ∈ R^(n_hc × d)`，首先将其展平并归一化：`X̂_l = RMSNorm(vec(X_l)) ∈ R^(1 × n_hc d)`。然后，我们遵循传统 HC 的方式，生成未受约束的原始参数 `Ã_l ∈ R^(1 × n_hc)`、`B̃_l ∈ R^(n_hc × n_hc)` 和 `C̃_l ∈ R^(n_hc × 1)`。随后再将其映射到满足约束的参数空间中。这里原文后续会继续给出具体公式。

### 2.3 结合 CSA 与 HCA 的混合注意力

随着上下文长度达到极端规模，注意力机制成为模型中的主要计算瓶颈。针对 DeepSeek-V4，我们设计了两种高效注意力架构，即 Compressed Sparse Attention（CSA）和 Heavily Compressed Attention（HCA），并采用二者交错的混合配置，从而在长文本场景下显著降低注意力计算成本。CSA 同时整合了压缩和稀疏注意力策略：它首先将每 `m` 个 token 的 Key-Value（KV）cache 压缩成一个条目，然后应用 DeepSeek Sparse Attention（DSA）（DeepSeek-AI, 2025），使每个 query token 只关注 `k` 个压缩后的 KV 条目。HCA 则以每 `m'`（远大于 `m`）个 token 的 KV cache 合并为一个条目的方式，追求极致压缩。CSA 和 HCA 的混合架构显著提升了 DeepSeek-V4 系列的长上下文效率，使得 100 万 token 上下文在实践中成为可能。本小节介绍我们混合注意力架构的核心技术，我们还提供了开源实现1，以便无歧义地说明更多细节。

#### 2.3.1 压缩稀疏注意力

CSA 的核心架构如图 3 所示。它首先将每 `m` 个 token 的 KV cache 压缩为一个条目，然后再应用 DeepSeek Sparse Attention 以进一步加速。

**压缩后的 Key-Value 条目**。设 `H ∈ R^(n×d)` 为一段输入隐藏状态序列，其中 `n` 为序列长度，`d` 为隐藏维度。CSA 首先计算两组 KV 条目 `C^a, C^b ∈ R^(n×c)` 及其对应的压缩权重 `Z^a, Z^b ∈ R^(n×c)`，其中 `c` 为 head dimension：

`C^a = H · W_KV^a`  (9)  
`C^b = H · W_KV^b`  
`Z^a = H · W_Z^a`  (10)  
`Z^b = H · W_Z^b`

其中，`W_KV^a, W_KV^b, W_Z^a, W_Z^b ∈ R^(d×c)` 为可训练参数。接下来，`C^a` 和 `C^b` 中每 `m` 个 KV 条目会依据各自的压缩权重以及可学习的位置偏置 `B_a, B_b ∈ R^(m×c)` 被压缩成一个条目，产生 `C^Comp ∈ R^((n/m)×c)`。每个压缩条目 `C_i^Comp ∈ R^c` 的计算方式为：

`[S^a_(mi:m(i+1)-1); S^b_(m(i-1):mi-1)] = Softmax_row([Z^a_(mi:m(i+1)-1) + B^a; Z^b_(m(i-1):mi-1) + B^b])`  (11)

`C_i^Comp = Σ_(j=mi)^(m(i+1)-1) S_j^a ⊙ C_j^a + Σ_(j=m(i-1))^(mi-1) S_j^b ⊙ C_j^b`  (12)

其中，`⊙` 表示 Hadamard 积；`Softmax_row(·)` 表示沿行维度执行 softmax，也就是在来自 `Z^a` 与 `Z^b` 的总计 `2m` 个元素上做归一化。当 `i = 0` 时，`Z^b_(m(i-1):mi-1)` 以负无穷填充，`C^b_(m(i-1):mi-1)` 以 0 填充。注意，每个 `C_i^Comp` 都来自 `2m` 个 KV 条目，但用于 `C_i^Comp` 的 `C^b` 下标与用于 `C_(i-1)^Comp` 的 `C^a` 下标存在重叠。因此，CSA 实际上将序列长度压缩为原来的 `1/m`。

**用于稀疏选择的闪电索引器（Lightning Indexer）**。得到压缩后的 KV 条目 `C^Comp` 之后，CSA 应用 DSA 策略，为核心注意力选择 top-k 个压缩 KV 条目。首先，CSA 使用与 `C^Comp` 相同的压缩操作，得到压缩后的索引器 key `K^{IComp} ∈ R^((n/m)×c^I)`，其中 `c^I` 是 indexer head dimension。然后，对于一个 query token `t`，我们以低秩方式生成索引器 query `{q_(t,1)^I, q_(t,2)^I, ..., q_(t,n_h^I)^I}`：

`c_t^Q = h_t · W_D^Q`  (13)  
`[q_(t,1)^I; q_(t,2)^I; ...; q_(t,n_h^I)^I] = q_t^I = c_t^Q · W_U^{IQ}`  (14)

其中，`h_t ∈ R^d` 是 query token `t` 的输入隐藏状态；`c_t^Q ∈ R^(d_c)` 是 query 的压缩潜变量；`d_c` 表示 query 压缩维度；`n_h^I` 表示 indexer query head 的数量；`W_D^Q ∈ R^(d×d_c)` 和 `W_U^{IQ} ∈ R^(d_c×(c^I n_h^I))` 分别是 indexer query 的降维矩阵和升维矩阵。接下来，query token `t` 与此前某个压缩块 `s`（`s < Floor(t/m)`）之间的索引分数 `I_(t,s) ∈ R` 计算如下：

`[w_(t,1)^I; w_(t,2)^I; ...; w_(t,n_h^I)^I] = w_t^I = h_t · W_w`  (15)

`I_(t,s) = Σ_(h=1)^(n_h^I) w_(t,h)^I · ReLU(q_(t,h)^I · K_s^{IComp})`  (16)

其中，`W_w ∈ R^(d×n_h^I)` 是可学习矩阵；`w_(t,h)^I ∈ R` 是第 `h` 个 indexer head 的权重。对于一个 query token `t`，给定其索引分数 `I_(t,:)`，我们使用一个 top-k 选择器，仅保留一部分压缩后的 KV 条目 `C_t^{SprsComp}` 供后续核心注意力使用：

`C_t^{SprsComp} = { C_s^{Comp} | I_(t,s) ∈ Top-k(I_(t,:)) }`  (17)

**共享 Key-Value 的 Multi-Query Attention**。选出稀疏 KV 条目之后，CSA 再以 Multi-Query Attention（MQA）（Shazeer, 2019）的方式执行核心注意力，其中 `C_t^{SprsComp}` 中的每个压缩 KV 条目同时作为 attention key 和 value。具体而言，对于 query token `t`，我们先从压缩 latent vector `c_t^Q` 生成注意力 query `{q_(t,1), q_(t,2), ..., q_(t,n_h)}`：

`[q_(t,1); q_(t,2); ...; q_(t,n_h)] = q_t = c_t^Q · W_U^Q`  (18)

其中，`n_h` 表示 query head 的数量；`W_U^Q ∈ R^(d_c×(cn_h))` 是 query 的升维矩阵。注意，latent query vector `c_t^Q` 与 indexer query 所使用的 latent vector 是共享的。接着，我们在 `{q_(t,i)}` 和 `C_t^{SprsComp}` 上执行 MQA：

`o_(t,i) = CoreAttn(query=q_(t,i), key=C_t^{SprsComp}, value=C_t^{SprsComp})`  (19)

其中，`o_(t,i) ∈ R^c` 是第 `t` 个 token 上第 `i` 个 head 的核心注意力输出；`CoreAttn(·)` 表示核心注意力操作。

**分组输出投影**。在 DeepSeek-V4 的配置中，`cn_h` 相当大。因此，若将核心注意力输出 `[o_(t,1); o_(t,2); ...; o_(t,n_h)] = o_t ∈ R^(cn_h)` 直接投影到 `d` 维隐藏状态，会带来很大计算负担。为缓解这一成本，我们设计了分组输出投影策略。具体而言，我们先将 `n_h` 个输出切分成 `g` 组，然后对每一组输出 `o_(t,i)^G ∈ R^(c n_h / g)`，投影到一个 `d_g` 维中间输出 `o_(t,i)^{G' } ∈ R^(d_g)`，其中 `d_g < c n_h / g`。最后，我们再将中间输出 `[o_(t,1)^{G' }; o_(t,2)^{G' }; ...; o_(t,g)^{G' }] ∈ R^(d_g g)` 投影到最终注意力输出 `ô_t ∈ R^d`。

#### 2.3.2 强压缩注意力

HCA 的核心架构如图 4 所示。它以更激进的方式压缩 KV cache，但不使用稀疏注意力。

**压缩后的 Key-Value 条目**。总体而言，HCA 的压缩策略与 CSA 类似，但采用更大的压缩倍率 `m'`（远大于 `m`），并且不执行重叠压缩。设 `H ∈ R^(n×d)` 为输入隐藏状态序列，HCA 首先计算原始 KV 条目 `C ∈ R^(n×c)` 及其对应的压缩权重 `Z ∈ R^(n×c)`：

`C = H · W_KV`  (20)  
`Z = H · W_Z`  (21)

其中，`W_KV, W_Z ∈ R^(d×c)` 为可训练参数。接着，`C` 中每 `m'` 个 KV 条目会依据压缩权重以及可学习的位置偏置 `B ∈ R^(m'×c)` 被压缩成一个条目，得到 `C^Comp ∈ R^((n/m')×c)`。每个压缩条目 `C_i^Comp ∈ R^c` 的计算方式如下：

`S_(m'i:m'(i+1)-1) = Softmax_row(Z_(m'i:m'(i+1)-1) + B)`  (22)

`C_i^Comp = Σ_(j=m'i)^(m'(i+1)-1) S_j ⊙ C_j`  (23)

通过这一压缩操作，HCA 将序列长度压缩为原来的 `1/m'`。

**共享 Key-Value 的 MQA 与分组输出投影**。HCA 也采用与 CSA 相同的共享 KV MQA 和分组输出投影策略。在 KV 压缩之后，对于 query token `t`，HCA 先以低秩方式生成注意力 query `{q_(t,1), q_(t,2), ..., q_(t,n_h)}`：

`c_t^Q = h_t · W_D^Q`  (24)  
`[q_(t,1); q_(t,2); ...; q_(t,n_h)] = q_t = c_t^Q · W_U^Q`  (25)

其中，`h_t ∈ R^d` 为 query token `t` 的输入隐藏状态；`n_h` 为 query head 数；`W_D^Q ∈ R^(d×d_c)` 和 `W_U^Q ∈ R^(d_c×(cn_h))` 分别是 query 的降维与升维矩阵。接下来，我们在 `{q_(t,i)}` 和 `C^Comp` 上执行 MQA：

`o_(t,i) = CoreAttn(query=q_(t,i), key=C^Comp, value=C^Comp)`  (26)

其中，`o_(t,i) ∈ R^c` 是第 `t` 个 token 上第 `i` 个 head 的核心注意力输出。随后，与 CSA 一样，HCA 会将 `n_h` 个输出分成 `g` 组，并对每组输出 `o_(t,i)^G ∈ R^(c n_h / g)` 投影到 `d_g` 维中间输出 `o_(t,i)^{G' } ∈ R^(d_g)`，其中 `d_g < c n_h / g`。最后，HCA 再把中间输出 `[o_(t,1)^{G' }; o_(t,2)^{G' }; ...; o_(t,g)^{G' }] ∈ R^(d_g g)` 投影到最终注意力输出 `ô_t ∈ R^d`。

#### 2.3.3 其他细节

除上述 CSA 和 HCA 的核心架构外，我们的混合注意力还纳入了若干其他技术。为使表述更清晰，我们在前文介绍中省略了这些附加技术，并将在本小节中简要说明。需要注意的是，本小节只关注它们的核心思想，为简洁起见，可能会省略一些很小的细节。我们鼓励读者参考我们的开源实现，以获得无歧义的细节。

**Query 与 Key-Value 条目归一化**。对于 CSA 和 HCA，我们会在核心注意力操作之前，对每个 head 的 query 以及压缩 KV 条目的唯一一个 head 额外执行一次 RMSNorm。这样的归一化可以避免 attention logit 爆炸，并可能提升训练稳定性。

**部分 Rotary Positional Embedding**。对于 CSA 和 HCA，我们会对 attention query、KV 条目以及核心注意力输出，部分使用 Rotary Positional Embedding（RoPE）（Su et al., 2024）。具体来说，对于 CSA 和 HCA 中使用的每个 query 向量和 KV 条目向量，我们会在其最后 64 维上施加 RoPE。由于 KV 条目同时充当 attention key 和 value，朴素的核心注意力输出 `{o_(t,i)}` 将携带绝对位置嵌入，这来自 KV 条目加权求和后的结果。作为应对，我们还会在每个 `o_(t,i)` 的最后 64 维上，以位置 `-i` 再施加一次 RoPE。这样一来，核心注意力的输出也会携带相对位置嵌入，也就是说，每个 KV 条目对核心注意力输出的贡献也会与 query 和 KV 条目之间的距离相关。

**滑动窗口注意力的附加分支**。为了在 CSA 和 HCA 中严格保持因果性，每个 query 只能关注此前的压缩 KV block。因此，query 无法访问与自己处于同一压缩 block 内的其他 token 的信息。与此同时，在语言建模中，最近的 token 往往与当前 query token 更相关。基于这些原因，我们为 CSA 和 HCA 都额外引入了一条以滑动窗口方式工作的补充注意力分支，以更好地建模局部依赖。具体来说，对于每个 query token，我们还会额外生成与最近 `n_win` 个 token 对应的 `n_win` 个未压缩 KV 条目。在 CSA 和 HCA 的核心注意力中，这些滑动窗口内的 KV 条目会与压缩 KV 条目一起使用。

**Attention Sink**。在 CSA 和 HCA 的核心注意力中，我们使用了 attention sink 技巧（OpenAI, 2025；Xiao et al., 2024）。具体来说，我们设置一组可学习的 sink logits `{z'_1, z'_2, ..., z'_(n_h)}`。对于第 `h` 个注意力 head，`Exp(z'_h)` 会被加到 attention score 的分母中：

`s_(h,i,j) = Exp(z_(h,i,j)) / (Σ_k Exp(z_(h,i,k)) + Exp(z'_h))`  (27)

其中，`s_(h,i,j), z_(h,i,j) ∈ R` 分别表示第 `h` 个注意力 head 上，第 `i` 个 query token 与第 `j` 个此前 token 或压缩 block 之间的 attention score 和 attention logit。该技术使得每个 query head 可以调节其总 attention score 不必等于 1，甚至可以接近 0。

#### 2.3.4 效率讨论

由于采用了混合 CSA 和 HCA，并结合低精度计算与存储，DeepSeek-V4 系列的注意力模块在 attention FLOPs 和 KV cache 大小方面都实现了显著效率，尤其是在长上下文场景中。首先，我们对 KV 条目采用混合存储格式：对于 Rotary Positional Embedding（RoPE）相关维度使用 BF16 精度，而剩余维度使用 FP8 精度。这种混合表示方式相比纯 BF16 存储，可将 KV cache 大小减少近一半。其次，lightning indexer 内部的注意力计算使用 FP4 精度，这会在超长上下文场景下加速注意力操作。第三，相较于 DeepSeek-V3.2，DeepSeek-V4 系列选择了更小的 attention top-k，从而提升了模型在短文本和中等长度文本上的效率。最后，也是最重要的一点，压缩注意力和混合注意力技术大幅降低了 KV cache 大小和计算 FLOPs。

以 head dimension 为 128 的 BF16 GQA8（Ainslie et al., 2023）作为基线，也就是大语言模型注意力的一种常见配置，在 100 万上下文设定下，DeepSeek-V4 系列的 KV cache 大小可以显著降低到该基线的大约 2%。此外，即便与已经相当高效的 DeepSeek-V3.2（DeepSeek-AI, 2025）相比，DeepSeek-V4 系列在效率上仍然具有显著优势。它们在推理 FLOPs 和 KV cache 大小上的对比，展示在图 1 的右侧。

### 2.4 Muon 优化器

由于 Muon（Jordan et al., 2024；Liu et al., 2025）具有更快的收敛速度和更好的训练稳定性，我们在 DeepSeek-V4 系列的大多数模块中采用该优化器。我们所使用的完整 Muon 优化算法总结于算法 1。

**基本配置**。对于 embedding 模块、prediction head 模块、mHC 模块中的静态偏置和 gating factors，以及所有 RMSNorm 模块的权重，我们保留 AdamW（Loshchilov and Hutter, 2017）优化器。其余所有模块则使用 Muon 更新。遵循 Liu et al. (2025)，我们也对 Muon 参数施加权重衰减，采用 Nesterov（Jordan et al., 2024；Nesterov, 1983）技巧，并对更新矩阵的 Root Mean Square（RMS）进行重新缩放，以便复用我们为 AdamW 设置的超参数。不同之处在于，我们使用 hybrid Newton-Schulz iterations 来完成正交化。

**Hybrid Newton-Schulz Iterations**。对于给定矩阵 `M`，设其奇异值分解（SVD）为 `M = UΣV^T`。Newton-Schulz 迭代的目标，是将 `M` 近似正交化为 `UV^T`。通常，会先将 `M` 归一化为 `M_0 = M / ||M||_F`，以确保其最大奇异值不超过 1。随后，每一步 Newton-Schulz 迭代执行如下操作：

`M_k = aM_(k-1) + b(M_(k-1) M_(k-1)^T) M_(k-1) + c(M_(k-1) M_(k-1)^T)^2 M_(k-1)`  (28)

我们的 hybrid Newton-Schulz 迭代共执行 10 步，分成两个阶段。在前 8 步中，我们采用系数 `(a, b, c) = (3.4445, -4.7750, 2.0315)`，以实现快速收敛，使奇异值靠近 1。在最后 2 步中，我们切换为系数 `(a, b, c) = (2, -1.5, 0.5)`，从而使奇异值稳定地精确收敛到 1。

**避免注意力 logit 爆炸**。DeepSeek-V4 系列的注意力架构允许我们直接对 attention query 和 KV 条目施加 RMSNorm，这能有效防止 attention logit 爆炸。因此，我们在 Muon 优化器中不再采用 QK-Clip 技术（Liu et al., 2025）。

---

## 3. 通用基础设施

### 3.1 专家并行中的细粒度通信-计算重叠

混合专家（Mixture-of-Experts, MoE）可以通过 Expert Parallelism（EP）加速。然而，EP 需要复杂的跨节点通信，并对互连带宽和延迟提出很高要求。为了缓解 EP 中的通信瓶颈，并在较低互连带宽需求下实现更高的端到端性能，我们提出了一种细粒度 EP 方案，将通信与计算融合到单一的流水化 kernel 中，以实现通信-计算重叠。

**通信延迟可以被隐藏**。我们这套 EP 方案的关键洞见是，在 MoE 层中，通信延迟可以有效隐藏在计算之下。正如图 5 所示，在 DeepSeek-V4 系列中，每个 MoE 层主要可以分解为四个阶段：两个通信受限阶段 Dispatch 和 Combine，以及两个计算受限阶段 Linear-1 和 Linear-2。我们的 profiling 表明，在单个 MoE 层内，总通信时间小于总计算时间。因此，在将通信与计算融合到统一流水线之后，计算仍然是主导瓶颈，这意味着系统可以容忍更低的互连带宽，而不会降低端到端性能。

**细粒度 EP 方案**。为了进一步降低互连带宽需求，并放大重叠带来的收益，我们引入了一种更细粒度的 expert 划分方案。受多项相关工作（Aimuyo et al., 2025；Zhang et al., 2025b）的启发，我们将 expert 切分并调度为多个 wave。每个 wave 由一小部分 expert 构成。一旦 wave 内所有 expert 完成通信，计算就可以立即开始，而无需等待其他 expert。在稳定状态下，当前 wave 的计算、下一 wave 的 token 传输，以及已完成 expert 的结果发送都会并发进行，如图 5 所示。这会在 expert 之间形成一条细粒度流水线，使计算和通信在整个 wave 过程中保持连续。基于 wave 的调度在强化学习（RL）rollout 这类极端场景下能带来加速，这类场景通常会遇到带长尾的小 batch。

**性能与开源 Mega-Kernel**。我们在 NVIDIA GPU 和华为 Ascend NPU 平台上验证了这一细粒度 EP 方案。与强基线的非融合实现相比，它在一般推理工作负载上可达到 `1.50 ~ 1.73×` 的加速，而在 RL rollout 和高速 agent serving 等延迟敏感场景中，最高可达到 `1.96×`。我们已经将基于 CUDA 的 mega-kernel 实现开源，并命名为 MegaMoE2，作为 DeepGEMM 的一个组件。

**观察与建议**。我们分享了一些来自 kernel 开发的观察与经验，并向硬件厂商提出若干建议，希望有助于实现更高效的硬件设计，并推动更好的软硬协同设计：

- **计算-通信比率**。完整的通信-计算重叠依赖于计算-通信比率，而不只是带宽本身。记峰值计算吞吐为 `C`，互连带宽为 `B`，当 `C / B ≤ V_comp / V_comm` 时，通信可以被完全隐藏，其中 `V_comp` 表示计算量，`V_comm` 表示通信量。对于 DeepSeek-V4-Pro，每个 token-expert 对需要 `6hd` FLOPs（SwiGLU gate、up、down projections），但只需要 `3h` 字节通信（FP8 Dispatch + BF16 Combine），于是该条件可化简为：

`C / B ≤ 2d = 6144 FLOPs/Byte`

也就是说，每 1 GBps 的互连带宽，就足以隐藏 6.1 TFLOP/s 计算所需的通信。一旦带宽达到这一阈值，它就不再是瓶颈，继续投入额外硅面积去进一步提高带宽，将会带来递减收益。我们鼓励未来硬件设计瞄准这样的平衡点，而不是无条件地提升带宽。

- **功耗预算**。极致的 kernel 融合会让计算、内存和网络同时处于高负载，因此功耗限频会成为关键性能限制因素。我们建议未来硬件设计为这种完全并发的工作负载预留足够的功耗裕量。

- **通信原语**。我们采用一种 pull-based 方法，由每个 GPU 主动从远端 GPU 读取数据，从而避免细粒度 push 带来的高通知延迟。若未来硬件能提供更低延迟的跨 GPU 信号机制，那么 push 将变得可行，也能支持更自然的通信模式。

- **激活函数**。我们建议将 SwiGLU 替换为一种低成本的逐元素激活函数，该函数不包含指数或除法运算。这会直接减轻 post-GEMM 处理负担；在相同参数预算下，去掉 gate projection 还能扩大中间维度 `d`，从而进一步降低带宽需求。

### 3.2 基于 TileLang 的灵活高效 Kernel 开发

在实践中，我们精细的模型架构本会产生数百个细粒度的 Torch ATen operator。我们采用 TileLang（Wang et al., 2026）开发了一组融合 kernel，用来替代其中绝大多数 operator，以很小的开发成本实现最优性能。它还允许我们在验证过程中快速原型化 attention variants 等算子。这些 kernel 在模型架构开发、大规模训练以及最终推理服务的生产部署中都扮演关键角色。作为一种领域专用语言（DSL），TileLang 在开发效率与运行时效率之间取得平衡，既支持快速开发，也支持在同一代码库中进行深入的、迭代式优化。此外，我们还与 TileLang 社区紧密合作，以促进更敏捷、更高效、更稳定的 kernel 开发工作流。

**通过 Host Codegen 降低调用开销**。随着加速器性能持续增长，CPU 侧编排开销正变得越来越突出。对于小而高度优化的 kernel 来说，这类固定的 host 开销很容易限制利用率和吞吐。该开销的一个常见来源在于，host 侧逻辑，例如运行时契约检查，通常出于灵活性而使用 Python 编写，因此会带来每次调用固定的额外成本。

我们通过 Host Codegen 来降低这一开销，即将大部分 host 侧逻辑移动到自动生成的 host 代码中。具体而言，我们先在 IR（Intermediate Representation）层共同生成 device kernel 和一个轻量级 host launcher，并将语言前端解析得到的必要元数据，例如数据类型、rank/shape 约束以及 stride/layout 假设，嵌入其中。随后，这个 launcher 会被 lower 成基于 TVM-FFI（Chen et al., 2018）框架的 host 源代码。TVM-FFI 的紧凑调用约定和 zero-copy tensor 互操作共同将 host 侧开销降到最低。在运行时，这段生成的 host 代码执行参数校验和参数封装，从而把所有逐次调用的检查都移出了 Python 执行路径。我们的测量表明，CPU 侧验证开销从每次调用数十到数百微秒，下降到每次调用不足一微秒。

**基于 SMT 求解器辅助的形式化整数分析**。TileLang kernel 涉及复杂的 tensor 索引整数运算，因此需要强大的形式化整数分析能力。在 layout inference、memory hazard detection 和 bound analysis 等编译过程阶段，编译器必须验证整数表达式是否满足特定性质，才能启用相应优化。因此，更强的形式分析能力可以解锁更高级、更复杂的优化机会。

为此，我们将 Z3 SMT solver（De Moura and Bjørner, 2008）集成到 TileLang 的代数系统中，为 tensor 程序中的大多数整数表达式提供形式分析能力。我们通过将 TileLang 的整数表达式翻译为 Z3 的无量词非线性整数算术（QF_NIA），在计算开销和形式表达能力之间取得平衡。基于整数线性规划（ILP）求解器，QF_NIA 能无缝处理 kernel 中常见的标准线性整数表达式。此外，其内在的非线性推理能力还可以有效解决例如可变 tensor shape 下的向量化这类高级问题。在合理资源限制下，Z3 显著提升了整体优化效果，同时将编译时间额外开销限制在几秒之内。它对多个编译阶段都产生了显著影响，包括向量化、barrier 插入和代码简化。

**数值精度与按位可复现性**。在生产环境中，数值正确性和可复现性与纯粹吞吐同样关键。因此，我们默认优先保证精度：在编译器层禁用 fast-math 优化，而所有可能影响精度的近似实现只会作为显式、可选的前端算子提供（例如 `T.__exp`、`T.__log` 和 `T.__sin`）。反过来，当需要严格的 IEEE-754 语义时，TileLang 也提供带显式舍入模式的 IEEE 合规 intrinsic（例如 `T.ieee_fsqrt`、`T.ieee_fdiv` 和 `T.ieee_add`），从而允许开发者精确指定数值行为。

我们还将 bitwise reproducibility 作为验证 kernel 相对于手写 CUDA baseline 的目标。我们使 TileLang 的代数化简和 lowering 规则与主流 CUDA 工具链（例如 NVCC）对齐，从而避免引入意外按位差异的变换。layout 标注（例如 `T.annotate_layout`）进一步允许用户固定与 layout 相关的 lowering 决策，使求值和累加顺序与参考 CUDA 实现保持一致，从而在需要时实现按位一致的输出。我们的评测表明，这些以精度和可复现性为导向的设计选择，并不会牺牲性能：在保守默认设置下，TileLang kernel 仍然具有竞争力，同时也为更高速度提供了可选择放宽数值约束的控制开关。

### 3.3 高性能、batch-invariant 且 deterministic 的 Kernel 库

为了实现高效训练与推理，我们开发了一整套高性能计算 kernel。除了基础功能和最大化硬件利用率之外，另一个关键设计目标，是确保预训练、后训练和推理流水线之间的训练可复现性和 bitwise 对齐。因此，我们实现了端到端、按位 batch-invariant 且 deterministic 的 kernel，并将性能开销控制到最小。这些 kernel 对调试、稳定性分析以及一致的后训练行为都很有帮助。

**Batch Invariance**。Batch invariance 保证任意给定 token 的输出，无论其在 batch 中的位置如何，都保持按位一致。为实现 batch invariance，主要挑战如下：

- **Attention**。为了实现 batch invariance，我们不能使用 split-KV 方法（Dao et al., 2023）。该方法会将单个序列上的注意力计算分发到多个 Streaming Multiprocessor（SM）上，以平衡 SM 的负载。然而，放弃这一技术会导致严重的 wave-quantization 问题3，这会对 GPU 利用率产生不利影响。为解决这一问题，我们开发了一种双 kernel 策略，用于 batch-invariant decoding。第一个 kernel 在单个 SM 内完成整个序列的 attention 输出计算，以保证在 wave 被完全填满时具有高吞吐。第二个 kernel 为了减少最后一个未填满 wave 的延迟，并从而缓解 wave-quantization，会让多个 SM 服务于单个序列。为了保证这两个 kernel 的按位一致性，我们仔细设计了第二个 kernel 的计算路径，确保其累加顺序与第一个 kernel 完全相同。此外，第二个 kernel 还使用 thread-block cluster 内部的 distributed shared memory4，以支持跨 SM 的高速数据交换。这种双 kernel 方法将 batch-invariant decoding 的额外开销有效限制到可以忽略的程度。

- **矩阵乘法**。传统的 cuBLAS 库（NVIDIA Corporation, 2024）无法实现 batch invariance。因此，我们在端到端上用 DeepGEMM（Zhao et al., 2025）替换它。此外，对于非常小的 batch size，常规实现通常会采用 split-k（Osama et al., 2023）技术来提升性能。不幸的是，split-k 技术无法保证 batch invariance，而这恰恰是 DeepSeek-V4 的一个关键特性。因此，我们在大多数场景中放弃了 split-k，不过这可能带来性能下降。为此，我们引入了一组优化，使我们的矩阵乘法实现在大多数主要场景中能够追平甚至超过标准 split-k 的性能。

**Determinism**。Deterministic training 对调试硬件或软件问题非常有帮助。此外，当训练出现异常，例如 loss spike 时，determinism 也能帮助研究者更容易定位数值原因，并进一步改进模型设计。训练中的 non-determinism 通常来自非确定性的累加顺序，而这往往与 atomic addition 指令的使用有关。这个问题主要出现在 backward pass 中，尤其体现在以下几个部分：

- **Attention Backward**。在稀疏注意力的反向传播常规实现中，我们使用 `atomicAdd` 来为 KV token 累加梯度。由于浮点加法不满足结合律，这会引入 non-determinism。为解决这一问题，我们为每个 SM 分配独立的累加 buffer，随后再在全局范围内对所有 buffer 执行 deterministic summation。

- **MoE Backward**。当来自不同 rank 的多个 SM 并发向接收 rank 上的同一个 buffer 写入数据时，对写入位置的协商也会引入 non-determinism。为解决这一问题，我们在每个单独 rank 内设计了 token 顺序预处理机制，并结合跨多个 rank 的 buffer 隔离。该策略确保了 expert parallelism 发送结果的确定性，以及 MoE backward 中累加顺序的确定性。

- **mHC 中的矩阵乘法**。mHC 涉及一个输出维度仅为 24 的矩阵乘法。在非常小的 batch size 下，我们不得不使用 split-k（Osama et al., 2023）算法，而其朴素实现会导致 non-determinism。为解决这一问题，我们将每个 split 部分分别输出，并在后续 kernel 中执行 deterministic reduction，从而同时保留性能与确定性。

### 3.4 FP4 量化感知训练

为了在部署时实现推理加速和内存节省，我们在后训练阶段引入 Quantization-Aware Training（QAT）（Jacob et al., 2018），使模型能够适应量化所引入的精度退化。我们将 FP4（MXFP4）量化（Rouhani et al., 2023）应用于两个组件：（1）MoE expert 权重，它们是 GPU 显存占用的主要来源之一（OpenAI, 2025）；（2）CSA 的 indexer 中的 Query-Key（QK）路径，在这一路径中，QK 激活会完全以 FP4 格式进行缓存、加载和乘法运算，从而在长上下文场景下加速 attention score 计算。此外，在这一 QAT 过程中，我们还将 index score `I_:,:` 从 FP32 进一步量化到 BF16。该优化让 top-k selector 获得了 `2×` 加速，同时仍保持 99.7% 的 KV 条目召回率。

对于 MoE expert 权重，遵循 QAT 的常见做法，优化器维护的 FP32 master weights 会先被量化到 FP4，然后再反量化回 FP8 用于计算。值得注意的是，我们的 FP4 到 FP8 反量化是无损的。这是因为 FP8（E4M3）相比 FP4（E2M1）多出 2 个指数位，因此拥有更大的动态范围。因此，只要每个 FP8 量化 block（`128 × 128` tile）内部各 FP4 子 block（`1 × 32` tile）的最大与最小 scale factor 之比不超过某个阈值，那么细粒度 scale 信息就可以被 FP8 扩展后的动态范围完全吸收。我们在经验上验证了当前权重满足这一条件。这使得整个 QAT 流水线能够在无需任何修改的情况下，完全复用现有的 FP8 训练框架。在 backward pass 中，梯度是相对于 forward pass 中相同的 FP8 权重计算的，并直接传回到 FP32 master weights，这等价于通过量化操作应用 Straight-Through Estimator（STE）。这也避免了重新量化转置权重的需要。

在 RL 训练的推理和 rollout 阶段，由于不涉及 backward pass，我们直接使用真实的 FP4 量化权重，而不是模拟量化。这确保了采样阶段的模型行为与在线部署完全一致，同时还能减少 kernel 的内存加载以获得实际加速，并显著降低内存消耗。对于 CSA 的 indexer 中的 QK 路径，我们也采用类似处理方式。

### 3.5 训练框架

我们的训练框架建立在为 DeepSeek-V3（DeepSeek-AI, 2024）开发的可扩展且高效的基础设施之上。在训练 DeepSeek-V4 时，我们继承了这一稳健基础，同时引入了若干关键创新，以适配其新的架构组件，特别是 Muon 优化器、mHC 和混合注意力机制，并同时保持较高的训练效率和稳定性。

#### 3.5.1 Muon 的高效实现

Muon 优化器需要完整的梯度矩阵来计算参数更新，这在与 Zero Redundancy Optimizer（ZeRO）（Rajbhandari et al., 2020）结合时带来了挑战。传统 ZeRO 是为 AdamW 这类逐元素优化器设计的，在这种情况下，一个参数矩阵可以被分割到多个 rank 上分别更新。为解决这一冲突，我们为 Muon 设计了一种混合式的 ZeRO bucket assignment 策略。

对于 dense 参数，我们限制 ZeRO 并行的最大规模，并采用 knapsack 算法把参数矩阵分配到这些 rank 上，以确保每个 rank 管理的负载大致均衡。每个 rank 上的 bucket 会被 padding 到与所有 rank 中最大 bucket 相同的大小，以便执行高效的 reduce-scatter 操作。在我们的配置中，每个 rank 管理不超过 5 个参数矩阵，因此这种 padding 通常只带来不到 10% 的内存开销。当 data parallelism 的总体规模超过 ZeRO 的限制时，我们会在额外的数据并行组上冗余计算 Muon update，以计算换取更低的总 bucket 内存。

对于 MoE 参数，我们独立优化每个 expert。我们首先把所有层中所有 expert 的 SwiGLU（Shazeer, 2020）down projection 矩阵展平，其后是所有 up projection 矩阵和 gate 矩阵。然后，我们对这个展平后的向量做 padding，确保能够在不切分任何逻辑独立矩阵的前提下，将其均匀分配到所有 rank 上。由于 expert 数量很多，我们不会对 MoE 参数施加 ZeRO 并行规模上限，同时 padding 开销也可以忽略不计。

此外，在每个 rank 上，形状相同且连续的参数会被自动合并，从而让 Newton-Schulz 迭代能够以 batched 方式执行，以获得更好的硬件利用率。进一步地，我们观察到，Muon 中的 Newton-Schulz 迭代在使用 BF16 矩阵乘法计算时仍然稳定。基于这一点，我们又将需要在数据并行 rank 之间同步的 MoE 梯度以 stochastic rounding 的方式量化到 BF16 精度，从而把通信量减半。为避免低精度加法器引入的累加误差，我们使用两阶段方案来替代传统的基于 tree 或 ring 的 reduce-scatter collective：首先，通过 all-to-all 操作在各 rank 之间交换本地梯度；然后，每个 rank 再在本地以 FP32 完成求和。该设计保持了数值鲁棒性。

#### 3.5.2 mHC 的低成本高内存效率实现

与传统残差连接相比，引入 mHC 会增加激活内存消耗以及 pipeline stage 之间的通信量。为了缓解这些成本，我们实现了若干优化策略。

首先，我们为训练和推理分别精心设计并实现了 mHC 的 fused kernels。其次，我们引入了一种重计算策略，对中间张量进行选择性 checkpoint。具体来说，我们会重计算层间的大多数隐藏状态以及所有归一化后的层输入，同时避免重计算那些计算代价高昂的操作。这样可以在内存节省与计算开销之间取得平衡。第三，我们调整了 DualPipe 1F1B 的重叠方案，以适配增加后的 pipeline 通信量，并使 mHC 中的部分操作能够并发执行。

总体而言，这些优化将 mHC 对重叠式 1F1B pipeline stage 所带来的 wall-time 开销限制在仅 6.7%。更多工程优化细节可参见专门的 mHC 论文（Xie et al., 2026）。

#### 3.5.3 面向长上下文注意力的上下文并行

传统的 Context Parallelism（CP）会沿序列维度切分，使每个 rank 维护连续的 `s` 个 token。这会给我们的压缩注意力机制（即 CSA 与 HCA）带来两个挑战。一方面，训练样本是由多个序列打包而成的，而每个序列都会独立地按 `m`（或 `m'`）进行压缩，尾部不足 `m` 的 token 会被丢弃。因此，压缩后的 KV 长度通常小于 `s/m`，并且在不同 rank 之间不一致。另一方面，压缩过程需要连续的 `m` 个 KV 条目，而这些条目可能跨越两个相邻 CP rank 的边界。

为了解决这些挑战，我们设计了一种两阶段通信方法。在第一阶段，每个 rank `i` 会把自己最后 `m` 个未压缩 KV 条目发送给 rank `i + 1`。随后，rank `i + 1` 会将接收到的部分条目与其本地的 `s` 个未压缩 KV 条目一起压缩，产生长度固定为 `s/m + 1` 的压缩条目，其中包含一些 padding 条目。在第二阶段，我们在所有 CP rank 之间执行 all-gather，以收集各 rank 本地压缩后的 KV 条目。接着，一个 fused select-and-pad operator 会将它们重组为完整的压缩 KV 条目集合，其总长度为 `cp_size · s/m`。所有 padding 条目都会放在末尾。对于 HCA 以及 CSA 中的 indexer，每个 query token 对压缩 KV 条目的可见范围都可以按照规则预先计算。对于 CSA 中的稀疏注意力，top-k selector 会显式给出每个 query 可见的压缩 KV 条目下标。

#### 3.5.4 用于灵活激活检查点的扩展自动微分

传统的激活检查点实现以整个模块为粒度，决定在 backward pass 期间是保留还是重计算其输出激活。这种粗粒度方式通常会在重计算成本与激活内存占用之间导致次优折中。另一种方法是手工实现整个层的 forward 和 backward 逻辑，并显式管理 tensor checkpointing 状态。虽然这样可以提供细粒度控制，但会失去自动微分框架带来的便利，并显著增加开发复杂度。

为了在不牺牲编程效率的前提下实现细粒度控制，我们实现了一种支持自动微分的 tensor-level activation checkpointing 机制。有了这套机制，开发者只需实现 forward pass，并有选择地标注需要自动 checkpoint 和重计算的单个张量。我们的框架利用 TorchFX（Reed et al., 2022）追踪完整的计算图。对于每个被标注的张量，它都会沿 backward 方向遍历，以找出重计算该张量所需的最小子图。我们将这些最小子图称为重计算图（recomputation graphs），并在对应梯度计算之前，把它们插入到 backward 逻辑中。

与手工实现相比，这种设计在训练过程中不会引入额外开销。该框架中的重计算是通过直接释放被标注张量的 GPU 内存，并复用重计算后张量的存储指针来实现的，不涉及任何 GPU 内存拷贝。此外，由于图追踪会对模型进行具体执行，我们能够跟踪每个张量底层的存储指针，从而对共享同一存储的张量（例如 reshape 操作的输入和输出）自动去重其重计算。这让开发者在标注重计算时，无需再自己推理底层内存细节。

### 3.6 推理框架

我们的推理框架大体继承自 DeepSeek-V3，但在 KV cache 管理方面存在一些差异。

#### 3.6.1 KV Cache 结构与管理

为了高效管理 DeepSeek-V4 中由混合注意力机制带来的异构 KV cache，我们设计了一个定制化的 KV cache 布局。其布局如图 6 所示，下面我们将详细展开。

**DeepSeek-V4 中的异构 KV 条目**。DeepSeek-V4 系列中的混合注意力机制引入了多种不同类型的 KV 条目，它们具有不同的 KV cache 大小和更新规则。用于稀疏选择的 lightning indexer 会为 KV cache 引入额外维度，而这些维度的 embedding size 与主注意力中的不同。CSA 和 HCA 中使用的压缩技术，分别将序列长度缩短为原来的 `1/m` 和 `1/m'`，从而减小总体 KV cache 大小。因此，不同层中的 KV cache 大小会有所不同。此外，Sliding Window Attention（SWA）层也具有不同的 KV cache 大小，并且拥有独立的 cache hit 与 eviction 策略。在压缩分支中，每 `m` 个 token 会产生一个 KV 条目。当剩余 token 数量不足以执行压缩时，所有待处理 token 及其相关隐藏状态都必须被保留在 buffer 中，直到压缩操作能够执行。这些缓冲 token 表示一种由位置上下文决定的序列状态，也会被纳入 KV cache 框架中管理。

**管理混合注意力 KV Cache 的挑战**。混合注意力机制打破了 PagedAttention 及其变体所依赖的一些基本假设。尽管最近已经出现了针对一般混合注意力模型或特定结构的混合 KV cache 管理算法（例如 Jenga（Zhang et al., 2025a）、Hymba（Dong et al., 2025）），但仍有两个主要障碍，使得无法在 PagedAttention 框架下将所有层的 KV cache 合并统一管理：

- 多样化的 cache 策略，例如 Sliding Window Attention 中使用的策略。
- 高性能 attention kernel 施加的约束，包括对齐要求。

为了高效管理 DeepSeek-V4 的 KV cache，我们设计了相应策略来克服这两个挑战。

**用于 SWA 和未压缩尾部 token 的状态缓存**。为解决第一个障碍，我们采用了一种替代性的 cache 管理机制。由于 SWA 的目标是在有限 KV cache 大小下增强性能，因此，将它与压缩分支中尚未压缩的尾部 token 一并视作 state-space model 是合理的。对应的 KV cache 因而可以被看作一种仅依赖当前位置的、序列特定状态。据此，我们预先分配一个固定大小且受限的 state cache 池，并动态地将其分配给每条序列。

**稀疏注意力 Kernel 协同设计**。关于第二个障碍，传统高性能 attention kernel 通常假设每个 block 中有固定数量 `B` 的 token，以优化性能，这在 CSA 中对应 `B·m` 个原始 token，在 HCA 中对应 `B·m'` 个原始 token。通过采用高性能稀疏注意力 kernel，不同层可以在不损失性能的情况下容纳可变的每 block token 数。实现这一点需要对 KV cache 布局与稀疏注意力 kernel 做联合设计。例如，为了与 cache line 对齐而对 block 做 padding，有助于提升性能。因此，对于压缩比为 `m` 的 CSA 和压缩比为 `m'` 的 HCA，每个 block 中原始 token 的数量都可以取 `lcm(m, m')` 的任意倍数，也就是这两个压缩比的最小公倍数。

#### 3.6.2 磁盘上的 KV Cache 存储

在部署 DeepSeek-V4 时，我们利用一种磁盘上的 KV cache 存储机制，以消除对 shared-prefix 请求的重复 prefilling。对于 CSA/HCA 中的压缩 KV 条目，以及 Sliding Window Attention（SWA）中的未压缩 KV 条目，我们分别设计了独立的存储管理方案。

对于 CSA 和 HCA，我们会直接将所有压缩后的 KV 条目存储到磁盘上。当一个请求命中已存储前缀时，我们会读取并复用与该前缀对应的压缩 KV 条目，直到最后一个完整压缩 block 为止。特别地，对于尾部那个未完成压缩 block 中的前缀 token，我们仍然需要重新计算它们，以恢复未压缩 KV 条目，因为 CSA 和 HCA 中的未压缩 KV 条目并不会被存储。

对于 SWA KV 条目，由于它们未经过压缩，并且存在于每一层中，其体量大约是压缩后的 CSA 和 HCA KV 条目的 8 倍。为高效处理这些庞大的 SWA KV 条目，我们提出并实现了三种不同的磁盘上 SWA KV 管理策略，每一种都在存储开销和重复计算之间提供不同折中：

- **完整 SWA 缓存（Full SWA Caching）**。该策略存储所有 token 的完整 SWA KV 条目，从而确保计算零冗余。在该策略下，命中前缀的 SWA KV 条目只需读取该前缀内部最后 `n_win` 个 token 的磁盘缓存即可重建。尽管实现了计算零冗余，但这种策略对于现代基于 SSD 的存储系统并不高效，因为每个命中请求只会访问所存 SWA KV cache 中很小的一部分，这会导致一种失衡的、以写入为主的访问模式。

- **周期性检查点（Periodic Checkpointing）**。该策略每隔 `p` 个 token，对最后 `n_win` 个 token 的 SWA KV 条目做一次检查点，其中 `p` 是可调参数。对于命中前缀，我们会加载最近的一个检查点状态，然后重计算剩余的尾部 token。通过调节 `p`，该策略可以按需在存储与计算之间做权衡。

- **零 SWA 缓存（Zero SWA Caching）**。该策略不存储任何 SWA KV 条目。对于命中前缀，我们需要进行更多重计算，以恢复 SWA KV 条目。具体来说，在每个 attention layer 中，每个 token 的 SWA KV 条目只依赖前一层中最近 `n_win` 个 token 的 SWA KV 条目。因此，借助已缓存的 CSA 和 HCA KV 条目，只需重计算最后 `n_win · L` 个 token，就足以为一个 `L` 层模型恢复最后 `n_win` 个 SWA KV 条目。

我们会根据具体部署场景，选择最合适的策略，以实现存储与计算之间的目标折中。

---

## 当前状态说明

本文件遵循“严格逐段、按章节对应”的翻译规范。

当前已完成：
- 摘要
- 第 1 章
- 第 2 章（截至现有可稳定抽取的内容）
- 第 3 章（按当前可稳定抽取文本落盘）

后续如果继续推进，应继续直接对照 PDF / 文本抽取结果，把第 2.3 之后和第 3 章后续细节进一步补全为更完整的逐段版。
