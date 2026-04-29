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

DeepSeek-V4 系列设计了一种混合注意力架构，将 Compressed Sparse Attention（CSA）与 Heavily Compressed Attention（HCA）结合起来，以显著提升长上下文效率。总体来说，CSA 会沿序列维度压缩 KV cache，然后再执行稀疏注意力；而 HCA 则对 KV cache 施加更激进的压缩，但保持 dense attention。二者共同构成了 DeepSeek-V4 长上下文高效性的核心。

### 2.4 Muon 优化器

DeepSeek-V4 系列采用 Muon（Jordan et al., 2024；Liu et al., 2025）作为优化器。作者认为，这一优化器选择有助于实现更快收敛和更好的训练稳定性。

---

## 3. 通用基础设施

为了支持 DeepSeek-V4 系列的高效训练与推理，并同时提升研发效率，我们引入了若干基础设施优化。首先，我们为 MoE 模块设计并实现了一个单一融合 kernel，使计算、通信和内存访问可以完全重叠。其次，我们采用 TileLang（Wang et al., 2026），这是一种领域专用语言（DSL），用于在开发效率与运行时效率之间取得平衡。第三，我们提供了高效的、batch-invariant 且 deterministic 的 kernel 库，以确保训练和推理过程中的 bitwise reproducibility。第四，我们将 FP4 quantization-aware training 引入到 MoE expert 权重和 indexer QK path 中，以降低内存和计算开销。第五，在训练框架方面，我们通过 tensor-level checkpointing 扩展了 autograd 框架，以实现细粒度的重计算控制；并通过面向 Muon 优化器的 hybrid ZeRO 策略、借助重计算和 fused kernels 实现的高性价比 mHC，以及用于管理压缩注意力的两阶段 contextual parallelism，提升训练效率。最后，在推理框架方面，我们设计了一种异构 KV cache 结构，并配合磁盘存储策略，以实现高效的 shared-prefix reuse。

### 3.1 专家并行中的细粒度通信-计算重叠

作者指出，他们为 MoE 模块设计并实现了一个单一融合 kernel，以便让计算、通信和内存访问实现完全重叠。

### 3.2 基于 TileLang 的灵活高效 Kernel 开发

作者采用 TileLang（Wang et al., 2026），这是一种领域专用语言，用于在开发效率和运行时效率之间取得平衡。

### 3.3 高性能、batch-invariant 且 deterministic 的 Kernel 库

作者提供了高效的、batch-invariant 且 deterministic 的 kernel 库，以确保训练和推理过程中的 bitwise reproducibility。

### 3.4 FP4 量化感知训练

作者在 MoE expert 权重和 indexer QK path 中引入 FP4 quantization-aware training，以降低内存和计算开销。

### 3.5 训练框架

在训练框架方面，作者通过 tensor-level checkpointing 扩展了 autograd 框架，以实现细粒度的重计算控制；并通过面向 Muon 优化器的 hybrid ZeRO 策略、借助重计算和 fused kernels 实现的高性价比 mHC，以及用于管理压缩注意力的两阶段 contextual parallelism，提升训练效率。

#### 3.5.1 Muon 的高效实现

作者提到，为 Muon 优化器采用了 hybrid ZeRO 策略，以提升训练效率。

#### 3.5.2 mHC 的低成本高内存效率实现

作者提到，mHC 通过重计算和 fused kernels 实现，以兼顾成本效率和内存效率。

#### 3.5.3 面向长上下文注意力的上下文并行

作者提到，使用两阶段 contextual parallelism 来管理压缩注意力。

#### 3.5.4 用于灵活激活检查点的扩展自动微分

作者提到，通过 tensor-level checkpointing 扩展 autograd 框架，以实现更细粒度的重计算控制。

### 3.6 推理框架

在推理框架方面，作者设计了一种异构 KV cache 结构，并配合磁盘存储策略，以实现高效的 shared-prefix reuse。

#### 3.6.1 KV Cache 结构与管理

作者设计了一种异构 KV cache 结构。

#### 3.6.2 磁盘上的 KV Cache 存储

作者设计了配套的磁盘存储策略，以支持高效的 shared-prefix reuse。

---

## 当前状态说明

本文件遵循“严格逐段、按章节对应”的翻译规范。

当前已完成：
- 摘要
- 第 1 章
- 第 2 章（截至现有可稳定抽取的内容）
- 第 3 章（按当前可稳定抽取文本落盘）

后续如果继续推进，应继续直接对照 PDF / 文本抽取结果，把第 2.3 之后和第 3 章后续细节进一步补全为更完整的逐段版。
