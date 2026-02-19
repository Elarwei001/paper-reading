# Causal Representation Learning: A Survey

**Date**: 2026-02-19  
**Field**: Causality × Deep Learning  
**Status**: Active research frontier

## 1. Introduction

### What is Causal Representation Learning?

Traditional deep learning learns **statistical correlations** from data. Causal Representation Learning (CRL) aims to learn **latent representations that reflect causal structure** — not just "what correlates with what" but "what causes what."

### Why Does It Matter?

| Statistical Learning | Causal Learning |
|---------------------|-----------------|
| Learns P(Y\|X) | Learns P(Y\|do(X)) |
| Breaks under distribution shift | Robust to interventions |
| Correlations may be spurious | Captures true mechanisms |
| Cannot answer "what if" | Supports counterfactual reasoning |

**Key insight**: A model that learns causal structure can generalize to new environments, interventions, and counterfactual scenarios — exactly what's needed for robust AI.

## 2. Foundational Concepts

### 2.1 Pearl's Causal Hierarchy

```
Level 3: Counterfactuals — "What if I had done X instead?"
    ↑
Level 2: Interventions — "What happens if I do X?"
    ↑
Level 1: Associations — "What is X related to?"
```

Most ML operates at Level 1. CRL aims for Level 2-3.

### 2.2 Structural Causal Models (SCM)

An SCM consists of:
- **Endogenous variables** V = {V₁, ..., Vₙ}
- **Exogenous variables** U = {U₁, ..., Uₙ}
- **Structural equations** Vᵢ = fᵢ(PAᵢ, Uᵢ)

The goal of CRL: learn the latent variables Z and their causal graph G from observations X.

### 2.3 Independent Causal Mechanisms (ICM)

**Principle**: The causal generative process consists of autonomous modules that don't inform each other.

Implications:
- P(effect | cause) is "simpler" than P(cause | effect)
- Changes in one mechanism don't affect others
- Basis for modular, recombinable representations

## 3. Key Research Directions

### 3.1 Identifiability Theory

**Core question**: Under what conditions can we recover the true causal variables from observations?

#### Key Results:

**ICA (Independent Component Analysis)**
- Classic result: Linear ICA is identifiable up to permutation and scaling
- Limitation: Nonlinear ICA is generally not identifiable

**Nonlinear ICA with Auxiliary Information** (Hyvärinen et al., 2019)
- Add auxiliary variable u (e.g., time index, domain label)
- If P(z|u) varies across u, nonlinear ICA becomes identifiable
- Foundation for many CRL methods

**iVAE** (Khemakhem et al., 2020)
- Identifiable VAE using auxiliary information
- Proves identifiability under specific conditions

**CausalVAE** (Yang et al., 2021)
- Incorporates causal graph structure into VAE
- Learns both representations and causal relationships

### 3.2 Disentanglement

**Goal**: Learn representations where each dimension corresponds to an independent factor of variation.

**Connection to Causality**: True causal variables are often disentangled (independent mechanisms).

#### Key Works:

| Paper | Key Idea |
|-------|----------|
| β-VAE (Higgins et al., 2017) | Increase KL penalty for disentanglement |
| FactorVAE (Kim & Mnih, 2018) | Adversarial total correlation penalty |
| Slot Attention (Locatello et al., 2020) | Object-centric disentanglement |

**Negative Result** (Locatello et al., 2019): Unsupervised disentanglement is impossible without inductive biases. Need auxiliary information or supervision.

### 3.3 Causal Discovery from Learned Representations

Once you have good representations, how do you discover the causal graph?

**Approaches**:
1. **Constraint-based**: PC algorithm, FCI
2. **Score-based**: GES, NOTEARS
3. **FCM-based**: LiNGAM, ANM

**NOTEARS** (Zheng et al., 2018)
- Reformulates discrete graph search as continuous optimization
- DAG constraint: tr(e^{W◦W}) - d = 0
- Enables gradient-based learning

**DAG-GNN** (Yu et al., 2019)
- Learns DAG structure with graph neural networks

### 3.4 Interventional & Counterfactual Learning

**Interventional Data**: Learn from data where variables are manipulated
- CausalGAN, Counterfactual Fairness

**Counterfactual Inference**: 
- Abduction → Action → Prediction
- Deep SCM, Counterfactual Latent Diffusion

## 4. Key Researchers & Groups

### Yoshua Bengio (Mila)

**Focus**: Connecting causality to generalization and consciousness

Key contributions:
- **System 2 Deep Learning**: Argues for explicit causal reasoning
- **GFlowNets**: Generative models for diverse causal structures
- **Causal Induction**: Learning to induce causal structure

Representative papers:
- "A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms" (2019)
- "Towards Causal Representation Learning" (2021, with Schölkopf)

### Bernhard Schölkopf (MPI Tübingen)

**Focus**: Theoretical foundations of causal representation learning

Key contributions:
- **ICM Principle**: Independent Causal Mechanisms
- **Nonlinear ICA**: Identifiability theory
- **Domain Adaptation**: Causal perspective

Representative papers:
- "Toward Causal Representation Learning" (2021)
- "Elements of Causal Inference" (2017, textbook)
- "Causality for Machine Learning" (2019)

### Kun Zhang (CMU)

**Focus**: Causal discovery algorithms

Key contributions:
- **PC-MCI**: Causal discovery for time series
- **Nonlinear ANM**: Additive noise models
- **Multi-domain learning**: Causal transfer

### Aapo Hyvärinen (Helsinki)

**Focus**: Nonlinear ICA identifiability

Key contributions:
- **TCL**: Time-contrastive learning
- **iVAE**: Identifiable VAE
- Foundation for CRL identifiability theory

## 5. Open Problems

### 5.1 Identifiability Gap
- Theory assumes clean conditions
- Real data is messy, conditions often violated
- How to bridge theory-practice gap?

### 5.2 Scaling
- Most methods tested on small synthetic datasets
- How to scale to real-world complexity?
- Integration with foundation models?

### 5.3 Latent Confounding
- What if there are unobserved confounders?
- Partial identifiability results?

### 5.4 Temporal Causal Learning
- Learning causal dynamics from video
- Connection to world models

## 6. Connection to World Models

**Key insight**: A good world model should have causal structure in its latent space.

```
Observations X → Encoder → Latent Z (causal structure) → Dynamics → Z' → Decoder → X'
                              ↑
                         This should capture
                         causal mechanisms
```

If Z captures true causal factors:
- Interventions in Z-space = interventions in real world
- Counterfactual reasoning becomes possible
- Model generalizes to new situations

**Challenge**: Current video world models (Sora, etc.) likely learn pixel correlations, not causal structure. How to inject causal inductive biases?

## 7. Recommended Reading Path

### Beginner
1. Pearl, "The Book of Why" (2018) — accessible introduction
2. Schölkopf, "Causality for Machine Learning" (2019) — ML perspective

### Intermediate
3. Schölkopf et al., "Toward Causal Representation Learning" (2021) — comprehensive survey
4. Locatello et al., "Challenging Common Assumptions in Unsupervised Disentanglement" (2019)

### Advanced
5. Khemakhem et al., "Variational Autoencoders and Nonlinear ICA" (2020)
6. Bengio et al., "A Meta-Transfer Objective for Learning to Disentangle" (2019)

## 8. Summary

Causal Representation Learning is attempting to solve a fundamental limitation of current deep learning: learning **what causes what**, not just **what correlates with what**.

Key progress:
- Identifiability theory maturing (nonlinear ICA)
- Practical algorithms emerging (CausalVAE, iVAE)
- Connection to disentanglement clarified

Key challenges:
- Scaling to real-world complexity
- Bridging theory-practice gap
- Integration with foundation models

**For World Models**: CRL provides the theoretical framework for what a "good" latent space should look like — one that captures causal mechanisms, not just statistical patterns.

---

*Survey compiled: 2026-02-19*
