# A Path Towards Autonomous Machine Intelligence (LeCun, 2022)

**Paper**: A Path Towards Autonomous Machine Intelligence (Version 0.9.2)
**Author**: Yann LeCun (NYU, Meta FAIR)
**Date**: June 27, 2022
**Type**: Position Paper
**PDF Read**: 2026-02-18

---

## Overview

This is a position paper proposing an architecture for autonomous intelligent agents. The core thesis: machines need to learn **world models** through self-supervised learning to achieve human-like intelligence. The paper introduces JEPA (Joint Embedding Predictive Architecture) as the key technical contribution.

---

## The Three Main Challenges

1. **Learning by observation**: How can machines learn to represent the world, predict, and act largely through observation (minimizing expensive/dangerous real-world trials)?

2. **Differentiable reasoning**: How can machines reason and plan in ways compatible with gradient-based learning?

3. **Hierarchical abstraction**: How can machines learn representations at multiple levels of abstraction and time scales?

---

## Proposed Architecture (Figure 2 in paper)

Six interconnected modules:

### 1. Configurator
- Executive control module
- Takes input from all other modules
- Configures/primes other modules for the task at hand
- Determines attention, parameter modulation, task decomposition

### 2. Perception
- Estimates current world state from sensors: `s[0] = Enc(x)`
- Produces hierarchical representations at multiple abstraction levels
- Configurable for task-relevant feature extraction

### 3. World Model
- **The core contribution of the paper**
- Two functions:
  - Estimate missing information about world state
  - Predict plausible future world states
- Must handle uncertainty (multiple possible futures)
- Operates in abstract representation space (not pixel space)

### 4. Cost Module
Two submodules:
- **Intrinsic Cost (IC)**: Immutable, hard-wired drives (pain, hunger, pleasure, curiosity)
- **Trainable Critic (TC)**: Learns to predict future intrinsic costs

Cost = IC(s) + TC(s), where terms can be weighted by configurator.

### 5. Short-Term Memory
- Associative memory for states and costs
- Used for critic training and world model updates
- Key-value memory network architecture

### 6. Actor
- Computes action sequences
- Two modes:
  - **Policy module**: Direct reactive output (Mode-1)
  - **Action optimizer**: Searches for cost-minimizing actions (Mode-2)

---

## Mode-1 vs Mode-2 (Kahneman's System 1/2)

### Mode-1: Reactive
```
s[0] = Enc(x)
a[0] = Policy(s[0])
```
- Fast, direct
- No world model consultation
- Habitual/learned behaviors

### Mode-2: Reasoning/Planning (Model-Predictive Control)
```
1. Perceive: s[0] = Enc(x)
2. Propose: Actor generates (a[0], ..., a[T])
3. Simulate: World model predicts (s[1], ..., s[T])
4. Evaluate: Cost = Σ C(s[t])
5. Optimize: Find action sequence minimizing cost
6. Act: Execute first action(s)
```
- Slow, deliberate
- Uses world model for simulation
- Gradient-based optimization possible

### Mode-2 → Mode-1 Transfer
Train policy module to imitate optimized Mode-2 actions:
```
D(ǎ[t], Policy(s[t])) → 0
```
This "compiles" deliberate planning into reactive skills.

---

## Energy-Based Models (EBM) Framework

The paper uses EBMs as the fundamental formalism:
- Energy function F(x, y) measures compatibility
- Low energy = compatible/plausible
- High energy = incompatible/implausible

### Why EBMs over Probabilistic Models?
- No need for normalized distributions
- Can represent multi-modal dependencies (multiple y compatible with x)
- More natural for prediction under uncertainty

### Latent Variables
For uncertain predictions: E(x, y, z) where z captures information about y not in x.
```
ž = argmin_z E(x, y, z)
F(x, y) = E(x, y, ž)
```

---

## JEPA: Joint Embedding Predictive Architecture

**The centerpiece of the paper.**

### Architecture
```
x → Enc_x → s_x → Predictor(s_x, z) → s̃_y
y → Enc_y → s_y
Energy = D(s_y, s̃_y)
```

### Key Innovation: Prediction in Representation Space
- Does NOT predict y directly (generative)
- Predicts s_y (representation of y) from s_x
- Allows ignoring irrelevant details (texture, noise)
- Encoder learns what information to preserve vs. discard

### Handling Multi-Modality
1. **Encoder invariance**: Many y map to same s_y
2. **Latent variable z**: Parameterizes which of multiple valid predictions

---

## Training JEPA (Non-Contrastive)

### Why Avoid Contrastive Methods?
- Require generating/selecting contrastive samples
- Curse of dimensionality: needs exponentially many samples in high dimensions

### Four Training Criteria
1. **Maximize info in s_x about x**: Prevent encoder collapse
2. **Maximize info in s_y about y**: Prevent encoder collapse
3. **Make s_y predictable from s_x**: Prediction loss D(s_y, s̃_y)
4. **Minimize info in z**: Prevent latent from bypassing prediction

### VICReg Method
- **Variance**: Keep std of each component above threshold
- **Invariance**: Minimize D(s_y, s̃_y)
- **Covariance**: Decorrelate components

### Regularizing z (Preventing Collapse)
If z has dimension ≥ dim(s_y), predictor could ignore s_x entirely.
Solutions:
- Discrete z (limited values)
- Low-dimensional z
- Sparse z (L1 regularization)
- Noisy/stochastic z (VAE-like)

---

## Hierarchical JEPA (H-JEPA)

### Motivation
- Low-level: detailed representations, short-term predictions
- High-level: abstract representations, long-term predictions
- Need multiple levels for different time horizons

### Architecture (Figure 15)
```
x_0 → JEPA-1 (low-level, short-term)
       ↓
     JEPA-2 (high-level, long-term)
```

### Example: Driving
- High-level: "Car will arrive at destination in ~30 minutes"
- Low-level: "Steering wheel angle in next 100ms"

---

## Hierarchical Planning (Figure 16)

Multi-scale world model enables multi-scale planning:

```
High level:  C(s2[4])  ← Goal
             ↑
       Pred2 → s2[2] → Pred2 → s2[4]
             ↑
     Actor2: a2[2], a2[4] (abstract "actions" = subgoals)
             ↓
Low level:   C(s[2])    C(s[4])  ← Subgoal costs
             ↑
       Pred1 → s[1] → Pred1 → ... → s[4]
             ↑
     Actor1: a[0], a[1], ...  (concrete actions)
```

High-level "actions" are really **conditions** that low-level states must satisfy.

---

## Handling Uncertainty (Figure 17)

Each prediction uses latent z_t:
```
s[t+1] = Pred(s[t], a[t], z[t])
```

- Sample z from regularizer-defined distribution
- Multiple samples → multiple trajectories
- Use MCTS-like search with pruning
- Optimize actions to minimize expected cost (possibly risk-adjusted)

---

## Intrinsic Motivation & Behavior

### Designing Behavior via IC
Rather than programming behaviors, design objectives:
- "Standing up" drive for legged robots
- "Influence world" drive for agency
- "Social interaction" drive
- "Curiosity" drive for exploration
- Safety guardrails (avoid heat, dangerous tools)

### Why IC Must Be Immutable
- Prevents behavioral collapse
- Ensures basic drives/safety cannot be hacked
- Trainable Critic handles learned costs

---

## Key Quotes

> "I submit that devising learning paradigms and architectures that would allow machines to learn world models in an unsupervised fashion, and to use those models to predict, to reason, and to plan is one of the main challenges of AI and ML today."

> "The main attraction of JEPAs is that they can be trained with non-contrastive methods."

> "Reasoning as Energy Minimization": Many forms of reasoning (probabilistic inference, constraint satisfaction) can be formulated as optimization.

---

## What's Missing / Open Questions

1. **Where do graphs/entities come from?** Paper assumes structured input but doesn't address segmentation.

2. **How does configurator learn?** Task decomposition into subgoals is left open.

3. **Real-world validation**: The architecture is theoretical; no empirical results.

4. **Scaling**: How do these ideas scale to real-world complexity?

---

## Connections to Other Work

- **Slot Attention / SAVi**: Could provide the entity-extraction layer LeCun assumes exists
- **Graph Networks**: World model predictor could use GN-style architecture for object interactions
- **Model-Based RL**: This is essentially model-based RL with learned world models
- **Transformers**: Mentioned as good architecture for object-based reasoning (permutation equivariance)

---

## Why This Matters

This paper provides a **blueprint** for autonomous AI:
1. Learn representations through prediction (not reconstruction)
2. Use hierarchical abstraction for long-term planning
3. Handle uncertainty with latent variables
4. Combine fast reactive behavior with slow deliberate reasoning
5. Ground behavior in intrinsic motivation, not external reward

It's a roadmap, not a solution—but a compelling one.

---

## Citation

```bibtex
@article{lecun2022path,
  title={A Path Towards Autonomous Machine Intelligence},
  author={LeCun, Yann},
  journal={Open Review},
  year={2022},
  url={https://openreview.net/pdf?id=BZ5a1r-kVsf}
}
```
