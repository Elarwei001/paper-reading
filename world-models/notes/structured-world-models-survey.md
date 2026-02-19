# Structured World Models: A Survey

**Date**: 2026-02-19  
**Field**: World Models × Structured Representations  
**Status**: Rapidly evolving research area

## 1. Introduction

### The Problem with Unstructured World Models

Current video generation models (Sora, Veo, etc.) predict future frames in **pixel space**. While impressive, they have fundamental limitations:

| Issue | Description |
|-------|-------------|
| **No object permanence** | Objects can appear/disappear inconsistently |
| **Physics violations** | Impossible motions, gravity errors |
| **Poor compositional generalization** | Can't recombine learned concepts |
| **Expensive inference** | Must generate every pixel |

### What are Structured World Models?

Structured World Models introduce **explicit structure** into the latent space:
- **Objects** as discrete entities
- **Relations** between objects
- **Physical properties** (mass, velocity)
- **Causal dependencies**

**Goal**: Learn a latent space that mirrors the compositional, object-centric nature of the physical world.

## 2. Architectural Paradigms

### 2.1 Object-Centric Learning

**Core idea**: Decompose scenes into discrete object representations ("slots").

```
Input Image → Encoder → Slots [s₁, s₂, ..., sₖ] → Decoder → Reconstruction
                           ↑
                    Each slot = one object
```

#### Slot Attention (Locatello et al., 2020)

**Mechanism**:
1. Initialize K slot vectors randomly
2. Iteratively compete for image features via attention
3. Each slot "claims" features from its object

**Key innovation**: Differentiable, unsupervised object discovery

**Limitations**:
- Struggles with complex real-world scenes
- Fixed number of slots
- No explicit dynamics

#### SAVi (Kipf et al., 2022) — Slot Attention for Video

**Extension**: Apply slot attention to video with temporal consistency

- Slots propagate across frames
- Learn to track objects
- Foundation for object-centric world models

#### STEVE (Singh et al., 2022)

**Improvement**: Better handling of complex textures using discrete codebook

### 2.2 Graph Neural Networks for Physics

**Core idea**: Represent world state as a graph, predict dynamics via message passing.

```
Nodes = Objects (position, velocity, type)
Edges = Relations (distance, contact)

GNN: G_t → G_{t+1}
```

#### Interaction Networks (Battaglia et al., 2016)

- First major work on learning physics with GNNs
- Objects as nodes, relations as edges
- Message passing predicts forces/accelerations

#### Graph Networks (Battaglia et al., 2018)

- Unified framework for relational reasoning
- Generalization across different physical systems
- Foundation for many subsequent works

#### Learning to Simulate (Sanchez-Gonzalez et al., 2020)

- Scale GNN physics to complex particle systems
- Learns to simulate fluids, rigid bodies, cloth
- Strong generalization to new scenarios

### 2.3 JEPA: Joint Embedding Predictive Architecture

**Yann LeCun's proposal** for next-generation world models.

**Core insight**: Don't predict pixels — predict in latent space.

```
Traditional: x_t → Encoder → z_t → Predictor → x̂_{t+1} (pixel prediction)

JEPA:        x_t → Encoder → z_t → Predictor → ẑ_{t+1} (latent prediction)
             x_{t+1} → Encoder → z_{t+1}
             
             Loss: ||ẑ_{t+1} - z_{t+1}||
```

**Advantages**:
- Avoids pixel-level details
- Focuses on semantically meaningful changes
- More efficient (no decoder needed for training)

**Challenge**: Representation collapse — model learns trivial constant representations

**Solutions**:
- **VICReg**: Variance-Invariance-Covariance regularization
- **I-JEPA** (Assran et al., 2023): Image JEPA with masking
- **V-JEPA** (Bardes et al., 2024): Video JEPA

### 2.4 Energy-Based Models

**Core idea**: Learn an energy function E(x, y) that scores compatibility.

**For World Models**:
- E(state, next_state, action) = compatibility score
- Inference: find next_state minimizing energy
- Can represent multiple possible futures

**Advantage**: Handles multimodality naturally (unlike MSE regression)

## 3. Key Research Groups & Progress

### 3.1 Yann LeCun / Meta FAIR

**Vision**: JEPA as the path to autonomous machine intelligence

**Key Works**:
| Paper | Year | Contribution |
|-------|------|--------------|
| "A Path Towards AMI" | 2022 | JEPA conceptual framework |
| I-JEPA | 2023 | Image-level implementation |
| V-JEPA | 2024 | Video-level extension |

**Philosophy**:
- Pixel prediction is wasteful
- Learn abstract representations
- Energy-based over generative

### 3.2 Thomas Kipf / DeepMind → Google

**Focus**: Object-centric learning and structured dynamics

**Key Works**:
| Paper | Year | Contribution |
|-------|------|--------------|
| Slot Attention | 2020 | Differentiable object discovery |
| SAVi | 2022 | Video extension of slots |
| SAVi++ | 2023 | Improved real-world performance |

**Research Direction**: Scale object-centric methods to real video

### 3.3 Danijar Hafner / DeepMind

**Focus**: World models for reinforcement learning

**The Dreamer Series**:
| Version | Year | Key Innovation |
|---------|------|----------------|
| Dreamer | 2020 | Latent dynamics for RL |
| DreamerV2 | 2021 | Discrete latents |
| DreamerV3 | 2023 | Scales across domains |

**Architecture**:
```
RSSM (Recurrent State-Space Model):
- Deterministic path: h_t → h_{t+1}
- Stochastic path: z_t ~ P(z|h_t)
- Combined: full state = (h_t, z_t)
```

**Key insight**: Discrete latents + KL balancing for stable training

### 3.4 Chelsea Finn / Stanford

**Focus**: Model-based RL, meta-learning

**Key Works**:
- Visual Foresight (2017): Learning to predict for control
- MAML (2017): Meta-learning framework
- Model-based adaptation

### 3.5 Peter Battaglia / DeepMind

**Focus**: Graph networks for physics simulation

**Key Works**:
- Interaction Networks (2016)
- Graph Networks (2018)
- Learning to Simulate (2020)

**Philosophy**: Explicit relational inductive biases improve generalization

## 4. Current Frontiers

### 4.1 Scaling Object-Centric Models

**Challenge**: Slot attention works on simple scenes, struggles on real video

**Recent Progress**:
- **DINOSAUR** (2023): Combine DINO features with slots
- **VideoSAUR** (2024): Temporal consistency for real video
- **SlotDiffusion** (2023): Diffusion models + slots

### 4.2 3D-Aware World Models

**Motivation**: Real world is 3D; 2D models lose information

**Approaches**:
- **NeRF + Dynamics**: Neural radiance fields with motion
- **3D Gaussian Splatting**: Efficient 3D with dynamics
- **Ego4D/Epic-Kitchens**: Egocentric 3D understanding

### 4.3 Language-Grounded World Models

**Goal**: Ground language in physical world understanding

**Examples**:
- **RT-2** (Google): Language-conditioned robot control
- **PaLM-E** (Google): Embodied language model
- **Gato** (DeepMind): Generalist agent

### 4.4 Long-Horizon Prediction

**Challenge**: Errors compound; hard to predict far future

**Approaches**:
- Hierarchical prediction (abstract → detailed)
- Landmark-based planning
- Multi-scale temporal abstraction

## 5. Open Problems

### 5.1 The Binding Problem
How do objects "bind" to their properties consistently over time?
- Color, shape, position must stay associated
- Biological visual system solves this; ML still struggles

### 5.2 Compositionality
Can we truly recombine learned concepts?
- "Red ball" + "blue cube" → "red cube" + "blue ball"
- Current models fail at systematic generalization

### 5.3 Physical Plausibility
How to enforce physics without hardcoding?
- Conservation laws (energy, momentum)
- Contact dynamics
- Soft vs rigid body

### 5.4 Real-World Gap
How to handle:
- Partial observability (occlusion)
- Sensor noise
- Unknown object categories

## 6. Connection to Causal Representation Learning

Structured world models and CRL are deeply connected:

```
CRL Question: What is the right latent structure?
SWM Answer: Objects + Relations + Dynamics

CRL Question: How to identify causal factors?
SWM Answer: Objects are natural causal units

CRL Question: How to predict interventions?
SWM Answer: Change object state, propagate through relations
```

**Synthesis**: An ideal world model has:
- **Object-centric structure** (from SWM)
- **Causal dynamics** (from CRL)
- **Efficient latent prediction** (from JEPA)

## 7. Recommended Reading Path

### Foundations
1. Battaglia et al., "Relational inductive biases, deep learning, and graph networks" (2018)
2. LeCun, "A Path Towards Autonomous Machine Intelligence" (2022)

### Object-Centric
3. Locatello et al., "Object-Centric Learning with Slot Attention" (2020)
4. Kipf et al., "Conditional Object-Centric Learning from Video" (2022)

### Dynamics & RL
5. Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination" (2020)
6. Hafner et al., "Mastering Diverse Domains through World Models" (2023)

### Physics Simulation
7. Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics with GNNs" (2020)

## 8. Summary

Structured World Models represent a promising path toward world models that learn **laws**, not just **patterns**.

**Key ideas**:
- Objects as fundamental units
- Relations as interaction channels
- Latent prediction over pixel prediction
- Graph structures for physical reasoning

**Progress**:
- Object-centric learning maturing (Slot Attention → SAVi → real video)
- JEPA framework gaining traction
- Dreamer shows RL applicability

**Challenges**:
- Scaling to real-world complexity
- Long-horizon prediction
- True physical plausibility

**Connection to "laws from phenomena"**: Structured world models provide the architectural framework; causal representation learning provides the theoretical foundation. Together, they offer a path toward AI that truly understands physical causality.

---

*Survey compiled: 2026-02-19*
