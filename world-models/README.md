# World Models & Object-Centric Learning

Foundational papers on object-centric representation learning, relational reasoning, and world models for autonomous agents.

## Papers in This Collection

### Completed ✅

| Paper | Year | Key Contribution | Notes |
|-------|------|------------------|-------|
| [Slot Attention](notes/slot-attention.md) | 2020 | Iterative attention mechanism for object discovery | Core building block |
| [Graph Networks](notes/graph-networks.md) | 2018 | Unifying framework for relational inductive biases | Foundational theory |
| [LeCun World Model](notes/lecun-world-model.md) | 2022 | JEPA architecture for autonomous agents | Position paper/vision |
| [SAVi](notes/savi.md) | 2022 | Conditional object-centric learning from video | Scales to realistic data |

### Reading Guide

**Start here**: [Deep Learning Overview Reading Guide](notes/deep_learning_overview_reading_guide.md)

**Recommended order**:
1. **Slot Attention** - Understand the core attention mechanism
2. **Graph Networks** - Broader framework for structured computation
3. **SAVi** - See how Slot Attention extends to video
4. **LeCun World Model** - High-level vision for how these pieces fit together

---

## Core Themes

### 1. Object-Centric Representations
The idea that perception should output **discrete entities** (objects) rather than pixel-level features.

- **Slot Attention**: Maps inputs to K output "slots" via competitive attention
- **SAVi**: Extends to video with temporal dynamics
- **Why it matters**: Enables compositional generalization, relational reasoning

### 2. Relational Inductive Biases
Structure built into architectures to encourage learning about entities and their relations.

- **Graph Networks**: Explicit nodes (entities), edges (relations), message passing
- **Slot Attention**: Implicit structure via slot competition
- **Why it matters**: Combinatorial generalization ("infinite use of finite means")

### 3. Learning World Models
Internal models of how the world works, enabling prediction and planning.

- **LeCun's JEPA**: Predict in representation space, not pixel space
- **Hierarchical abstraction**: Different time scales, different levels of detail
- **Why it matters**: Sample-efficient learning, safe exploration, reasoning

---

## Key Equations

### Slot Attention
```
attn[i,j] = exp(M[i,j]) / Σ_l exp(M[i,l])  # softmax over SLOTS
```

### Graph Network Update
```
e'_k = φ^e(e_k, v_r, v_s, u)     # edge update
v'_i = φ^v(ρ(E'_i), v_i, u)      # node update
u'   = φ^u(ρ(E'), ρ(V'), u)      # global update
```

### JEPA Energy
```
E(x, y, z) = D(Enc_y(y), Pred(Enc_x(x), z))
```

---

## Architecture Comparisons

| Aspect | Slot Attention | SAVi | Graph Network |
|--------|---------------|------|---------------|
| Input | Image features | Video frames | Graph (V, E, u) |
| Output | K slot vectors | K slots + masks | Graph (V', E', u') |
| Dynamics | None | Transformer predictor | GRU/Transformer |
| Supervision | Reconstruction | Optical flow | Task-dependent |
| Permutation | Equivariant | Equivariant | Equivariant |

---

## Connections to PhD Research

These papers are foundational for:

1. **Memory-Augmented Transformers**: Object-centric slots as memory items
2. **Causal Representation Learning**: Entities enable causal abstraction
3. **Cognitive AI**: LeCun's architecture as cognitive model

### Key Questions
- How to extract entities from raw perception (without supervision)?
- How to represent entity interactions dynamically?
- How to learn hierarchical world models from video?

---

## PDFs

All PDFs stored in `~/clawd/paper-reading/pdfs/`:
- `slot-attention.pdf` (3.4 MB)
- `graph-networks.pdf` (673 KB)
- `lecun-world-model.pdf` (729 KB)
- `savi.pdf` (4.3 MB)

---

## Related Collections

- [Causal Representation Learning](../causal-representation-learning/)
- [Memory-Augmented Transformers](../memory-augmented-transformers/)
