# Relational Inductive Biases, Deep Learning, and Graph Networks (Battaglia et al., 2018)

**Paper**: Relational Inductive Biases, Deep Learning, and Graph Networks
**Authors**: Peter W. Battaglia et al. (28 authors from DeepMind, Google Brain, MIT, Edinburgh)
**Date**: October 2018 (arXiv:1806.01261v3)
**PDF Read**: 2026-02-18

---

## Core Thesis

**Combinatorial generalization must be a top priority for AI.** The paper argues that:
1. Deep learning's success comes despite (not because of) its weak relational inductive biases
2. Structured representations and computations are key to human-like generalization
3. Graph Networks provide a framework that combines deep learning flexibility with strong relational structure

---

## Relational Reasoning: Definitions

**Entities**: Elements with attributes (e.g., physical objects with mass, position)

**Relations**: Properties between entities (e.g., "heavier than", "connected to")
- Can have attributes (e.g., spring constant)
- Can depend on global context

**Rules**: Functions mapping entities/relations to other entities/relations

**Relational Reasoning**: Manipulating structured representations of entities and relations using compositional rules.

---

## Relational Inductive Biases in Standard Architectures

| Component | Entities | Relations | Relational Bias | Invariance |
|-----------|----------|-----------|-----------------|------------|
| Fully Connected | Units | All-to-all | Weak | - |
| Convolutional | Grid elements | Local | Locality | Translation |
| Recurrent | Timesteps | Sequential | Sequentiality | Time translation |
| **Graph Network** | **Nodes** | **Edges** | **Arbitrary** | **Node/edge permutation** |

Key insight: CNNs and RNNs have implicit relational biases (locality, sequentiality), but they're fixed by architecture. Graph networks learn arbitrary relational structure.

---

## The Graph Network Framework

### Graph Definition

A graph G = (u, V, E) where:
- **u**: Global attributes (e.g., gravitational field)
- **V = {v_i}**: Set of N^v nodes with attributes
- **E = {(e_k, r_k, s_k)}**: Set of N^e edges with attributes, receiver index r_k, sender index s_k

### GN Block: Graph-to-Graph Transformation

**Update Functions (φ)**: Learnable, often MLPs
```
e'_k = φ^e(e_k, v_r_k, v_s_k, u)     # Edge update
v'_i = φ^v(ē'_i, v_i, u)             # Node update  
u'   = φ^u(ē', v̄', u)               # Global update
```

**Aggregation Functions (ρ)**: Must be permutation-invariant (sum, mean, max)
```
ē'_i = ρ^(e→v)(E'_i)    # Aggregate edges to node i
ē'   = ρ^(e→u)(E')      # Aggregate all edges to global
v̄'   = ρ^(v→u)(V')      # Aggregate all nodes to global
```

### Computation Order
1. Compute all edge updates (parallel over edges)
2. Aggregate edges per node → ē'_i
3. Compute all node updates (parallel over nodes)
4. Aggregate edges globally → ē'
5. Aggregate nodes globally → v̄'
6. Compute global update

---

## Key Properties

### Relational Inductive Biases
1. **Arbitrary structure**: Graph topology determines interactions (not architecture)
2. **Permutation invariance/equivariance**: No ordering imposed on nodes/edges
3. **Reuse**: Same φ^e applied to all edges, same φ^v to all nodes → **combinatorial generalization**

### Combinatorial Generalization
Because GNs operate on entities and relations with shared functions:
- Can generalize to graphs of different sizes
- Can handle novel configurations of familiar components
- Supports "infinite use of finite means"

---

## Architecture Variants

### Full GN Block
Uses all update and aggregation functions. Example (Hamrick et al., Sanchez-Gonzalez et al.):
```python
φ^e(e_k, v_r, v_s, u) = NN_e([e_k, v_r, v_s, u])  # concatenate, apply MLP
φ^v(ē'_i, v_i, u)     = NN_v([ē'_i, v_i, u])
φ^u(ē', v̄', u)        = NN_u([ē', v̄', u])
```

### Message-Passing Neural Network (MPNN)
- Edge function M_t (no global u)
- Aggregation: sum
- Node function U_t
- Readout function R (global prediction, no edge aggregation)

### Non-Local Neural Network (NLNN) / Self-Attention
- φ^e factored into attention weight α(v_r, v_s) and value β(v_s)
- ρ^(e→v): attention-weighted sum with normalization
- Captures "self-attention" style methods (Transformer, etc.)

### Relation Network
- Only edge updates and global aggregation
- No node updates
- Used for relational reasoning on question-answering

### Deep Sets
- No edge updates
- Only node updates and global aggregation
- For permutation-invariant functions on sets

---

## Design Principles

### 1. Flexible Representations
- Attributes can be vectors, tensors, sequences, sets, even other graphs
- Node/edge/global outputs can be used selectively based on task

### 2. Configurable Within-Block Structure
- Can simplify by removing unused φ or ρ functions
- Can add recurrence (RNN-based φ)
- Can use typed edges for different relation types

### 3. Composable Multi-Block Architectures

**Encode-Process-Decode**:
```
G^0 = GN_enc(G_input)
G^m = GN_core(G^(m-1))  # repeat M times
G_output = GN_dec(G^M)
```

**Message Passing**: Information propagates m hops after m steps of shared GN_core.

**Recurrent**: Maintain hidden graph state, combine with new inputs at each timestep.

---

## Applications Demonstrated

- **Physical simulation**: Predicting dynamics of balls, springs, rigid bodies
- **Molecular property prediction**: Chemical graphs
- **Visual reasoning**: Scene understanding, few-shot learning
- **Combinatorial optimization**: TSP, SAT
- **Multi-agent systems**: Communication, coordination
- **Program synthesis**: AST manipulation
- **Knowledge graphs**: Relation prediction

---

## Limitations (Authors' Discussion)

1. **Graph structure must be given or inferred**: How to extract graphs from raw sensory data?
2. **Fixed structure**: How to handle dynamic graph topology (splitting, merging)?
3. **Message passing expressivity**: Cannot distinguish some non-isomorphic graphs
4. **Beyond graphs**: Programs, recursion, control flow need additional structure

---

## Key Insight: Combining Structure and Learning

The paper advocates against the false dichotomy of "hand-engineering" vs. "end-to-end learning":

> "Just as biology uses nature and nurture cooperatively, we reject the false choice between 'hand-engineering' and 'end-to-end' learning."

Graph Networks:
- Provide **structure** through explicit entities, relations, and compositional computation
- Allow **flexibility** through learned φ and ρ functions
- Enable **generalization** through shared parameters across entities/relations

---

## Connections to Other Work

- **Slot Attention**: Slots as nodes, attention as implicit edges, can be seen as a specific GN
- **Transformers**: Self-attention is a special case (fully-connected NLNN)
- **World Models**: GNs as dynamics models for entity-based physics simulation
- **SAVi**: Uses Transformer predictor (permutation-equivariant like GN) for slot dynamics

---

## Mathematical Formulation

The full GN block computation (Algorithm 1 from paper):

```
function GraphNetwork(E, V, u):
    for k in 1..N^e:
        e'_k ← φ^e(e_k, v_{r_k}, v_{s_k}, u)
    
    for i in 1..N^v:
        E'_i = {(e'_k, r_k, s_k) : r_k = i}
        ē'_i ← ρ^(e→v)(E'_i)
        v'_i ← φ^v(ē'_i, v_i, u)
    
    V' = {v'_i}
    E' = {(e'_k, r_k, s_k)}
    ē' ← ρ^(e→u)(E')
    v̄' ← ρ^(v→u)(V')
    u' ← φ^u(ē', v̄', u)
    
    return (E', V', u')
```

---

## Why This Matters

Graph Networks provide a **unifying framework** for understanding and designing neural architectures with relational inductive biases. They show that:

1. Structure and learning are complementary, not competing
2. Explicit relational reasoning enables compositional generalization
3. Many successful architectures are special cases of this framework

This is foundational for building world models that can:
- Learn object-centric representations
- Model object interactions
- Generalize to novel scenes and dynamics

---

## Citation

```bibtex
@article{battaglia2018relational,
  title={Relational inductive biases, deep learning, and graph networks},
  author={Battaglia, Peter W and Hamrick, Jessica B and Bapst, Victor and Sanchez-Gonzalez, Alvaro and Zambaldi, Vinicius and Malinowski, Mateusz and Tacchetti, Andrea and Raposo, David and Santoro, Adam and Faulkner, Ryan and others},
  journal={arXiv preprint arXiv:1806.01261},
  year={2018}
}
```
