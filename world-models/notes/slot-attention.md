# Object-Centric Learning with Slot Attention (Locatello, Kipf et al., NeurIPS 2020)

**Paper**: Object-Centric Learning with Slot Attention
**Authors**: Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, Thomas Kipf
**Venue**: NeurIPS 2020
**PDF Read**: 2026-02-18

---

## Core Contribution

Slot Attention is an architectural component that maps from N input features to K output "slots" through iterative attention. The key innovation is that it enables **permutation-equivariant** binding of inputs to slots without requiring supervision, making it suitable for unsupervised object discovery.

---

## Technical Details

### The Slot Attention Module

**Input**: N feature vectors from encoder (e.g., CNN output)
**Output**: K slot vectors representing distinct entities

**Algorithm** (T iterations, typically T=3):
```
inputs ∈ R^(N×D_in)
slots ← μ + σ · N(0,1)  # K slots, randomly initialized
for t in 1..T:
    slots_prev = slots
    slots = LayerNorm(slots)
    q = Linear(slots)           # K×D
    k, v = Linear(inputs), Linear(inputs)  # N×D
    
    attn = softmax(k·q^T / √D, axis=slots)  # N×K, normalized OVER SLOTS
    updates = WeightedMean(attn, v)         # K×D, weighted mean NOT sum
    slots = GRU(updates, slots_prev)        # recurrent update
    slots = slots + MLP(LayerNorm(slots))   # optional residual MLP
```

### Key Design Choices

1. **Softmax over slots (not inputs)**: This forces competition - each input position must decide which single slot it belongs to. Standard cross-attention normalizes over inputs, which allows a slot to attend to everything equally.

2. **Weighted mean aggregation**: Using weighted mean instead of weighted sum ensures slots receive balanced input regardless of how many positions attend to them. Prevents slots from dominating simply by attending to more positions.

3. **Common representational format**: Unlike Capsules where different capsules have specialized roles, slots use identical learned parameters (shared μ, σ initialization). Slots are truly exchangeable.

4. **Iterative refinement**: Starting from random initialization, slots compete and specialize over T iterations. Authors find T=3 sufficient.

### Properties

- **Permutation invariant** to input ordering
- **Permutation equivariant** to slot ordering (swapping slots in input swaps them in output)
- **No explicit object supervision required**

---

## Applications in Paper

### 1. Set Prediction (Supervised)
- Task: CLEVR property prediction
- Architecture: CNN encoder → Slot Attention → MLP per slot → Hungarian matching loss
- Result: Near-perfect (99.1% accuracy), matches specialized models

### 2. Object Discovery (Unsupervised)
- Task: Segment objects without supervision
- Architecture: CNN encoder → Slot Attention → Spatial Broadcast Decoder per slot
- Decoder: Each slot broadcasts to all positions + position encoding, then CNN → RGB + alpha mask
- Reconstruction: Weighted sum of slot outputs using softmax-normalized alpha masks

**Results**:
- CLEVR (≤6 objects): 98.8% ARI
- CLEVR (≤10 objects): 91.5% ARI
- Competitive with IODINE/MONet but **10× faster, 8× less memory**

---

## Spatial Broadcast Decoder

Important component for unsupervised segmentation:
```
For each slot s_k:
    1. Tile s_k to spatial grid (H×W×D)
    2. Concatenate position encodings
    3. Apply CNN decoder
    4. Output: RGB prediction + alpha mask
Combine: y = Σ softmax(α_k) · RGB_k
```

This forces each slot to explain a spatial region, encouraging object-like decomposition.

---

## Comparison with Related Work

| Method | Per-slot Params | Iterative | Supervision |
|--------|----------------|-----------|-------------|
| Capsules | Specialized | Yes | Often supervised |
| IODINE | Shared | Yes (expensive) | Unsupervised |
| MONet | Specialized (sequential) | No | Unsupervised |
| Slot Attention | Shared | Yes (cheap) | Either |

**Advantages over IODINE**: No need for separate posterior network; attention mechanism is simpler and faster than amortized variational inference.

---

## Insights & Limitations

### What Makes It Work
1. The slot competition via softmax normalization
2. Random initialization breaks symmetry
3. Iterative refinement allows slots to specialize
4. Shared parameters enable generalization

### Limitations (from paper)
- Struggles with complex real-world textures
- Fixed number of slots K
- No explicit 3D reasoning
- Permutation equivariance means no "slot identity"

### Why This Matters for World Models
Slot Attention provides a differentiable mechanism to extract object-centric representations from raw pixels. This is exactly what's needed for:
- Learning object-level dynamics
- Compositional generalization
- Relational reasoning

---

## Key Equations

**Attention (the critical innovation)**:
$$\text{attn}_{i,j} = \frac{\exp(M_{i,j})}{\sum_l \exp(M_{i,l})}$$

Where $M = \frac{1}{\sqrt{D}} k(x) \cdot q(s)^T$

Note: softmax is over slots (axis $l$), not inputs (axis $i$).

**Update**:
$$\text{updates}_j = \frac{1}{|\{i: \arg\max_l \text{attn}_{i,l} = j\}|} \sum_i \text{attn}_{i,j} \cdot v(x_i)$$

(In practice: weighted mean using normalized attention weights)

---

## Connections to Other Work

- **Graph Networks**: Slots can be seen as nodes; attention defines edges; update is message passing
- **Transformers**: Slot Attention is cross-attention with axis-swapped softmax
- **Binding Problem**: Provides a differentiable solution to feature binding
- **SAVi**: Direct extension to video with temporal dynamics

---

## Citation

```bibtex
@inproceedings{locatello2020object,
  title={Object-centric learning with slot attention},
  author={Locatello, Francesco and Weissenborn, Dirk and Unterthiner, Thomas and Mahendran, Aravindh and Heigold, Georg and Uszkoreit, Jakob and Dosovitskiy, Alexey and Kipf, Thomas},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
