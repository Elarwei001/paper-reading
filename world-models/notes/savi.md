# Conditional Object-Centric Learning from Video (Kipf et al., ICLR 2022)

**Paper**: Conditional Object-Centric Learning from Video
**Authors**: Thomas Kipf, Gamaleldin F. Elsayed, Aravindh Mahendran, Austin Stone, Sara Sabour, Georg Heigold, Rico Jonschkowski, Alexey Dosovitskiy, Klaus Greff
**Venue**: ICLR 2022
**Link**: https://arxiv.org/abs/2111.12594
**PDF Read**: 2026-02-18

---

## Core Contribution

SAVi (Slot Attention for Video) extends Slot Attention to video with two key innovations:
1. **Temporal dynamics**: Predictor-corrector architecture for tracking objects
2. **Conditional initialization**: Using weak hints (bounding boxes, center of mass) to guide object discovery

This enables object segmentation and tracking in significantly more realistic synthetic videos than prior work.

---

## Architecture Overview

### Processing Pipeline (per frame)

```
Input frame x_t
    ↓
CNN Encoder → h_t ∈ R^(N×D_enc)
    ↓
[Corrector: Slot Attention update]
    ↓
Slots Ŝ_t
    ↓
[Predictor: Transformer self-attention]
    ↓
Slots S_{t+1} (passed to next frame)
    ↓
Spatial Broadcast Decoder
    ↓
Optical flow prediction + segmentation masks
```

### Slot Initialization

**Conditional** (the paper's main focus):
- Encode hints (bounding box, center of mass, segmentation) via MLP/CNN
- Hints only provided for first frame
- Slots without hints get fixed placeholder value (-1)

**Unconditional**:
- Random Gaussian sampling (like original Slot Attention)
- Or learned initial slot vectors

---

## The Corrector (Slot Attention)

Updates slot representations based on visual features:

```python
# Slot Attention (slot-normalized cross-attention)
attn = softmax(k(h_t) · q(S_t)^T / √D, axis=slots)  # N×K
U_t = WeightedMean(attn, v(h_t))  # K×D
Ŝ_t = GRU(U_t, S_t)  # recurrent update
Ŝ_t = Ŝ_t + MLP(LayerNorm(Ŝ_t))  # optional residual
```

Key: Softmax over slots (not inputs) → competition for input positions.

---

## The Predictor (Transformer)

Models temporal dynamics and object interactions:

```python
# Transformer encoder block
S̃_t = LayerNorm(MultiHeadSelfAttn(Ŝ_t) + Ŝ_t)
S_{t+1} = LayerNorm(MLP(S̃_t) + S̃_t)
```

Why Transformer?
- Permutation equivariant (preserves slot symmetry)
- Allows information exchange between slots (interaction modeling)
- More memory efficient than Graph Networks

---

## Decoder: Spatial Broadcast

Same as original Slot Attention:
```python
for each slot s_k:
    tiled = tile(s_k, [H, W])
    tiled = concat(tiled, position_encoding)
    m̂_k, y_k = CNN_decoder(tiled)  # alpha mask + RGB/flow

# Combine with soft attention
m = softmax([m̂_1, ..., m̂_K], axis=slots)
output = Σ_k m_k * y_k
```

---

## Training Objective

**Optical flow prediction** (not RGB reconstruction):

$$L_{rec} = \sum_{t=1}^{T} \|y_t - y_t^{true}\|^2$$

### Why Optical Flow?
1. Flow directly encodes **motion**, the key cue for object segmentation
2. Avoids learning texture/appearance details that don't help segmentation
3. Provides implicit supervision for "what moves together"
4. Can use estimated flow (SMURF) when ground truth unavailable

---

## Datasets

### MOVi (simple)
- CLEVR-like 3D shapes
- Rigid body physics with collisions
- Simple backgrounds

### MOVi++ (realistic)
- 380 HDR real-world background photos
- 1028 3D scanned everyday objects (shoes, toys, household items)
- Significantly harder than prior work

### CATER
- CLEVR objects with complex temporal reasoning
- Used for comparison with prior work

---

## Key Results

### Conditional + Flow Training on MOVi++
| Method | FG-ARI | mIoU |
|--------|--------|------|
| SAVi + Center of Mass | **78.3%** | 43.5% |
| SAVi + Bounding Box | 77.4% | **45.9%** |
| SAVi + Segmentation | 70.4% | 43.0% |
| T-VOS (supervised baseline) | - | 46.4% |
| CRW (contrastive random walk) | - | 50.9% |

**Remarkable finding**: Center of mass (just one point per object!) works as well as full segmentation masks!

### Unconditional on CATER
| Method | FG-ARI |
|--------|--------|
| Slot Attention (image) | 7.3% |
| MONet | 41.2% |
| SIMONe | 91.8% |
| **SAVi (unconditional)** | **92.8%** |

---

## Ablation Studies

### Flow vs RGB reconstruction
- MOVi: Flow unnecessary (RGB works)
- MOVi++: Flow **critical** (RGB fails completely)

### Estimated vs Ground-truth Flow
- Estimated flow (SMURF) works nearly as well as ground truth
- Important for real-world applicability

### Robustness to Noisy Conditioning
- Gaussian noise up to ~20% of object size: minimal degradation
- Model is robust to imprecise hints

---

## Generalization Results

### Longer Sequences
- Trained on 6 frames, tested on 24 frames
- FG-ARI remains stable or even improves (more time to break symmetry)
- Unlike propagation methods which accumulate errors

### Novel Objects and Backgrounds
- <2% drop when testing on entirely new objects/backgrounds
- MOVi++ model transfers well to MOVi

### Part-Whole Segmentation
**Emergent capability**: Model can track parts OR wholes depending on conditioning granularity!
- Single bounding box around two fists → tracks both as one object
- Two bounding boxes → tracks each fist separately
- Model learns implicit part-whole hierarchy

---

## Why This Matters for World Models

SAVi demonstrates that:

1. **Object-centric representations can scale** to realistic visual complexity with the right inductive biases

2. **Weak supervision is sufficient**: Just knowing where objects are (roughly) enables full segmentation and tracking

3. **Motion is a powerful learning signal**: Optical flow provides object-level supervision without object labels

4. **Conditional interfaces enable flexible querying**: The same model can segment at different granularities

This provides the **perception module** that LeCun's world model architecture assumes exists.

---

## Connections to Other Papers

### vs Slot Attention
- Adds temporal dynamics (predictor)
- Adds conditional initialization
- Uses flow instead of reconstruction
- Scales to much more realistic data

### vs Graph Networks
- Predictor is a Transformer (special case of GN)
- Slots = nodes, self-attention = all-to-all edges
- Permutation equivariant like GNs

### vs LeCun's Architecture
- SAVi provides the **Perception** module
- Predictor resembles **World Model**
- Could use conditioning as **Configurator** interface

---

## Limitations

1. **Requires optical flow at training time** (though estimated flow works)
2. **Only rigid objects** with simple physics
3. **Only moving objects** (static object segmentation is harder)
4. **Synthetic data only** (real-world gap remains)

---

## Technical Details

- **Encoder**: 5-layer CNN with ReLU, position encoding at layer 4
- **Slots**: K=11, D=128
- **Attention iterations**: 1 (conditional), 2 (unconditional)
- **Training**: 100k steps, batch size 64, Adam with lr=2×10⁻⁴
- **Sequence length**: 6 frames during training

---

## Key Insight

> "Since our model achieves very reasonable segmentation and tracking performance, this is evidence that object-centric representation learning is not primarily limited by model capacity."

The bottleneck is **learning signal and inductive biases**, not model capacity. SAVi shows that:
- Flow as objective (captures "what moves together")
- Conditional initialization (guides decomposition granularity)
- Slot competition (ensures non-overlapping segments)

Together provide sufficient signal for object-centric learning at scale.

---

## Citation

```bibtex
@inproceedings{kipf2022conditional,
  title={Conditional Object-Centric Learning from Video},
  author={Kipf, Thomas and Elsayed, Gamaleldin F and Mahendran, Aravindh and Stone, Austin and Sabour, Sara and Heigold, Georg and Jonschkowski, Rico and Dosovitskiy, Alexey and Greff, Klaus},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
