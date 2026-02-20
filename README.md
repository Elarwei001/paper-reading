# 📚 Paper Reading Notes

Academic paper reading notes, organized by research topic.

## Structure

```
paper-reading/
├── README.md           # This file (central index)
├── TEMPLATE.md         # Template for new notes
├── <topic>/            # Topic directories
│   ├── paper1.md       # Paper notes (flat structure)
│   ├── paper2.md
│   └── papers/         # Original PDFs (optional)
```

---

## 🌍 World Models & Cognitive AI

Internal representations for simulating and predicting environment changes. Core component of general intelligence.

### Foundations
| Paper | Authors | Year | Notes |
|-------|---------|------|-------|
| Relational Inductive Biases, Deep Learning, and Graph Networks | Battaglia et al. (DeepMind) | 2018 | [relational_inductive_biases.md](world-models/relational_inductive_biases.md) |
| A Path Towards Autonomous Machine Intelligence | LeCun | 2022 | [lecun_autonomous_intelligence.md](world-models/lecun_autonomous_intelligence.md) |

### Object-Centric Learning
| Paper | Authors | Year | Notes |
|-------|---------|------|-------|
| Object-Centric Learning with Slot Attention | Locatello, Kipf et al. | 2020 | [slot_attention.md](world-models/slot_attention.md) |
| Conditional Object-Centric Learning from Video (SAVi) | Kipf et al. | 2022 | [savi_conditional_object_centric.md](world-models/savi_conditional_object_centric.md) |

### Surveys & Reading Guides
| Topic | Notes |
|-------|-------|
| World Models Survey | [world-model-survey.md](world-models/world-model-survey.md) |
| Timeline of Key Developments | [world-model-timeline.md](world-models/world-model-timeline.md) |
| Structured World Models (Object-centric, JEPA) | [structured-world-models-survey.md](world-models/structured-world-models-survey.md) |
| Jim Fan: Second Pre-training Paradigm | [jim-fan-second-pretraining-paradigm.md](world-models/jim-fan-second-pretraining-paradigm.md) |
| Research Landscape: Laws from Phenomena | [research-landscape-laws-from-phenomena.md](world-models/research-landscape-laws-from-phenomena.md) |
| Deep Learning Overview Reading Guide | [deep_learning_overview_reading_guide.md](world-models/deep_learning_overview_reading_guide.md) |

---

## 🔗 Causal Representation Learning

Bridging machine learning and causal inference. Learning representations that capture causal structure for robust generalization.

### Key Papers
| Paper | Authors | Year | Notes |
|-------|---------|------|-------|
| Causality for Machine Learning | Schölkopf | 2019 | [causality_for_ml.md](causal-representation-learning/causality_for_ml.md) |
| A Meta-Transfer Objective for Learning to Disentangle Causal Mechanisms | Bengio et al. | 2019 | [meta_transfer_causal.md](causal-representation-learning/meta_transfer_causal.md) |
| Towards Causal Representation Learning | Schölkopf, Bengio et al. | 2021 | [towards_causal_rep_learning.md](causal-representation-learning/towards_causal_rep_learning.md) |
| Elements of Causal Inference (Textbook) | Peters, Janzing, Schölkopf | 2017 | [elements_of_causal_inference.md](causal-representation-learning/elements_of_causal_inference.md) |

### Core Concepts
- **ICM (Independent Causal Mechanisms)**: Causal mechanisms are modular and independent
- **Sparse Mechanism Shift**: Distribution shifts typically affect only a few mechanisms
- **Causal Discovery**: Inferring causal structure from data

### Survey
| Topic | Notes |
|-------|-------|
| Causal Representation Learning Survey | [causal-representation-learning-survey.md](world-models/causal-representation-learning-survey.md) |

---

## 🧬 Structural Biology & Drug Discovery

AI for protein structure, molecular interactions, and drug design.

### Papers
| Paper | Authors | Year | Notes |
|-------|---------|------|-------|
| AlphaGenome (Regulatory Variant Prediction) | DeepMind | 2026 | [alphagenome-review.md](alphagenome/alphagenome-review.md) |

### Resources
| Topic | Notes |
|-------|-------|
| Boltz - Latent Space Podcast Summary | [boltz-latent-space-podcast.md](structural-biology/boltz-latent-space-podcast.md) |

### Key Concepts
- **AlphaFold 2/3**: Protein structure prediction breakthrough
- **Co-evolution**: Using evolutionary information for structure prediction
- **Diffusion Models**: Generative modeling for molecular design

---

## 🧠 Memory-Augmented Transformers

Extending transformer context via external memory and retrieval. Key for 1M+ token context.

### Research Directions (To Read)
- **Memorizing Transformer** (Google, 2022): External KV store + retrieval
- **Landmark Attention** (2023): Landmark tokens for hierarchical retrieval
- **LongMem** (2023): Decoupling memory from reasoning
- **StreamingLLM** (2023): Attention sink for infinite context
- **H2O** (2023): Heavy-hitter KV cache eviction

### Core Idea
> Not storing everything in VRAM, retrieve on demand — like foveal vision focusing on relevant parts.

---

## Key Researchers

### World Models & Physical AI
- **Yann LeCun** (Meta FAIR) — JEPA, cognitive architectures
- **Jim Fan** (NVIDIA) — Physical AI, robotics
- **Danijar Hafner** (DeepMind) — Dreamer series
- **Thomas Kipf** (Google DeepMind) — Slot Attention, object-centric learning
- **Peter Battaglia** (DeepMind) — Graph networks for physics

### Causal ML
- **Yoshua Bengio** (Mila) — GFlowNets, causal induction
- **Bernhard Schölkopf** (MPI Tübingen) — Causal ML theory
- **Jonas Peters** (ETH) — Causal discovery

### Structural Biology
- **John Jumper** (DeepMind) — AlphaFold
- **David Baker** (UW) — Protein design

---

## How to Add Notes

1. Find the appropriate topic directory (or create one)
2. Copy `TEMPLATE.md` → `<topic>/<paper-name>.md`
3. Fill in the template sections
4. Add entry to this README under the relevant section
