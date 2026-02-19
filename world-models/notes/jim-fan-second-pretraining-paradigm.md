# The Second Pre-training Paradigm

**Author**: Jim Fan (@DrJimFan), Senior Research Scientist at NVIDIA  
**Source**: [X/Twitter Article](https://x.com/DrJimFan/status/2018754323141054786)  
**Date**: 2026-02-03  
**Type**: Opinion/Essay

## Summary

Jim Fan argues we are witnessing the second major paradigm shift in AI pre-training:

| First Paradigm | Second Paradigm |
|----------------|-----------------|
| Next Word Prediction | Next Physical State Prediction |
| Language Models | World Models |
| Text-first | Vision-first |

## Key Arguments

### 1. What Are World Models?

World Models predict the next plausible world state (or sequence of states) conditioned on an action.

- Video generative models are one instantiation
- "Next states" = sequence of RGB frames (8-10 seconds to minutes)
- "Action" = textual description of what to do
- Training involves modeling future changes in billions of hours of video pixels

**Core insight**: Video WMs are **learnable physics simulators and rendering engines**. They capture counterfactuals — reasoning about how the future would unfold differently given alternative actions.

### 2. VLM vs World Model: Fundamental Difference

**VLMs are language-first:**
- Vision enters at encoder, routes into language backbone
- Vision is "second-class citizen"
- Most parameters allocated to **knowledge** (e.g., "this is a Coca Cola brand")

**World Models are vision-first:**
- Parameters allocated to **physics** (e.g., "tipping the bottle spreads liquid, stains cloth, ruins motor")
- Directly model the sensorimotor loop
- No language bottleneck required

### 3. Current Physical AI Landscape (2025)

VLAs (Vision-Language-Action models) dominate:
- Graft robot motor action decoder on pre-trained VLM
- Really "LVAs": Language > Vision > Action (decreasing citizenship)
- Head-heavy in wrong places for physical reasoning

### 4. Biological Evidence

- ~1/3 of human cortex devoted to visual processing
- Language relies on relatively compact area
- Vision is highest-bandwidth channel for sensorimotor loop

**The Ape Argument:**
> "I've seen apes drive golf carts and change brake pads with screwdrivers like human mechanics. Their language understanding is no more than BERT or GPT-1, yet their physical skills are far beyond anything our SOTA robots can do."

**Conclusion**: Nature proves highly dexterous physical intelligence is possible with minimal language capability.

### 5. The Future of World Modeling

**New Pre-training:**
- Beyond RGB: 3D spatial motions, proprioception, tactile sensing
- YouTube + smart glasses will capture visual streams at scale beyond all text ever trained on

**New Reasoning:**
- Chain of thought in **visual space** rather than language space
- Solve physical puzzles by simulating geometry and contact
- No translation to strings required

**Key Quote:**
> "Language is a bottleneck, a scaffold, not a foundation."

### 6. Open Questions

1. With perfect future simulation, how should motor actions be decoded?
2. Is pixel reconstruction the best objective, or should we use alternative latent spaces?
3. How much robot data is needed? Is scaling teleoperation still the answer?
4. Are we finally approaching the "GPT-3 moment" for robotics?

## Memorable Quotes

> "Supervision is the opium of the AI researcher." — Jitendra Malik

> "Language is a bottleneck, a scaffold, not a foundation."

> "AGI has not converged. We are back to the age of research, and nothing is more thrilling than challenging first principles."

## Relevance

This article connects to several research threads:

1. **World Models for Robotics**: Vision-first approach to physical AI
2. **Video Generation**: Sora, Veo, and other video models as physics simulators
3. **Embodied AI**: Closing the sensorimotor loop without language intermediary
4. **Scaling Laws**: Whether video/world model scaling follows similar patterns to LLMs

## Personal Notes

The "ape argument" is particularly compelling — it's an existence proof that physical intelligence doesn't require sophisticated language. This challenges the VLM-centric approach that dominates current robotics research.

The prediction that 2026 marks the foundation year for Large World Models in robotics is bold but aligns with rapid progress in video generation quality.

---

*Added: 2026-02-19*
