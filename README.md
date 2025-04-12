# Modular Gated Attention: Adaptive Architecture for Flexible Sequence Modeling

This repository contains the official PyTorch implementation for the paper: **"Modular Gated Attention: Adaptive Architecture for Flexible Sequence Modeling"** by Zung-Ru Lin (University of Pennsylvania).

## Overview

Standard Transformers, despite their success, face limitations in efficiency for long sequences and lack inherent inductive biases like locality or recurrence. Modular Gated Attention (MGA) addresses this by introducing a novel architecture incorporating three parallel computational paths per layer:

1.  **Local Attention Path:** Focuses on detailed contextual patterns using self-attention (optionally windowed).
2.  **Global Latent-Bottleneck Path:** Aggregates broader context efficiently via a low-rank bottleneck using latent vectors.
3.  **Recurrent State Path:** Captures sequential dependencies and persistent memory using a GRU module.

These paths are dynamically combined via a learned, token-wise gating mechanism, allowing the model to adapt its processing strategy based on context and task demands. This repository provides the code to reproduce the experiments presented in the paper on synthetic benchmarks (Selective Copy, Character Palindrome) and a semi-realistic task (Noisy WordPiece Palindrome).


<img width="578" alt="Screenshot 2025-04-12 at 5 45 16 AM" src="https://github.com/user-attachments/assets/ab7bc127-798b-4665-80bb-6ada24b912db" />
