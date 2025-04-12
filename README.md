# Modular Gated Attention: Adaptive Architecture for Flexible Sequence Modeling

This repository contains the official PyTorch implementation for the paper: **"Modular Gated Attention: Adaptive Architecture for Flexible Sequence Modeling"** by Zung-Ru Lin (University of Pennsylvania).

## Overview

Standard Transformers, despite their success, face limitations in efficiency for long sequences and lack inherent inductive biases like locality or recurrence. Modular Gated Attention (MGA) addresses this by introducing a novel architecture incorporating three parallel computational paths per layer:

1.  **Local Attention Path:** Focuses on detailed contextual patterns using self-attention (optionally windowed).
2.  **Global Latent-Bottleneck Path:** Aggregates broader context efficiently via a low-rank bottleneck using latent vectors.
3.  **Recurrent State Path:** Captures sequential dependencies and persistent memory using a GRU module.

These paths are dynamically combined via a learned, token-wise gating mechanism, allowing the model to adapt its processing strategy based on context and task demands. This repository provides the code to reproduce the experiments presented in the paper on synthetic benchmarks (Selective Copy, Character Palindrome) and a semi-realistic task (Noisy WordPiece Palindrome).


<img width="1000" alt="Screenshot 2025-04-12 at 5 45 16 AM" src="https://github.com/user-attachments/assets/ab7bc127-798b-4665-80bb-6ada24b912db" />



<img width="1500" alt="Screenshot 2025-04-12 at 6 00 29 AM" src="https://github.com/user-attachments/assets/39074694-ea89-4c37-b003-70851285f6ab" />



## Results

**Selective Copy:** MGA significantly outperforms the baseline on OOD generalization (Table 1, Figure 2).
Baseline:
<img width="613" alt="1 baseline_acc" src="https://github.com/user-attachments/assets/2cb009fc-689b-41fa-a863-fa4361c39f2b" /> 
MGA:
<img width="617" alt="1 mga_acc" src="https://github.com/user-attachments/assets/143f6d1e-2c63-428f-9550-4db8fb23be5c" /> <img width="663" alt="1 visual" src="https://github.com/user-attachments/assets/246b9574-7d0c-48cd-8d8b-9cb167dbcc7b" />




**Character Palindrome:** MGA achieves near-perfect accuracy while the baseline struggles (Table 2, Figure 3).
Baseline:
<img width="601" alt="2 baseline_acc" src="https://github.com/user-attachments/assets/dffbb65e-872b-4ba4-a146-f9306a12b340" /> 
MGA:
<img width="610" alt="2 mga_acc" src="https://github.com/user-attachments/assets/66a091ea-0e75-4ea2-8a21-77fac1613bc9" /> <img width="672" alt="2 visual-1" src="https://github.com/user-attachments/assets/c7620bfb-78ae-4fcf-b554-3b9c5a4d2ebb" />





**WordPiece Palindrome:** MGA demonstrates robustness to noise and subword effects, achieving high accuracy where the baseline plateaus early (Table 3, Figure 4).
Baseline:
<img width="639" alt="3 baseline_acc" src="https://github.com/user-attachments/assets/c3de7274-6288-493d-821b-a24080f0c6df" />
MGA:
<img width="600" alt="3 mga_acc" src="https://github.com/user-attachments/assets/0f7562f7-9021-4be5-b662-4ef5bf8175e2" />
<img width="657" alt="3 visual-1" src="https://github.com/user-attachments/assets/33d17b9b-62be-456b-8bbc-ca018445cdb6" />




Gating Analysis: The code logs average gate weights, allowing analysis and showing task-dependent routing.

(Note: Minor variations in results due to library versions or hardware differences are possible).


