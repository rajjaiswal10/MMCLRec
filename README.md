# MMCLGCN — Multimodal Contrastive LightGCN for Recommendation

A graph-based recommendation model that combines collaborative filtering (LightGCN) with multimodal item features (image, text description, metadata) and contrastive learning.

---

## Overview

Two variants are provided:

| Version | Adjacency Matrix | Key Difference |
|---------|-----------------|----------------|
| **v1** (`train.py`) | Binary (0/1 edges) | Simple interaction graph |
| **v2** (`train_rating.py`) | Rating-weighted edges | Edge weights scaled by normalized ratings (rating / 5.0) |

Both share the same model architecture and training loop.

---

## How It Works

### 1. Graph Construction
User–item interactions are modelled as a bipartite graph. A symmetric normalized adjacency matrix is built so that each node aggregates signals from its neighbours, weighted by degree (and rating in v2).

### 2. LightGCN Propagation
User and item ID embeddings are propagated through `NUM_LAYERS` (default: 2) graph convolution steps. The final embedding is the mean of all layer outputs — no activation functions or weight matrices, keeping it lightweight.

### 3. Multimodal Fusion
Each item has three side-feature embeddings (image, description, metadata). These are projected to a common dimension and concatenated with the graph embedding, then passed through a small MLP to produce a single **fused item embedding**.

### 4. Training Objectives
- **BPR Loss** — encourages the score of a positive (interacted) item to be higher than a sampled negative item.
- **InfoNCE / Contrastive Loss** — aligns the graph embedding of an item with each of its modality embeddings, pulling matching pairs together and pushing non-matching pairs apart.

Total loss: `L = L_BPR + λ_CL * L_InfoNCE`

### 5. Evaluation
Recall@20 is computed every 10 epochs. Training stops early if recall does not improve for `PATIENCE` consecutive checks.

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `G_DIM` | 64 | Collaborative embedding size |
| `FUSED_DIM` | 64 | Projected/fused embedding size |
| `NUM_LAYERS` | 2 | LightGCN propagation depth |
| `CL_WEIGHT` | 0.1 | Contrastive loss weight |
| `TEMP` | 0.5 | InfoNCE temperature |
| `LR` | 1e-3 | AdamW learning rate |
| `REG_LAMBDA` | 1e-4 | L2 regularisation |
| `EPOCHS` | 70/80 | Max training epochs |
| `BATCH_SIZE` | 8192 | Interactions per step |
| `PATIENCE` | 10 | Early stopping checks |

---

## Required Files

```
train.csv               # user, item (+ rating for v2)
test.csv
encoders.npz            # LabelEncoder classes
image_embeddings.npy    # (num_items, img_dim)
desc_embeddings.npy     # (num_items, txt_dim)
metadata_embeddings.npy # (num_items, meta_dim)
```

---

## Running

```bash
# v1 — binary adjacency
python train.py

# v2 — rating-weighted adjacency
python train_rating.py
```

The best model checkpoint is saved to `best_model.pt` based on Recall@20 on the test set.

---

## Architecture Summary

```
User ID ─┐
          ├─► LightGCN (2 layers) ─► user embedding
Item ID ──┘                     └─► item graph embedding ─┐
                                                            ├─► Fusion MLP ─► fused item embedding
Image emb ──► Linear ─────────────────────────────────────┤
Text emb  ──► Linear ─────────────────────────────────────┤
Meta emb  ──► Linear ─────────────────────────────────────┘

Score = dot(user embedding, fused item embedding)
```