# TopologyAware CAD Encoder

Topology-aware semantic encoder for STEP/B-Rep CAD models.

## Overview

This project converts CAD models into continuous latent embeddings.

Pipeline:

STEP CAD
→ B-Rep Parsing
→ Face UV Sampling
→ Face Adjacency Graph
→ CNN Face Encoder
→ Graph Attention
→ Hierarchical Pooling
→ CAD Latent Vector

The learned latent space supports:

- CAD retrieval
- CAD clustering
- Semantic embedding
- Future CAD generation conditioning

---

## Architecture

STEP
↓
B-Rep
↓
Face UV Grid
↓
CNN
↓
Face Tokens
↓
Graph Attention
↓
Hierarchical Pooling
↓
64D Latent

---

## Latent Evaluation

The latent space was evaluated using:

- PCA
- Residual PCA
- t-SNE
- Complexity Correlation Analysis

Results show:

- Low complexity bias
- Stable latent clusters
- Continuous semantic structure

---

## Repository Structure

preprocess/
STEP → B-Rep preprocessing

training/
CAD encoder training

export/
latent export

examples/
evaluation figures

---

## License

MIT License