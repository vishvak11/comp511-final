# comp511-final

# Learning Unified Graph-Based Representations for Spatiotemporal Transcriptomics

A graph-based representation learning framework for spatiotemporal transcriptomics that trains shared Graph Autoencoders (GAE) and Variational Graph Autoencoders (VGAE) across multiple timepoints to learn a unified latent space.

# Overview
This project builds spatial k-NN graphs from spatiotemporal transcriptomics data and trains graph autoencoders with shared weights across timepoints. The learned embeddings capture both gene expression and spatial neighborhood structure in a common latent space, enabling downstream tasks like clustering and trajectory inference.
We evaluate three GNN encoder variants (GCN, GraphSAGE, GAT), two training regimes (shared vs. separate encoders), and compare against non-graph baselines (PCA, standard AE, VAE). Downstream trajectory inference is performed using CellOT.

# Datasets
MOSTA — Mouse Organogenesis Spatiotemporal Transcriptomic Atlas (E9.5–E16.5)
Axolotl brain regeneration — Stereo-seq spatial transcriptomics (2–20 days post-injury)
