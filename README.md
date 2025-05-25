# Neural Graph Collaborative Filtering for Supermarket Purchase Recommendation

This repository provides a Jupyter notebook implementation of the **Neural Graph Collaborative Filtering (NGCF)** model, adapted for a real-world recommendation system to suggest items for supermarket customers. The NGCF model leverages the structure of the user-item interaction graph to generate high-quality user and product embeddings, improving prediction accuracy through graph-based representation learning.

This work is based on the original paper:

> Wang, Xiang, et al. ["Neural Graph Collaborative Filtering."](https://dl.acm.org/doi/10.1145/3331184.3331267) *Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval*. 2019.

---

## Overview

The NGCF model captures higher-order collaborative signals by propagating user and item embeddings through a bipartite graph formed by their interactions. This notebook-based implementation includes:

- Graph-based modeling of user-item interactions  
- Multi-layer message passing and embedding propagation  
- Hard negative sampling to improve training robustness  
- Mixed precision training for computational efficiency  
- Evaluation using the **Hit Rate@K** metric  

---

## Datasets

As this project is based on a **real-world industry use case**, the datasets are not publicly available due to safety and privacy concerns. However, the notebook is structured in a way that it can be adapted to other datasets with similar schema.

Three datasets were used in this implementation:

- **Train Data**: Contains historical transaction records from **2022 and 2023** for **100,000 customers**. Each record includes customer identifiers and purchase information.

- **Product Data**: Includes detailed metadata about products, such as their characteristics, categories, and other properties relevant for recommendation.

- **Test Data**: Contains **actual purchases made in 2024** by the first **80,000 customers** from the training set, used to evaluate model performance.

---

## Notebook Workflow

All implementation logic is contained in a single Jupyter notebook, organized into the following steps:

### 1. Data Preprocessing
- Loading and transforming interaction data  
- Creating the user-item bipartite graph  
- Initializing embeddings for users and items  
- Preparing training samples with hard negative sampling  

### 2. Model Architecture
- `EmbeddingPropagationLayer`: Basic graph message-passing layer  
- `GraphEmbeddingPropagation`: Multi-layer graph encoder  
- `NGCF`: Full model including forward pass and prediction  
- BPR (Bayesian Personalized Ranking) loss for training  

### 3. Training Pipeline
- Training loop with early stopping  
- Mixed precision support  
- Hyperparameter tuning options  

### 4. Evaluation
- Generation of top-K recommendations per user  
- Evaluation via Hit Rate@K  
- Submission file creation for offline scoring  

---

## Reference

Wang, Xiang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua.  
**Neural Graph Collaborative Filtering.**  
*In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '19), 2019.*  
[DOI: 10.1145/3331184.3331267](https://doi.org/10.1145/3331184.3331267)
