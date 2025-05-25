# Are we cooking? Or are we cooked? A comparative Analysis

This repository contains a series of 10 Jupyter notebooks, datasets and images, where each notebook corresponds to one step in the process, from raw data filtering through an in-depth analysis of the misclassification errors. The goal was to execute a systematic comparison of an LLM-based classifier and a BERT-based classifier in terms of feature importance. 
---

## Table of Contents

1. [Notebook Overview](#notebook-overview)
2. [Datasets Overview](#dataset-overview)
3. [Results & Figures](#results--figures)  

---

## Notebook Overview

### 1. Dataset Preparation 1  
*Filename:* 1_dataset_preparation_1.ipynb  
*Goal:* Reduce the raw Kaggle recipe dataset from ≈2.2 M recipes down to ~298 K.  
*Key steps:*  
- Filter by cooking direction length (IQR).  
- Remove titles with unusual punctuation.  
- Exclude recipes with >50 ingredients.  
- Deduplicate via TF–IDF + nearest-neighbour (cosine < 0.2).

### 2. Dataset Preparation 2  
*Filename:* 2_dataset_preparation_2.ipynb  
*Goal:* Further shrink to ~25 K recipes for manageable downstream analysis.  
*Key steps:*  
- Keep only top 90% most frequent ingredients.  
- Extract verbs from directions (lemmatize + POS-tag).  
- Retain the 90% most common verbs.  
- Final dataset: 25 758 recipes (see preprocessing details in Appendix A).

### 3. Embeddings with RecipeNLP  
*Filename:* 3_embeddings_recipenlp.ipynb  
*Goal:* Compute unified BERT embeddings for each recipe.  
*Key steps:*  
- Title embeddings: all-MiniLM-L6-v2.  
- Ingredient & direction embeddings: TF–IDF + weighted BERT.  
- Concatenate into 1 152-dim vectors.

### 4. Clustering with RecipeNLP  
*Filename:* 4_cluster_recipenlp.ipynb  
*Goal:* Group recipes into 7 clusters via hierarchical clustering.  
*Key steps:*  
- Dimensionality reduction with UMAP (60 components).  
- Agglomerative clustering (Ward linkage, k = 7).  
- Manual label assignment based on top log-odds features.

### 5. Dish Categories (LLM)  
*Filename:* 5_dishcategories.ipynb  
*Goal:* Classify each recipe using an LLM (e.g., Mistral-7B-Instruct).  
*Key steps:*  
- Zero-shot prompt with chain-of-thought.  
- Constrain choices to the 7 cluster labels.  
- Greedy decoding, 60-token limit.

### 6. BERT vs. LLM (Overview)  
*Filename:* 6_bertvsllm.ipynb  
*Goal:* Compare overall performance of BERT clustering vs. LLM classification.  
*Key steps:*  
- Compute accuracy, precision, recall, F1 against human-labeled sample (1 000 recipes).  
- Summarize major differences in a high-level dashboard.

### 7. Human Label Extraction  
*Filename:* 7_extraction_human_answer.ipynb  
*Goal:* Extract & format the human gold-standard labels.  
*Key steps:*  
- Load manual annotations.  
- Align with model outputs for later comparison.

### 8. Combined Log-Odds Analysis  
*Filename:* 8_combined_log_odds.ipynb  
*Goal:* Analyze feature importance via smoothed log-odds vs. human standard.  
*Key steps:*  
- Compute mean & variance of log-odds for each feature block.  
- Compare BERT vs. LLM log-odds distributions.

### 9. LLM Feature Importance  
*Filename:* 9_llm_feature_importance.ipynb  
*Goal:* Rank and visualize which tokens the LLM used in its chain-of-thought.  
*Key steps:*  
- Token-level log-odds within rationale keywords.  
- Rank features by discriminative strength.

### 10. BERT vs. LLM II (Error Cases)  
*Filename:* 10_bertvsllm_2.ipynb  
*Goal:* Dive into misclassification cases to understand model failures.  
*Key steps:*  
- Identify where BERT and LLM disagree with humans.  
- Case studies of high-overlap clusters (e.g., breakfast vs. desserts).  
- Outlier analysis & robustness checks.

---

## Results & Figures

- *Cluster purity* and *UMAP plots* (Notebook 4)  
- *Classification reports* (Notebooks 6 & 10)  
- *Log-odds heatmaps* and *histograms* (Notebooks 8 & 9)  

_All figures are saved under /figures._
