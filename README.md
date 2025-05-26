# From Transformers to the Table: How BERT and LLMs Cook Up Features Differently

This repository contains a series of 10 Jupyter notebooks, datasets and images, where each notebook corresponds to one step in the process, from raw data filtering through an in-depth analysis of the misclassification errors. The goal was to execute a systematic comparison of an LLM-based classifier and a BERT-based classifier in terms of feature importance. 

## Table of Contents


1. [Notebook Overview](#notebook-overview)
2. [Datasets Overview](#datasets-overview)
3. [Results & Figures](#results--figures)

---

## Notebook Overview

### 1. [Dataset Preparation](notebooks/1_DatasetPreparation.ipynb)
*Filename:* 1_DatasetPreparation.ipynb  
*Goal:* Reduce the raw Kaggle recipe dataset from ≈2.2 M recipes down to ~298 K.  
*Key steps:*  
- Filter by cooking direction length (IQR).  
- Remove titles with unusual punctuation.  
- Exclude recipes with >50 ingredients.  
- Deduplicate via TF–IDF + nearest-neighbour (cosine < 0.2) with Union-Find algorithm.

### 2. [Dataset Preparation, phase 2](notebooks/2_DatasetPreparation_P2.ipynb)
*Filename:* 2_DatasetPreparation_P2.ipynb  
*Goal:* Further shrink to ~25 K recipes for manageable downstream analysis.  
*Key steps:*  
- Keep only top 90% most frequent ingredients.  
- Extract verbs from directions (lemmatize + POS-tag).  
- Retain the 90% most common verbs.  
- Final dataset: 25 758 recipes (see preprocessing details in Appendix A).

### 3. [Embeddings with RecipeNLP](notebooks/3_Embeddings_RecipeNLP.ipynb)
*Filename:* 3_Embeddings_RecipeNLP.ipynb
*Goal:* Compute unified BERT embeddings for each recipe.  
*Key steps:*  
- Title embeddings: `all-MiniLM-L6-v2`.  
- Ingredient & direction embeddings: TF–IDF + weighted BERT.  
- Concatenate into 1 152-dim vectors.

### 4. [Clustering with RecipeNLP](notebooks/4_Clusters_RecipeNLP.ipynb)
*Filename:* 4_cluster_recipenlp.ipynb  
*Goal:* Group recipes into 7 clusters via hierarchical clustering.  
*Key steps:*  
- Dimensionality reduction with UMAP (60 components).  
- Agglomerative clustering (Ward linkage, k = 7).  
- Manual label assignment based on top log-odds features.

### 5. [Dish Categories (LLM)](notebooks/5_DishCategories.ipynb)
*Filename:* 5_DishCategories.ipynb
*Goal:* Classify each recipe using an LLM (`Mistral-7B-Instruct-v0.2`).  
*Key steps:*  
- Zero-shot prompt with chain-of-thought.  
- Constrain choices to the 7 cluster labels.  
- Greedy decoding, 60-token limit.

### 6. [BERT vs. LLM (Overview)](notebooks/6_BERTvsLLM.ipynb)
*Filename:* 6_BERTvsLLM.ipynb
*Goal:* Compare overall performance of BERT clustering vs. LLM classification.  
*Key steps:*  
- Compute accuracy, precision, recall, F1.  
- Summarize major differences in a high-level dashboard.

### 7. [Human Label Extraction](notebooks/7_ExtractionHumanAnswers.ipynb)
*Filename:* 7_ExtractionHumanAnswers.ipynb
*Goal:* Extract & format the human gold-standard labels.  
*Key steps:*  
- Load manual annotations.  
- Align with model outputs for later comparison.

### 8. [Combined Log-Odds Analysis](notebooks/8_Combined_Log_odds.ipynb)
*Filename:* 8_Combined_Log_odds.ipynb
*Goal:* Analyze feature importance via smoothed log-odds, compare performance of BERT and LLM against human standard.  
*Key steps:*  
- Confusion matrix analysis for BERT and LLM.
- Compute mean & variance of log-odds for each feature block.
- Compare BERT vs. LLM log-odds distributions.
- Outlier analysis & robustness checks.

### 9. [LLM Feature Importance](notebooks/9_LLMFeatureImportance.ipynb)
*Filename:* 9_LLMFeatureImportance.ipynb
*Goal:* Rank and visualize which tokens the LLM used in its chain-of-thought at a local, categorical, and global level.  
*Key steps:*  
- Features matching to Ingredients, Titles, Verbs, Prompt information, and out-prompt information.
- Token-level log-odds within rationale keywords.
- Rank features by discriminative strenght and categorical uniqueness.

### 10. [BERT vs. LLM II (Error Cases)](notebooks/10_BERTvsLLM_P2.ipynb)
*Filename:* 10_BERTvsLLM_P2.ipynb
*Goal:* Investigate further into misclassification cases to understand model behavior and failure.  
*Key steps:*  
- Identify where BERT and LLM disagree with humans.
- Case studies of high-overlap clusters (breakfast vs. desserts).  

---
## Datasets Overview

The datasets are stored in the `datasets` as CSVs, while others can be downloaded on request through url. It follows a brief description of each:

| File                                   | Description                                                                                       |
|----------------------------------------|---------------------------------------------------------------------------------------------------|
| [RecipeNLG_dataset.csv](datasets/References.md)     | Original raw Kaggle recipe dump (~2.2 M recipes) after basic formatting and column harmonization. |
| [reduced_dataset.csv](datasets/References.md)          | Output of *Notebook 1*: filtered down to ~298 K recipes.                                         |
| [ds_verbs.csv](datasets/References.md)                  | Refined version of the `reduced_dataset.csv` filtered down to ~25 K recipes. (used in *Notebook 2*).                    |
| [hierarchical_clusters.csv](datasets/References.md)     | UMAP‐reduced embeddings and 7 hierarchical cluster labels for each recipe (*Notebook 4*).       |
| [dish_type_results_25758.csv](datasets/References.md)   | Datasets of 25 758 recipes populated with LLM outputs (output of *Notebook 5*).                                |
| [llm_bert_clusters.csv](datasets/llm_bert_clusters.csv)         | Merged BERT and LLM cluster assignments for every recipe (from *Notebooks 6*).              |
| [llm_clusters_human.csv](datasets/llm_clusters_human.csv)        | Combining merged datasets with human categorisation (*Notebook 7*).       |
| [human_labelling.csv](datasets/human_labelling.csv)           | Gold-standard human annotations for the 1 000-recipe evaluation set (used in *Notebook 7*).              |
| [llm_logodds.csv](datasets/llm_logodds.csv)               | Token-level log-odds scores extracted from the LLM’s chain-of-thought outputs (*Notebook 8*).    |
| [recipes_with_logodds_lists.csv](datasets/References.md)| Combined log-odds statistics (mean & variance) per recipe for both BERT & LLM features (*Notebook 8*). | [ingredient_embeddings_by_title.npz](datasets/References.md)| Ingredients embeddings by title (result of *Notebook 4*). |
| [title_embeddings_by_title.npz](datasets/References.md)| Title embeddings (result of *Notebook 4*). |
| [verb_embeddings_by_title.npz](datasets/References.md)| Ingredients embeddings by title (result of *Notebook 4*). |
---

## Results & Figures

- *Cluster purity* and *UMAP plots* (Notebook 4)  
- *Classification reports* (Notebooks 6 & 10)  
- *Log-odds heatmaps* and *histograms* (Notebooks 8 & 9)  

_All figures are saved in the folder [`images`](images)._
