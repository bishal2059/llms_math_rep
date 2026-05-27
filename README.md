# Layer-wise Evolution of Mathematical Concept Representations:
## Comparing Specialist and Generalist Language Models

This repository contains the official implementation for the research project:

> **Layer-wise Evolution of Mathematical Concept Representations: Comparing Specialist and Generalist Language Models**

The project investigates how mathematical concepts evolve across transformer layers in large language models using embedding-space analysis, clustering, and linear probing.

---

# Research Objective

This work studies how mathematical knowledge is represented internally across layers of transformer-based language models.

We compare:

- Qwen2.5-7B
- Llama-3.1-8B-Instruct

using a balanced mathematical dataset consisting of four domains:

- Algebra
- Arithmetic
- Calculus
- Probability

The project analyzes:

1. Layer-wise embedding evolution
2. Cluster separability of mathematical concepts
3. Linear probing performance across layers
4. Representation geometry differences between models

---

# Dataset

The dataset contains:

- 2000 mathematical questions
- 4 balanced classes
- 500 samples per class

## Dataset Format

```json
[
  {
    "question": "Solve -5*l + 8*l - 3 = 0 for l.",
    "answer_domain": "Algebra"
  },
  {
    "question": "Find the derivative of x^2 + 3x.",
    "answer_domain": "Calculus"
  }
]
```

---

# Repository Structure

```text
math-concept-representations/
├── configs/
├── data/
├── results/
├── scripts/
├── src/
├── notebooks/
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

# Methodology

The research pipeline consists of three phases.

---

## Phase 1 — Layer-wise Embedding Extraction

For each model:

- Load transformer hidden states
- Extract representations from every layer
- Apply mean pooling
- Save embeddings as `.npz`

Each `.npz` file contains:

- embeddings
- labels
- sample IDs
- questions
- model name
- layer index

---

## Phase 2 — Clustering Analysis

For every layer:

- K-Means clustering is performed
- Silhouette score is computed
- Representation separability is analyzed

This phase measures how well mathematical domains separate geometrically inside the embedding space.

---

## Phase 3 — Linear Probing

For every layer:

- Logistic regression is trained
- Layer embeddings are used as features
- Classification performance is measured

Metrics include:

- Accuracy
- Macro F1
- Weighted F1

---

# Models

| Model | Type |
|---|---|
| Qwen2.5-7B | Specialist |
| Llama-3.1-8B-Instruct | Generalist |

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/math-concept-representations.git

cd math-concept-representations
```

---

## 2. Create Environment

### Using venv

```bash
python -m venv venv
```

### Activate Environment

#### Linux / macOS

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\\Scripts\\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

---

## Step 1 — Validate Dataset

```bash
python scripts/00_validate_dataset.py
```

Checks:

- valid JSON
- required keys
- empty values
- label consistency

---

## Step 2 — Preprocess Dataset

```bash
python scripts/01_preprocess_dataset.py
```

Creates:

```text
data/processed/dataset.parquet
```

---

## Step 3 — Create Train/Test Split

```bash
python scripts/02_split_dataset.py
```

Creates:

```text
data/splits/train_ids.csv
data/splits/test_ids.csv
```

Split strategy:

- 400 train samples per class
- 100 test samples per class

---

## Step 4 — Extract Layer Embeddings

```bash
python scripts/03_extract_embeddings.py
```

Creates:

```text
results/embeddings/
```

Each layer is saved separately.

Example:

```text
results/embeddings/qwen/layer_00.npz
results/embeddings/qwen/layer_01.npz
```

---

## Step 5 — Run Clustering

```bash
python scripts/04_run_clustering.py
```

Creates:

```text
results/clustering/silhouette_results.csv
```

---

## Step 6 — Run Linear Probing

```bash
python scripts/05_run_probing.py
```

Creates:

```text
results/probing/layerwise_results.csv
```

---

## Step 7 — Generate Final Analysis

```bash
python scripts/06_generate_analysis.py
```

Creates:

```text
results/tables/
results/figures/
```

---

## Run Complete Pipeline

```bash
python scripts/07_run_full_pipeline.py
```

---

# Output Files

---

## Embeddings

```text
results/embeddings/
```

Contains layer-wise `.npz` files.

---

## Clustering Results

```text
results/clustering/silhouette_results.csv
```

Columns:

- model_name
- layer_idx
- silhouette_score

---

## Probing Results

```text
results/probing/layerwise_results.csv
```

Columns:

- accuracy
- macro_f1
- weighted_f1

---

## Figures

```text
results/figures/
```

Includes:

- accuracy vs layer
- silhouette vs layer

---

# Experimental Details

| Setting | Value |
|---|---|
| Dataset Size | 2000 |
| Classes | 4 |
| Train/Test Split | 400/100 per class |
| Pooling Strategy | Mean Pooling |
| Clustering Algorithm | K-Means |
| Probe Model | Logistic Regression |
| Random Seed | 42 |

---

# Reproducibility

To ensure reproducibility:

- fixed random seed
- same split across all experiments
- same preprocessing
- same pooling strategy
- same evaluation protocol

---

# Research Contributions

This work contributes:

1. Layer-wise analysis of mathematical representations
2. Comparison between specialist and generalist LLMs
3. Embedding-space analysis of mathematical concepts
4. Linear probing evaluation across transformer depth
5. Representation geometry analysis of mathematical reasoning

---

# Citation

```bibtex
@article{yourpaper2026,
  title={Layer-wise Evolution of Mathematical Concept Representations: Comparing Specialist and Generalist Language Models},
  author={Author Names},
  year={2026}
}
```

---

# License

MIT License