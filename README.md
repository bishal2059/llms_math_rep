# Layer-wise Evolution of Mathematical Concept Representations

## Overview

This repository contains the official implementation for the research project:

**"Layer-wise Evolution of Mathematical Concept Representations: Comparing Specialist and Generalist Language Models"**

The project investigates how mathematical concepts are represented across layers of:

- Qwen2.7
- Llama3.1

using a curated mathematical dataset containing:

- Algebra
- Arithmetic
- Calculus
- Probability

The research consists of three major phases:

1. Layer-wise embedding extraction
2. Clustering analysis with silhouette scoring
3. Logistic regression classification across layers

---

## Research Goals

This work aims to:

- Analyze how mathematical concepts evolve across transformer layers
- Compare specialist and generalist language models
- Measure semantic separability using clustering metrics
- Evaluate layer-wise predictive power using classification
- Study representation geometry in mathematical reasoning

---

## Dataset

The dataset contains:

- 2000 mathematical samples
- 4 balanced mathematical categories
- 500 samples per category

### Categories

| Class | Samples |
|---|---|
| Algebra | 500 |
| Arithmetic | 500 |
| Calculus | 500 |
| Probability | 500 |

---

## Methodology

### Phase 1: Embedding Extraction

For each model:

- Extract hidden states from every transformer layer
- Apply pooling strategy
- Save embeddings layer-wise

### Phase 2: Clustering Analysis

For each layer:

- Perform clustering
- Compute silhouette score
- Evaluate cluster separability

### Phase 3: Classification Analysis

For each layer:

- Train logistic regression classifier
- Evaluate accuracy and F1 score
- Compare layer-wise performance trends

---

## Installation

```bash
git clone git@github.com:bishal2059/llms_math_rep.git
cd llms_math_rep

pip install -r requirements.txt