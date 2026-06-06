# Layer-wise Evolution of Mathematical Concept Representations

Official code for studying how mathematical concepts are encoded across transformer layers in specialist and generalist language models.

**Paper:** *Layer-wise Evolution of Mathematical Concept Representations: Comparing Specialist and Generalist Language Models*

---

## Abstract

We analyze internal representations of mathematical reasoning in large language models by extracting layer-wise hidden states, measuring geometric cluster separability, and training linear probes to classify problem domains. Two models are compared under identical conditions:

| Model | Hugging Face ID | Role |
|-------|-----------------|------|
| Qwen2.5-Math-7B-Instruct | `Qwen/Qwen2.5-Math-7B-Instruct` | Math specialist |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | Generalist |

Experiments use a balanced corpus of **2,000** math questions across four domains (Algebra, Arithmetic, Calculus, Probability). The pipeline reports layer-wise silhouette scores, probing accuracy, and low-dimensional embedding projections, with paired statistical tests between models.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing Experiments](#reproducing-experiments)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Google Colab](#google-colab)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Requirements

| Component | Version |
|-----------|---------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1 |
| CUDA GPU (recommended) | ≥ 16 GB VRAM |
| Hugging Face account | Required for Llama 3.1 access |

**Hardware notes.** Embedding extraction loads 7B–8B models with `device_map="auto"`. A GPU with at least **16 GB VRAM** (e.g. T4, RTX 4090, A10) is recommended. CPU execution is supported but impractically slow at this scale.

---

## Installation

```bash
git clone https://github.com/bishal2059/llms_math_rep.git
cd llms_math_rep

python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
pip install -e .
```

The editable install (`pip install -e .`) registers the `src` package so pipeline scripts can be run from the repository root.

**Llama access.** Accept the model license at [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), then authenticate:

```bash
huggingface-cli login
```

---

## Quick Start

Run the full experiment end-to-end:

```bash
python scripts/07_run_full_pipeline.py
```

This executes all stages below in order and writes results to `results/`.

---

## Reproducing Experiments

### Dataset

The raw dataset is provided at `data/raw/math_dataset.json`:

- **2,000** questions, **500** per domain
- Fields: `question`, `answer_domain`

```json
[
  {
    "question": "Solve -5*l + 8*l - 3 = 0 for l.",
    "answer_domain": "Algebra"
  }
]
```

### Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 0 | `scripts/00_validate_dataset.py` | Validate JSON schema and non-empty fields |
| 1 | `scripts/01_preprocess_dataset.py` | Normalize labels → `data/processed/dataset.parquet` |
| 2 | `scripts/02_split_dataset.py` | Fixed split → `data/splits/train_ids.csv`, `test_ids.csv` |
| 3 | `scripts/03_extract_embeddings.py` | Extract mean-pooled hidden states per layer (GPU) |
| 4 | `scripts/04_run_clustering.py` | K-Means (k=4) + silhouette score per layer |
| 5 | `scripts/05_run_probing.py` | Logistic regression probe per layer |
| 6 | `scripts/06_generate_analysis.py` | Summary tables and layer-wise metric plots |
| 7 | `scripts/07_run_full_pipeline.py` | Run steps 0–6, 8–10 sequentially |
| 8 | `scripts/08_run_significance_tests.py` | Paired Wilcoxon / t-tests (Qwen vs Llama) |
| 9 | `scripts/09_plot_umap_layers.py` | UMAP projections per layer |
| 10 | `scripts/10_plot_pca_layers.py` | PCA projections per layer |

### Methodology

**Phase 1 — Embedding extraction**

- Models loaded via Hugging Face `AutoModel` with `output_hidden_states=True`
- Each question tokenized (max length 256), passed through the model
- **Mean pooling** over non-padding tokens at every layer (including the embedding layer)
- One `.npz` file saved per layer per model

**Phase 2 — Clustering**

- K-Means with `k = 4` on full-dataset embeddings
- Silhouette score measures unsupervised domain separability per layer

**Phase 3 — Linear probing**

- `StandardScaler` + multinomial logistic regression
- Trained on **1,600** samples (400 per class), evaluated on **400** held-out samples (100 per class)
- Fixed split defined in step 2; same split used for all layers and models

**Phase 4 — Statistical comparison**

- Layer-aligned paired tests between Qwen and Llama on accuracy, macro F1, and silhouette
- Reports Wilcoxon statistic, paired t-test, Cohen's d, and bootstrap confidence intervals

### Experimental Settings

| Setting | Value |
|---------|-------|
| Dataset size | 2,000 |
| Classes | 4 (Algebra, Arithmetic, Calculus, Probability) |
| Train / test per class | 400 / 100 |
| Pooling | Mean over tokens |
| Max sequence length | 256 |
| Batch size (extraction) | 8 |
| Clustering | K-Means, k = 4 |
| Probe | Logistic regression |
| Random seed | 42 |

---

## Project Structure

```text
llms_math_rep/
├── configs/
│   ├── models.yaml          # Model IDs and device preference
│   ├── data.yaml            # Dataset paths, labels, split settings
│   ├── paths.yaml           # Output directory paths
│   ├── clustering.yaml
│   └── probing.yaml
├── data/
│   └── raw/
│       └── math_dataset.json
├── notebooks/
│   └── colab_run.ipynb      # Google Colab reproduction notebook
├── scripts/                 # Pipeline entry points (00–10)
├── src/
│   ├── data/                # Validation, preprocessing, splitting
│   ├── embeddings/          # Model loading, extraction, pooling
│   ├── clustering/          # K-Means and silhouette
│   ├── probing/             # Layer-wise logistic regression
│   └── analysis/            # Figures, tables, significance tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Configuration

Key settings in `configs/models.yaml`:

```yaml
device: "cuda_if_available"

models:
  qwen:
    model_name: "Qwen/Qwen2.5-Math-7B-Instruct"
    type: "specialist"
  llama:
    model_name: "meta-llama/Llama-3.1-8B-Instruct"
    type: "generalist"
```

To run a single model during development, comment out one entry under `models`.

---

## Outputs

After a full run, `results/` contains:

```text
results/
├── embeddings/
│   ├── qwen/layer_XX.npz
│   └── llama/layer_XX.npz
├── clustering/silhouette_results.csv
├── probing/layerwise_results.csv
├── tables/final_summary.csv
├── figures/
│   ├── accuracy_by_layer.png
│   ├── silhouette_by_layer.png
│   ├── umap/
│   └── pca/
└── stats/
    ├── accuracy_significance.json
    ├── macro_f1_significance.json
    └── silhouette_significance.json
```

Each `.npz` embedding file stores: `embeddings`, `labels`, `sample_ids`, `questions`, `model_name`, `layer_idx`.

---

## Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload `notebooks/colab_run.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Run all cells (Hugging Face token required for Llama).
4. Download `results.zip` or copy `results/` to Google Drive.

Colab sessions may disconnect after several hours. Save embeddings to Drive after step 3 if the run is interrupted.

---

## Reproducibility

- Fixed random seed (`42`) for splitting, clustering, probing, and projections
- Identical train/test IDs across all layers and models
- Same preprocessing and pooling for both models
- Questions passed as raw text (no chat template) for a controlled comparison

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{yourpaper2026,
  title   = {Layer-wise Evolution of Mathematical Concept Representations: Comparing Specialist and Generalist Language Models},
  author  = {Author Names},
  year    = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
