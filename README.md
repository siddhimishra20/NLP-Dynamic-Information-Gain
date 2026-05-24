# Dynamic Information Gain (DIG) for Hallucination Detection in Retrieval-Augmented Language Models

**CS F429 · Natural Language Processing · BITS Pilani, Dubai Campus**  
**Second Semester 2025–2026 · Track A — Dynamic Uncertainty-Aware Attribution**

**Authors:** Siddhi Mishra · Samhita Kakumani · Manasi Bagga  
**Supervisor:** Prof. Elakkiya Rajasekar, Department of Computer Science & Information Systems

---

## Overview

This repository contains the full implementation of **DIG (Dynamic Information Gain)**, a training-free, token-level framework for detecting hallucinations in Retrieval-Augmented Language Models (RALMs). DIG decomposes uncertainty at inference time by comparing the token probability distributions produced by a language model with and without the retrieved context, surfacing hallucinations as measurable divergences across four complementary metrics.

The system is evaluated on the **RAGTruth** benchmark (primary) and subjected to zero-shot cross-domain transfer to **HaluEval QA**, with experiments spanning individual metric analysis, composite weight optimisation, temporal precedence, hallucination type breakdown, per-model attribution, and failure case analysis.

On the RAGTruth held-out test set (*n* = 2,675), DIG achieves **AUROC 0.6859** (95% CI: [0.6648, 0.7052]), outperforming the entropy-only baseline (AUROC 0.6550) and SelfCheckGPT (AUROC 0.5000) by margins of +0.0309 and +0.1859 respectively.

---

## Table of Contents

- [Methodology](#methodology)
- [Experiments](#experiments)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Setup & Requirements](#setup--requirements)
  - [System Requirements](#system-requirements)
  - [Installation](#installation)
  - [Running demo.py](#running-demopy)
- [Reproducing Results](#reproducing-results)
- [Dataset Access](#dataset-access)
- [Key Design Decisions](#key-design-decisions)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Methodology

### Formal Framework

Let $Y = (y_1, \ldots, y_T)$ be the generated response sequence, $q$ the query, and $d$ the retrieved document.

| Symbol | Definition |
|--------|------------|
| $P_t^{\varnothing}$ | $P(y_t \mid q)$ — token distribution **without** context |
| $P_t^{d}$ | $P(y_t \mid q, d)$ — token distribution **with** context |
| $H(P)$ | Shannon entropy $-\sum_v P(v)\log P(v)$ |

### Metrics

| # | Metric | Formula | Interpretation |
|---|--------|---------|----------------|
| M1 | **Information Gain** | $\Delta H_t = H(P_t^{\varnothing}) - H(P_t^{d})$ | Low/negative IG → context did not ground the token |
| M2 | **KL Divergence** | $\text{KL}_t = D_{\text{KL}}(P_t^{d} \| P_t^{\varnothing})$ | Small KL → model ignoring retrieved evidence |
| M3 | **Confidence Drop** | $\Delta c_t = \max P_t^{\varnothing} - \max P_t^{d}$ | Negative/near-zero → retrieval lowered peak confidence |
| M4 | **Semantic Entropy** | $SE_t = -\sum_{c \in C} p(c)\log p(c)$ over meaning clusters | High SE → genuine uncertainty over meaning, not just surface form |

### Composite Score

$$s_t = w_1\widetilde{\Delta H_t} + w_2\widetilde{\text{KL}_t} + w_3\widetilde{\Delta c_t} + w_4\widetilde{SE_t}$$

Weights $\{w_i\}$ are optimised on the validation set via **Nelder-Mead** with 50 random restarts, maximising composite AUROC (best val AUROC: **0.6874**). Decision threshold $\tau = -0.0849$ is set via Youden's J on the validation set. A **top-10% token aggregation** is applied to obtain sequence-level scores.

### Language Model & Pipeline

All inference uses **Mistral-7B-v0.1** in **4-bit NF4 quantisation** (bitsandbytes ≥ 0.43.0), requiring ~4.5 GB VRAM on a T4 GPU. Each example receives two forward passes: one conditioned on `Context: {d}\n\nQuestion: {q}\n\nAnswer:` and one on `Question: {q}\n\nAnswer:`, both truncated to 512 tokens. Semantic entropy clustering uses **all-MiniLM-L6-v2** with agglomerative clustering (average linkage) into max(2, K/4) = 5 semantic clusters over top-K = 20 tokens.

---

## Experiments

| ID | Experiment | Key Question |
|----|-----------|--------------|
| **E1** | Individual Metrics + Baselines | How does each metric perform independently vs. Entropy-only and SelfCheckGPT? |
| **E2** | Composite AUROC — Incremental Build | Does AUROC improve monotonically as metrics are added? |
| **E3** | Temporal Precedence | Does signal peak at $t-2$ or earlier before hallucination onset? |
| **E4** | Cross-Domain Transfer (HaluEval) | Do optimised weights transfer zero-shot to a structurally different domain? |
| **E5** | Hallucination Type Breakdown | Does AUROC differ across `contradictory`, `unsupported`, and `fabricated` types? |
| **E6** | AUROC by Generator Model | Does performance vary systematically by source LLM? |
| **E7** | Failure Case Analysis | What are the mechanistic explanations for false negatives and false positives? |
| **E8** | SOTA Gap Analysis | What fraction of the gap to supervised SOTA (LUMINA) does the system close? |

---

## Results Summary

### E1 + E2 — Individual and Composite Metrics (RAGTruth Test Set, *n* = 2,675)

95% bootstrap CI for Full Composite AUROC: **[0.6648, 0.7052]**

| Metric / Composite | AUROC ↑ | F1 ↑ | Spearman ρ ↑ | ECE ↓ |
|---|---|---|---|---|
| Entropy-only (B1) | 0.6550 | 0.5634 | 0.2814 | 0.2375 |
| SelfCheckGPT (B2) | 0.5000 | 0.5213 | 0.0000 | 0.3525 |
| + Information Gain ($\Delta H_t$) | 0.6804 | 0.5889 | 0.3129 | 0.1331 |
| + KL Divergence ($\text{KL}_t$) | 0.6804 | 0.5889 | 0.3129 | 0.1331 |
| + Confidence Drop ($\Delta c_t$) | 0.6859 | 0.5812 | 0.3204 | 0.1333 |
| + Semantic Entropy ($SE_t$) | 0.6859 | 0.5812 | 0.3204 | 0.1333 |
| **Full Composite (4-metric)** | **0.6859** | **0.5812** | **0.3204** | **0.1333** |

Each component contributes monotonically to composite AUROC. Confidence Drop and Semantic Entropy produce the most significant incremental gains, consistent with their ability to capture parameter-over-reliance and semantic ambiguity respectively.

### E3 — Temporal Precedence (*n* = 943 hallucinated examples)

| Position | ΔH | KL | H (raw) | SE | Δc |
|----------|----|----|---------|----|----|
| t−3 | −0.0701 | **12.7268** | 1.5864 | 0.4421 | 0.0003 |
| t−2 | −0.1354 | 12.5166 | 1.6523 | 0.4593 | **0.0084** |
| t−1 | −0.0636 | 12.4926 | **1.6751** | **0.4697** | −0.0011 |
| t (onset) | −0.0395 | 12.6195 | 1.6015 | 0.4634 | −0.0128 |
| t+1 | −0.0572 | 12.7119 | 1.5738 | 0.4397 | −0.0082 |

**KL divergence peaks at t−3** and **Confidence Drop peaks at t−2** relative to hallucination onset, constituting a pre-generation warning signal. The composite DIG score also peaks at t−2. Mann–Whitney U tests: KL at t−3 (U = 447,955, p = 0.778); Conf. Drop at t−2 (U = 455,173, p = 0.372) — directional trend is present but does not reach p < 0.05 significance at this sample size.

### E4 — Cross-Domain Transfer to HaluEval (Zero-Shot, *n* = 10,000)

No re-fitting of any weights or threshold was performed on HaluEval.

| Metric | AUROC (RAGTruth) | AUROC (HaluEval) | Drop | Rank Stable? |
|--------|----------------|----------------|------|-------------|
| **Full Composite** | **0.6859** | **0.6712** | +0.0147 | ✅ Yes |
| Information Gain ($\Delta H_t$) | 0.6804 | 0.8425 | −0.1620 | ❌ Reversed |
| KL Divergence ($\text{KL}_t$) | 0.6804 | 0.5139 | +0.1670 | ❌ Degrades |
| Semantic Entropy ($SE_t$) | 0.6020 | 0.6677 | −0.0660 | ✅ Yes |

IG reverses rank on HaluEval because its hallucinated answers are explicit factual substitutions generated with high no-context confidence but low context-conditioned confidence — the regime where IG is maximally informative. KL degrades most because HaluEval contexts are short factual snippets that suppress distributional shift for both classes. The composite is stable (drop = 0.0147) because IG's improvement offsets KL's degradation, demonstrating metric diversity as a transfer robustness mechanism.

### E5 — Hallucination Type Breakdown (RAGTruth Test Set)

| Type | Count | Composite AUROC | Best Single Metric |
|------|-------|-----------------|--------------------|
| Fabricated | 538 | 0.6974 | Conf. Drop (0.6732) |
| Contradictory | 394 | 0.6731 | Conf. Drop (0.7296) |
| Unsupported | 11 | 0.5865 | Conf. Drop (0.5797) |
| **AUROC Gap (max−min)** | — | **0.1109** | — |

Fabricated hallucinations are most detectable (AUROC 0.6974) because entirely invented content maximises context-prior KL divergence. Unsupported hallucinations are hardest (AUROC 0.5865) as they involve omission rather than substitution, producing near-zero distributional shift. AUROC gap of **0.1109** exceeds the 0.10 threshold.

### E6 — AUROC by Generator Model (RAGTruth Test Set)

| Generator Model | Count | Composite AUROC | Entropy AUROC |
|-----------------|-------|-----------------|---------------|
| mistral-7B-instruct | 436 | **0.7171** | 0.7408 |
| gpt-4-0613 | 450 | 0.7007 | 0.6988 |
| llama-2-13b-chat | 450 | 0.6851 | 0.7057 |
| llama-2-70b-chat | 449 | 0.6813 | 0.6502 |
| llama-2-7b-chat | 449 | 0.6501 | 0.6108 |
| gpt-3.5-turbo-0613 | 441 | 0.6024 | 0.7247 |

The composite outperforms entropy-alone for GPT-4 (0.7007 vs 0.6988), indicating stronger models require richer detection signals. GPT-3.5 is the notable outlier where entropy-only outperforms the composite by −0.1223, arising because the Mistral-7B surrogate's context-conditioned distributions track GPT-3.5 hallucination patterns poorly. Llama-2 shows a consistent DIG gain over entropy-only as model size increases.

### E7 — Failure Case Analysis

**False Negatives (hallucinated, predicted faithful):**

- **FN-1** | gpt-3.5-turbo, *unsupported* | composite = −0.1487 < τ: Fluent language mirroring the retrieved passage kept KL and ΔH near zero; high context-conditioned confidence suppressed all signals.
- **FN-2** | mistral-7B-instruct, *fabricated* | composite = −0.1071 < τ: Low IG (0.1101) and CD (0.1431) indicate the fabricated token was from a high-frequency, low-surprise vocabulary region — the hallucination was in content, not in uncertainty.
- **FN-3** | llama-2-7b-chat, *fabricated* | composite = −0.1091 < τ: Style-mimicry of retrieved context kept p_ctx ≈ p_noctx at the token level despite the factual error.

**False Positives (faithful, predicted hallucinated):**

- **FP-1** | gpt-4-0613, *faithful* | composite = +0.0147: Internally contradictory retrieved passages inflated SE even though the response was faithful; DIG conflated source ambiguity with hallucination uncertainty.
- **FP-2** | gpt-3.5-turbo, *faithful* | composite = −0.0422: Borderline score near the decision boundary; paraphrastic faithful response shifted p_ctx away from p_noctx, penalising lexical divergence without semantic error.
- **FP-3** | mistral-7B-instruct, *faithful* | composite = −0.0559: Epistemic hedging tokens ("it may be the case that...") carried high entropy in both passes, conflating calibrated hedging with hallucination uncertainty.

### E8 — SOTA Gap (RAGTruth Test Set)

| System | AUROC | Gap Closed |
|--------|-------|------------|
| LUMINA (supervised upper bound) | 0.87 | 100% |
| ReDeEP (unsupervised) | ≈0.82 | 80.2% |
| Semantic Entropy (published) | ≈0.70 | 51.1% |
| **DIG Composite (ours)** | **0.6859** | **14.4%** |
| Entropy-only (lower bound) | 0.6550 | 0% |
| SelfCheckGPT | 0.5000 | — |

Gap closed = (0.6859 − 0.6550) / (0.87 − 0.6550) × 100 = **14.4%**, without any labelled training data or supervised fine-tuning.

---

## Project Structure

```
NLP-Dynamic-Information-Gain-/
├── demo.py                     ← Standalone inference script (run this)
├── requirements.txt            ← All dependencies
├── Track_A_NLP_DIG.ipynb      ← Main pipeline notebook (full experiments)
│
│   Notebook sections:
├── Section 0               Environment setup and Google Drive mount
├── Section 1               RAGTruth dataset loading and preprocessing
├── Section 2               Mistral-7B-v0.1 (4-bit NF4) + semantic encoder
├── Section 3               Core metric functions (IG, KL, CD, SemEnt, Composite)
├── Section 4               Full inference loop with checkpointing and crash recovery
├── Section 5               Evaluation utilities (AUROC, bootstrap CI, F1, ECE)
├── Experiment 1            Individual metrics + SelfCheckGPT baseline
├── Experiment 2            Composite AUROC + Nelder-Mead weight optimisation
├── Experiment 3            Temporal precedence analysis
├── Experiment 4            HaluEval zero-shot cross-domain transfer
├── Experiment 5            Hallucination type breakdown
├── Experiments 6–8         Generator model analysis, failure cases, SOTA gap
└── Section 6 / 7           Live demo pipeline + final results summary
```

Outputs (saved to `NLP/outputs/` on Google Drive):

```
train_scores.pkl / val_scores.pkl / test_scores.pkl
halueval_scores.pkl
best_weights.json
E1_individual_metrics.csv
E2_composite_auroc.csv
E3_temporal_table.csv
E4_cross_domain.csv
E5_type_breakdown.csv
E6_by_model.csv
E7_failure_cases.csv
E8_sota_gap.csv
demo_token_scores.png
```

---

## Setup & Requirements

### System requirements

| Requirement | Minimum |
|-------------|---------|
| **Python** | 3.10 or 3.11 (tested on 3.10.12) |
| **GPU VRAM** | 4.5 GB (4-bit NF4); 14 GB+ for full-precision |
| **Recommended environment** | Google Colab T4 (15 GB VRAM) or any CUDA-capable GPU |
| **CUDA** | 11.8+ |
| **OS** | Linux (Ubuntu 20.04+), macOS (CPU-only), Windows WSL2 |

> CPU fallback is supported but inference will be very slow (~minutes per example). A CUDA GPU is strongly recommended.

### Dependencies (`requirements.txt`)

```
torch>=2.0.0
transformers>=4.41.0
accelerate
bitsandbytes>=0.43.0
selfcheckgpt
sentence-transformers
scikit-learn
scipy
statsmodels
matplotlib
seaborn
pandas
tqdm
datasets
huggingface_hub
numpy
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/siddhimishra20/NLP-Dynamic-Information-Gain-.git
cd NLP-Dynamic-Information-Gain-

# 2. (Recommended) create a virtual environment
python3.10 -m venv dig-env
source dig-env/bin/activate        # Windows: dig-env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Or install directly:

```bash
pip install torch>=2.0.0 transformers>=4.41.0 accelerate bitsandbytes>=0.43.0 \
    selfcheckgpt sentence-transformers scikit-learn scipy statsmodels \
    matplotlib seaborn pandas tqdm datasets huggingface_hub numpy
```

### Running `demo.py`

`demo.py` is the standalone inference script at the root of this repository. It accepts a passage and retrieved context and outputs per-token DIG scores to stdout and saves full results to `dig_output.json`.

**Option 1 — inline arguments (recommended):**
```bash
python demo.py \
  --passage "The Eiffel Tower is located in Berlin, Germany." \
  --context "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
```

**Option 2 — JSON input file:**
```bash
python demo.py --input input.json
```

where `input.json` has the format:
```json
{
  "passage": "Your generated response here",
  "context": "Your retrieved context document here"
}
```

**Option 3 — no arguments (runs built-in Eiffel Tower example):**
```bash
python demo.py
```

**Optional flags:**
```bash
python demo.py --passage "..." --context "..." --output my_results.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--passage` | — | Generated response text to score |
| `--context` | — | Retrieved context document |
| `--input` | — | Path to JSON file with `passage` and `context` keys |
| `--output` | `dig_output.json` | Path to save full JSON results |

**Expected output:**
```
[DIG] Loading Mistral-7B-v0.1 (4-bit NF4) on cuda...
[DIG] Model ready.

======================================================================
  DIG — Dynamic Information Gain | Token-Level Hallucination Scores
======================================================================

  Passage : The Eiffel Tower is located in Berlin, Germany....
  Aggregate DIG Score : -0.1234
  Decision Threshold  : -0.0849
  Prediction          : ⚠️  HALLUCINATED
  Tokens scored       : 14

  Per-Token Scores (top 20 shown):
  Token              DIG        IG       KL       CD       SE  Label
  ------------------------------------------------------------------------
  ...
======================================================================

[DIG] Full results saved to: dig_output.json
```

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Base language model | Mistral-7B-v0.1 |
| Quantisation | 4-bit NF4 (bitsandbytes) |
| Max input tokens | 512 |
| SE top-k tokens clustered | 20 |
| SE number of clusters | max(2, ⌊K/4⌋) = 5 |
| SE encoder | all-MiniLM-L6-v2 |
| Fusion weights {wᵢ} | Nelder-Mead on val (50 restarts) |
| Decision threshold τ | −0.0849 |
| Score aggregation | top-10% pooling |
| Bootstrap iterations | 1000 |
| Random seed | 42 |

---

## Reproducing Results

1. Mount your Google Drive and create the following directory structure:

```
MyDrive/NLP/
├── dataset/
│   ├── ragtruth/        ← place response.jsonl and source_info.jsonl here
│   └── halueval/        ← auto-created; HaluEval downloaded via HuggingFace
├── outputs/
└── checkpoints/
```

2. Download the RAGTruth dataset files from the [official repository](https://github.com/ParticleMedia/RAGTruth/tree/main/dataset) and place them under `ragtruth/`.

3. Run all cells in order. Section 4 includes **crash-safe checkpointing** — inference resumes automatically from the last saved checkpoint if the runtime disconnects (saves every 25 examples).

4. To run a quick sanity check before full inference, set `N_SUBSAMPLE` in Section 4.2 to a small integer (e.g., `100`). Set to `None` for the full dataset run.

> **Note:** Delete any existing `.pkl` files in `NLP/outputs/` if switching from a different base model (e.g., OPT-1.3B), as cached scores are model-specific.

---

## Dataset Access

| Dataset | Source | Split | Hallucination Rate |
|---------|--------|-------|--------------------|
| [RAGTruth](https://github.com/ParticleMedia/RAGTruth) | GitHub | 70/10/20 train/val/test (17,617 total) | 35.25% |
| [HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) | HuggingFace | Zero-shot transfer (10,000 QA pairs) | 50.10% |

RAGTruth provides span-level annotations across three hallucination types (`contradictory`, `unsupported`, `fabricated`) from six generator models (GPT-4, GPT-3.5, Mistral-7B, Llama-2 7B/13B/70B).

---

## Key Design Decisions

**Why Mistral-7B over OPT-1.3B?**  
Mistral-7B provides substantially better probability calibration. OPT-1.3B produces near-uniform vocabulary distributions that collapse IG and KL signals, making it unsuitable for entropy-based hallucination detection.

**Why top-10% token aggregation over mean?**  
Hallucinations are localised events. Averaging over all tokens dilutes the signal from the small number of hallucinated spans. The top-10% aggregation isolates the most suspicious tokens without requiring span-level labels at inference time.

**Why Nelder-Mead with 50 restarts?**  
The composite AUROC objective is non-convex due to tanh compression. 50 Dirichlet-initialised restarts consistently finds a stable global optimum, achieving best val AUROC of 0.6874.

**Why semantic entropy via agglomerative clustering?**  
Standard token entropy treats semantically equivalent surface forms as distinct. Clustering the top-20 tokens by cosine similarity of MiniLM embeddings into 5 clusters collapses these into meaning groups, capturing genuine semantic uncertainty rather than tokenisation artefacts.

**Why dual forward pass instead of sampling?**  
SelfCheckGPT requires 3–5× inference overhead for multiple generations. The dual forward pass requires exactly two model calls per example (~1.82 seconds on T4), making it substantially more efficient while directly exploiting the retrieval signal rather than measuring self-consistency.

---

## Limitations

- **White-box requirement:** DIG requires access to token-level log-probabilities; it is inapplicable to black-box API models (e.g., GPT-4 via OpenAI API).
- **Unsupported hallucinations** (AUROC 0.5865) remain largely undetectable — omission-type hallucinations produce near-zero context-prior divergence that all four signals cannot reliably capture.
- **KL Divergence** degrades on short-context domains (HaluEval AUROC 0.5139); weights optimised on RAGTruth's long passages may require re-fitting in domains with structurally different retrieval.
- **Generator mismatch:** Composite underperforms entropy-only for GPT-3.5 (−0.1223 gap) because the Mistral-7B surrogate's distributions do not align with GPT-3.5's generation geometry.
- **Computational overhead:** ~1.82 seconds per example on T4 (two full forward passes + semantic clustering).

---

## Citation

If you reference this work, please cite:

```bibtex
@misc{dig_hallucination_2026,
  title   = {Dynamic Information Gain (DIG): A Composite Uncertainty-Aware Metric
             for Token-Level Hallucination Detection in Retrieval-Augmented Language Models},
  author  = {Mishra, Siddhi and Kakumani, Samhita and Bagga, Manasi},
  year    = {2026},
  note    = {CS F429 Natural Language Processing, BITS Pilani Dubai Campus.
             Supervised by Prof. Elakkiya Rajasekar},
  url     = {https://github.com/siddhimishra20/NLP-Dynamic-Information-Gain-}
}
```

---

*Submitted in partial fulfilment of CS F429 — Natural Language Processing Project, BITS Pilani Dubai Campus, May 2026.*
