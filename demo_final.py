#!/usr/bin/env python3
"""
demo.py — DIG: Dynamic Information Gain for Token-Level Hallucination Detection
CS F429 NLP Project | Track A | BITS Pilani Dubai Campus

Usage:
    python demo.py --passage "Your passage text here" --context "Retrieved context here"
    python demo.py --input input.json
    python demo.py  (uses built-in example)

input.json format: 

json
{
  "passage": "Your generated response here",
  "context": "Your retrieved context document here"
}

Output:
    Per-token DIG composite scores + hallucination labels printed to stdout.
    JSON output saved to dig_output.json
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

# ─── Constants (from trained run) ────────────────────────────────────────────
MODEL_NAME       = "mistralai/Mistral-7B-v0.1"
MAX_INPUT_TOKENS = 512
SEED             = 42
DECISION_TAU     = -0.0849          # Youden's J on validation set
TOP_K_SE         = 20               # top-k tokens for semantic entropy
N_CLUSTERS_SE    = 5                # semantic entropy clusters
W1, W2, W3, W4  = 0.25, 0.25, 0.25, 0.25   # fusion weights (Nelder-Mead)

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load model (lazy, once) ─────────────────────────────────────────────────
_model     = None
_tokenizer = None
_sem_enc   = None

def _load_model():
    global _model, _tokenizer, _sem_enc
    if _model is not None:
        return

    print(f"[DIG] Loading Mistral-7B-v0.1 (4-bit NF4) on {DEVICE}...")
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                               BitsAndBytesConfig)

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "left"

    if DEVICE == "cuda":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb, device_map="auto",
            trust_remote_code=False,
        )
    else:
        # CPU fallback (slow but functional)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="cpu", torch_dtype=torch.float32,
            trust_remote_code=False,
        )

    _model.eval()

    from sentence_transformers import SentenceTransformer
    _sem_enc = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    print("[DIG] Model ready.")


# ─── Core: dual forward pass ─────────────────────────────────────────────────
def _get_log_probs(text: str) -> torch.Tensor:
    enc = _tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=MAX_INPUT_TOKENS, padding=False,
    ).to(DEVICE)
    with torch.no_grad():
        out = _model(**enc)
    return torch.log_softmax(out.logits[0].float(), dim=-1)  # [T, V]


def _semantic_entropy(p_ctx: torch.Tensor) -> np.ndarray:
    """Compute semantic entropy over top-K tokens at each position."""
    from sklearn.cluster import AgglomerativeClustering

    top_k_ids = torch.argsort(p_ctx, dim=-1, descending=True)[:, :TOP_K_SE]  # [R, K]
    se_scores = []
    for r in range(p_ctx.shape[0]):
        tokens = [_tokenizer.decode([tid.item()]).strip() for tid in top_k_ids[r]]
        if not any(t for t in tokens):
            se_scores.append(0.0)
            continue
        embs = _sem_enc.encode(tokens, show_progress_bar=False,
                               convert_to_numpy=True)
        try:
            clust = AgglomerativeClustering(
                n_clusters=N_CLUSTERS_SE, metric="cosine", linkage="average"
            )
            labels = clust.fit_predict(embs)
            probs_c = np.bincount(labels, minlength=N_CLUSTERS_SE) / TOP_K_SE
            se = float(-np.sum(probs_c * np.log(probs_c + 1e-12)))
        except Exception:
            se = 0.0
        se_scores.append(se)
    return np.array(se_scores)


def score_passage(passage: str, context: str) -> dict:
    """
    Main entry point.

    Parameters
    ----------
    passage : str   The generated response / passage to score.
    context : str   Retrieved context document.

    Returns
    -------
    dict with keys:
        tokens          - list of decoded token strings
        ig              - Information Gain per token  (higher = more informative)
        kl              - KL Divergence per token
        conf_drop       - Confidence Drop per token
        sem_entropy     - Semantic Entropy per token
        dig_composite   - DIG composite score per token
        labels          - 1 = hallucinated, 0 = faithful, per token
        aggregate_score - single scalar DIG score for the whole passage
        is_hallucinated - overall binary prediction
    """
    _load_model()

    # Build prompts
    prompt_ctx    = f"Context: {context}\n\nQuestion: {passage}\n\nAnswer:"
    prompt_no_ctx = f"Question: {passage}\n\nAnswer:"

    lp_ctx    = _get_log_probs(prompt_ctx)
    lp_noctx  = _get_log_probs(prompt_no_ctx)

    # Align to response tokens
    resp_enc = _tokenizer(
        passage, return_tensors="pt", truncation=True,
        max_length=MAX_INPUT_TOKENS, add_special_tokens=False,
    )
    R = resp_enc["input_ids"].shape[1]
    if R == 0:
        return {"error": "Empty passage after tokenisation."}

    def _extract(lp):
        T = lp.shape[0]
        start = max(0, T - R - 1)
        return lp[start: T - 1]

    lp_c  = _extract(lp_ctx).cpu()
    lp_nc = _extract(lp_noctx).cpu()
    R_use = min(lp_c.shape[0], lp_nc.shape[0])
    if R_use == 0:
        return {"error": "Could not align response tokens."}

    lp_c  = lp_c[:R_use]
    lp_nc = lp_nc[:R_use]
    p_c   = torch.exp(lp_c).float()
    p_nc  = torch.exp(lp_nc).float()

    # ── Metrics ────────────────────────────────────────────────────────────
    def entropy(p):
        return (-torch.sum(p * torch.log(p + 1e-12), dim=-1)).numpy()

    H_ctx   = entropy(p_c)    # [R_use]
    H_noctx = entropy(p_nc)

    ig         = H_noctx - H_ctx                              # positive = context helps
    kl         = torch.sum(p_c * (lp_c - lp_nc), dim=-1).numpy()
    conf_drop  = (p_nc.max(dim=-1).values - p_c.max(dim=-1).values).numpy()
    sem_ent    = _semantic_entropy(p_c)

    # ── Min-max normalisation ───────────────────────────────────────────────
    def _norm(x):
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-8)

    ig_n   = _norm(-ig)        # negate: low IG = hallucination risk
    kl_n   = _norm(kl)
    cd_n   = _norm(conf_drop)
    se_n   = _norm(sem_ent)

    composite = W1 * ig_n + W2 * kl_n + W3 * cd_n + W4 * se_n

    # ── Aggregation (top-10% pooling) ───────────────────────────────────────
    k = max(1, int(0.10 * R_use))
    agg_score = float(np.sort(composite)[-k:].mean())

    # ── Token labels ─────────────────────────────────────────────────────────
    token_labels = (composite < DECISION_TAU).astype(int)

    # ── Decode tokens ────────────────────────────────────────────────────────
    ids = resp_enc["input_ids"][0][:R_use].tolist()
    tokens = [_tokenizer.decode([tid]) for tid in ids]

    return {
        "tokens":          tokens,
        "ig":              ig.tolist(),
        "kl":              kl.tolist(),
        "conf_drop":       conf_drop.tolist(),
        "sem_entropy":     sem_ent.tolist(),
        "dig_composite":   composite.tolist(),
        "labels":          token_labels.tolist(),
        "aggregate_score": agg_score,
        "is_hallucinated": bool(agg_score < DECISION_TAU),
        "decision_tau":    DECISION_TAU,
        "n_tokens_scored": R_use,
    }


# ─── Pretty print ─────────────────────────────────────────────────────────────
def _print_results(result: dict, passage: str):
    if "error" in result:
        print(f"\n[DIG] ERROR: {result['error']}")
        return

    print("\n" + "="*70)
    print("  DIG — Dynamic Information Gain | Token-Level Hallucination Scores")
    print("="*70)
    print(f"\n  Passage : {passage[:80]}{'...' if len(passage)>80 else ''}")
    print(f"\n  Aggregate DIG Score : {result['aggregate_score']:+.4f}")
    print(f"  Decision Threshold  : {result['decision_tau']:+.4f}")
    print(f"  Prediction          : {'⚠️  HALLUCINATED' if result['is_hallucinated'] else '✅  FAITHFUL'}")
    print(f"  Tokens scored       : {result['n_tokens_scored']}")

    print("\n  Per-Token Scores (top 20 shown):")
    print(f"  {'Token':<18} {'DIG':>8} {'IG':>8} {'KL':>8} {'CD':>8} {'SE':>8}  Label")
    print("  " + "-"*72)

    tokens   = result["tokens"]
    composite = result["dig_composite"]
    ig        = result["ig"]
    kl        = result["kl"]
    cd        = result["conf_drop"]
    se        = result["sem_entropy"]
    labels    = result["labels"]

    for i in range(min(20, len(tokens))):
        tok   = repr(tokens[i])[:16]
        flag  = "⚠️ HALL" if labels[i] else "  ok  "
        print(f"  {tok:<18} {composite[i]:+8.4f} {ig[i]:+8.4f} {kl[i]:8.4f}"
              f" {cd[i]:+8.4f} {se[i]:8.4f}  {flag}")

    if len(tokens) > 20:
        print(f"  ... ({len(tokens)-20} more tokens not shown)")
    print("="*70)


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="DIG: Token-Level Hallucination Detection"
    )
    parser.add_argument("--passage", type=str, default=None,
                        help="Generated passage / response text to score")
    parser.add_argument("--context", type=str, default=None,
                        help="Retrieved context document")
    parser.add_argument("--input",   type=str, default=None,
                        help="JSON file with keys 'passage' and 'context'")
    parser.add_argument("--output",  type=str, default="dig_output.json",
                        help="Output JSON file path (default: dig_output.json)")
    args = parser.parse_args()

    # ── Resolve inputs ────────────────────────────────────────────────────────
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        passage = data.get("passage", data.get("response", ""))
        context = data.get("context", data.get("source_info", ""))
    elif args.passage and args.context:
        passage = args.passage
        context = args.context
    else:
        # Built-in example
        print("[DIG] No input provided — running built-in example.")
        passage = (
            "The Eiffel Tower was built in 1887 and is located in Berlin, "
            "Germany. It stands 330 metres tall and was designed by Gustave Eiffel."
        )
        context = (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
            "in Paris, France. It was constructed from 1887 to 1889 as the centerpiece "
            "of the 1889 World's Fair. The tower is 330 metres tall and named after "
            "engineer Gustave Eiffel, whose company designed and built the structure."
        )

    # ── Score ─────────────────────────────────────────────────────────────────
    result = score_passage(passage, context)
    _print_results(result, passage)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[DIG] Full results saved to: {args.output}")


if __name__ == "__main__":
    main()
