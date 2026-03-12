"""
Evaluation metrics for OCR: CTC decoding, CER, WER.
"""

from __future__ import annotations
import numpy as np
import torch


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    idx2char: dict,
    blank: int = 0,
) -> list[str]:
    """
    Greedy best-path CTC decode.

    Parameters
    ----------
    log_probs : (B, T, V) log-softmax outputs
    idx2char  : index → character mapping
    blank     : blank token index

    Returns
    -------
    list[str]  — one decoded string per sample
    """
    preds = log_probs.argmax(-1).cpu().numpy()   # (B, T)
    results = []
    for seq in preds:
        chars, prev = [], None
        for idx in seq:
            if idx != blank and idx != prev:
                chars.append(idx2char.get(int(idx), ""))
            prev = idx
        results.append("".join(chars))
    return results


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(preds: list[str], targets: list[str]) -> float:
    """Character Error Rate (lower is better)."""
    total_dist = sum(_edit_distance(p, t) for p, t in zip(preds, targets))
    total_len  = sum(len(t) for t in targets)
    return total_dist / max(total_len, 1)


def compute_wer(preds: list[str], targets: list[str]) -> float:
    """Word Error Rate (lower is better) — splits on whitespace."""
    total_dist = sum(
        _edit_distance(p.split(), t.split()) for p, t in zip(preds, targets)
    )
    total_len = sum(len(t.split()) for t in targets)
    return total_dist / max(total_len, 1)
