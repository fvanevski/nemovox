"""Evaluation metrics (WER, CER, alignment coverage) with explicit Unicode normalization."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_SPECIAL_TAG_RE = re.compile(r"<[^<>\s]+>")


def normalize_latin_words(text: str) -> list[str]:
    """Normalize French/English text for WER: NFC, strip tags/punctuation, lowercase, split."""
    s = unicodedata.normalize("NFC", text or "")
    s = _SPECIAL_TAG_RE.sub(" ", s)
    s = s.replace("’", "'").replace("'", " ").replace("-", " ")
    s = _PUNCT_RE.sub(" ", s)
    return [w for w in s.lower().split() if w]


def normalize_zh_chars(text: str) -> list[str]:
    """Normalize Mandarin Chinese text for CER: NFKC, strip tags/punctuation/whitespace, char list."""
    s = unicodedata.normalize("NFKC", text or "")
    s = _SPECIAL_TAG_RE.sub("", s)
    s = _PUNCT_RE.sub("", s)
    return [ch for ch in s if not ch.isspace()]


def _levenshtein(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Return (edits, substitutions, deletions, insertions) between token sequences."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[(0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        ops[i][0] = (0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = j
        ops[0][j] = (0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = ops[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + 1
                del_ = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, del_, ins)
                dp[i][j] = best
                if best == sub:
                    s, d, k = ops[i - 1][j - 1]
                    ops[i][j] = (s + 1, d, k)
                elif best == del_:
                    s, d, k = ops[i - 1][j]
                    ops[i][j] = (s, d + 1, k)
                else:
                    s, d, k = ops[i][j - 1]
                    ops[i][j] = (s, d, k + 1)
    s, d, k = ops[n][m]
    return dp[n][m], s, d, k


def compute_wer(reference: str, hypothesis: str) -> dict[str, Any]:
    ref_tokens = normalize_latin_words(reference)
    hyp_tokens = normalize_latin_words(hypothesis)
    edits, sub, dele, ins = _levenshtein(ref_tokens, hyp_tokens)
    denom = max(len(ref_tokens), 1)
    return {
        "wer": round(edits / denom, 4),
        "ref_words": len(ref_tokens),
        "hyp_words": len(hyp_tokens),
        "edits": edits,
        "substitutions": sub,
        "deletions": dele,
        "insertions": ins,
    }


def compute_cer(reference: str, hypothesis: str) -> dict[str, Any]:
    ref_chars = normalize_zh_chars(reference)
    hyp_chars = normalize_zh_chars(hypothesis)
    edits, sub, dele, ins = _levenshtein(ref_chars, hyp_chars)
    denom = max(len(ref_chars), 1)
    return {
        "cer": round(edits / denom, 4),
        "ref_chars": len(ref_chars),
        "hyp_chars": len(hyp_chars),
        "edits": edits,
        "substitutions": sub,
        "deletions": dele,
        "insertions": ins,
    }
