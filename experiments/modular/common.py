"""Shared corpus paths, normalization, alignment auditing, and diarization metrics."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path("/workspace/nvidia-transcribe")
PYANNOTE_ROOT = Path("/workspace/pyannote")
MODELS_ROOT = Path("/workspace/models")

NEMOTRON_35_DIR = MODELS_ROOT / "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
NEMOTRON_DIAR_DIR = MODELS_ROOT / "Nemotron-3-Diarization"
QWEN_ASR_DIR = MODELS_ROOT / "Qwen3-ASR-1.7B"
QWEN_ALIGNER_DIR = MODELS_ROOT / "Qwen3-ForcedAligner-0.6B"

MODULAR_OUT = WORKSPACE_ROOT / "outputs/modular-eval"
MODULAR_EVAL_DIR = WORKSPACE_ROOT / "evaluations/modular-bakeoff"

FRENCH_FULL_EPISODE_WEBM = Path(
    "/opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm"
)
FRENCH_FULL_EPISODE_WAV = WORKSPACE_ROOT / "outputs/audio/fr/fr_full_episode.wav"


_LANG_MAP = {
    "fr": "fr",
    "french": "fr",
    "français": "fr",
    "francais": "fr",
    "fr-fr": "fr",
    "zh": "zh",
    "zh-cn": "zh",
    "zh_cn": "zh",
    "cmn": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "mandarin chinese": "zh",
    "en": "en",
    "english": "en",
    "en-us": "en",
    "en-gb": "en",
    "yue": "yue",
    "cantonese": "yue",
    "auto": "auto",
}


def normalize_language(label: str | None) -> str:
    """Normalize ASR/aligner language names into ISO-style codes without silent defaults."""
    if label is None:
        return "unknown"
    cleaned = unicodedata.normalize("NFKC", str(label)).strip().lower()
    if not cleaned:
        return "unknown"
    # Handle Qwen output like "French" or comma-separated multi-language tags
    if cleaned in _LANG_MAP:
        return _LANG_MAP[cleaned]
    primary = cleaned.split(",")[0].strip()
    if primary in _LANG_MAP:
        return _LANG_MAP[primary]
    return primary


def preserve_unicode_text(text: str | None) -> str:
    """Preserve source-language French/CJK text verbatim without translation or lossy ASCII folding."""
    if text is None:
        return ""
    return str(text)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _levenshtein_counts(ref_seq: list[str], hyp_seq: list[str]) -> dict[str, int]:
    """Compute substitutions, deletions, and insertions via dynamic programming."""
    n, m = len(ref_seq), len(hyp_seq)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"
    for i in range(1, n + 1):
        ri = ref_seq[i - 1]
        for j in range(1, m + 1):
            hj = hyp_seq[j - 1]
            if ri == hj:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                if best == sub:
                    op[i][j] = "S"
                elif best == dele:
                    op[i][j] = "D"
                else:
                    op[i][j] = "I"
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        cur = op[i][j]
        if cur == "M":
            i -= 1
            j -= 1
        elif cur == "S":
            subs += 1
            i -= 1
            j -= 1
        elif cur == "D":
            dels += 1
            i -= 1
        elif cur == "I":
            ins += 1
            j -= 1
        else:
            break
    return {"edits": dp[n][m], "substitutions": subs, "deletions": dels, "insertions": ins}


def normalize_for_wer(text: str, language: str = "fr") -> list[str]:
    cleaned = unicodedata.normalize("NFKC", text).lower()
    cleaned = cleaned.replace("’", "'").replace("‘", "'").replace("'", " ")
    # Separate punctuation
    cleaned = re.sub(r"[^\w\s]", " ", cleaned, flags=re.UNICODE)
    tokens = [t for t in cleaned.split() if t]
    return tokens


def normalize_for_cer(text: str) -> list[str]:
    cleaned = unicodedata.normalize("NFKC", text).lower()
    # Keep letters and digits across Latin and CJK, drop whitespace and punctuation
    chars = [ch for ch in cleaned if not ch.isspace() and unicodedata.category(ch)[0] not in ("P", "S")]
    return chars


def compute_wer_detailed(ref_text: str, hyp_text: str, language: str = "fr") -> dict[str, Any]:
    ref_tokens = normalize_for_wer(ref_text, language=language)
    hyp_tokens = normalize_for_wer(hyp_text, language=language)
    counts = _levenshtein_counts(ref_tokens, hyp_tokens)
    ref_len = len(ref_tokens)
    return {
        "wer": round(counts["edits"] / ref_len, 4) if ref_len else (0.0 if not hyp_tokens else 1.0),
        "ref_words": ref_len,
        "hyp_words": len(hyp_tokens),
        **counts,
    }


def compute_cer_detailed(ref_text: str, hyp_text: str) -> dict[str, Any]:
    ref_chars = normalize_for_cer(ref_text)
    hyp_chars = normalize_for_cer(hyp_text)
    counts = _levenshtein_counts(ref_chars, hyp_chars)
    ref_len = len(ref_chars)
    return {
        "cer": round(counts["edits"] / ref_len, 4) if ref_len else (0.0 if not hyp_chars else 1.0),
        "ref_chars": ref_len,
        "hyp_chars": len(hyp_chars),
        **counts,
    }


def _strip_lexical_chars(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKC", text)
        if not ch.isspace() and unicodedata.category(ch)[0] not in ("P", "S")
    )


def validate_alignment_items(
    items: list[dict[str, Any]],
    *,
    expected_text: str,
    audio_duration_s: float,
    source_offset_s: float = 0.0,
) -> dict[str, Any]:
    """Audit forced-alignment units for Issue #197 zero-durations, inversions, bounds, and omissions.

    Never clamps, invents, or silently drops invalid units.
    """
    valid_items = 0
    zero_duration_items: list[dict[str, Any]] = []
    negative_or_inverted_items: list[dict[str, Any]] = []
    out_of_bounds_items: list[dict[str, Any]] = []
    past_duration_items: list[dict[str, Any]] = []
    non_monotonic_items: list[dict[str, Any]] = []
    missing_timing_items: list[dict[str, Any]] = []
    repeated_identical_spans: list[dict[str, Any]] = []

    prev_start: float | None = None
    prev_span: tuple[float, float] | None = None

    rel_tol = 1e-3
    max_abs_end = source_offset_s + audio_duration_s + rel_tol

    for idx, item in enumerate(items):
        tok = str(item.get("text", ""))
        st = item.get("start")
        en = item.get("end")
        if st is None or en is None:
            missing_timing_items.append({"index": idx, "text": tok, "start": st, "end": en})
            continue
        st_f = float(st)
        en_f = float(en)
        dur = en_f - st_f
        item_ok = True

        if st_f < source_offset_s - rel_tol or en_f < source_offset_s - rel_tol or dur < 0:
            negative_or_inverted_items.append({"index": idx, "text": tok, "start": st_f, "end": en_f})
            out_of_bounds_items.append({"index": idx, "text": tok, "start": st_f, "end": en_f})
            item_ok = False
        if en_f > max_abs_end or st_f > max_abs_end:
            past_duration_items.append(
                {"index": idx, "text": tok, "start": st_f, "end": en_f, "max_end": round(max_abs_end, 4)}
            )
            out_of_bounds_items.append({"index": idx, "text": tok, "start": st_f, "end": en_f})
            item_ok = False
        if abs(dur) <= 1e-6:
            zero_duration_items.append({"index": idx, "text": tok, "start": st_f, "end": en_f})
            item_ok = False
        if prev_start is not None and st_f + 1e-6 < prev_start:
            non_monotonic_items.append({"index": idx, "text": tok, "start": st_f, "prev_start": prev_start})
            item_ok = False
        if prev_span is not None and abs(st_f - prev_span[0]) <= 1e-6 and abs(en_f - prev_span[1]) <= 1e-6:
            repeated_identical_spans.append({"index": idx, "text": tok, "start": st_f, "end": en_f})
        prev_start = st_f
        prev_span = (st_f, en_f)
        if item_ok:
            valid_items += 1

    joined_aligned = _strip_lexical_chars("".join(str(x.get("text", "")) for x in items))
    expected_stripped = _strip_lexical_chars(expected_text)
    char_diff = _levenshtein_counts(list(expected_stripped), list(joined_aligned))

    return {
        "total_items": len(items),
        "valid_timed_items": valid_items,
        "valid_ratio": round(valid_items / len(items), 4) if items else 0.0,
        "zero_duration_count": len(zero_duration_items),
        "zero_duration_items": zero_duration_items[:20],
        "negative_or_inverted_count": len(negative_or_inverted_items),
        "negative_or_inverted_items": negative_or_inverted_items[:20],
        "out_of_bounds_count": len(out_of_bounds_items),
        "past_duration_count": len(past_duration_items),
        "past_duration_items": past_duration_items[:20],
        "non_monotonic_count": len(non_monotonic_items),
        "non_monotonic_items": non_monotonic_items[:20],
        "missing_timing_count": len(missing_timing_items),
        "repeated_identical_span_count": len(repeated_identical_spans),
        "repeated_identical_spans": repeated_identical_spans[:20],
        "expected_lexical_chars": len(expected_stripped),
        "aligned_lexical_chars": len(joined_aligned),
        "lexical_preservation_edits": char_diff["edits"],
        "lexical_missing_chars_approx": char_diff["deletions"],
        "lexical_extra_chars_approx": char_diff["insertions"],
        "text_preserved_exact": expected_stripped == joined_aligned,
    }


def offset_intervals(
    intervals: list[dict[str, Any]],
    source_offset_s: float,
    *,
    default_unknown_label: str = "unknown",
) -> list[dict[str, Any]]:
    """Apply recording offset without fabricating missing timestamps or collapsing unknown speakers."""
    out: list[dict[str, Any]] = []
    for iv in intervals:
        raw_spk = iv.get("speaker")
        if raw_spk is None or str(raw_spk).strip() == "":
            spk = default_unknown_label
        else:
            spk = str(raw_spk).strip()
        st = iv.get("start")
        en = iv.get("end")
        out.append(
            {
                "speaker": spk,
                "start": None if st is None else round(float(st) + source_offset_s, 4),
                "end": None if en is None else round(float(en) + source_offset_s, 4),
                **{k: v for k, v in iv.items() if k not in ("speaker", "start", "end")},
            }
        )
    return out


def compute_overlap_der(
    ref_intervals: list[tuple[float, float, str]],
    hyp_intervals: list[tuple[float, float, str]],
    *,
    total_duration_s: float | None = None,
    step_s: float = 0.01,
) -> dict[str, Any]:
    """Compute exact discrete-frame overlap-aware DER and JER via optimal bipartite mapping."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    max_t = max(
        [total_duration_s or 0.0]
        + [e for _, e, _ in ref_intervals]
        + [e for _, e, _ in hyp_intervals]
    )
    n_frames = max(1, int(round(max_t / step_s)))
    ref_spks = sorted({spk for _, _, spk in ref_intervals})
    hyp_spks = sorted({spk for _, _, spk in hyp_intervals})

    ref_mat = np.zeros((len(ref_spks), n_frames), dtype=bool)
    hyp_mat = np.zeros((len(hyp_spks), n_frames), dtype=bool)
    r_idx = {s: i for i, s in enumerate(ref_spks)}
    h_idx = {s: i for i, s in enumerate(hyp_spks)}

    for s, e, spk in ref_intervals:
        a = max(0, int(round(s / step_s)))
        b = min(n_frames, int(round(e / step_s)))
        if b > a:
            ref_mat[r_idx[spk], a:b] = True

    for s, e, spk in hyp_intervals:
        a = max(0, int(round(s / step_s)))
        b = min(n_frames, int(round(e / step_s)))
        if b > a:
            hyp_mat[h_idx[spk], a:b] = True

    overlap_frames = np.zeros((len(ref_spks), len(hyp_spks)), dtype=np.int64)
    if ref_spks and hyp_spks:
        for i in range(len(ref_spks)):
            for j in range(len(hyp_spks)):
                overlap_frames[i, j] = int(np.sum(ref_mat[i] & hyp_mat[j]))
        row_ind, col_ind = linear_sum_assignment(-overlap_frames)
        mapping = {hyp_spks[j]: ref_spks[i] for i, j in zip(row_ind, col_ind) if overlap_frames[i, j] > 0}
    else:
        mapping = {}

    ref_speech_total = int(ref_mat.sum())
    # Per-frame DER accounting: missed = max(0, r - h), fa = max(0, h - r), conf = matched_slots - correct
    ref_counts = ref_mat.sum(axis=0)
    hyp_counts = hyp_mat.sum(axis=0)

    # Map hypothesis frames to ref indices
    mapped_hyp_mat = np.zeros_like(ref_mat)
    for j, h_spk in enumerate(hyp_spks):
        if h_spk in mapping:
            mapped_hyp_mat[r_idx[mapping[h_spk]]] |= hyp_mat[j]

    correct = int((ref_mat & mapped_hyp_mat).sum())
    missed = int(np.maximum(0, ref_counts - hyp_counts).sum())
    false_alarm = int(np.maximum(0, hyp_counts - ref_counts).sum())
    matched_speaker_slots = int(np.minimum(ref_counts, hyp_counts).sum())
    confusion = max(0, matched_speaker_slots - correct)
    der = (missed + false_alarm + confusion) / ref_speech_total if ref_speech_total > 0 else 0.0

    # JER
    jer_terms = []
    for i, r_spk in enumerate(ref_spks):
        inv_map = [j for j, h_spk in enumerate(hyp_spks) if mapping.get(h_spk) == r_spk]
        r_active = ref_mat[i]
        if not inv_map:
            jer_terms.append(1.0)
        else:
            h_active = hyp_mat[inv_map[0]]
            union = int((r_active | h_active).sum())
            inter = int((r_active & h_active).sum())
            jer_terms.append(1.0 - (inter / union) if union > 0 else 0.0)
    jer = sum(jer_terms) / len(jer_terms) if jer_terms else 0.0

    # Frame-level overlap (>1 active speaker) metrics
    ref_ov = ref_counts > 1
    hyp_ov = hyp_counts > 1
    ov_tp = int((ref_ov & hyp_ov).sum())
    ov_fp = int((~ref_ov & hyp_ov).sum())
    ov_fn = int((ref_ov & ~hyp_ov).sum())
    ov_prec = ov_tp / (ov_tp + ov_fp) if (ov_tp + ov_fp) > 0 else (1.0 if ov_fn == 0 else 0.0)
    ov_rec = ov_tp / (ov_tp + ov_fn) if (ov_tp + ov_fn) > 0 else (1.0 if ov_fp == 0 else 0.0)
    ov_f1 = (2 * ov_prec * ov_rec / (ov_prec + ov_rec)) if (ov_prec + ov_rec) > 0 else 0.0

    return {
        "der": round(float(der), 4),
        "jer": round(float(jer), 4),
        "missed_s": round(missed * step_s, 3),
        "false_alarm_s": round(false_alarm * step_s, 3),
        "confusion_s": round(confusion * step_s, 3),
        "ref_speech_s": round(ref_speech_total * step_s, 3),
        "ref_speakers": len(ref_spks),
        "hyp_speakers": len(hyp_spks),
        "speaker_mapping": mapping,
        "overlap_ref_s": round(int(ref_ov.sum()) * step_s, 3),
        "overlap_hyp_s": round(int(hyp_ov.sum()) * step_s, 3),
        "overlap_precision": round(float(ov_prec), 4),
        "overlap_recall": round(float(ov_rec), 4),
        "overlap_f1": round(float(ov_f1), 4),
    }


def load_shared_corpus() -> dict[str, Any]:
    """Return authoritative shared audio fixtures and verified references without inventing labels."""
    zh_manifest = {
        item["clip_id"]: item
        for item in json.loads((WORKSPACE_ROOT / "outputs/audio/zh/zh_manifest.json").read_text(encoding="utf-8"))
    }
    fr_fleurs = {
        item["clip_id"]: item
        for item in json.loads((WORKSPACE_ROOT / "outputs/audio/overlap/fr_fleurs_manifest.json").read_text(encoding="utf-8"))
    }
    ov_manifest = json.loads((WORKSPACE_ROOT / "outputs/audio/overlap/overlap_manifest.json").read_text(encoding="utf-8"))
    overlap_eval = json.loads(
        (PYANNOTE_ROOT / ".superpowers/audits/subtitle-quality/overlap-recovery/benchmark.json").read_text(encoding="utf-8")
    )["cases"]

    return {
        "french_regions": {
            "F1": {
                "path": str(WORKSPACE_ROOT / "outputs/audio/fr/fr_0316_0355_child_voices.wav"),
                "offset_s": 196.0,
                "duration_s": 39.0,
                "purpose": "Similar child voices and previously merged speaker IDs (03:16-03:55)",
                "verified_ref": None,
            },
            "F2": {
                "path": str(WORKSPACE_ROOT / "outputs/audio/fr/fr_0422_0439_rapid_turns.wav"),
                "offset_s": 262.0,
                "duration_s": 17.0,
                "purpose": "Rapid speaker transitions (04:22-04:39)",
                "verified_ref": None,
            },
            "F3": {
                "path": str(WORKSPACE_ROOT / "outputs/audio/fr/fr_0539_0542_overlap_nonlex.wav"),
                "offset_s": 339.0,
                "duration_s": 3.0,
                "purpose": "Speech with nonlexical overlap (05:39-05:42)",
                "verified_ref": None,
            },
            "F4": {
                "path": str(WORKSPACE_ROOT / "outputs/audio/fr/fr_0550_0552_transition.wav"),
                "offset_s": 350.0,
                "duration_s": 2.0,
                "purpose": "Short adjacent speaker turns (05:50-05:52)",
                "verified_ref": None,
            },
            "F5": {
                "path": str(WORKSPACE_ROOT / "outputs/audio/fr/fr_1031_1033_music_fp.wav"),
                "offset_s": 631.0,
                "duration_s": 2.0,
                "purpose": "False-positive speaker activity over music (10:31-10:33)",
                "verified_ref": None,
            },
            "F6": {
                "path": str(WORKSPACE_ROOT / "outputs/audio/fr/fr_1731_1735_overlap_vox.wav"),
                "offset_s": 1051.0,
                "duration_s": 4.0,
                "purpose": "Overlapping yelp (17:31-17:35)",
                "verified_ref": None,
            },
            "F7": {
                "path": str(WORKSPACE_ROOT / "outputs/audio/fr/fr_2208_2211_high_pitch.wav"),
                "offset_s": 1328.0,
                "duration_s": 3.0,
                "purpose": "High-pitched vocalization (22:08-22:11)",
                "verified_ref": None,
            },
            "F_100s": {
                "path": str(WORKSPACE_ROOT / "outputs/audio/fr/fr_0305_0445_cont_100s.wav"),
                "offset_s": 185.0,
                "duration_s": 100.0,
                "purpose": "Continuous 100s passage spanning F1 and F2",
                "verified_ref": None,
            },
            "F_300s": {
                "path": str(WORKSPACE_ROOT / "outputs/audio/fr/fr_0230_0730_300s.wav"),
                "offset_s": 150.0,
                "duration_s": 300.0,
                "purpose": "Continuous 5-min recurring voices passage (02:30-07:30)",
                "verified_ref": None,
            },
            "F_full": {
                "path": str(FRENCH_FULL_EPISODE_WAV),
                "offset_s": 0.0,
                "duration_s": 1379.901,
                "purpose": "Full 23-min French episode",
                "verified_ref": None,
            },
        },
        "french_verified_overlap": {
            "O1": {
                "path": str(PYANNOTE_ROOT / "outputs/eval_overlap/real/O1_mono_16k.wav"),
                "offset_s": 336.0,
                "duration_s": 9.0,
                "reference_text": overlap_eval["O1"]["reference_french_lexical"],
                "reference_speakers": overlap_eval["O1"]["reference_speakers"],
                "reference_lexical_speakers": overlap_eval["O1"]["reference_lexical_speakers"],
                "classification": overlap_eval["O1"]["classification"],
            },
            "O2": {
                "path": str(PYANNOTE_ROOT / "outputs/eval_overlap/real/O2_mono_16k.wav"),
                "offset_s": 346.5,
                "duration_s": 9.0,
                "reference_text": overlap_eval["O2"]["reference_french_lexical"],
                "reference_speakers": overlap_eval["O2"]["reference_speakers"],
                "reference_lexical_speakers": overlap_eval["O2"]["reference_lexical_speakers"],
                "classification": overlap_eval["O2"]["classification"],
            },
            "O3_music_neg": {
                "path": str(PYANNOTE_ROOT / "outputs/eval_overlap/real/O3_mono_16k.wav"),
                "offset_s": 627.5,
                "duration_s": 9.0,
                "reference_text": overlap_eval["O3"]["reference_french_lexical"],
                "reference_speakers": overlap_eval["O3"]["reference_speakers"],
                "reference_lexical_speakers": overlap_eval["O3"]["reference_lexical_speakers"],
                "classification": overlap_eval["O3"]["classification"],
            },
            "O4": {
                "path": str(PYANNOTE_ROOT / "outputs/eval_overlap/real/O4_mono_16k.wav"),
                "offset_s": 1048.0,
                "duration_s": 9.0,
                "reference_text": overlap_eval["O4"]["reference_french_lexical"],
                "reference_speakers": overlap_eval["O4"]["reference_speakers"],
                "reference_lexical_speakers": overlap_eval["O4"]["reference_lexical_speakers"],
                "classification": overlap_eval["O4"]["classification"],
            },
            "S1": {
                "path": str(PYANNOTE_ROOT / "outputs/eval_overlap/synthetic/S1_balanced.wav"),
                "duration_s": 4.8,
                "speaker_a_ref": overlap_eval["S1"]["speaker_a_ref"],
                "speaker_b_ref": overlap_eval["S1"]["speaker_b_ref"],
                "ref_diarization": [(0.925, 3.878, "A"), (1.15, 3.60, "B")],
                "classification": overlap_eval["S1"]["classification"],
            },
            "S2": {
                "path": str(PYANNOTE_ROOT / "outputs/eval_overlap/synthetic/S2_quiet_secondary.wav"),
                "duration_s": 5.18,
                "speaker_a_ref": overlap_eval["S2"]["speaker_a_ref"],
                "speaker_b_ref": overlap_eval["S2"]["speaker_b_ref"],
                "ref_diarization": [(0.976, 4.165, "A"), (1.35, 3.47, "B")],
                "classification": overlap_eval["S2"]["classification"],
            },
            "S3": {
                "path": str(PYANNOTE_ROOT / "outputs/eval_overlap/synthetic/S3_interrupted_0_9s.wav"),
                "duration_s": 7.42,
                "speaker_a_ref": overlap_eval["S3"]["speaker_a_ref"],
                "speaker_b_ref": overlap_eval["S3"]["speaker_b_ref"],
                "ref_diarization": [(0.993, 4.72, "A"), (3.82, 5.515, "B")],
                "classification": overlap_eval["S3"]["classification"],
            },
        },
        "french_fleurs": fr_fleurs,
        "fr_2spk_overlap": {
            **ov_manifest["fr_2spk_overlap"],
            "ref_diarization": [
                (0.0, 8.28, "A"),
                (2.0, 12.2, "B"),
            ],
        },
        "mandarin_fleurs": zh_manifest,
        "zh_2spk_overlap": {
            **ov_manifest["zh_2spk_overlap"],
            "ref_diarization": [
                (0.0, 10.38, "A"),
                (2.0, 9.62, "B"),
            ],
        },
        "zh_2spk_seq": {
            "path": str(WORKSPACE_ROOT / "outputs/audio/overlap/synthetic_zh_2spk_seq.wav"),
            "duration_s": 27.08,
            "ref_diarization": [
                (0.0, 10.38, "A"),
                (10.88, 18.50, "B"),
                (19.00, 27.08, "A"),
            ],
            "speaker_a_ref": zh_manifest["zh_fleurs_00_1906"]["raw_transcription"],
            "speaker_b_ref": zh_manifest["zh_fleurs_05_1805"]["raw_transcription"],
            "speaker_a_ref_2": zh_manifest["zh_fleurs_01_2006"]["raw_transcription"],
        },
        "zh_multichunk_synthetic": {
            "path": str(WORKSPACE_ROOT / "outputs/audio/zh/synthetic_zh_multichunk.wav"),
            "duration_s": 69.78,
        },
    }
