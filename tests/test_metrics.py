"""CPU-only unit tests for WER and CER metrics."""

from __future__ import annotations

from nvidia_transcribe.metrics import compute_cer, compute_wer


def test_compute_wer_french_normalization() -> None:
    res = compute_wer(
        "C'était l'été à Paris !",
        "<spk:0> C'etait l'été à Paris.",
    )
    assert res["edits"] == 1  # 'etait' vs 'était'
    assert res["ref_words"] == 6


def test_compute_cer_mandarin_chinese() -> None:
    res = compute_cer("你好，世界！", "你好世界。")
    assert res["cer"] == 0.0
    assert res["ref_chars"] == 4

    res2 = compute_cer("今天天气很好", "今天天气不错")
    assert res2["edits"] == 2
    assert res2["cer"] == round(2 / 6, 4)
