"""Source-faithful speaker and English review safeguards (CPU-only)."""

from __future__ import annotations

import json

from nvidia_transcribe.speakers import _near_threshold_single_embedding_candidate
from nvidia_transcribe.translation_review import (
    TranslationReviewConfig,
    review_translated_utterances,
)


class ReviewClient:
    def __init__(self, reviews):
        self.reviews = reviews
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        return {"choices": [{"message": {"content": json.dumps({"reviews": self.reviews})}}]}


def _review(text, source, primary):
    item = {
        "id": "u-neg", "source_text": source, "translated_text": primary,
        "source_start": 1.0, "source_end": 3.0, "speaker_id": "SPK_A",
    }
    client = ReviewClient([{
        "utterance_id": "u-neg", "changed": True,
        "corrected_text": text, "reason": "review candidate",
    }])
    out, summary = review_translated_utterances(
        [item], source_language="fr", text_client=client,
        config=TranslationReviewConfig(profile="balanced"),
    )
    return item, out[0], summary


def test_single_near_threshold_embedding_cannot_force_merge():
    assert _near_threshold_single_embedding_candidate(0.6536, 0.65)
    assert _near_threshold_single_embedding_candidate(0.6637, 0.65)
    assert not _near_threshold_single_embedding_candidate(0.78, 0.65)
    assert not _near_threshold_single_embedding_candidate(0.649, 0.65)
    assert not _near_threshold_single_embedding_candidate(0.70, 0.65)


def test_negation_inversion_is_rejected_without_modifying_original():
    before, after, summary = _review(
        "We discovered the map.", "Non, nous n'avons pas découvert la carte.",
        "We did not discover the map.",
    )
    assert after["source_text"] == before["source_text"]
    assert after["machine_translated_text"] == before["translated_text"]
    assert after["translated_text"] == before["translated_text"]
    assert after["translation_reviewed_text"] == "We discovered the map."
    assert after["translation_review_status"] == "review_rejected_polarity_preserved_machine_translation"
    assert summary["changed"] == 0 and summary["rejected_polarity"] == 1


def test_semantically_equivalent_negative_rephrasing_is_allowed():
    _, after, summary = _review(
        "Neither person saw it.", "Personne ne l'a vu.",
        "Not one person or the other saw it.",
    )
    assert after["translated_text"] == "Neither person saw it."
    assert summary["changed"] == 1 and summary["rejected_polarity"] == 0


def test_french_ne_plus_negative_is_preserved():
    _, after, summary = _review(
        "I will continue going.", "Je ne veux plus partir.",
        "I do not want to go anymore.",
    )
    assert after["translated_text"] == "I do not want to go anymore."
    assert summary["changed"] == 0 and summary["rejected_polarity"] == 1


def test_positive_sentence_can_receive_supported_review():
    _, after, summary = _review(
        "They reached the city.", "Ils ont atteint la cité.",
        "They got to it.",
    )
    assert after["translated_text"] == "They reached the city."
    assert summary["changed"] == 1 and summary["rejected_polarity"] == 0
