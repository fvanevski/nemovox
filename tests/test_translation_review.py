"""CPU-only tests for independent machine-translation post-edit."""
from __future__ import annotations

import json

from nvidia_transcribe.translation_review import TranslationReviewConfig, review_translated_utterances
from nvidia_transcribe.refinement import GlossaryConfig


class FakeClient:
    def __init__(self, *batches):
        self.responses = list(batches)
        self.calls = []

    def complete(self, *, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"choices": [{"message": {"content": json.dumps(item)}}]}


def _utts():
    return [
        {"id": "u0", "speaker_id": "A", "source_text": "Bonjour, Esteban.",
         "translated_text": "Hello, Estban.", "source_start": 1.0, "source_end": 2.0},
        {"id": "u1", "speaker_id": "B", "source_text": "Salut!",
         "translated_text": "Hi!", "source_start": 2.0, "source_end": 3.0},
    ]


def _cfg(**kw):
    return TranslationReviewConfig(**{"profile": "balanced", "resource_fingerprint": "text-one", **kw})


def test_accepts_supported_correction_only_and_preserves_identity():
    client = FakeClient({"reviews": [
        {"utterance_id": "u0", "changed": True, "corrected_text": "Hello, Esteban.", "reason": "Source name"},
        {"utterance_id": "u1", "changed": False, "corrected_text": "Hi!", "reason": "No change"},
    ]})
    original = _utts()
    result, summary = review_translated_utterances(
        original, source_language="fr", config=_cfg(),
        glossary=GlossaryConfig(characters=("Esteban",)), text_client=client)
    assert (summary["changed"], summary["reviewed"]) == (1, 2)
    assert result[0]["machine_translated_text"] == "Hello, Estban."
    assert result[0]["translated_text"] == "Hello, Esteban."
    assert result[1]["translation_review_status"] == "unchanged"
    for old, new in zip(original, result):
        for field in ("id", "speaker_id", "source_text", "source_start", "source_end"):
            assert old[field] == new[field]
    assert "Esteban" in client.calls[0]["messages"][1]["content"]


def test_missing_or_duplicate_review_ids_preserve_primary_translation():
    client = FakeClient({"reviews": [
        {"utterance_id": "u0", "changed": True, "corrected_text": "invented", "reason": "none"},
    ]})
    result, summary = review_translated_utterances(_utts(), source_language="fr",
                                                     config=_cfg(), text_client=client)
    assert summary["changed"] == 0 and summary["errors"]
    assert [x["translated_text"] for x in result] == ["Hello, Estban.", "Hi!"]
    assert all(x["translation_review_status"] == "review_error_preserved_machine_translation" for x in result)


def test_invalid_edits_and_flag_contradictions_fail_closed():
    client = FakeClient({"reviews": [
        {"utterance_id": "u0", "changed": True, "corrected_text": "", "reason": "delete"},
        {"utterance_id": "u1", "changed": True, "corrected_text": "Hi!", "reason": "mismatch"},
    ]})
    result, summary = review_translated_utterances(_utts(), source_language="fr",
                                                     config=_cfg(), text_client=client)
    assert len(summary["errors"]) == 2 and summary["changed"] == 0
    assert all(x["translation_review_status"] == "review_invalid_preserved_machine_translation" for x in result)


def test_review_cache_is_bound_to_model_style_glossary_and_machine_translation(tmp_path):
    cache = tmp_path / "review.json"
    unchanged = {"reviews": [
        {"utterance_id": "u0", "changed": False, "corrected_text": "Hello, Estban.", "reason": "fine"},
        {"utterance_id": "u1", "changed": False, "corrected_text": "Hi!", "reason": "fine"},
    ]}
    client = FakeClient(unchanged, unchanged, unchanged, unchanged)
    for cfg, gl in [
        (_cfg(), GlossaryConfig()),
        (_cfg(), GlossaryConfig()),
        (_cfg(style="literal"), GlossaryConfig()),
        (_cfg(resource_fingerprint="text-two"), GlossaryConfig()),
        (_cfg(), GlossaryConfig(characters=("Esteban",))),
    ]:
        review_translated_utterances(_utts(), source_language="fr", config=cfg,
                                    glossary=gl, text_client=client, checkpoint_path=cache)
    assert len(client.calls) == 4
    altered = _utts()
    altered[0]["translated_text"] = "Different."
    client.responses.append({"reviews": [
        {"utterance_id": "u0", "changed": False, "corrected_text": "Different.", "reason": "fine"},
        {"utterance_id": "u1", "changed": False, "corrected_text": "Hi!", "reason": "fine"},
    ]})
    review_translated_utterances(altered, source_language="fr", config=_cfg(),
                                  text_client=client, checkpoint_path=cache)
    assert len(client.calls) == 5


def test_off_does_not_contact_text_model():
    result, summary = review_translated_utterances(_utts(), source_language="fr")
    assert summary["status"] == "disabled"
    assert result[0]["translated_text"] == result[0]["machine_translated_text"]



def test_programmatic_postedit_rejects_legacy_primary_translator(tmp_path):
    import pytest
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
    from nvidia_transcribe.translation import TranslationConfig
    config = SubtitlePipelineConfig(
        translation=TranslationConfig(backend="openai_chat"),
        translation_review=TranslationReviewConfig(profile="balanced"),
    )
    with pytest.raises(ValueError, match="requires dedicated Riva"):
        generate_subtitles(recording_contract_path=tmp_path / "not-needed",
                           output_dir=tmp_path / "new-output", config=config)
    assert not (tmp_path / "new-output").exists()
