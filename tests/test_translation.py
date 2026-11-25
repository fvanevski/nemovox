"""CPU tests for batched translation, CLI subtitle generation, and resume without ASR (Work Packages C & E)."""

from __future__ import annotations

from contextlib import contextmanager
import json

import pytest
from pathlib import Path
from typing import Any

from nvidia_transcribe import translation as translation_module
from nvidia_transcribe.cli import build_parser
from nvidia_transcribe.refinement import GlossaryConfig, RefinementConfig
from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
from nvidia_transcribe.translation import TranslationConfig, map_language_for_riva, translate_utterances


class _StubTranslationClient:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"messages": messages, **kwargs})
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def test_map_language_for_riva() -> None:
    assert map_language_for_riva("fr") == "fr-en"
    assert map_language_for_riva("zh") == "zh-en"
    assert map_language_for_riva("zh-TW") == "zh-tw-en"
    assert map_language_for_riva("ja") == "ja-en"
    assert map_language_for_riva("unknown-lang") is None



@pytest.mark.parametrize(
    ("device", "requires_lease"),
    [("auto", True), ("cuda:0", True), ("cpu", False)],
)
def test_riva_worker_admission_spans_auto_device_selection(monkeypatch, device, requires_lease):
    """An auto device can become CUDA only after the shared lease is acquired."""
    events = []

    class TracedLifecycle:
        @contextmanager
        def worker(self, *, resource_name):
            assert resource_name == "riva_local"
            events.append("lease_enter")
            try:
                yield
            finally:
                events.append("lease_exit")

    def run_worker(command, **kwargs):
        assert command[-1] == str(translation_module.RIVA_WORKER_SCRIPT)
        assert json.loads(kwargs["input"])["device"] == device
        events.append("worker")
        return type("Completed", (), {
            "returncode": 0,
            "stdout": json.dumps({
                "status": "ok",
                "device_used": "cpu" if device == "cpu" else "cuda:0",
                "results": [{"utterance_id": "u1", "translated_text": "Hello"}],
            }),
            "stderr": "",
        })()

    monkeypatch.setattr(translation_module.subprocess, "run", run_worker)
    result, summary = translate_utterances(
        [{"utterance_id": "u1", "source_text": "Bonjour"}],
        source_language="fr",
        config=TranslationConfig(backend="riva", riva_device=device),
        lifecycle=TracedLifecycle(),
    )
    assert result[0]["translated_text"] == "Hello"
    assert summary["backend"] == "riva"
    assert events == (
        ["lease_enter", "worker", "lease_exit"] if requires_lease else ["worker"]
    )


def test_riva_auto_rejected_lease_prevents_worker_spawn(monkeypatch):
    from nvidia_transcribe.resource_lifecycle import ResourceLifecycleError

    class RejectedLifecycle:
        @contextmanager
        def worker(self, *, resource_name):
            assert resource_name == "riva_local"
            raise ResourceLifecycleError("resource_lease_timeout")
            yield  # Keep this a contextmanager; admission must fail on entry.

    monkeypatch.setattr(
        translation_module.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("Riva worker spawned without admission"),
    )
    with pytest.raises(ResourceLifecycleError) as exc:
        translate_utterances(
            [{"utterance_id": "u1", "source_text": "Bonjour"}],
            source_language="fr",
            config=TranslationConfig(backend="riva", riva_device="auto"),
            lifecycle=RejectedLifecycle(),
        )
    assert exc.value.status == "resource_lease_timeout"


def test_riva_auto_releases_lease_when_worker_fails(monkeypatch):
    events = []

    class TracedLifecycle:
        @contextmanager
        def worker(self, *, resource_name):
            events.append("lease_enter")
            try:
                yield
            finally:
                events.append("lease_exit")

    def fail_worker(*args, **kwargs):
        events.append("worker")
        raise RuntimeError("simulated subprocess failure")

    monkeypatch.setattr(translation_module.subprocess, "run", fail_worker)
    with pytest.raises(RuntimeError, match="simulated subprocess failure"):
        translate_utterances(
            [{"utterance_id": "u1", "source_text": "Bonjour"}],
            source_language="fr",
            config=TranslationConfig(backend="riva", riva_device="auto"),
            lifecycle=TracedLifecycle(),
        )
    assert events == ["lease_enter", "worker", "lease_exit"]


def test_translate_utterances_preserves_ids_glossary_and_never_merges_distinct_speakers() -> None:
    utts = [
        {"id": "utt-0000", "speaker_id": "SPK_00", "source_start": 1.0, "source_end": 2.5, "source_text": "Esteban, viens voir !", "refined_text": "Esteban, viens voir !"},
        {"id": "utt-0001", "speaker_id": "SPK_00", "source_start": 2.7, "source_end": 4.2, "source_text": "On a trouvé le double médaillon.", "refined_text": "On a trouvé le double médaillon."},
        {"id": "utt-0002", "speaker_id": "SPK_01", "source_start": 4.4, "source_end": 6.0, "source_text": "Vraiment ? Montre-moi !", "refined_text": "Vraiment ? Montre-moi !"},
    ]
    stub = _StubTranslationClient(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "translations": [
                                        {"utterance_id": "utt-0000", "translated_text": "Esteban, come look!", "grouped_with_prev": False, "notes": ""},
                                        {"utterance_id": "utt-0001", "translated_text": "We found the double medallion.", "grouped_with_prev": True, "notes": "same speaker continuation"},
                                        # Deliberately attempt unsafe grouped_with_prev=True across DIFFERENT speaker SPK_01!
                                        {"utterance_id": "utt-0002", "translated_text": "Really? Show me!", "grouped_with_prev": True, "notes": "bad cross-speaker group attempt"},
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
        ]
    )
    out, summary = translate_utterances(
        utts,
        source_language="fr",
        config=TranslationConfig(backend="swiftu", style="natural", batch_size=6),
        glossary=GlossaryConfig(characters=("Esteban", "Zia"), terms=("double medallion",)),
        text_client=stub,
    )
    assert len(stub.calls) == 1
    assert "Esteban" in stub.calls[0]["messages"][1]["content"]
    # Same-speaker SPK_00 utt-0001 allowed to group cleanly into utt-0000
    assert out[0]["grouped_utterance_ids"] == ["utt-0000", "utt-0001"]
    assert "Esteban, come look! We found the double medallion." == out[0]["translated_text"]
    assert out[0]["source_end"] == 4.2
    assert out[1]["merged_into"] == "utt-0000"
    # Different speaker SPK_01 MUST NOT be merged into SPK_00!
    assert "merged_into" not in out[2]
    assert out[2]["translated_text"] == "Really? Show me!"
    assert out[2]["source_text"] == "Vraiment ? Montre-moi !"
    assert summary["regrouped_adjacent_utterances"] == 1


def test_translate_utterances_missing_id_and_malformed_retry_fallback() -> None:
    utts = [
        {"id": "utt-0000", "speaker_id": "SPK_00", "source_start": 0.0, "source_end": 1.0, "source_text": "钙、钾等元素"},
        {"id": "utt-0001", "speaker_id": "SPK_01", "source_start": 2.0, "source_end": 3.0, "source_text": "这是金属。"},
    ]
    stub = _StubTranslationClient(
        [
            {"choices": [{"message": {"content": "NOT JSON"}}]},  # batch fails -> halves to 1+1
            {"choices": [{"message": {"content": json.dumps({"translations": [{"utterance_id": "utt-0000", "translated_text": "Elements such as calcium and potassium", "grouped_with_prev": False, "notes": ""}]})}}]},
            {"choices": [{"message": {"content": json.dumps({"translations": [{"utterance_id": "WRONG_ID", "translated_text": "bad", "grouped_with_prev": False, "notes": ""}]})}}]},
        ]
    )
    out, summary = translate_utterances(utts, source_language="zh", config=TranslationConfig(batch_size=2), text_client=stub)
    assert out[0]["translated_text"] == "Elements such as calcium and potassium"
    # Missing/wrong ID gracefully falls back to source text without crashing or losing the cue!
    assert out[1]["translated_text"] == "这是金属。"
    assert out[1]["translation_status"] == "fallback_missing_id"
    assert "missing_translation_id:utt-0001" in summary["errors"]


@pytest.mark.parametrize("bound", [False, True])
def test_translate_vetoed_regrouping_never_leaves_untranslated_source_orphan(bound: bool) -> None:
    utts = [
        {"id": "utt-0000", "speaker_id": "SPK_00", "source_start": 100.0, "source_end": 101.2, "source_text": "C'est la"},
        {"id": "utt-0001", "speaker_id": "SPK_00", "source_start": 107.0, "source_end": 110.5, "source_text": "pyramide de Mu !"},
    ]
    if bound:
        # Within the ordinary merge distance, but distinct acoustic ownership.
        utts[1].update(source_start=101.3, source_end=103.5)
        for n, row in enumerate(utts):
            row.update(source_occurrence_ids=[f"occ-{n}"], source_unit_ids=[f"unit-{n}"])
    stub = _StubTranslationClient(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "translations": [
                                        {"utterance_id": "utt-0000", "translated_text": "It's the pyramid of Mu!", "grouped_with_prev": False, "notes": ""},
                                        {"utterance_id": "utt-0001", "translated_text": "", "grouped_with_prev": True, "notes": ""},
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "translations": [
                                        {"utterance_id": "utt-0000", "translated_text": "It's the", "grouped_with_prev": False, "notes": ""},
                                        {"utterance_id": "utt-0001", "translated_text": "pyramid of Mu!", "grouped_with_prev": False, "notes": ""},
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        ]
    )
    out, summary = translate_utterances(utts, source_language="fr", config=TranslationConfig(batch_size=4), text_client=stub)
    assert len(stub.calls) == 2
    assert summary["regrouped_adjacent_utterances"] == 0
    assert out[0]["translated_text"] == "It's the"
    assert out[1]["translated_text"] == "pyramid of Mu!"
    assert "merged_into" not in out[1]


@pytest.mark.parametrize("backend", ["openai_chat", "swiftu"])
@pytest.mark.parametrize("reply_kind", ["ok", "request_error", "missing_id"])
def test_subtitle_translation_stage_reports_fallback_errors_without_aborting(
    tmp_path: Path, backend: str, reply_kind: str,
) -> None:
    """A completed render must not disguise a degraded translation stage."""
    contract = tmp_path / "recording.contract.json"
    contract.write_text(json.dumps({
        "source_language": "fr",
        "source": {"path": "/synthetic/dialogue.wav"},
        "utterances": [
            {"id": "utt-0000", "speaker_id": "SPK_00",
             "source_start": 1.0, "source_end": 3.0, "source_text": "Bonjour."},
        ],
    }), encoding="utf-8")

    translated_reply = {"choices": [{"message": {"content": json.dumps({
        "translations": [{"utterance_id": "utt-0000", "translated_text": "Hello.",
                          "grouped_with_prev": False, "notes": ""}],
    })}}]}
    missing_id_reply = {"choices": [{"message": {"content": json.dumps({
        "translations": [{"utterance_id": "unknown", "translated_text": "wrong",
                          "grouped_with_prev": False, "notes": ""}],
    })}}]}
    reply: dict[str, Any] | Exception = (
        translated_reply if reply_kind == "ok" else
        RuntimeError("synthetic translation endpoint unavailable")
        if reply_kind == "request_error" else missing_id_reply
    )
    progress: list[str] = []
    output = tmp_path / "output"
    metadata = generate_subtitles(
        recording_contract_path=contract,
        output_dir=output,
        config=SubtitlePipelineConfig(
            translation=TranslationConfig(backend=backend, batch_size=1),
            formats=("srt",),
        ),
        text_client=_StubTranslationClient([reply]),
        progress_callback=progress.append,
    )
    events = [json.loads(line[len("NEMOVOX_EVENT "):])
              for line in progress if line.startswith("NEMOVOX_EVENT ")]
    trans_events = [event for event in events if event["stage"] == "translation"]
    expected_status = "complete" if reply_kind == "ok" else "failed"
    assert [event["status"] for event in trans_events] == ["running", expected_status]
    assert all(event["schema"] == "nemovox-progress/v1" for event in trans_events)
    assert [event["status"] for event in events if event["stage"] == "render"] == [
        "running", "complete",
    ]

    persisted = json.loads((output / "translated.contract.json").read_text(encoding="utf-8"))
    assert persisted["translation"]["errors"] == metadata["translation"]["errors"]
    if reply_kind == "ok":
        assert not metadata["translation"]["errors"]
        assert trans_events[-1]["detail"] == ""
        assert persisted["utterances"][0]["translated_text"] == "Hello."
    else:
        assert metadata["translation"]["errors"]
        assert "completed_with_errors" in trans_events[-1]["detail"]
        assert "1 translation error(s)" in trans_events[-1]["detail"]
        assert persisted["utterances"][0]["translated_text"] == "Bonjour."
        assert persisted["utterances"][0]["translation_status"] == (
            "fallback_source_after_error" if reply_kind == "request_error"
            else "fallback_missing_id"
        )
    assert Path(metadata["files"]["srt"]).is_file()


def test_explicit_legacy_source_refinement_and_rerender_without_asr(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate-subtitles",
            "--recording-contract",
            "rec.json",
            "--output-dir",
            str(tmp_path / "subs"),
            "--refinement",
            "balanced",
            "--translator",
            "swiftu",
            "--style",
            "natural",
            "--formats",
            "srt,ass",
        ]
    )
    assert args.refinement == "balanced"
    assert args.translator == "swiftu"

    contract = {
        "schema_version": "nemotron-recording-contract/v1",
        "source": {"path": "/tmp/Episode_01.mkv", "duration_s": 10.0},
        "source_language": "zh",
        "utterances": [
            {"id": "utt-0000", "speaker_id": "SPK_00", "source_start": 1.0, "source_end": 4.0, "source_text": "钙、钾等元素属于金属。"}
        ],
    }
    rc_path = tmp_path / "recording.contract.json"
    rc_path.write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")

    stub = _StubTranslationClient(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "corrections": [
                                        {
                                            "utterance_id": "utt-0000",
                                            "original_text": "钙、钾等元素属于金属。",
                                            "refined_text": "钙、钾等元素属于金属。",
                                            "changed": False,
                                            "edit_type": "none",
                                            "rationale": "ok",
                                            "confidence": 0.99,
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "translations": [
                                        {
                                            "utterance_id": "utt-0000",
                                            "translated_text": "Elements like calcium and potassium are metals.",
                                            "grouped_with_prev": False,
                                            "notes": "",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        ]
    )
    out_dir = tmp_path / "out_subs"
    meta = generate_subtitles(
        recording_contract_path=rc_path,
        output_dir=out_dir,
        # This legacy test deliberately opts in to the old source-language
        # correction to retain coverage. Ordinary subtitle defaults use "off".
        config=SubtitlePipelineConfig(
            refinement=RefinementConfig(backend="openai_chat", profile="balanced"),
        ),
        text_client=stub,
    )
    assert len(stub.calls) == 2
    assert Path(meta["files"]["srt"]).name == "Episode_01.en.srt"
    assert Path(meta["files"]["ass"]).name == "Episode_01.en.ass"
    srt_text = Path(meta["files"]["srt"]).read_text(encoding="utf-8")
    assert "Elements like calcium\nand potassium are metals." in srt_text
    assert "calcium and potassium" in Path(meta["files"]["final_english_subtitles"]).read_text(encoding="utf-8")

    # Now re-render with narrower max_chars completely offline with ZERO model calls!
    meta_re = generate_subtitles(
        output_dir=out_dir,
        config=SubtitlePipelineConfig(re_render_only=True),
        text_client=None,
    )
    assert meta_re["refinement"]["status"] == "reused_translated_contract"
    assert len(stub.calls) == 2
