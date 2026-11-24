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



def test_riva_first_pipeline_runs_separate_postedit_and_preserves_raw_mt(tmp_path, monkeypatch):
    """No actual model or media file: verify stage order, saved contracts and rerender."""
    from pathlib import Path
    from types import SimpleNamespace
    from nvidia_transcribe import translation
    from nvidia_transcribe.translation import TranslationConfig
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles

    contract = {
        "schema_version": "nemotron-recording-contract/v1",
        "source": {"path": "/sample/French_episode.mkv", "duration_s": 10.0},
        "source_language": "fr",
        "utterances": [{"id": "utt-0000", "speaker_id": "SPK_00",
                        "source_start": 1.0, "source_end": 3.0,
                        "source_text": "Bonjour, Esteban."}],
    }
    recording = tmp_path / "recording.contract.json"
    recording.write_text(json.dumps(contract), encoding="utf-8")
    worker_calls = []

    def fake_riva(command, **kwargs):
        req = json.loads(kwargs["input"])
        worker_calls.append(req)
        assert req["device"] == "cuda:0"
        assert req["items"][0]["lang_pair"] == "fr-en"
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "status": "ok", "device_used": "cuda:0", "results": [
                {"utterance_id": "utt-0000", "translated_text": "Hello, Estban."},
            ],
        }))
    monkeypatch.setattr(translation.subprocess, "run", fake_riva)

    review_client = FakeClient({"reviews": [
        {"utterance_id": "utt-0000", "changed": True,
         "corrected_text": "Hello, Esteban.", "reason": "Correct proper name from French source."},
    ]})
    outdir = tmp_path / "subtitle-output"
    config = SubtitlePipelineConfig(
        source_language="fr",
        translation=TranslationConfig(backend="riva", riva_device="cuda:0"),
        translation_review=_cfg(),
        condense_high_cps=False,
        formats=("srt",),
    )
    meta = generate_subtitles(
        recording_contract_path=recording, output_dir=outdir,
        config=config, text_client=review_client,
    )
    assert len(worker_calls) == 1
    assert len(review_client.calls) == 1
    assert meta["translation"]["backend"] == "riva"
    assert meta["translation_review"]["changed"] == 1
    assert meta["refinement"]["profile"] == "off"  # no source-text model call in the normal pipeline
    translated = json.loads((outdir / "translated.contract.json").read_text(encoding="utf-8"))
    u = translated["utterances"][0]
    assert u["machine_translated_text"] == "Hello, Estban."
    assert u["translated_text"] == "Hello, Esteban."
    assert u["translation_review_status"] == "corrected"
    assert u["source_text"] == "Bonjour, Esteban."
    assert u["speaker_id"] == "SPK_00" and (u["source_start"], u["source_end"]) == (1.0, 3.0)
    assert "Hello, Esteban." in Path(meta["files"]["srt"]).read_text(encoding="utf-8")

    # Rerender must not touch either Riva worker or reviewer.
    offline = generate_subtitles(output_dir=outdir,
                                 config=SubtitlePipelineConfig(re_render_only=True, formats=("srt",)))
    assert offline["translation_review"]["status"] == "reused_translated_contract"
    assert len(worker_calls) == 1 and len(review_client.calls) == 1


def test_riva_rejects_missing_or_duplicate_worker_items(tmp_path, monkeypatch):
    """The dedicated translator must not pass malformed MT through to review."""
    import pytest
    from types import SimpleNamespace
    from nvidia_transcribe import translation
    from nvidia_transcribe.translation import TranslationConfig, translate_utterances

    def bad_worker(command, **kwargs):
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "status": "ok", "device_used": "cuda:0", "results": [
                {"utterance_id": "wrong-id", "translated_text": "Incorrect"},
            ],
        }))
    monkeypatch.setattr(translation.subprocess, "run", bad_worker)
    with pytest.raises(RuntimeError, match="invalid or duplicate utterance IDs"):
        translate_utterances(
            [{"id": "id1", "source_text": "Bonjour"}], source_language="fr",
            config=TranslationConfig(backend="riva", riva_device="cuda:0"),
        )



def test_reviewer_uses_original_asr_and_uncorrected_english_neighbors_not_legacy_refined_text():
    initial = _utts()
    initial[0]["refined_text"] = "Invented altered source-language hypothesis"
    initial[1]["refined_text"] = "Another invented hypothesis"
    client = FakeClient({"reviews": [
        {"utterance_id": "u0", "changed": False,
         "corrected_text": "Hello, Estban.", "reason": "No supported correction"},
        {"utterance_id": "u1", "changed": False,
         "corrected_text": "Hi!", "reason": "No supported correction"},
    ]})
    out, summary = review_translated_utterances(
        initial, source_language="fr", config=_cfg(), text_client=client,
    )
    assert summary["changed"] == 0
    assert [u["translated_text"] for u in out] == [u["translated_text"] for u in initial]
    prompt = json.loads(client.calls[0]["messages"][1]["content"])
    targets = prompt["target_utterances"]
    assert [u["source_text"] for u in targets] == [
        "Bonjour, Esteban.", "Salut!",
    ]
    assert [u["machine_translation"] for u in targets] == ["Hello, Estban.", "Hi!"]
    assert "Invented altered source-language hypothesis" not in client.calls[0]["messages"][1]["content"]
    assert "Another invented hypothesis" not in client.calls[0]["messages"][1]["content"]
    instructions = client.calls[0]["messages"][0]["content"]
    assert "Only edit TARGET translated English cues" in instructions
    assert "NOT a fresh translation request" in instructions
    assert "Stylistic preferences alone do not justify edits" in instructions


def test_english_source_skips_riva_worker_and_text_review(tmp_path, monkeypatch):
    """No en→en translation or source/English-review rewrite in default subtitle jobs."""
    from pathlib import Path
    from nvidia_transcribe import translation
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
    from nvidia_transcribe.translation import TranslationConfig, is_english_source_to_english

    assert is_english_source_to_english("en-US", "en")
    assert is_english_source_to_english("eng", "English")
    assert not is_english_source_to_english("fr", "en")
    assert not is_english_source_to_english(None, "en")

    def forbidden_worker(*args, **kwargs):
        raise AssertionError("Already-English speech must never invoke Riva")
    monkeypatch.setattr(translation.subprocess, "run", forbidden_worker)

    class ForbiddenTextClient:
        def complete(self, **kwargs):
            raise AssertionError("Already-English speech must not invoke a text model")

    record = tmp_path / "recording.contract.json"
    record.write_text(json.dumps({
        "source_language": "en", "source": {"path": "/sample/english.mkv", "duration_s": 6.0},
        "utterances": [{"id": "eng-1", "source_text": "I can tell you what happened.",
                        "source_start": 1.0, "source_end": 3.0,
                        "speaker_id": "SPK_00"}],
    }), encoding="utf-8")
    outdir = tmp_path / "english-output"
    config = SubtitlePipelineConfig(
        translation=TranslationConfig(backend="riva"),
        translation_review=_cfg(),
        formats=("srt",),
    )
    meta = generate_subtitles(
        recording_contract_path=record, output_dir=outdir,
        config=config, text_client=ForbiddenTextClient(),
    )
    assert meta["translation"]["status"] == "bypassed_same_language"
    assert meta["translation_review"]["status"] == "skipped_same_language"
    assert meta["condensation"]["status"] == "disabled"
    contract = json.loads((outdir / "translated.contract.json").read_text(encoding="utf-8"))
    u = contract["utterances"][0]
    assert u["source_text"] == u["translated_text"] == "I can tell you what happened."
    assert u["translation_status"] == "bypassed_same_language"
    assert u["translation_review_status"] == "skipped_same_language"
    assert "machine_translated_text" not in u
    assert "I can tell you what happened." in Path(meta["files"]["srt"]).read_text(encoding="utf-8")
    offline = generate_subtitles(output_dir=outdir, config=SubtitlePipelineConfig(
        re_render_only=True, formats=("srt",),
    ))
    assert offline["subtitles"]["renderable_cues"] == 1


def test_cli_riva_english_contract_does_not_require_local_worker(tmp_path, monkeypatch, capsys):
    """Riva path preflight cannot reject an English contract with a missing worker."""
    from nvidia_transcribe import cli
    from nvidia_transcribe import translation

    def forbidden_worker(*args, **kwargs):
        raise AssertionError("English contract must not invoke a Riva subprocess")
    monkeypatch.setattr(translation.subprocess, "run", forbidden_worker)
    # The standard resource registry's Riva checkpoint is intentionally not
    # required. No configured text-model credentials or live endpoint are used.
    recording = tmp_path / "recording.contract.json"
    recording.write_text(json.dumps({
        "source_language": "eng", "source": {"path": "/example/en.webm", "duration_s": 5.0},
        "utterances": [{"id": "one", "source_text": "We should go.",
                        "source_start": 1.0, "source_end": 2.0}],
    }), encoding="utf-8")
    resource_config = tmp_path / "resources.json"
    resource_config.write_text(json.dumps({
        "version": 1,
        "resources": {"riva_local": {
            "checkpoint_path": str(tmp_path / "definitely-missing-riva-checkpoint"),
            "worker_python": str(tmp_path / "definitely-missing-riva-python"),
        }},
    }), encoding="utf-8")
    status = cli.main([
        "generate-subtitles", "--recording-contract", str(recording),
        "--output-dir", str(tmp_path / "subtitles"),
        "--model-resources", str(resource_config),
        "--translator", "riva", "--translation-review", "balanced",
        "--formats", "srt",
    ])
    assert status == 0, capsys.readouterr().err
    contract = json.loads((tmp_path / "subtitles" / "translated.contract.json").read_text())
    assert contract["translation"]["status"] == "bypassed_same_language"



def test_required_review_endpoint_failure_is_not_a_successful_studio_job(tmp_path, monkeypatch):
    """Intermediate Riva MT survives a failed required English post-edit."""
    import pytest
    from types import SimpleNamespace
    from nvidia_transcribe import translation
    from nvidia_transcribe.translation import TranslationConfig
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles

    contract = {
        "schema_version": "nemotron-recording-contract/v1",
        "source": {"path": "/synthetic/french.webm", "duration_s": 4.0},
        "source_language": "fr",
        "utterances": [{"id": "utt-0000", "speaker_id": "SPK_00",
                        "source_start": 0.1, "source_end": 1.8,
                        "source_text": "Bonjour."}],
    }
    source = tmp_path / "recording.contract.json"
    source.write_text(json.dumps(contract), encoding="utf-8")

    def fake_riva(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "status": "ok", "device_used": "cuda:0",
            "results": [{"utterance_id": "utt-0000", "translated_text": "Hello."}],
        }))
    monkeypatch.setattr(translation.subprocess, "run", fake_riva)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="Required contextual English review did not complete"):
        generate_subtitles(
            recording_contract_path=source, output_dir=output,
            config=SubtitlePipelineConfig(
                source_language="fr",
                translation=TranslationConfig(backend="riva", riva_device="cuda:0"),
                translation_review=_cfg(), formats=("srt", "ass"),
            ),
            text_client=FakeClient(RuntimeError("review endpoint connection refused")),
        )
    saved = json.loads((output / "translated.contract.json").read_text(encoding="utf-8"))
    assert saved["translation_review"]["status"] == "completed_with_errors"
    assert saved["utterances"][0]["machine_translated_text"] == "Hello."
    assert saved["utterances"][0]["translated_text"] == "Hello."
    assert not list(output.glob("*.srt"))



def test_blocked_review_resource_preserves_all_mt_and_propagates_typed_failure(tmp_path, monkeypatch):
    import pytest
    from types import SimpleNamespace
    from nvidia_transcribe import translation
    from nvidia_transcribe.translation import TranslationConfig
    from nvidia_transcribe.resource_lifecycle import ResourceLifecycleError
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles

    utterances = [{"id": f"utt-{i:04d}", "speaker_id": "SPK_A",
                   "source_start": float(i), "source_end": float(i) + .6,
                   "source_text": "Bonjour."} for i in range(7)]
    contract = {
        "schema_version": "nemotron-recording-contract/v1",
        "source": {"path": "/synthetic/french.webm", "duration_s": 10.0},
        "source_language": "fr", "utterances": utterances,
    }
    source = tmp_path / "recording.contract.json"
    source.write_text(json.dumps(contract), encoding="utf-8")

    def riva_worker(_command, **kwargs):
        requested = json.loads(kwargs["input"])
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "status": "ok", "device_used": "cuda:0",
            "results": [{"utterance_id": row["utterance_id"], "translated_text": "Hello."}
                        for row in requested["items"]],
        }))
    monkeypatch.setattr(translation.subprocess, "run", riva_worker)

    blocked = FakeClient(ResourceLifecycleError("managed_unit_not_configured"))
    output = tmp_path / "output"
    with pytest.raises(ResourceLifecycleError) as error:
        generate_subtitles(
            recording_contract_path=source, output_dir=output,
            config=SubtitlePipelineConfig(
                source_language="fr",
                translation=TranslationConfig(backend="riva", riva_device="cuda:0"),
                translation_review=_cfg(), formats=("srt", "ass"),
            ),
            text_client=blocked,
        )
    assert error.value.status == "managed_unit_not_configured"
    saved = json.loads((output / "translated.contract.json").read_text(encoding="utf-8"))
    assert saved["translation_review"]["resource_blocked_reason"] == "managed_unit_not_configured"
    assert len(saved["utterances"]) == 7
    assert all(u["machine_translated_text"] == "Hello." for u in saved["utterances"])
    assert all(u["translated_text"] == "Hello." for u in saved["utterances"])
    assert len(blocked.calls) == 1  # no seven-batch startup retry storm


def test_messages_comparative_review_rejects_worse_proposal_and_preserves_original():
    class Client(FakeClient):
        def complete(self, **kwargs):
            response = super().complete(**kwargs)
            response["choices"][0]["finish_reason"] = "stop"
            return response
    rows = [{"id": "u1", "source_text": "Tu crois cette histoire ?",
             "translated_text": "Do you believe that story?"}]
    proposed = {"reviews": [{"utterance_id": "u1", 
                             "corrected_text": "Because do you believe that story?",
                             "reason": "Add conversational conjunction."}]}
    decision = {"decisions": [{"utterance_id": "u1", "choice": "original",
                               "reason": "The proposal introduces an ungrammatical question."}]}
    client = Client(proposed, decision)
    out, report = review_translated_utterances(
        rows, source_language="fr", config=_cfg(wire_protocol="ninfer_messages",
                                               thinking_budget=0, reasoning_effort="none"),
        text_client=client)
    assert out[0]["translated_text"] == out[0]["machine_translated_text"] == rows[0]["translated_text"]
    assert out[0]["translation_review_status"] == "review_rejected_comparative_preserved_machine_translation"
    assert report["comparative_calls"] == report["comparative_rejections"] == 1
    assert client.calls[1]["thinking_budget"] == 1024
    assert client.calls[1]["reasoning_effort"] == "xhigh"


def test_messages_comparative_review_accepts_contextual_repair():
    class Client(FakeClient):
        def complete(self, **kwargs):
            response = super().complete(**kwargs)
            response["choices"][0]["finish_reason"] = "stop"
            return response
    rows = [{"id": "u1", "source_text": "Ils se sont fichus de nous !",
             "translated_text": "They don't care about us!"}]
    proposed = {"reviews": [{"utterance_id": "u1", 
                             "corrected_text": "They made fools of us!",
                             "reason": "Contextual idiom."}]}
    decision = {"decisions": [{"utterance_id": "u1", "choice": "proposal",
                               "reason": "The past reflexive idiom denotes ridicule or deception."}]}
    out, report = review_translated_utterances(
        rows, source_language="fr", config=_cfg(wire_protocol="ninfer_messages",
                                               thinking_budget=0, reasoning_effort="none"),
        text_client=Client(proposed, decision))
    assert out[0]["translated_text"] == "They made fools of us!"
    assert out[0]["machine_translated_text"] == rows[0]["translated_text"]
    assert report["changed"] == 1


def test_messages_incomplete_comparative_review_does_not_accept_candidate():
    class Client(FakeClient):
        def complete(self, **kwargs):
            response = super().complete(**kwargs)
            response["choices"][0]["finish_reason"] = "stop"
            return response
    rows = [{"id": "u1", "source_text": "Bonjour", "translated_text": "Hello."}]
    proposed = {"reviews": [{"utterance_id": "u1", 
                             "corrected_text": "Good evening.", "reason": "Greeting."}]}
    out, report = review_translated_utterances(
        rows, source_language="fr", config=_cfg(wire_protocol="ninfer_messages",
                                               thinking_budget=0, reasoning_effort="none"),
        text_client=Client(proposed, {"decisions": []}))
    assert out[0]["translated_text"] == rows[0]["translated_text"]
    assert report["status"] == "completed_with_errors"



def test_messages_source_first_and_comparator_receive_coherent_cross_turn_passage():
    class Client(FakeClient):
        def complete(self, **kwargs):
            response=super().complete(**kwargs)
            response["choices"][0]["finish_reason"]="stop"
            return response
    rows=[{"id":"u0","source_text":"Il a fermé la","translated_text":"He closed it",
           "source_start":1,"source_end":3,"speaker_id":"A"},
          {"id":"u1","source_text":"porte avant de partir.","translated_text":"Door before leaving.",
           "source_start":3,"source_end":5,"speaker_id":"A"}]
    proposed={"reviews":[{"utterance_id":"u0","corrected_text":"He closed the","reason":"Object follows"},
                         {"utterance_id":"u1","corrected_text":"door before leaving.","reason":"Continuation"}]}
    decision={"decisions":[{"utterance_id":uid,"choice":"proposal","reason":"Coherent complete passage"}
                            for uid in ("u0","u1")]}
    client=Client(proposed,decision)
    out,report=review_translated_utterances(rows,source_language="fr",
        config=_cfg(wire_protocol="ninfer_messages",thinking_budget=2048,reasoning_effort="xhigh"),
        text_client=client)
    first=json.loads(client.calls[0]["messages"][1]["content"])
    assert all("machine_translation" not in x for x in first["target_utterances"])
    assert "He closed it" not in client.calls[0]["messages"][1]["content"]
    assert client.calls[0]["thinking_budget"]==2048
    assert "changed" not in client.calls[0]["response_format"]["schema"]["properties"]["reviews"]["items"]["required"]
    compare=json.loads(client.calls[1]["messages"][1]["content"])
    assert [x["utterance_id"] for x in compare["complete_target_passage"]]==["u0","u1"]
    assert [x["translated_text"] for x in out]==["He closed the","door before leaving."]
    assert [x["machine_translated_text"] for x in out]==[x["translated_text"] for x in rows]
    assert report["changed"]==2


def test_restrictive_construction_is_not_negative_action():
    from nvidia_transcribe.translation_review import _unsupported_polarity_inversion
    assert not _unsupported_polarity_inversion("Ils ont menti. Il n'a que deux amis.",
                                               "They don't care. He has only two friends.",
                                               "They lied. He has only two friends.","fr")
    assert _unsupported_polarity_inversion("Il n'a pas de bateau.","He has no boat.","He has a boat.","fr")
    assert _unsupported_polarity_inversion("Il n'a que deux amis. Il ne vient jamais.",
                                           "He has two friends. He never comes.",
                                           "He has two friends. He comes.","fr")



def test_nonrestrictive_subordinate_clause_does_not_remove_negative_guard():
    from nvidia_transcribe.translation_review import _unsupported_polarity_inversion
    assert _unsupported_polarity_inversion("Je n'ose dire que c'est vrai.",
                                           "I don't dare say it is true.",
                                           "I dare say it is true.","fr")
