"""CPU-only tests for independent machine-translation post-edit."""
from __future__ import annotations

import json

import pytest

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
        text_client=Client(proposed, {"decisions": []}, RuntimeError("comparison transport failed")))
    assert out[0]["translated_text"] == rows[0]["translated_text"]
    assert report["status"] == "completed_with_errors"
    assert not report["deferred_comparison_ids"]



def test_messages_source_first_and_comparator_receive_coherent_cross_turn_passage():
    class Client(FakeClient):
        def complete(self, **kwargs):
            request=json.loads(kwargs["messages"][1]["content"])
            response=super().complete(**kwargs)
            if "passage_checks" in request:
                payload=json.loads(response["choices"][0]["message"]["content"])
                payload["decisions"].extend(
                    {"utterance_id":item["utterance_id"],"choice":"proposal","reason":"Coherent complete passage"}
                    for item in request["passage_checks"]
                )
                response["choices"][0]["message"]["content"]=json.dumps(payload)
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
    assert compare["joined_primary_passage"]=="He closed it Door before leaving."
    assert compare["joined_candidate_passage"]=="He closed the door before leaving."
    assert client.calls[0]["retry_max_tokens"]==16384
    assert client.calls[1]["retry_max_tokens"]==16384
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



class _LoopReviewClient(FakeClient):
    """Native response shapes, including an exhausted, deceptively valid prefix."""
    def complete(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if item == "loop":
            return {"choices": [{"finish_reason": "length",
                                "message": {"content": '{"decisions":[{"utterance_id":"u0","choice":"proposal"' + "\n " * 4096}}],
                    "_client_request_diagnostics": {"native_stop_reason": "max_tokens",
                                                     "completion_attempts": [{"stop_reason": "max_tokens"}]}}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(item)}}]}


def _loop_proposals():
    return {"reviews": [{"utterance_id": "u0", "corrected_text": "Hello, Esteban.", "reason": "Source name."},
                        {"utterance_id": "u1", "corrected_text": "Hi!", "reason": "No change."}]}


def _loop_cfg():
    return _cfg(wire_protocol="ninfer_messages", thinking_budget=2048,
                reasoning_effort="xhigh", max_tokens=8192)


def test_comparison_loop_uses_compact_response_same_evidence_and_caches_success(tmp_path):
    client = _LoopReviewClient(_loop_proposals(), "loop",
                               {"reason": "The name matches the source.", "verdicts": ["u0=proposal"]})
    checkpoint = tmp_path / "review.json"
    out, report = review_translated_utterances(_utts(), source_language="fr",
        config=_loop_cfg(), text_client=client, checkpoint_path=checkpoint)
    assert report["status"] == "completed" and report["reviewed"] == 2
    assert report["api_calls"] == 3 and report["comparative_calls"] == 2
    assert out[0]["translated_text"] == "Hello, Esteban."
    assert out[0]["translation_comparative_review"]["response_format"] == "compact_passage"
    assert client.calls[1]["messages"][1] == client.calls[2]["messages"][1]
    assert client.calls[1]["response_format"] != client.calls[2]["response_format"]
    assert "retry_max_tokens" not in client.calls[2]
    assert report["comparison_recoveries"][0]["outcome"] == "complete"
    replay = _LoopReviewClient()
    again, stats = review_translated_utterances(_utts(), source_language="fr",
        config=_loop_cfg(), text_client=replay, checkpoint_path=checkpoint)
    assert not replay.calls and stats["cache_hits"] == 2
    assert again == out


def test_comparison_loop_terminal_exhaustion_preserves_primary_and_remains_deferred(tmp_path):
    client = _LoopReviewClient(_loop_proposals(), "loop", "loop")
    checkpoint = tmp_path / "review.json"
    out, report = review_translated_utterances(_utts(), source_language="fr",
        config=_loop_cfg(), text_client=client, checkpoint_path=checkpoint)
    assert report["status"] == "completed_with_deferred_comparisons"
    assert report["deferred_comparison_ids"] == ["u0"] and report["reviewed"] == 1
    assert report["changed"] == 0 and report["errors"] == []
    assert report["comparison_recoveries"][0]["outcome"] == "deferred_preserved_primary"
    assert out[0]["translated_text"] == out[0]["machine_translated_text"] == "Hello, Estban."
    assert out[0]["translation_review_status"] == "review_deferred_comparative_preserved_machine_translation"
    assert "translation_comparative_review" not in out[0]
    assert not checkpoint.exists()


def test_compact_invalid_ids_defer_but_transport_failure_remains_error(tmp_path):
    for failed in ({"reason": "Invalid ID.", "verdicts": ["other=proposal"]},
                   RuntimeError("transport failed")):
        client = _LoopReviewClient(_loop_proposals(), "loop", failed)
        path=tmp_path/"review.json"
        out, report = review_translated_utterances(_utts(), source_language="fr",
            config=_loop_cfg(), text_client=client,checkpoint_path=path)
        if isinstance(failed,Exception):
            assert report["status"] == "completed_with_errors"
            assert report["errors"] and not report["deferred_comparison_ids"]
        else:
            assert report["status"]=="completed_with_deferred_comparisons"
            assert report["deferred_comparison_ids"]==["u0"] and not report["errors"]
            assert "compact_rejected_response" in report["comparison_recoveries"][0]
        assert out[0]["translated_text"] == "Hello, Estban."
        assert "translation_comparative_review" not in out[0]
        assert not path.exists()



@pytest.mark.parametrize("responses",[
    ("loop","loop"),
    ({"decisions":[{"utterance_id":"unknown","choice":"proposal","reason":"Bad ID"}]},
     {"reason":"Bad ID","verdicts":["unknown=proposal"]}),
])
def test_deferred_comparator_renders_without_claiming_complete_review(tmp_path, monkeypatch,responses):
    from pathlib import Path
    from types import SimpleNamespace
    from nvidia_transcribe import translation
    from nvidia_transcribe.translation import TranslationConfig
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles

    record = tmp_path / "recording.contract.json"
    record.write_text(json.dumps({
        "source_language": "fr", "source": {"path": "/sample/french.mkv", "duration_s": 5.0},
        "utterances": _utts(),
    }), encoding="utf-8")
    def fake_riva(_command, **kwargs):
        original = {u["id"]: u["translated_text"] for u in _utts()}
        items = json.loads(kwargs["input"])["items"]
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "status": "ok", "device_used": "cuda:0",
            "results": [{"utterance_id": item["utterance_id"],
                         "translated_text": original[item["utterance_id"]]} for item in items],
        }))
    monkeypatch.setattr(translation.subprocess, "run", fake_riva)
    client = _LoopReviewClient(_loop_proposals(), *responses)
    root = tmp_path / "rendered"
    cfg = SubtitlePipelineConfig(
        source_language="fr", translation=TranslationConfig(backend="riva", riva_device="cuda:0"),
        translation_review=_loop_cfg(), formats=("srt", "ass"))
    metadata = generate_subtitles(recording_contract_path=record, output_dir=root,
                                  config=cfg, text_client=client)
    assert metadata["translation_review"]["status"] == "completed_with_deferred_comparisons"
    assert metadata["translation_review"]["deferred_comparison_ids"] == ["u0"]
    contract_path = root / "translated.contract.json"
    contract_bytes = contract_path.read_bytes()
    saved = json.loads(contract_bytes)
    assert saved["utterances"][0]["translated_text"] == "Hello, Estban."
    assert saved["utterances"][0]["translation_review_status"] == "review_deferred_comparative_preserved_machine_translation"
    files = {name: Path(metadata["files"][name]) for name in ("srt", "ass")}
    before = {name: path.read_bytes() for name, path in files.items()}
    assert all(before.values())
    assert b"Hello, Estban." in before["srt"]
    assert len(client.calls) == 3
    replay = generate_subtitles(output_dir=root, config=SubtitlePipelineConfig(
        re_render_only=True, formats=("srt", "ass")), text_client=_LoopReviewClient())
    assert replay["translation_review"]["retained_review_status"] == "completed_with_deferred_comparisons"
    assert replay["translation_review"]["deferred_comparison_ids"] == ["u0"]
    assert contract_path.read_bytes() == contract_bytes
    assert {name: path.read_bytes() for name, path in files.items()} == before



def test_mixed_comparative_verdict_does_not_break_one_source_sentence():
    class Native(FakeClient):
        def complete(self, **kwargs):
            reply=super().complete(**kwargs)
            reply["choices"][0]["finish_reason"]="stop"
            return reply
    rows=[{"id":"a","speaker_id":"A","source_text":"Il a fermé la","translated_text":"He closed it"},
          {"id":"b","speaker_id":"A","source_text":"porte avant de partir.","translated_text":"Door before leaving."}]
    proposals={"reviews":[{"utterance_id":"a","corrected_text":"He closed the","reason":"Continuation"},
                          {"utterance_id":"b","corrected_text":"door before leaving.","reason":"Continuation"}]}
    from nvidia_transcribe.translation_review import text_digest
    comparison={"decisions":[{"utterance_id":"a","choice":"proposal","reason":"Better"},
                            {"utterance_id":"b","choice":"original","reason":"Keep"},
                            {"utterance_id":"passage:"+text_digest(json.dumps(["a","b"])),
                             "choice":"proposal","reason":"Joined candidate is coherent"}]}
    out, report=review_translated_utterances(rows,source_language="fr",
        config=_cfg(wire_protocol="ninfer_messages",thinking_budget=2048,reasoning_effort="xhigh"),
        text_client=Native(proposals,comparison))
    assert [r["translated_text"] for r in out]==[r["translated_text"] for r in rows]
    assert report["dependent_passage_rejections"]==1 and report["comparative_rejections"]==1


def test_rereview_preserves_original_mt_and_bound_context():
    from nvidia_transcribe.semantic_context import accepted_row_context
    row={"id":"u","source_text":"Nadia revient.","machine_translated_text":"Nadia returns.",
         "translated_text":"Nadia comes back.","source_start":1.,"source_end":2.}
    frame={"id":"f","inside_event":True,"visible_text":"Nadia","text_kind":"entity_label",
           "literal_observation_status":"model_observed",
           "uncertainty":"","sha256":"a"*64,"requested_seek_s":1.5}
    row["semantic_context"]=accepted_row_context(row,[],[frame])
    client=FakeClient({"reviews":[{"utterance_id":"u","corrected_text":"Nadia returns.",
                                   "changed":False,"reason":"Keep"}]})
    out,report=review_translated_utterances([row],source_language="fr",config=_cfg(),text_client=client)
    payload=json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["target_utterances"][0]["accepted_source_context"]==row["semantic_context"]
    assert out[0]["translated_text"]==out[0]["machine_translated_text"]=="Nadia returns."
    assert row["translated_text"]=="Nadia comes back."


def test_riva_operator_examples_are_literal_bound_and_alternating(tmp_path):
    from nvidia_transcribe.refinement import load_glossary
    from nvidia_transcribe.semantic_context import literal_examples
    from nvidia_transcribe._riva_translate_worker import translation_messages
    path=tmp_path/"glossary.json"
    path.write_text(json.dumps({"translation_examples":[{"source":"porte","target":"door"},
                                                        {"source":"port","target":"harbor"}]}))
    gl=load_glossary(path)
    item={"lang_pair":"fr-en","text":"Ferme la porte.",
          "context_examples":literal_examples({"source_text":"Ferme la porte."},gl)}
    assert len(item["context_examples"])==1
    assert translation_messages(item)==[{"role":"system","content":"fr-en"},
        {"role":"user","content":"porte"},{"role":"assistant","content":"door"},
        {"role":"user","content":"Ferme la porte."}]
    assert not literal_examples({"source_text":"portefeuille"},gl)



def test_native_length_subdivision_preserves_all_primary_rows_and_counts_attempts():
    class Native:
        def complete(self, **kwargs):
            request=json.loads(kwargs["messages"][1]["content"])
            targets=request["target_utterances"]
            if len(targets)>1:
                return {"choices":[{"finish_reason":"length","message":{"content":"partial"}}],
                        "_client_request_diagnostics":{"native_stop_reason":"max_tokens",
                        "completion_attempts":[{"stop_reason":"max_tokens","usage":{"input_tokens":50,"output_tokens":8192}}]}}
            target=targets[0]
            text={"u0":"Hello, Estban.","u1":"Hi!"}[target["utterance_id"]]
            return {"choices":[{"finish_reason":"stop","message":{"content":json.dumps(
                {"reviews":[{"utterance_id":target["utterance_id"],"corrected_text":text,"reason":"Keep"}]})}}],
                "_client_request_diagnostics":{"native_stop_reason":"end_turn",
                "completion_attempts":[{"stop_reason":"end_turn","usage":{"input_tokens":25,"output_tokens":10}}]}}
    out,report=review_translated_utterances(_utts(),source_language="fr",config=_loop_cfg(),text_client=Native())
    assert report["status"]=="completed" and report["api_calls"]==3
    assert report["reviewed"]==2 and report["changed"]==0
    assert len(report["native_attempts"])==3 and len(report["subdivisions"])==1
    assert [x["translated_text"] for x in out]==[x["translated_text"] for x in _utts()]


def test_riva_example_receipt_and_cache_bind_actual_primary_input(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from nvidia_transcribe import translation
    from nvidia_transcribe.translation import TranslationConfig, translate_utterances
    calls=[]
    def run(command,**kwargs):
        payload=json.loads(kwargs["input"])
        calls.append(payload)
        return SimpleNamespace(returncode=0,stderr="",stdout=json.dumps(
            {"status":"ok","device_used":"cuda:0","results":[
                {"utterance_id":item["utterance_id"],"translated_text":"Close the door.",
                 "input_tokens":33,"context_example_ids":[ex["id"] for ex in item["context_examples"]]}
                for item in payload["items"]]}))
    monkeypatch.setattr(translation.subprocess,"run",run)
    rows=[{"id":"a","source_text":"Ferme la porte.","machine_translated_text":"stale"}]
    cfg=TranslationConfig(backend="riva",riva_device="cuda:0")
    cache=tmp_path/"mt.json"
    for target in ("door","door","gate"):
        gl=GlossaryConfig(translation_examples=(("porte",target),))
        out,report=translate_utterances(rows,source_language="fr",config=cfg,glossary=gl,checkpoint_path=cache)
        assert report["context_example_rows"]==1 and report["native_input_tokens"]==33
        assert out[0]["machine_translated_text"]=="Close the door."
    assert len(calls)==2 and calls[-1]["items"][0]["context_examples"][0]["target"]=="gate"



def test_joint_source_and_mt_negation_veto_even_if_native_comparator_approves():
    class Native(FakeClient):
        def complete(self,**kwargs):
            response=super().complete(**kwargs)
            response["choices"][0]["finish_reason"]="stop"
            return response
    rows=[{"id":"negative","source_text":"Il ne ferme pas la porte.",
           "translated_text":"He does not close the door."}]
    prop={"reviews":[{"utterance_id":"negative","corrected_text":"He closes the door.","reason":"Bad proposal"}]}
    verdict={"decisions":[{"utterance_id":"negative","choice":"proposal","reason":"Bad approval"}]}
    result,report=review_translated_utterances(rows,source_language="fr",
        config=_loop_cfg(),text_client=Native(prop,verdict))
    assert result[0]["translated_text"]==rows[0]["translated_text"]
    assert report["rejected_polarity"]==1
    assert result[0]["translation_review_status"]=="review_rejected_polarity_preserved_machine_translation"



def test_dependent_edits_from_separate_native_comparisons_are_not_spliced():
    class Native(FakeClient):
        def complete(self,**kwargs):
            result=super().complete(**kwargs)
            result["choices"][0]["finish_reason"]="stop"
            return result
    rows=[{"id":"a","speaker_id":"A","source_text":"Il a fermé la","translated_text":"He closed it"},
          {"id":"b","speaker_id":"A","source_text":"porte.","translated_text":"Door."}]
    client=Native(
        {"reviews":[{"utterance_id":"a","corrected_text":"He closed the","reason":"Object follows"}]},
        {"decisions":[{"utterance_id":"a","choice":"proposal","reason":"Approve"}]},
        {"reviews":[{"utterance_id":"b","corrected_text":"door.","reason":"Continuation"}]},
        {"decisions":[{"utterance_id":"b","choice":"proposal","reason":"Approve"}]})
    cfg=_cfg(wire_protocol="ninfer_messages",thinking_budget=2048,reasoning_effort="xhigh",batch_size=1)
    out,report=review_translated_utterances(rows,source_language="fr",config=cfg,text_client=client)
    assert [r["translated_text"] for r in out]==[r["translated_text"] for r in rows]
    assert report["dependent_passage_rejections"]==2 and report["changed"]==0



def test_joined_passage_verdict_vetoes_individually_approved_fragments(tmp_path):
    class Native:
        def complete(self, **kwargs):
            request=json.loads(kwargs["messages"][1]["content"])
            if "target_utterances" in request:
                response={"reviews":[
                    {"utterance_id":"a","corrected_text":"He headed straight for","reason":"Direction"},
                    {"utterance_id":"b","corrected_text":"the course west.","reason":"Direction"}]}
            else:
                passage,=request["passage_checks"]
                assert passage["joined_proposal"]=="He headed straight for the course west."
                assert passage["joined_source"]=="Il a mis directement le cap à l'ouest."
                response={"decisions":[{"utterance_id":uid,"choice":"proposal","reason":"Cue only"}
                    for uid in ("a","b")]+[{"utterance_id":passage["utterance_id"],
                    "choice":"original","reason":"Joined phrase is ungrammatical"}]}
            return {"choices":[{"finish_reason":"stop","message":{"content":json.dumps(response)}}]}
    rows=[{"id":"a","speaker_id":"A","source_text":"Il a mis directement","translated_text":"He headed"},
          {"id":"b","speaker_id":"A","source_text":"le cap à l'ouest.","translated_text":"west."}]
    cfg=_cfg(wire_protocol="ninfer_messages",thinking_budget=2048,reasoning_effort="xhigh")
    path=tmp_path/"review.json"
    for client in (Native(),FakeClient()):
        out,report=review_translated_utterances(rows,source_language="fr",config=cfg,text_client=client,
                                                checkpoint_path=path)
        assert report["joined_passage_rejections"]==2 and report["changed"]==0
        assert [r["translated_text"] for r in out]==["He headed","west."]
        assert all(r["translation_review_passage"]["verdict"]["choice"]=="original" for r in out)



def test_missing_joined_passage_verdict_preserves_primary_and_not_cached(tmp_path):
    class Native(FakeClient):
        def complete(self,**kwargs):
            result=super().complete(**kwargs)
            result["choices"][0]["finish_reason"]="stop"
            return result
    rows=[{"id":"a","speaker_id":"A","source_text":"Il ferme la","translated_text":"He closes it"},
          {"id":"b","speaker_id":"A","source_text":"porte.","translated_text":"Door."}]
    prop={"reviews":[{"utterance_id":"a","corrected_text":"He closes the","reason":"Object follows"},
                     {"utterance_id":"b","corrected_text":"door.","reason":"Object"}]}
    incomplete={"decisions":[{"utterance_id":"a","choice":"proposal","reason":"Approve"},
                             {"utterance_id":"b","choice":"proposal","reason":"Approve"}]}
    path=tmp_path/"review.json"
    result,report=review_translated_utterances(rows,source_language="fr",
        config=_cfg(wire_protocol="ninfer_messages",thinking_budget=2048,reasoning_effort="xhigh"),
        text_client=Native(prop,incomplete,{"reason":"Incomplete","verdicts":["a=proposal","b=proposal"]}),checkpoint_path=path)
    assert report["status"]=="completed_with_deferred_comparisons" and report["changed"]==0
    assert report["deferred_comparison_ids"]==["a","b"]
    assert report["comparison_recoveries"][0]["reason"]=="missing_comparative_decision_ids"
    assert [r["translated_text"] for r in result]==[r["translated_text"] for r in rows]
    assert not path.exists()



def test_completed_malformed_comparison_recovers_on_identical_evidence(tmp_path):
    invalid=[
        ({"decisions":[{"utterance_id":"unknown","choice":"proposal","reason":"x"}]},"unexpected_decision_id:0"),
        ({"decisions":[{"utterance_id":"u0","choice":"proposal","reason":"x"}]*2},"duplicate_decision_id:1"),
        ({"decisions":[{"utterance_id":"u0","choice":"maybe","reason":"x"}]},"invalid_decision_choice:0"),
        ({"decisions":[{"utterance_id":"u0","choice":"proposal","reason":" "}]},"invalid_decision_reason:0"),
        ({"decisions":[]},"missing_comparative_decision_ids"),
        ({"decisions":[{"utterance_id":"u0","choice":"proposal"}]},"invalid_decision_shape:0"),
    ]
    for i,(bad,code) in enumerate(invalid):
        client=_LoopReviewClient(_loop_proposals(),bad,
                                {"reason":"Source spelling","verdicts":["u0=proposal"]})
        path=tmp_path/f"review-{i}.json"
        result,report=review_translated_utterances(_utts(),source_language="fr",
            config=_loop_cfg(),text_client=client,checkpoint_path=path)
        assert report["status"]=="completed" and report["changed"]==1
        assert result[0]["translated_text"]=="Hello, Esteban."
        receipt,=report["comparison_recoveries"]
        assert receipt["reason"]==code and receipt["outcome"]=="complete"
        assert json.loads(receipt["initial_rejected_response"]["answer_text"])==bad
        assert client.calls[1]["messages"][1]==client.calls[2]["messages"][1]
        assert report["api_calls"]==3 and path.is_file()


def test_uncertain_source_hallucinated_locations_veto_even_after_comparator_approval():
    from nvidia_transcribe.review_uncertainty import bind_review_uncertainty, current_uncertainty
    from copy import deepcopy
    row={"id":"bus","source_text":"巴士开往北港，在岩角停靠。","source_unit_ids":["u1"],
         "translated_text":"The bus goes to Northport, stopping at Rock Point."}
    prop={"proposal_id":"p","kind":"hallucination","target_source_id":"bus",
          "status":"advisory_only","source_truth_accepted":False,"citations":[
          {"source_id":"bus","quote":"巴士","character_span":[0,2],"source_row_unit_ids":["u1"]}]}
    quality={"schema":"nemovox-current-source-quality/v1","recording_sha256":"record",
        "recording_context":{"recording_sha256":"record","proposals":[prop]},
        "context_proposal_dispositions":[{"proposal_id":"p","kind":"hallucination",
            "source_ids":["bus"],"status":"hypothesis_unresolved"}],
        "scene_evidence":{"decisions":[{"source_id":"bus","outcome":"unresolved"}]}}
    bound=bind_review_uncertainty([row],quality)
    assert "review_source_uncertainty" not in row
    assert current_uncertainty(bound[0])["nominations"][0]["kind"]=="hallucination"
    guessed="The bus goes to Singapore, stopping at Rochor."
    client=_LoopReviewClient({"reviews":[{"utterance_id":"bus","corrected_text":guessed,"reason":"Likely geography"}]},
        {"decisions":[{"utterance_id":"bus","choice":"proposal","reason":"Plausible setting"}]})
    result,report=review_translated_utterances(bound,source_language="zh",config=_loop_cfg(),text_client=client)
    assert result[0]["translated_text"]==row["translated_text"]
    assert result[0]["translation_comparative_review"]["choice"]=="proposal"
    assert report["unresolved_source_rejections"]==1
    assert "unresolved_source_evidence" in json.loads(client.calls[0]["messages"][1]["content"])["target_utterances"][0]
    assert json.loads(client.calls[1]["messages"][1]["content"])["comparisons"][0]["source_and_original"]["unresolved_source_evidence"]
    # A stale reference cannot acquire a new row or altered source text.
    stale=deepcopy(bound[0]);stale["source_text"]="不同的话。"
    assert current_uncertainty(stale) is None
    stale_rows=bind_review_uncertainty([{**row,"source_unit_ids":["other"]}],quality)
    assert not current_uncertainty(stale_rows[0])["nominations"]
    # Uncertain rows are not frozen: a complete explicit operator mapping
    # or a lexical-preserving grammatical improvement remains admissible.
    from nvidia_transcribe.review_uncertainty import uncertainty_veto
    assert uncertainty_veto(bound[0],row["translated_text"],
        "A bus goes to Northport, stopping at Rock Point!",GlossaryConfig()) is None
    assert uncertainty_veto(bound[0],row["translated_text"],guessed,
        GlossaryConfig(translation_examples=((row["source_text"],guessed),))) is None


def test_direction_construction_veto_on_joined_exact_candidate():
    from nvidia_transcribe.translation_review import _bad_joined_direction
    assert _bad_joined_direction("He headed straight for west.")
    assert _bad_joined_direction("He headed straight for the course west.")
    assert not _bad_joined_direction("He headed straight west.")
    assert not _bad_joined_direction("He headed for the west coast.")
    class Native:
        def complete(self,**kwargs):
            payload=json.loads(kwargs["messages"][1]["content"])
            if "target_utterances" in payload:
                body={"reviews":[{"utterance_id":"a","corrected_text":"He headed straight","reason":"Direction"},
                                 {"utterance_id":"b","corrected_text":"for west.","reason":"Continuation"}]}
            else:
                ids=["a","b"]+[p["utterance_id"] for p in payload["passage_checks"]]
                body={"decisions":[{"utterance_id":uid,"choice":"proposal","reason":"Approve"} for uid in ids]}
            return {"choices":[{"finish_reason":"stop","message":{"content":json.dumps(body)}}]}
    rows=[{"id":"a","source_text":"Il a mis directement","translated_text":"He headed"},
          {"id":"b","source_text":"le cap à l'ouest.","translated_text":"west."}]
    out,report=review_translated_utterances(rows,source_language="fr",config=_loop_cfg(),text_client=Native())
    assert report["direction_grammar_rejections"]==2 and report["changed"]==0
    assert [r["translated_text"] for r in out]==["He headed","west."]


def test_uncertain_entity_anchors_compare_lexical_occurrences_not_capitalization():
    from nvidia_transcribe.review_uncertainty import _anchors_preserved
    assert _anchors_preserved("Change course twice!", "change course twice!")
    assert _anchors_preserved("It's us the crew!", "We're the crew!")
    assert _anchors_preserved("You're brilliant! That's our Nakaal.", "You're brilliant! Our Nakaal.")
    assert _anchors_preserved("in China, Japan and Arabia.", "In China, in Japan and in Arabia.")
    assert _anchors_preserved("Nadia returns.", "nadia returns.")
    assert not _anchors_preserved("Nadia returns.", "Isabella returns.")
    assert not _anchors_preserved("Nadia returns.", "nadia and Nadia return.")
    assert not _anchors_preserved("Nadia meets Tao.", "Tao meets Nadia.")
    assert not _anchors_preserved("Route 403 to Rock Point.", "Route 404 to Rock Point.")
    assert not _anchors_preserved("Nadia's boat.", "Tao's boat.")
    assert not _anchors_preserved("the bus stops.", "the bus stops at Rochor.")


def test_scene_uncertainty_allows_case_only_anchor_change_in_joined_passage():
    from nvidia_transcribe.review_uncertainty import bind_review_uncertainty
    rows = [
        {"id":"a","speaker_id":"A","source_text":"Il a mis directement le cap à",
         "translated_text":"He went straight to the"},
        {"id":"b","speaker_id":"A","source_text":"l'ouest.","translated_text":"West."},
    ]
    quality = {"schema":"nemovox-current-source-quality/v1",
               "scene_evidence":{"decisions":[{"source_id":"b","outcome":"unresolved"}]}}
    bound = bind_review_uncertainty(rows, quality)
    class Client:
        def complete(self, **kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            if "target_utterances" in payload:
                body = {"reviews":[
                    {"utterance_id":"a","corrected_text":"He headed straight","reason":"Direction"},
                    {"utterance_id":"b","corrected_text":"west.","reason":"Continuation"}]}
            else:
                ids = ["a","b"] + [p["utterance_id"] for p in payload["passage_checks"]]
                body = {"decisions":[{"utterance_id":uid,"choice":"proposal","reason":"Faithful passage"} for uid in ids]}
            return {"choices":[{"finish_reason":"stop","message":{"content":json.dumps(body)}}]}
    out, report = review_translated_utterances(bound, source_language="fr",
        config=_loop_cfg(), text_client=Client())
    assert [r["translated_text"] for r in out] == ["He headed straight","west."]
    assert report["unresolved_source_rejections"] == report["dependent_passage_rejections"] == 0
    assert report["changed"] == 2


def test_scene_uncertainty_retains_names_numbers_and_material_wording():
    from nvidia_transcribe.review_uncertainty import bind_review_uncertainty, uncertainty_veto
    row = {"id":"u","source_text":"Nadia revient.","translated_text":"Nadia returns on bus 403."}
    quality = {"schema":"nemovox-current-source-quality/v1",
               "scene_evidence":{"decisions":[{"source_id":"u","outcome":"unresolved"}]}}
    bound, = bind_review_uncertainty([row], quality)
    original = row["translated_text"]
    assert uncertainty_veto(bound, original, "nadia returns on bus 403.", GlossaryConfig()) is None
    for candidate in ("Isabella returns on bus 403.", "Nadia returns on bus 404."):
        assert uncertainty_veto(bound, original, candidate, GlossaryConfig()) == "unresolved_source_name_or_number_change"
    bound["review_source_uncertainty"]["source_quality_status"] = "unresolved_wording"
    assert uncertainty_veto(bound, original, "Nadia leaves on bus 403.", GlossaryConfig()) == "unresolved_source_material_lexical_change"


@pytest.mark.parametrize("compact_reply,expected_status,expected_text,cache_written", [
    ({"reason": "Source name.", "verdicts": ["u0=proposal"]},
     "completed", "Hello, Esteban.", True),
    ("exhausted", "completed_with_deferred_comparisons", "Hello, Estban.", False),
    (RuntimeError("transport failed"), "completed_with_errors", "Hello, Estban.", False),
])
def test_terminal_native_comparison_exhaustion_recovers_without_weakening_review(
        tmp_path, compact_reply, expected_status, expected_text, cache_written):
    class ExhaustionClient(_LoopReviewClient):
        def complete(self, *, messages, **kwargs):
            if self.responses and self.responses[0] == "exhausted":
                self.calls.append({"messages": messages, **kwargs})
                self.responses.pop(0)
                return {"choices": [{"finish_reason": "length",
                                     "message": {"content": '{"decisions":[{"utterance_id":"u0"'}}],
                        "_client_request_diagnostics": {"native_stop_reason": "max_tokens",
                            "completion_attempts": [{"stop_reason": "max_tokens"}]}}
            return super().complete(messages=messages, **kwargs)

    proposal = {"reviews": [_loop_proposals()["reviews"][0]]}
    client = ExhaustionClient(proposal, "exhausted", compact_reply)
    checkpoint = tmp_path / "review.json"
    out, report = review_translated_utterances(_utts()[:1], source_language="fr",
        config=_loop_cfg(), text_client=client, checkpoint_path=checkpoint)
    assert report["status"] == expected_status
    assert out[0]["translated_text"] == expected_text
    assert out[0]["machine_translated_text"] == "Hello, Estban."
    assert report["api_calls"] == 3 and len(client.calls) == 3
    assert report["subdivisions"] == []
    receipt, = report["comparison_recoveries"]
    assert receipt["reason"] == "terminal_comparative_output_exhausted"
    assert client.calls[1]["messages"][1] == client.calls[2]["messages"][1]
    assert checkpoint.exists() is cache_written
    if expected_status == "completed":
        assert receipt["outcome"] == "complete"
        assert out[0]["translation_comparative_review"]["choice"] == "proposal"
        assert report["reviewed"] == 1 and report["changed"] == 1
    elif expected_status == "completed_with_deferred_comparisons":
        assert receipt["outcome"] == "deferred_preserved_primary"
        assert report["deferred_comparison_ids"] == ["u0"] and not report["errors"]
        assert "translation_comparative_review" not in out[0]
    else:
        assert report["errors"] and not report["deferred_comparison_ids"]
        assert "translation_comparative_review" not in out[0]


@pytest.mark.parametrize("blocked", [False, True])
def test_failed_review_public_offline_salvage_preserves_failure_and_exact_contract(
        tmp_path, monkeypatch, capsys, blocked):
    from pathlib import Path
    from types import SimpleNamespace
    from nvidia_transcribe import cli, translation
    from nvidia_transcribe.translation import TranslationConfig
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
    from nvidia_transcribe.render_receipt import current_render_metadata

    record = tmp_path / "recording.contract.json"
    record.write_text(json.dumps({"source_language": "fr",
        "source": {"path": "/sample/french.mkv", "duration_s": 5.0},
        "utterances": _utts()[:1]}), encoding="utf-8")
    def fake_riva(*args, **kwargs):
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "status": "ok", "device_used": "cuda:0",
            "results": [{"utterance_id": "u0", "translated_text": "Hello, Estban."}]}))
    monkeypatch.setattr(translation.subprocess, "run", fake_riva)
    job = tmp_path / "job"
    job.mkdir()
    metadata_path = job / "subtitles.metadata.json"
    metadata_path.write_text('{"subtitles":{"renderable_cues":99}}')
    cfg = SubtitlePipelineConfig(source_language="fr",
        translation=TranslationConfig(backend="riva", riva_device="cuda:0"),
        translation_review=_cfg(), formats=("srt", "ass"))
    from nvidia_transcribe.resource_lifecycle import ResourceLifecycleError
    error = (ResourceLifecycleError("managed_unit_not_configured") if blocked
             else RuntimeError("transport failed"))
    expected = ("Required English review model unavailable" if blocked
                else "Required contextual English review did not complete")
    with pytest.raises(RuntimeError, match=expected):
        generate_subtitles(recording_contract_path=record, output_dir=job,
            config=cfg, text_client=FakeClient(error))
    contract = job / "translated.contract.json"
    original = contract.read_bytes()
    retained = json.loads(original)
    assert retained["translation_review"]["status"] == "completed_with_errors"
    assert not metadata_path.exists() and not list(job.glob("*.srt"))
    assert current_render_metadata(job, retained) is None

    def no_inference(*args, **kwargs):
        raise AssertionError("Offline salvage must not invoke inference or service commands")
    monkeypatch.setattr(translation.subprocess, "run", no_inference)
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({"version": 1, "resources": {"riva_local": {
        "checkpoint_path": str(tmp_path / "missing-checkpoint"),
        "worker_python": str(tmp_path / "missing-worker")}}}))
    args = ["generate-subtitles", "--output-dir", str(job), "--re-render",
            "--model-resources", str(resources), "--formats", "srt,ass"]
    assert cli.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "OK"  # command succeeded; required review did not
    assert result["render_status"] == "rendered_with_review_errors"
    assert result["translation_review"]["status"] == "reused_translated_contract"
    assert result["translation_review"]["retained_review_status"] == "completed_with_errors"
    assert result["translation_review"]["errors"] == retained["translation_review"]["errors"]
    assert result["translation_review"]["resource_blocked_reason"] == (
        "managed_unit_not_configured" if blocked else None)
    assert contract.read_bytes() == original
    assert current_render_metadata(job, retained)["render_status"] == "rendered_with_review_errors"
    files = {fmt: Path(result["files"][fmt]) for fmt in ("srt", "ass")}
    before = {fmt: path.read_bytes() for fmt, path in files.items()}
    assert b"Hello, Estban." in before["srt"]
    assert cli.main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["salvage_render_receipt"] == result["salvage_render_receipt"]
    assert contract.read_bytes() == original
    assert {fmt: path.read_bytes() for fmt, path in files.items()} == before
    files["srt"].write_bytes(before["srt"] + b"modified")
    assert current_render_metadata(job, retained) is None
    files["srt"].write_bytes(before["srt"])
    changed = dict(retained, source_language="es")
    contract.write_text(json.dumps(changed))
    assert current_render_metadata(job, changed) is None
    contract.write_bytes(original)
    assert current_render_metadata(job, retained) is not None
    # The failed contract must stay unchanged even on a rejected overlay.
    with pytest.raises(ValueError, match="overlay not allowed"):
        generate_subtitles(output_dir=job, config=SubtitlePipelineConfig(
            re_render_only=True, validation_overlay_path=tmp_path / "overlay.json"))
    assert contract.read_bytes() == original


def test_terminal_comparison_non_output_stop_is_not_a_recoverable_exhaustion(tmp_path):
    class ContextStopClient(_LoopReviewClient):
        def complete(self, *, messages, **kwargs):
            if self.responses and self.responses[0] == "context_stop":
                self.calls.append({"messages": messages, **kwargs})
                self.responses.pop(0)
                return {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
                        "_client_request_diagnostics": {
                            "native_stop_reason": "model_context_window_exceeded"}}
            return super().complete(messages=messages, **kwargs)
    client = ContextStopClient({"reviews": [_loop_proposals()["reviews"][0]]}, "context_stop")
    path = tmp_path / "review.json"
    out, report = review_translated_utterances(_utts()[:1], source_language="fr",
        config=_loop_cfg(), text_client=client, checkpoint_path=path)
    assert report["status"] == "completed_with_errors"
    assert report["errors"] and not report["comparison_recoveries"]
    assert not report["deferred_comparison_ids"] and not path.exists()
    assert out[0]["translated_text"] == "Hello, Estban."
    assert len(client.calls) == 2


def test_salvage_never_binds_render_to_contract_modified_during_render(tmp_path, monkeypatch):
    from nvidia_transcribe import subtitle_pipeline as pipeline
    contract = tmp_path / "translated.contract.json"
    row = dict(_utts()[0], machine_translated_text="Hello, Estban.")
    data = {"source_language": "fr", "source": {"path": "/sample/french.mkv",
            "duration_s": 5.0}, "utterances": [row],
            "translation_review": {"status": "completed_with_errors", "errors": ["transport"]}}
    contract.write_text(json.dumps(data))
    metadata = tmp_path / "subtitles.metadata.json"
    metadata.write_text('{"subtitles":{"renderable_cues":99}}')
    real_render = pipeline.render_subtitles
    def competing_write(*args, **kwargs):
        result = real_render(*args, **kwargs)
        contract.write_text(json.dumps({**data, "source_language": "es"}))
        return result
    monkeypatch.setattr(pipeline, "render_subtitles", competing_write)
    with pytest.raises(RuntimeError, match="changed during offline salvage"):
        pipeline.generate_subtitles(output_dir=tmp_path, config=pipeline.SubtitlePipelineConfig(
            re_render_only=True, formats=("srt",)))
    assert not metadata.exists()


@pytest.mark.parametrize("candidate,rejected", [
    ("Paul arrives on bus 403.", True),
    ("John arrives on bus 404.", True),
    ("john arrives on bus 403.", False),
    ("John is arriving on bus 403.", False),
])
def test_deferred_scene_uncertainty_preserves_anchors_after_comparator_approval(
        candidate, rejected):
    from nvidia_transcribe.review_uncertainty import bind_review_uncertainty, current_uncertainty
    row = {"id": "u", "speaker_id": "A", "source_text": "Jean arrive dans le bus 403.",
           "source_start": 1.0, "source_end": 2.0,
           "translated_text": "John arrives on bus 403.", "source_unit_ids": ["unit"]}
    quality = {"schema": "nemovox-current-source-quality/v1",
               "scene_evidence": {"decisions": [], "deferred_source_ids": ["u"]}}
    bound, = bind_review_uncertainty([row], quality)
    uncertainty = current_uncertainty(bound)
    assert uncertainty["scene_outcome"] == "deferred"
    assert uncertainty["nominations"] == []
    assert uncertainty["source_quality_status"] == ""
    client = _LoopReviewClient(
        {"reviews": [{"utterance_id": "u", "corrected_text": candidate,
                      "reason": "Candidate interpretation"}]},
        {"decisions": [{"utterance_id": "u", "choice": "proposal",
                        "reason": "Approve candidate"}]})
    out, report = review_translated_utterances(
        [bound], source_language="fr", config=_loop_cfg(), text_client=client)
    assert len(client.calls) == 2
    assert report["status"] == "completed"
    assert out[0]["machine_translated_text"] == row["translated_text"]
    assert out[0]["translation_comparative_review"]["choice"] == "proposal"
    assert report["unresolved_source_rejections"] == int(rejected)
    assert report["changed"] == int(not rejected)
    if rejected:
        assert out[0]["translation_review_rejection_reason"] == "unresolved_source_name_or_number_change"
        assert out[0]["translated_text"] == row["translated_text"]
    else:
        assert out[0]["translation_review_status"] == "corrected"
        assert out[0]["translated_text"] == candidate


def test_deferred_scene_guard_is_not_material_veto_or_operator_override():
    from nvidia_transcribe.review_uncertainty import bind_review_uncertainty, uncertainty_veto
    row = {"id": "u", "source_text": "Jean arrive.", "translated_text": "John arrives."}
    quality = {"schema": "nemovox-current-source-quality/v1",
               "scene_evidence": {"decisions": [], "deferred_source_ids": ["u"]}}
    bound, = bind_review_uncertainty([row], quality)
    assert uncertainty_veto(bound, "John arrives.", "Paul arrives.", GlossaryConfig()) == (
        "unresolved_source_name_or_number_change")
    assert uncertainty_veto(bound, "John arrives.", "John is arriving.", GlossaryConfig()) is None
    assert uncertainty_veto(bound, "John arrives.", "Paul arrives.",
        GlossaryConfig(translation_examples=((row["source_text"], "Paul arrives."),))) is None
    # An actually resolved, non-deferred case has no scene uncertainty.
    resolved, = bind_review_uncertainty([row], {
        **quality, "scene_evidence": {"decisions": [{"source_id": "u",
            "outcome": "context_supported"}], "deferred_source_ids": []}})
    assert uncertainty_veto(resolved, "John arrives.", "Paul arrives.", GlossaryConfig()) is None


def test_completed_malformed_native_proposal_recovers_with_exact_keyed_target_mapping():
    from copy import deepcopy
    bad=deepcopy(_loop_proposals())
    bad["reviews"][1]["utterance_id"]="u0"
    alternate={"reviews":{"r01":{"corrected_text":"Hello, Esteban.","reason":"Source name."},
                          "r02":{"corrected_text":"Hi!","reason":"No change."}}}
    comparator={"decisions":[{"utterance_id":"u0","choice":"proposal","reason":"Source spelling."}]}
    client=_LoopReviewClient(bad,alternate,comparator)
    out,report=review_translated_utterances(_utts(),source_language="fr",config=_loop_cfg(),text_client=client)
    assert report["status"]=="completed" and report["reviewed"]==2 and report["api_calls"]==3
    receipt=report["proposal_recoveries"][0]
    assert receipt["initial_rejected_response"]["validation_failure"]=="duplicate_proposal_id:1"
    assert json.loads(receipt["initial_rejected_response"]["answer_text"])==bad
    first=json.loads(client.calls[0]["messages"][-1]["content"])
    second=json.loads(client.calls[1]["messages"][-1]["content"])
    assert second.pop("response_keys")=={"r01":"u0","r02":"u1"} and second==first
    assert receipt["outcome"]=="complete"
    assert out[0]["translated_text"]=="Hello, Esteban."


def test_invalid_keyed_proposal_remains_failed_and_uncached(tmp_path):
    bad={"reviews":[{"utterance_id":"invented","corrected_text":"x","reason":"x"}]}
    alternate={"reviews":{"r01":{"corrected_text":"invented","reason":"x"}}}
    client=_LoopReviewClient(bad,alternate)
    checkpoint=tmp_path/"review.json"
    out,report=review_translated_utterances(_utts(),source_language="fr",config=_loop_cfg(),
        text_client=client,checkpoint_path=checkpoint)
    assert report["status"]=="completed_with_errors"
    assert report["proposal_recoveries"][0]["initial_rejected_response"]["validation_failure"]=="unexpected_proposal_id:0"
    assert report["proposal_recoveries"][0]["alternate_rejected_response"]["validation_failure"]=="invalid_keyed_proposal_ids"
    assert [r["translated_text"] for r in out]==[r["translated_text"] for r in _utts()]
    assert not checkpoint.exists()


def test_keyed_proposal_rejects_duplicate_json_keys():
    from nvidia_transcribe.translation_review import _native_proposals, ProposalDecisionError
    reply={"choices":[{"message":{"content":'{"reviews":{"r01":{"corrected_text":"A","reason":"x"},"r01":{"corrected_text":"B","reason":"y"}}}'}}]}
    with pytest.raises(ProposalDecisionError,match="duplicate_json_key"):
        _native_proposals(reply,[{"utterance_id":"u0","machine_translation":"A"}],{"r01":"u0"})


def test_offline_untimed_dialogue_is_not_a_successful_render(tmp_path):
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
    from nvidia_transcribe.render_receipt import current_render_metadata

    row = dict(_utts()[0], source_start=None, source_end=None, source_words=[], source_characters=[])
    retained = {"source_language": "fr", "source": {"duration_s": 5.0}, "utterances": [row]}
    contract = tmp_path / "translated.contract.json"
    contract.write_text(json.dumps(retained), encoding="utf-8")
    original = contract.read_bytes()
    with pytest.raises(RuntimeError, match="no usable subtitle timing"):
        generate_subtitles(output_dir=tmp_path, config=SubtitlePipelineConfig(
            re_render_only=True, formats=("srt", "ass")))
    assert contract.read_bytes() == original
    metadata = json.loads((tmp_path / "subtitles.metadata.json").read_text())
    assert metadata["render_status"] == "failed_no_renderable_dialogue"
    assert current_render_metadata(tmp_path, retained) is None


def test_timing_only_recovery_permits_supported_relative_duration_without_freezing_bad_mt():
    from copy import deepcopy
    from nvidia_transcribe.review_uncertainty import bind_review_uncertainty, uncertainty_veto
    from nvidia_transcribe.semantic_context import current_row_context, text_digest
    from nvidia_transcribe.semantic_context import _lex
    source="将于一月后举行仪式"
    tokens=list(_lex(source))
    units=[{"source_unit_id":f"u{i}","text":t} for i,t in enumerate(tokens)]
    ids=[u["source_unit_id"] for u in units]
    row={"id":"u","source_text":source,"source_unit_ids":ids,
        "source_quality_status":"event_timing_recovered_original_wording",
        "wording_status":"model_variants_unresolved","source_start":10.,"source_end":13.,
        "translated_text":"The ceremony will be held in January.",
        "prior_source_hypotheses":[{"source_characters":units}],
        "source_lineage":{"localized_wording_evidence":{"source_id":"u","source_text_sha256":text_digest(source),
            "source_unit_ids":ids,"status":"advisory_model_hypotheses","entity_identity_accepted":False,
            "alternatives":[{"backend":backend,"tokens":tokens,"source_unit_range":[0,len(ids)],
                             "audio_sha256":sha*64,"observation_ids":[backend]}
                            for backend,sha in (("nemotron","a"),("independent_asr","b"))]}}}
    quality={"schema":"nemovox-current-source-quality/v1","scene_evidence":{"decisions":[],"deferred_source_ids":["u"]}}
    bound,=bind_review_uncertainty([row],quality)
    assert current_row_context(bound)["localized_wording_alternatives"]
    good="The ceremony will be held in one month."
    assert uncertainty_veto(bound,row["translated_text"],good,GlossaryConfig()) is None
    assert uncertainty_veto(bound,row["translated_text"],"The ceremony will be held in two months.",GlossaryConfig())
    assert uncertainty_veto(bound,row["translated_text"],"Paul will arrive in one month.",GlossaryConfig())
    client=_LoopReviewClient({"reviews":[{"utterance_id":"u","corrected_text":good,"reason":"Relative interval."}]},
        {"decisions":[{"utterance_id":"u","choice":"proposal","reason":"Source duration."}]})
    out,report=review_translated_utterances([bound],source_language="zh",config=_loop_cfg(),text_client=client)
    assert report["status"]=="completed" and out[0]["translated_text"]==good
    # Do not use another range or stale source evidence to bypass the guard.
    stale=deepcopy(bound)
    stale["source_lineage"]["localized_wording_evidence"]["alternatives"][1]["source_unit_range"]=[len(ids)-2,len(ids)]
    assert uncertainty_veto(stale,row["translated_text"],good,GlossaryConfig())
