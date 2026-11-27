"""CPU successor regressions for the normal native connected workflow."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from nvidia_transcribe import connected_units as cu, translation
from nvidia_transcribe.refinement import GlossaryConfig
from nvidia_transcribe.translation import TranslationConfig, translate_utterances
from nvidia_transcribe.translation_review import TranslationReviewConfig, review_translated_utterances
from nvidia_transcribe.subtitles import render_subtitles
from nvidia_transcribe.utterances import apply_validation_overlay


def row(i, source, start, end, english="primary", speaker="S01"):
    return {"id": f"event-{i:05d}", "source_text": source, "speaker_id": speaker,
            "source_start": start, "source_end": end, "translated_text": english,
            "machine_translated_text": english, "grouped_utterance_ids": [f"event-{i:05d}"],
            "timestamp_provenance": {"backend": "moss_events", "native_event_index": i,
                                     "timing_status": "observed_segment"}}


def attach(rows, joint="joint alternative"):
    for group in cu.nominate(rows):
        record = {**cu.request_record(group, []), "joint_mt": joint, "joint_input_tokens": 20}
        for member in group:
            member["translation_unit_candidate"] = deepcopy(record)
    return rows


def config(**kw):
    return TranslationReviewConfig(**{"profile": "balanced", "wire_protocol": "ninfer_messages",
        "thinking_budget": 2048, "reasoning_effort": "xhigh", "max_tokens": 4096, **kw})


class Review:
    """Return explicit fixture decisions; inspect the actual native wire payload."""
    def __init__(self, proposals, *, reject=(), reject_joint=False):
        self.proposals = proposals
        self.reject = set(reject)
        self.reject_joint = reject_joint
        self.calls = []

    def complete(self, *, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        self.calls.append(payload)
        if "target_utterances" in payload:
            body = {"reviews": [{"utterance_id": t["utterance_id"],
                "corrected_text": self.proposals[t["utterance_id"]], "reason": "fixture source meaning"}
                for t in payload["target_utterances"]]}
        else:
            body = {"decisions": [{"utterance_id": c["utterance_id"],
                "choice": "original" if c["utterance_id"] in self.reject else "proposal",
                "reason": "fixture comparison"} for c in payload["comparisons"]]
                + [{"utterance_id": c["utterance_id"],
                    "choice": "original" if self.reject_joint else "proposal",
                    "reason": "fixture complete allocation comparison"} for c in payload["passage_checks"]]}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(body)}}]}


def review(rows, client, **kwargs):
    return review_translated_utterances(rows, source_language=kwargs.pop("language", "zh"),
        config=kwargs.pop("cfg", config()), text_client=client, **kwargs)


@pytest.mark.parametrize("second", [
    row(1, "下去", 2.1, 3, speaker="S02"),
    row(1, "下去", None, None),
    row(2, "下去", 2.1, 3),
    row(1, "下去", 1.8, 3),
    row(1, "下去", 2.5, 3),
    row(1, "他说", 2.1, 3),
    {**row(1, "下去", 2.1, 3), "source_quality_status": "unresolved_wording"},
])
def test_protected_boundaries(second):
    assert not cu.nominate([row(0, "他说", 1, 2), second])


def test_intervening_overlap_and_complete_sentence():
    rows = [row(0, "他说", 1, 2), row(1, "下去", 2.1, 3),
            row(2, "别走", 1.8, 2.2, speaker="S02")]
    assert not cu.nominate(rows)
    assert not cu.nominate([row(0, "他说。", 1, 2), row(1, "下去", 2.1, 3)])


def test_cjk_hypothesis_never_replaces_observed_boundaries():
    rows = attach([row(0, "那不是他", 1, 2, "That is not him."),
                   row(1, "老师，这", 2.1, 3, "Teacher, this...")],
                  joint="That is not his teacher.")
    source = deepcopy(rows)
    assert cu.groups(rows) == [[0, 1]]
    candidate = rows[0]["translation_unit_candidate"]
    assert candidate["joint_input"] == "那不是他老师，这"
    assert [x["source_text"] for x in candidate["constituents"]] == ["那不是他", "老师，这"]
    proposals = {rows[0]["id"]: "That is not his teacher,", rows[1]["id"]: "this..."}
    client = Review(proposals, reject_joint=True)
    out, summary = review(rows, client)
    assert summary["status"] == "completed" and summary["changed"] == 0
    assert [x["translated_text"] for x in out] == ["That is not him.", "Teacher, this..."]
    assert all(x["translation_unit"]["selection"] == "independent" for x in out)
    assert rows == source
    targets = client.calls[0]["target_utterances"]
    assert [t["source_text"] for t in targets] == ["那不是他", "老师，这"]
    assert targets[0]["connected_hypothesis"]["joint_mt"] == "That is not his teacher."
    assert not client.calls[1]["comparisons"]
    assert client.calls[1]["passage_checks"][0]["atomic_connected_allocation"]
    bad = deepcopy(rows)
    bad[1]["source_text"] = "changed"
    with pytest.raises(ValueError, match="membership"):
        cu.groups(bad)


def test_atomic_acceptance_does_not_revert_unrelated_units_and_resumes(tmp_path):
    rows = attach([row(0, "他说", 1, 2, "He said."),
                   row(1, "向西走", 2.1, 3, "Go west."),
                   row(2, "还有别的", 6, 7, "Something else."),
                   row(3, "继续走", 12, 13, "Keep going.")])
    proposals = dict(zip([r["id"] for r in rows], ["He said to", "go west.", "There is more.",
                                                  "Continue walking."]))
    client = Review(proposals, reject=[rows[2]["id"]])
    ckpt = tmp_path / "review.json"
    out, summary = review(rows, client, checkpoint_path=ckpt)
    assert summary["status"] == "completed"
    assert summary["changed"] == 3 and summary["dependent_passage_rejections"] == 0
    assert summary["jointly_adjudicated_count"] == 1
    assert [r["translated_text"] for r in out] == ["He said to", "go west.", "Something else.",
                                                 "Continue walking."]
    assert out[0]["translation_unit"] == out[1]["translation_unit"]
    assert out[2]["translation_unit"] != out[3]["translation_unit"]
    assert [r["machine_translated_text"] for r in out] == [r["machine_translated_text"] for r in rows]
    comparison = client.calls[1]
    assert {c["utterance_id"] for c in comparison["comparisons"]} == {rows[2]["id"], rows[3]["id"]}
    assert comparison["passage_checks"][0]["target_ids"] == [rows[0]["id"], rows[1]["id"]]
    replay_client = Review({})
    replay, result = review(rows, replay_client, checkpoint_path=ckpt)
    assert not replay_client.calls and replay == out and result["cache_hits"] == len(rows)
    assert cu.validate_selected_units(out, require_complete=True) is None


def test_atomic_group_rolls_back_if_local_polarity_veto_rejects_one_part():
    rows = attach([row(0, "Il ne veut pas", 1, 2, "He does not want"),
                   row(1, "partir", 2.1, 3, "to leave.")])
    client = Review({rows[0]["id"]: "He wants", rows[1]["id"]: "to go."})
    out, summary = review(rows, client, language="fr")
    assert summary["rejected_polarity"] == 1
    assert summary["dependent_passage_rejections"] == 1 and summary["changed"] == 0
    assert [r["translated_text"] for r in out] == ["He does not want", "to leave."]
    assert all(r["translation_unit"]["selection"] == "independent" for r in out)


def test_candidate_is_indivisible_at_batch_and_selective_boundaries():
    rows = attach([row(0, "他说", 1, 2, "He said."),
                   row(1, "向西走", 2.1, 3, "Go west.")])
    client = Review({rows[0]["id"]: "He said to", rows[1]["id"]: "go west."})
    out, summary = review(rows, client, cfg=config(batch_size=1))
    assert summary["changed"] == 2 and len(client.calls) == 2
    assert len(client.calls[0]["target_utterances"]) == 2
    with pytest.raises(ValueError, match="complete connected candidate"):
        review(rows, Review({}), target_utterance_ids={rows[0]["id"]})
    assert out[0]["translation_unit"] == out[1]["translation_unit"]


def fake_worker(monkeypatch, translations):
    calls = []
    def run(command, **kwargs):
        request = json.loads(kwargs["input"])
        calls.append(request)
        results = [{"utterance_id": item["utterance_id"],
                    "translated_text": (translations[item["utterance_id"]]
                        if item["utterance_id"] in translations else "He said to go west."),
                    "context_example_ids": [ex["id"] for ex in item["context_examples"]],
                    "input_tokens": 20} for item in request["items"]]
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
            {"status": "ok", "device_used": request["device"], "results": results}))
    monkeypatch.setattr(translation.subprocess, "run", run)
    return calls


def test_riva_requests_both_hypotheses_once_and_cache_is_bound(tmp_path, monkeypatch):
    rows = [row(0, "他说", 1, 2), row(1, "向西走", 2.1, 3)]
    calls = fake_worker(monkeypatch, {rows[0]["id"]: "He said.", rows[1]["id"]: "Go west."})
    cfg = TranslationConfig(backend="riva", connected_hypotheses=True)
    gl = GlossaryConfig(translation_examples=(("他说", "he said"), ("向西", "west")))
    ckpt = tmp_path / "mt.json"
    out, summary = translate_utterances(rows, source_language="zh", config=cfg,
                                       glossary=gl, checkpoint_path=ckpt)
    assert len(calls) == 1 and len(calls[0]["items"]) == 3
    assert summary["connected_candidate_count"] == 1
    assert len(calls[0]["items"][-1]["context_examples"]) == 2
    assert [r["machine_translated_text"] for r in out] == ["He said.", "Go west."]
    assert cu.groups(out) == [[0, 1]]
    again, summary = translate_utterances(rows, source_language="zh", config=cfg,
                                         glossary=gl, checkpoint_path=ckpt)
    assert len(calls) == 1 and summary["cache_hits"] == 3
    assert again[0]["translation_unit_candidate"] == out[0]["translation_unit_candidate"]
    separate, summary = translate_utterances(rows, source_language="zh",
        config=replace(cfg, connected_hypotheses=False), glossary=gl, resume=False)
    assert len(calls[-1]["items"]) == 2 and not summary["connected_candidate_count"]
    assert not any("translation_unit_candidate" in r for r in separate)


def test_native_pipeline_and_offline_render_use_exact_unit_lineage(tmp_path, monkeypatch):
    from nvidia_transcribe.moss_recording import build_native_recording
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles

    recording = build_native_recording({"path": "clip.wav"}, "zh", {"model": "local"},
        {"text": "[1][S01]他说[2][2.1][S01]向西走[3]",
         "generated_tokens": 120, "max_new_tokens": 4096, "prompt_len": 200,
         "max_input_tokens": 131072}, 30.0, "audio")
    raw = tmp_path / "recording.contract.json"
    raw.write_text(json.dumps(recording), encoding="utf-8")
    a, b = [r["id"] for r in recording["utterances"]]
    calls = fake_worker(monkeypatch, {a: "He said.", b: "Go west."})
    client = Review({a: "He said to", b: "go west."})
    cfg = SubtitlePipelineConfig(translation=TranslationConfig(backend="riva"),
        translation_review=config(), source_quality="off")
    outdir = tmp_path / "job"
    meta = generate_subtitles(recording_contract_path=raw, output_dir=outdir, config=cfg,
                              text_client=client)
    assert len(calls) == 1 and len(calls[0]["items"]) == 3
    assert meta["translation_review"]["jointly_adjudicated_count"] == 1
    assert {c["occurrence_id"] for c in meta["cue_lineage"]} == {a, b}
    assert len({c["translation_unit_id"] for c in meta["cue_lineage"]}) == 1
    assert all(c["unit_occurrence_ids"] == [a, b] for c in meta["cue_lineage"])
    assert {tuple(c["actual_event_interval"]) for c in meta["cue_lineage"]} == {(1, 2), (2.1, 3)}
    path = outdir / "translated.contract.json"
    retained = path.read_bytes()
    again = generate_subtitles(output_dir=outdir, config=replace(cfg, re_render_only=True))
    assert path.read_bytes() == retained and len(calls) == 1
    assert again["cue_lineage"] == meta["cue_lineage"]
    contract = json.loads(retained)
    broken = deepcopy(contract)
    broken["utterances"][1].pop("translation_unit")
    with pytest.raises(ValueError, match="occurrence mismatch"):
        render_subtitles(broken, output_dir=outdir, episode_stem="invalid")
    overlaid, summary = apply_validation_overlay(contract, {"entries": [
        {"id": a, "translated_text": "He told us to"}]})
    assert len(summary["invalidated_translation_units"]) == 1
    assert all("translation_unit" not in r and "pre_overlay_translation_unit" in r
               for r in overlaid["utterances"])
    render_subtitles(overlaid, output_dir=outdir, episode_stem="operator_override")
