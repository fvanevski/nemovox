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
                "corrected_text": self.proposals[t["utterance_id"]], "reason": "fixture source meaning",
                "discourse_terms": [], "evidence_request": {"kind":"none","reference_id":"","question":"","reason":"fixture text sufficient","focus":"none"}}
                for t in payload["target_utterances"]]}
        else:
            body = {"decisions": {c["key"]: {
                "choice": "original" if (self.reject_joint if c["atomic"] else c["target_ids"][0] in self.reject) else "proposal",
                "reason": "fixture complete allocation comparison" if c["atomic"] else "fixture comparison"}
                for c in payload["comparisons"]}}
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
    assert len(client.calls[1]["comparisons"])==1
    assert client.calls[1]["comparisons"][0]["atomic"] is True
    assert client.calls[1]["comparisons"][0]["target_ids"]==[r["id"] for r in rows]
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
    assert {c["target_ids"][0] for c in comparison["comparisons"] if not c["atomic"]} == {rows[2]["id"], rows[3]["id"]}
    assert [c["target_ids"] for c in comparison["comparisons"] if c["atomic"]] == [[rows[0]["id"], rows[1]["id"]]]
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


@pytest.mark.parametrize("case,new_interval,retire_first", [
    ("intervening", (1.5, 1.8), True),
    ("recovered", (1.5, 1.8), True),
    ("overlap_before", (0.5, 1.5), True),
    ("overlap_after", (2.9, 3.5), True), # starts after the last member, overlaps its end
    ("touching", (3, 3.5), False),
    ("outside", (4.5, 5.5), False),
])
def test_overlay_reordering_preserves_or_retires_exact_joint_units(
        tmp_path, monkeypatch, case, new_interval, retire_first):
    from nvidia_transcribe import subtitle_pipeline as pipeline
    rows = [
        row(0, "他说", 1, 2, "He said."),
        row(1, "向西走", 2.1, 3, "Go west."),
        row(2, "别走。", 4, 5, "Don't leave.", speaker="S02"),
        row(3, "他要", 7, 8, "He wants."),
        row(4, "回家", 8.1, 9, "Go home."),
    ]
    if case == "recovered":
        rows[2]["source_start"] = rows[2]["source_end"] = None
        rows[2]["timestamp_provenance"]["timing_status"] = "unavailable"
        rows.append(rows.pop(2))
    rows = attach(rows)
    proposals = {r["id"]: r["translated_text"] for r in rows}
    proposals.update({"event-00000":"He said to", "event-00001":"go west.",
                      "event-00003":"He wants to", "event-00004":"go home."})
    selected, report = review(rows, Review(proposals))
    assert report["status"] == "completed" and report["jointly_adjudicated_count"] == 2
    contract = {"source":{"path":"clip.wav","duration_s":12},"source_language":"zh",
                "target_language":"en","utterances":selected,"translation_review":report}
    before = deepcopy(contract)
    by_id = {r["id"]:r for r in selected}
    a,b,c,d,e = (f"event-{i:05d}" for i in range(5))
    first_unit = by_id[a]["translation_unit"]
    second_unit = by_id[d]["translation_unit"]
    assert first_unit == by_id[b]["translation_unit"]
    assert second_unit == by_id[e]["translation_unit"]
    assert len(first_unit["constituents"]) == len(second_unit["constituents"]) == 2
    cu.validate_selected_units(selected, require_complete=True)
    outdir = tmp_path/"job"
    outdir.mkdir()
    path = outdir/"translated.contract.json"
    path.write_text(json.dumps(contract),encoding="utf-8")
    overlay = {"entries":[{"id":c,"proposed_start_s":new_interval[0],
                            "proposed_end_s":new_interval[1]}]}
    overlay_path = tmp_path/"overlay.json"
    overlay_path.write_text(json.dumps(overlay),encoding="utf-8")
    def forbidden(*args,**kwargs):
        pytest.fail("offline overlay must not invoke inference")
    monkeypatch.setattr(pipeline,"translate_utterances",forbidden)
    monkeypatch.setattr(pipeline,"review_translated_utterances",forbidden)
    cfg = pipeline.SubtitlePipelineConfig(re_render_only=True,
        validation_overlay_path=overlay_path)
    meta = pipeline.generate_subtitles(output_dir=outdir,config=cfg)
    overlaid = json.loads(path.read_text())
    direct, summary = apply_validation_overlay(contract,overlay)
    assert direct == overlaid and contract == before
    assert meta["validation_overlay"] == summary
    expected = {by_id[c]["translation_unit"]["unit_id"]}
    # Strict interval overlap matters even when adjacency remains.
    if retire_first:
        expected.add(first_unit["unit_id"])
    assert set(summary["invalidated_translation_units"]) == expected
    after = {r["id"]:r for r in overlaid["utterances"]}
    assert set(after) == set(by_id)
    for uid in (a,b,c,d,e):
        for field in ("id","source_text","speaker_id","machine_translated_text",
                      "translated_text","translation_review_status"):
            assert after[uid][field] == by_id[uid][field]
        if uid != c:
            assert cu.interval(after[uid]) == cu.interval(by_id[uid])
            assert after[uid]["timestamp_provenance"] == by_id[uid]["timestamp_provenance"]
        if by_id[uid]["translation_unit"]["unit_id"] in expected:
            assert "translation_unit" not in after[uid]
            assert after[uid]["pre_overlay_translation_unit"] == by_id[uid]["translation_unit"]
        else:
            assert after[uid] == by_id[uid]
    if case != "recovered":
        assert (after[c]["raw_source_start"],after[c]["raw_source_end"]) == (4,5)
    assert cu.interval(after[c]) == list(new_interval)
    cu.validate_selected_units(overlaid["utterances"], require_complete=True)
    lineage = meta["cue_lineage"]
    assert {r["occurrence_id"] for r in lineage} == set(by_id)
    for entry in lineage:
        uid = entry["occurrence_id"]
        assert entry["actual_event_interval"] == cu.interval(by_id[uid])
        assert entry["effective_event_interval"] == cu.interval(after[uid])
        active = after[uid].get("translation_unit")
        assert entry["translation_unit_id"] == (active["unit_id"] if active else None)
        assert entry["unit_occurrence_ids"] == (
            [r["id"] for r in active["constituents"]] if active else [uid])
    retained = path.read_bytes()
    files = {fmt: (outdir / f"clip.en.{fmt}").read_bytes() for fmt in ("srt","ass")}
    again = pipeline.generate_subtitles(output_dir=outdir,
        config=replace(cfg,validation_overlay_path=None))
    assert path.read_bytes() == retained and again["cue_lineage"] == lineage
    assert all((outdir / f"clip.en.{fmt}").read_bytes() == data for fmt,data in files.items())


@pytest.mark.parametrize("change",[
    {"proposed_start_s":1.2,"proposed_end_s":2},
    {"translated_text":"He told us to"},
    {"suppress":True},
])
def test_overlay_direct_constituent_change_retires_complete_unit(tmp_path,change):
    rows=attach([row(0,"他说",1,2,"He said."),row(1,"向西走",2.1,3,"Go west.")])
    a,b=[r["id"] for r in rows]
    selected,report=review(rows,Review({a:"He said to",b:"go west."}))
    contract={"utterances":selected,"translation_review":report}
    before=deepcopy(contract)
    overlaid,summary=apply_validation_overlay(contract,{"entries":[{"id":a,**change}]})
    unit=selected[0]["translation_unit"]
    assert summary["invalidated_translation_units"]==[unit["unit_id"]]
    assert all(r.get("pre_overlay_translation_unit")==unit and "translation_unit" not in r
               for r in overlaid["utterances"])
    assert contract==before
    assert all(r["machine_translated_text"]==old["machine_translated_text"]
               for r,old in zip(overlaid["utterances"],selected))
    cu.validate_selected_units(overlaid["utterances"],require_complete=True)
    render_subtitles(overlaid,output_dir=tmp_path,episode_stem="direct")


def test_invalid_overlay_result_rejected_before_contract_publication(tmp_path,monkeypatch):
    from nvidia_transcribe import subtitle_pipeline as pipeline
    rows=attach([row(0,"他说",1,2,"He said."),row(1,"向西走",2.1,3,"Go west.")])
    a,b=[r["id"] for r in rows]
    selected,report=review(rows,Review({a:"He said to",b:"go west."}))
    contract={"source":{"path":"clip.wav"},"utterances":selected,"translation_review":report}
    path=tmp_path/"translated.contract.json"
    path.write_text(json.dumps(contract),encoding="utf-8")
    cfg=pipeline.SubtitlePipelineConfig(re_render_only=True)
    pipeline.generate_subtitles(output_dir=tmp_path,config=cfg)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    overlay_path=tmp_path/"overlay.json"
    overlay_path.write_text(json.dumps({"entries":[{"id":a,"reason":"fixture"}]}))
    def broken(contract,overlay):
        result,summary=apply_validation_overlay(contract,overlay)
        result["utterances"][1].pop("translation_unit")
        return result,summary
    monkeypatch.setattr(pipeline,"apply_validation_overlay",broken)
    def forbidden(*args,**kwargs):
        pytest.fail("invalid overlay must be rejected before render")
    monkeypatch.setattr(pipeline,"render_subtitles",forbidden)
    with pytest.raises(ValueError,match="Selected unit occurrence mismatch"):
        pipeline.generate_subtitles(output_dir=tmp_path,
            config=replace(cfg,validation_overlay_path=overlay_path))
    assert all((tmp_path/name).read_bytes()==data for name,data in before.items())
    assert {p.name for p in tmp_path.iterdir() if p.is_file()}==set(before)|{"overlay.json"}
