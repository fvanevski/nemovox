"""CPU regressions for the recognized lexical spine and immutable occurrence IDs."""
from types import SimpleNamespace

import pytest

from nvidia_transcribe.alignment import recover_chunk_alignment
from nvidia_transcribe.source_occurrences import bind_occurrence_inventory, source_observation_id
from nvidia_transcribe.utterances import reconstruct_speaker_utterances


def word(text, start, speaker="0"):
    return {"word": text, "speaker": speaker, "start": start, "end": start + .2}


def recover(text, words, error=None):
    return recover_chunk_alignment(raw_text=text, ctc_payload={"words": words, "error": error},
                                   source_offset_s=0, chunk_index=0, model_path=None)


def bind(rec, raw, projected=None):
    spec = SimpleNamespace(chunk_index=0, infer_start_s=0, infer_end_s=20,
                           core_start_s=0, core_end_s=20)
    return bind_occurrence_inventory(observation_id=source_observation_id("media", "a"*64, spec),
        raw_text=raw, projected_text=raw if projected is None else projected, units=rec["units"],
        native_observations=rec["diagnostic"]["native_ctc_observations"])


def test_noncontiguous_same_tag_has_distinct_turns_and_provisional_times():
    raw = "<spk:0> first <spk:1> answer <spk:0> again"
    rec = recover(raw, [word("first", 1), word("again", 2.5), word("answer", 5, "1")])
    inv = bind(rec, raw)
    units = rec["units"]
    assert [u["text"] for u in units] == ["first", "answer", "again"]
    assert [u["start"] for u in units] == [1, 5, 2.5]
    assert len({u["recognized_turn_id"] for u in units}) == 3
    assert all(u["ctc_association_status"].endswith("provisional") for u in units)
    assert not inv["acoustic_locality_certified"]
    assert {o["source_unit_id"] for o in inv["native_ctc_observations"]} == {u["source_unit_id"] for u in units}
    ids = {u["source_text"]: u["id"] for u in reconstruct_speaker_utterances(units)}
    assert ids == {u["source_text"]: u["id"] for u in reconstruct_speaker_utterances(list(reversed(units)))}


def test_partial_repeated_token_is_not_guessed_and_extra_ctc_is_retained():
    raw = "<spk:0> yes yes encore"
    rec = recover(raw, [word("yes", 1), word("extra", 2), word("encore", 3)])
    inv = bind(rec, raw)
    assert [u["text"] for u in rec["units"]] == ["yes", "yes", "encore"]
    assert [u["start"] for u in rec["units"]] == [None, None, 3]
    assert len(inv["native_ctc_observations"]) == 3
    assert inv["native_ctc_observations"][0]["association_status"] == "ambiguous_sequence"
    assert inv["native_ctc_observations"][1]["association_status"] == "unmatched"


def test_ctc_error_and_loop_projection_keep_raw_inventory():
    raw = "<spk:0> 你好。 <spk:0> 好"
    rec = recover(raw, [word("你", 1), word("好", 2)], error="alignment_failed")
    inv = bind(rec, raw)
    assert all(u["start"] is None for u in rec["units"])
    assert len(inv["recognized_turns"]) == 2
    assert len(inv["native_ctc_observations"]) == 2
    assert all(o["association_status"] == "native_error" for o in inv["native_ctc_observations"])
    assert {t["id"] for turn in inv["recognized_turns"] for t in turn["tokens"]} == {
        tid for unit in rec["units"] for tid in unit["raw_recognized_token_ids"]}
    changed = recover("<spk:0> 你好。", [])
    inv2 = bind(changed, raw, "<spk:0> 你好。")
    assert inv2["projection_status"] == "loop_cleaned_unmapped"
    assert len(inv2["recognized_turns"]) == 2
    assert all(not u["raw_recognized_token_ids"] for u in changed["units"])


def test_single_request_contract_uses_recognized_text_and_preserves_extras():
    from nvidia_transcribe.contract import build_transcript_contract

    raw = "<spk:0> first <spk:1> answer <spk:0> again"
    contract = build_transcript_contract(raw_text=raw, source_language="fr",
        ctc_payload={"words": [word("first", 1), word("again", 2.5),
                                word("extra", 3), word("answer", 5, "1")]})
    assert {u["source_text"] for u in contract["utterances"]} == {"first", "answer", "again"}
    assert len(contract["native_ctc_observations"]) == 4
    assert contract["native_ctc_observations"][2]["association_status"] == "unmatched"


def test_overlapping_sot_fragments_coalesce_without_losing_turn_identity():
    raw = "<spk:0> hello <spk:1> good <spk:0> there <spk:1> morning"
    rec = recover(raw, [word("hello", 1), word("good", 1.1, "1"),
                        word("there", 1.3), word("morning", 1.4, "1")])
    bind(rec, raw)
    rows = reconstruct_speaker_utterances(rec["units"])
    assert {r["source_text"] for r in rows} == {"hello there", "good morning"}
    assert all(len(r["recognized_turn_ids"]) == 2 for r in rows)
    assert {uid for r in rows for uid in r["source_unit_ids"]} == {
        u["source_unit_id"] for u in rec["units"]}
    assert max(r["source_start"] for r in rows) < min(r["source_end"] for r in rows)


def test_locality_long_pause_is_review_only_and_never_drops_text():
    from nvidia_transcribe.occurrence_locality import assess_occurrence_locality
    raw = "<spk:0> hello there goodbye friend"
    rec = recover(raw, [word("hello", 1), word("there", 1.3),
                        word("goodbye", 14), word("friend", 14.3)])
    bind(rec, raw)
    audited, report = assess_occurrence_locality(rec["units"], infer_start_s=0, infer_end_s=20)
    assert all(u["valid_timing"] for u in audited)
    assert len(report["spans"]) == 2
    assert [f["reason"] for f in report["findings"]] == ["large_intra_turn_gap_review_only"]
    assert {uid for span in report["spans"] for uid in span["source_unit_ids"]} == {
        u["source_unit_id"] for u in rec["units"]}


def test_locality_quarantines_compressed_phrase_before_optional_alignment():
    from nvidia_transcribe.occurrence_locality import assess_occurrence_locality
    from nvidia_transcribe.selective_alignment import AdmissionPolicy, plan_selective_recovery
    words = ["marvellous", "mountains", "ancient", "gardens", "extraordinary",
             "journeys", "familiar", "travelers", "captivating", "beautiful", "children"]
    raw = "<spk:0> " + " ".join(words)
    rec = recover(raw, [word(t, 5+i*.105) for i,t in enumerate(words)])
    bind(rec, raw)
    audited, report = assess_occurrence_locality(rec["units"], infer_start_s=0, infer_end_s=20)
    assert [u["text"] for u in audited] == words
    assert all(not u["valid_timing"] and u["requires_acoustic_recovery"] for u in audited)
    assert all(u["ctc_hypothesis_start_s"] is not None for u in audited)
    assert report["spans"][0]["locality_status"] == "disputed"
    plans, decisions = plan_selective_recovery(
        native={**rec, "units": audited}, raw_text=raw, chunk_start_s=0, chunk_end_s=20,
        policy=AdmissionPolicy())
    assert not plans
    assert decisions[0]["status"] == "skipped_disputed_occurrence_requires_acoustic_evidence"


def test_coherent_boundary_span_not_truncated_and_context_alternative_retained():
    from nvidia_transcribe.occurrence_locality import assess_occurrence_locality
    from nvidia_transcribe.chunking import ChunkSpec, reconcile_chunk_boundaries
    left = ChunkSpec(0, 0, 10, 0, 14)
    right = ChunkSpec(1, 10, 20, 6, 20)
    raw = "<spk:0> the city remains safe"
    rec = recover(raw, [word("the", 9), word("city", 9.3),
                        word("remains", 9.6), word("safe", 10.1)])
    bind(rec, raw)
    units, report = assess_occurrence_locality(rec["units"], infer_start_s=0, infer_end_s=14)
    merged, diag = reconcile_chunk_boundaries([units, []], [left, right])
    assert [u["text"] for u in merged] == ["the", "city", "remains", "safe"]
    assert len(diag["source_dispositions"]) == 4
    assert all(d["status"] == "selected_provisional" for d in diag["source_dispositions"])
    rec2 = recover("<spk:0> uncorroborated", [word("uncorroborated", 13.5)])
    bind(rec2, "<spk:0> uncorroborated")
    context, _ = assess_occurrence_locality(rec2["units"], infer_start_s=0, infer_end_s=14)
    merged, diag = reconcile_chunk_boundaries([context, []], [left, right])
    assert not merged
    assert diag["source_dispositions"][0]["status"] == "context_candidate_unresolved"
    assert diag["source_dispositions"][0]["source_unit_id"] == context[0]["source_unit_id"]


def test_two_real_repeats_keep_both_core_occurrences_with_bound_references():
    from nvidia_transcribe.chunking import ChunkSpec, reconcile_chunk_boundaries
    from nvidia_transcribe.occurrence_locality import assess_occurrence_locality
    words = ["we", "must", "leave", "this", "place", "now"]
    raw = "<spk:0> " + " ".join(words) + " <spk:0> " + " ".join(words)
    specs = [ChunkSpec(0, 0, 10, 0, 14), ChunkSpec(1, 10, 20, 6, 20)]
    chunks = []
    for spec, shift in zip(specs, (0, .1)):
        native = recover_chunk_alignment(raw_text=raw, ctc_payload={"words": [
            word(t, onset+i*.28+shift) for onset in (8, 10)
            for i, t in enumerate(words)]}, source_offset_s=0, chunk_index=spec.chunk_index,
            model_path=None)
        bind_occurrence_inventory(observation_id=source_observation_id("media",
            str(spec.chunk_index+1)*64, spec), raw_text=raw, projected_text=raw,
            units=native["units"], native_observations=native["diagnostic"]["native_ctc_observations"])
        units, _ = assess_occurrence_locality(native["units"],
            infer_start_s=spec.infer_start_s, infer_end_s=spec.infer_end_s)
        for u in units:
            u["global_speaker_id"] = "SPK_A"
        chunks.append(units)
    merged, diag = reconcile_chunk_boundaries(chunks, specs)
    assert [u["text"] for u in merged] == words + words
    assert all(u["valid_timing"] for u in merged)
    selected = {u["source_unit_id"] for u in merged}
    decisions = diag["source_dispositions"]
    assert len(decisions) == 24
    assert {d["source_unit_id"] for d in decisions if d["status"].startswith("selected_")} == selected
    alternatives = [d for d in decisions if d["status"] == "alternate_not_selected"]
    assert len(alternatives) == 12
    assert all(d["related_source_unit_ids"] and not d["occurrence_truth_accepted"]
               for d in alternatives)


def test_optional_alignment_refresh_preserves_span_identity():
    from nvidia_transcribe.occurrence_locality import assess_occurrence_locality, refresh_span_ownership
    raw = "<spk:0> missing timing"
    rec = recover(raw, [])
    bind(rec, raw)
    units, report = assess_occurrence_locality(rec["units"], infer_start_s=0, infer_end_s=20)
    before = [u["occurrence_span_id"] for u in units]
    for i, unit in enumerate(units):
        unit.update(start=2+i*.3, end=2.2+i*.3, valid_timing=True,
                    alignment_status="external_qwen_recovered")
    refresh_span_ownership(units, report)
    assert [u["occurrence_span_id"] for u in units] == before
    assert all(u["occurrence_bounds_s"] == [2, 2.5] for u in units)
    assert report["spans"][0]["locality_status"] == "unlocalized"


def test_boundary_dispositions_do_not_assert_transitive_or_ownerless_coverage():
    from nvidia_transcribe.occurrence_dispositions import finalize_boundary_dispositions

    def row(uid, status, refs=()):
        return {"source_unit_id": uid, "occurrence_span_id": "span-"+uid,
                "status": status, "reason": "original_reason",
                "related_source_unit_ids": list(refs),
                "timing_truth_accepted": False, "occurrence_truth_accepted": False}
    rows = [row("a", "alternate_not_selected", ["b"]),
            row("b", "alternate_not_selected", ["c"]),
            row("c", "selected_unlocalized"),
            row("d", "alternate_not_selected", ["e"]),
            row("e", "context_candidate_unresolved"),
            row("f", "alternate_not_selected"),
            row("g", "selected_provisional")]
    result = finalize_boundary_dispositions(rows)
    by_id = {d["source_unit_id"]: d for d in result}
    assert {uid for uid,d in by_id.items() if d["status"] == "context_candidate_unresolved"} == {"a","d","e","f"}
    assert by_id["b"] == rows[1]
    assert by_id["c"] == rows[2] and by_id["g"] == rows[-1]
    for i in (0,3,5):
        actual = result[i]
        assert actual["related_source_unit_ids"] == rows[i]["related_source_unit_ids"]
        assert actual["prior_selection_reason"] == rows[i]["reason"]
        assert actual["reason"] == "provisional_correspondence_without_selected_counterpart"
        assert rows[i]["status"] == "alternate_not_selected"
        assert not actual["occurrence_truth_accepted"] and not actual["timing_truth_accepted"]
    assert finalize_boundary_dispositions(result) == result


def test_boundary_matching_two_context_spans_exposes_unselected_counterpart():
    from nvidia_transcribe.chunking import ChunkSpec, reconcile_chunk_boundaries
    from nvidia_transcribe.occurrence_locality import assess_occurrence_locality

    specs = [ChunkSpec(0,0,10,0,14), ChunkSpec(1,10,20,6,20)]
    chunks = []
    for spec, raw, native_words in (
        (specs[0], "<spk:0> again now", [word("again",10.2),word("now",10.5)]),
        (specs[1], "<spk:0> before again", [word("before",9.4),word("again",10.2)]),
    ):
        native = recover_chunk_alignment(raw_text=raw, ctc_payload={"words":native_words},
            source_offset_s=0, chunk_index=spec.chunk_index, model_path=None)
        bind_occurrence_inventory(observation_id=source_observation_id("media",
            str(spec.chunk_index+1)*64,spec), raw_text=raw, projected_text=raw,
            units=native["units"], native_observations=native["diagnostic"]["native_ctc_observations"])
        units, _ = assess_occurrence_locality(native["units"],
            infer_start_s=spec.infer_start_s,infer_end_s=spec.infer_end_s)
        for u in units:
            u["global_speaker_id"] = "SPK_A"
        chunks.append(units)
    left, right = chunks[0][0], chunks[1][-1]
    assert left["occurrence_bounds_s"][0] > 10
    assert sum(right["occurrence_bounds_s"])/2 < 10
    merged, diag = reconcile_chunk_boundaries(chunks,specs)
    selected = {u["source_unit_id"] for u in merged}
    decisions = {d["source_unit_id"]: d for d in diag["source_dispositions"]}
    assert left["source_unit_id"] not in selected and right["source_unit_id"] not in selected
    assert decisions[left["source_unit_id"]]["status"] == "context_candidate_unresolved"
    assert decisions[left["source_unit_id"]]["related_source_unit_ids"] == [right["source_unit_id"]]
    assert decisions[right["source_unit_id"]]["status"] == "context_candidate_unresolved"
    assert len(decisions) == 4
    assert all(d["related_source_unit_ids"] and set(d["related_source_unit_ids"]) <= selected
               for d in decisions.values() if d["status"] == "alternate_not_selected")


def test_boundary_disposition_invalid_reference_fails_closed():
    import pytest
    from nvidia_transcribe.occurrence_dispositions import finalize_boundary_dispositions

    row = {"source_unit_id":"a", "status":"alternate_not_selected",
           "related_source_unit_ids":["missing"], "reason":"test"}
    with pytest.raises(ValueError, match="Invalid boundary correspondence"):
        finalize_boundary_dispositions([row])
    with pytest.raises(ValueError, match="Duplicate source-unit"):
        finalize_boundary_dispositions([row,row])


@pytest.mark.parametrize("labels", [("0",)*4, ("<spk:0>",)*4, (None,"0",None,"1"), (None,)*4])
def test_untagged_recognition_retains_ctc_timing_without_speaker_attribution(labels):
    from nvidia_transcribe.contract import build_transcript_contract
    from nvidia_transcribe.occurrence_locality import assess_occurrence_locality
    from nvidia_transcribe.speakers import normalize_chunk_speakers

    raw = "Bonjour tout le monde"
    words = [word(text, 1+i*.3, label)
             for i, (text, label) in enumerate(zip(raw.split(), labels))]
    rec = recover(raw, words)
    inv = bind(rec, raw)
    units = rec["units"]
    assert [u["text"] for u in units] == raw.split()
    assert [u["start"] for u in units] == [w["start"] for w in words]
    assert all(u["valid_timing"] and u["speaker"] is None and u["raw_speaker_tag"] is None
               for u in units)
    assert [u["ctc_observed_speaker"] for u in units] == list(labels)
    method = ("complete_untagged_sequence_provisional" if any(labels)
              else "complete_per_tag_sequence_provisional")
    assert all(u["ctc_association_status"] == method for u in units)
    observations = inv["native_ctc_observations"]
    assert [o["raw_word"] for o in observations] == words
    assert [o["source_unit_id"] for o in observations] == [u["source_unit_id"] for u in units]
    assert len({o["source_unit_id"] for o in observations}) == 4
    assert not inv["acoustic_locality_certified"]
    audited, locality = assess_occurrence_locality(units, infer_start_s=0, infer_end_s=20)
    assert all(u["valid_timing"] for u in audited)
    assert all(not span["acoustic_truth_accepted"] for span in locality["spans"])
    normalized, profiles = normalize_chunk_speakers(chunk_index=0, units=audited,
        diarization=[], speaker_mapping={"0": 0, "1": 1})
    assert list(profiles) == ["chunk_000:unknown"]
    assert all(u["attribution_status"] == "unknown" and u["diarization_speaker_id"] is None
               for u in normalized)
    result = build_transcript_contract(raw_text=raw, source_language="fr",
        ctc_payload={"words": words}, source_offset_s=100.)
    row, = result["utterances"]
    assert row["source_text"] == raw and row["speaker_id"] == "unknown"
    assert row["source_start"] == 101. and row["source_end"] == 102.1
    assert all(u["speaker"] is None for u in row["source_words"])
    assert [u["ctc_observed_speaker"] for u in row["source_words"]] == list(labels)


def test_untagged_partial_stream_uses_all_native_labels_for_ambiguity():
    raw = "yes yes encore fin"
    words = [word("yes",1,None), word("extra",2,"0"), word("encore",3,"1"),
             word("fin",4,"0")]
    rec = recover(raw, words)
    inv = bind(rec, raw)
    assert [u["start"] for u in rec["units"]] == [None,None,3,4]
    assert all(u["speaker"] is None for u in rec["units"])
    assert [o["association_status"] for o in inv["native_ctc_observations"]] == [
        "ambiguous_sequence","unmatched","partial_untagged_unique_ordered_anchor_provisional",
        "partial_untagged_unique_ordered_anchor_provisional"]
    # A duplicate native word under another CTC label also competes.
    duplicate = recover("yes encore", [word("yes",1,None),word("yes",2,"0"),word("encore",3,"1")])
    assert [u["start"] for u in duplicate["units"]] == [None,3]
    # Crossing anchors remain ambiguous even when labels differ.
    reversed_native = recover("yes encore", [word("encore",1,"0"),word("yes",2,"1")])
    assert all(not u["valid_timing"] for u in reversed_native["units"])


def test_untagged_fallback_does_not_steal_explicit_tagged_observation():
    rec = recover("bonjour <spk:0> encore",
                  [word("bonjour",1,"0"),word("encore",2,"0")])
    inv = bind(rec, "bonjour <spk:0> encore")
    assert [u["start"] for u in rec["units"]] == [None,2]
    assert [u["speaker"] for u in rec["units"]] == [None,"0"]
    assert "source_unit_id" not in inv["native_ctc_observations"][0]
    assert inv["native_ctc_observations"][1]["source_unit_id"] == rec["units"][1]["source_unit_id"]


def test_untagged_native_error_and_invalid_interval_do_not_become_usable():
    words = [word("bonjour",1), {**word("encore",2), "end":1.5}]
    rec = recover("bonjour encore", words)
    assert [u["valid_timing"] for u in rec["units"]] == [True,False]
    assert rec["units"][1]["start"] is None and rec["units"][1]["end"] is None
    failed = recover("bonjour encore", words, error="alignment_failed")
    assert all(not u["valid_timing"] for u in failed["units"])


def test_untagged_cjk_localized_observation_keeps_order_and_unknown_voice():
    from nvidia_transcribe.acoustic_observations import reconstruct_acoustic_observation
    raw = {"choices":[{"message":{"content":"你好世界"},"finish_reason":"stop"}],
           "ctc_timestamps":{"words":[word(char,1+i*.3) for i,char in enumerate("你好世界")]}}
    rows, inv = reconstruct_acoustic_observation(raw, audio_sha256="a"*64,
                                                 crop=(100.,110.), language="zh")
    row, = rows
    assert row["source_text"] == "你好世界" and row["speaker_id"] == "unknown"
    assert row["source_start"] == 101. and row["source_end"] == 102.1
    assert len(row["source_characters"]) == 4 and not row["source_words"]
    assert all(u["speaker"] is None for u in row["source_characters"])
    assert [u["text"] for u in row["source_characters"]] == list("你好世界")
    assert row["source_unit_ids"] == sorted(inv["working_source_unit_ids"])
