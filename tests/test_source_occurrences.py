"""CPU regressions for the recognized lexical spine and immutable occurrence IDs."""
from types import SimpleNamespace

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
    rec = recover(raw, [word("first", 1), word("again", 2), word("answer", 5, "1")])
    inv = bind(rec, raw)
    units = rec["units"]
    assert [u["text"] for u in units] == ["first", "answer", "again"]
    assert [u["start"] for u in units] == [1, 5, 2]
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
        ctc_payload={"words": [word("first", 1), word("again", 2),
                                word("extra", 3), word("answer", 5, "1")]})
    assert {u["source_text"] for u in contract["utterances"]} == {"first", "answer", "again"}
    assert len(contract["native_ctc_observations"]) == 4
    assert contract["native_ctc_observations"][2]["association_status"] == "unmatched"
