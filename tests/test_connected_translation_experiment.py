from copy import deepcopy
import pytest

from experiments.connected_translation import (
    adapter_row, make_unit, partition, validate_riva,
)


def row(i, text, start, end, speaker="S0"):
    return {"id": f"event-{i:05d}", "source_text": text, "speaker_id": speaker,
            "source_start": start, "source_end": end,
            "timestamp_provenance": {"native_event_index": i}}


def test_connected_unit_preserves_occurrences_and_actual_intervals():
    rows = [row(0, "Nous allons", 1, 2), row(1, "vers l'ouest.", 2.1, 3)]
    before = deepcopy(rows)
    groups, edges = partition(rows, rows, "strict")
    assert len(groups) == 1 and edges[0]["reason"] == "eligible"
    unit = make_unit(groups[0])
    assert unit["occurrence_ids"] == [r["id"] for r in rows]
    assert unit["supported_intervals"] == [(1, 2), (2.1, 3)]
    projected = adapter_row(unit, "We're going west.")
    assert projected["grouped_utterance_ids"] == unit["occurrence_ids"]
    assert projected["timestamp_provenance"]["timing_status"] == "experimental_display_envelope"
    assert rows == before


@pytest.mark.parametrize("second", [
    row(1, "suite", 2.1, 3, "S1"),
    row(1, "suite", None, None),
    row(2, "suite", 2.1, 3),
    row(1, "suite", 1.5, 3),
    row(1, "suite", 4, 5),
    row(1, "début", 2.1, 3),
    {**row(1, "suite", 2.1, 3), "semantic_context": {"unresolved": True}},
])
def test_protected_boundaries(second):
    rows = [row(0, "début", 1, 2), second]
    for variant in ("strict", "expanded"):
        assert len(partition(rows, rows, variant)[0]) == 2


def test_overlapping_event_outside_window_blocks_grouping():
    rows = [row(0, "début", 1, 2), row(1, "suite", 2.1, 3)]
    all_rows = [*rows, row(2, "other", 0.5, 2.5, "S1")]
    assert len(partition(rows, all_rows, "strict")[0]) == 2


def test_candidate_difference_and_cjk():
    rows = [row(0, "他说。", 1, 2), row(1, "向西走", 2.1, 3)]
    assert len(partition(rows, rows, "strict")[0]) == 2
    assert len(partition(rows, rows, "expanded")[0]) == 1
    rows[0]["source_text"] = "他说"
    assert make_unit(rows)["source_text"] == "他说向西走"


def test_riva_receipt_rejects_wrong_ids_or_examples():
    plan = {"requests": [{"utterance_id": "a", "context_examples": [{"id": "x"}]}]}
    good = {"status": "ok", "device_used": "cuda:0", "results": [
        {"utterance_id": "a", "translated_text": "west", "context_example_ids": ["x"],
         "input_tokens": 20}]}
    assert validate_riva(plan, good, "cuda:0") == {"a": "west"}
    for key, value in (("utterance_id", "b"), ("context_example_ids", []),
                       ("translated_text", ""), ("input_tokens", True)):
        bad = deepcopy(good)
        bad["results"][0][key] = value
        with pytest.raises(ValueError):
            validate_riva(plan, bad, "cuda:0")
    bad = deepcopy(good)
    bad["results"] *= 2
    with pytest.raises(ValueError):
        validate_riva(plan, bad, "cuda:0")


def test_plan_native_contract_controls_examples_and_complete_occurrence_inventory():
    from nvidia_transcribe.moss_recording import build_native_recording
    from nvidia_transcribe.refinement import GlossaryConfig
    from experiments.connected_translation import plan_windows

    contract = build_native_recording(
        {"path": "clip.wav"}, "zh", {"model": "local"},
        {"text": "[1][S01]他说[2][2.1][S01]向西走[3][4][S02]好！[5]",
         "generated_tokens": 120, "max_new_tokens": 4096,
         "prompt_len": 200, "max_input_tokens": 131072},
        30.0, "audio",
    )
    rows = contract["utterances"]
    before = deepcopy(contract)
    glossary = GlossaryConfig(translation_examples=(("他说", "he said"), ("向西", "west")))
    plan = plan_windows(contract, [(rows[0]["id"], rows[-1]["id"])], glossary)
    case = plan["cases"][0]
    requests = {r["utterance_id"]: r for r in plan["requests"]}
    units = case["variants"]["strict"]["units"]
    assert len(units) == 2
    assert [uid for u in units for uid in u["occurrence_ids"]] == [r["id"] for r in rows]
    group = units[0]
    joined = requests[group["request_id"]]
    assert joined["text"] == "他说向西走"
    assert len(joined["context_examples"]) == 2
    for rid in group["matched_constituent_requests"]:
        assert requests[rid]["context_examples"] == joined["context_examples"]
    assert contract == before
    with pytest.raises(ValueError, match="disjoint"):
        plan_windows(contract, [(rows[0]["id"], rows[1]["id"]),
                                (rows[1]["id"], rows[2]["id"])], glossary)
