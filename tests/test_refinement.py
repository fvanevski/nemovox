"""CPU unit tests for experimental refinement proposal contract (Phase C)."""

from __future__ import annotations

import json
from pathlib import Path

from nvidia_transcribe.refinement import (
    PatchProposal,
    build_c2_text_probe_messages,
    evaluate_refinement_against_reference,
    parse_c2_probe_response,
    select_suspicious_utterances,
)


def test_select_suspicious_utterances_flags_oov_mixed_script_and_glossary_near_miss() -> None:
    utts = [
        {
            "utterance_id": "u0",
            "source_text": "Tout va bien dans cette longue phrase parfaitement normale.",
            "timestamp_provenance": {"timing_status": "full"},
        },
        {
            "utterance_id": "u1",
            "source_text": "Estebane doit retrouver Mendoza bientôt.",
            "timestamp_provenance": {"timing_status": "full"},
        },
        {
            "utterance_id": "u2",
            "source_text": "Sintra 路在罗卡角。",
            "timestamp_provenance": {"timing_status": "failed"},
        },
    ]
    flagged = select_suspicious_utterances(utts, glossary_terms=["Esteban", "Mendoza"])
    ids = [f["utterance_id"] for f in flagged]
    assert ids == ["u1", "u2"]
    assert any("glossary_near_miss" in r for r in flagged[0]["reasons"])
    assert "timing_failed" in flagged[1]["reasons"]
    assert "mixed_cjk_latin" in flagged[1]["reasons"]


def test_parse_c2_probe_response_handles_reasoning_and_invalid_json() -> None:
    msgs = build_c2_text_probe_messages(
        target_text="Estebane part demain",
        language="fr",
        glossary_terms=["Esteban"],
    )
    assert msgs[0]["role"] == "system"
    assert "Esteban" in msgs[1]["content"]

    ok_raw = {
        "choices": [
            {
                "message": {
                    "reasoning_content": "Glossary says Esteban, not Estebane.",
                    "content": json.dumps(
                        {
                            "changed": True,
                            "proposed_text": "Esteban part demain",
                            "confidence": 0.92,
                            "rationale": "proper noun spelling",
                        }
                    ),
                }
            }
        ]
    }
    parsed = parse_c2_probe_response(ok_raw, fallback_text="Estebane part demain")
    assert parsed["status"] == "ok"
    assert parsed["changed"] is True
    assert parsed["proposed_text"] == "Esteban part demain"
    assert "Glossary says Esteban" in (parsed["reasoning_content"] or "")

    bad_raw = {"choices": [{"message": {"content": "not json at all"}}]}
    parsed_bad = parse_c2_probe_response(bad_raw, fallback_text="Estebane part demain")
    assert parsed_bad["status"] == "parse_error"
    assert parsed_bad["changed"] is False
    assert parsed_bad["proposed_text"] == "Estebane part demain"

    dup_raw = {
        "choices": [
            {
                "message": {
                    "reasoning": "trace",
                    "content": '{"changed": false, "proposed_text": "Esteban part demain", "rationale": "fix"}\n\n{"changed": false, "proposed_text": "Esteban part demain", "rationale": "fix"}',
                }
            }
        ]
    }
    parsed_dup = parse_c2_probe_response(dup_raw, fallback_text="Estebane part demain")
    assert parsed_dup["status"] == "ok"
    assert parsed_dup["degenerate_repetition"] is True
    assert parsed_dup["flag_contradiction"] is True
    assert parsed_dup["changed"] is False


def test_evaluate_refinement_measures_fix_and_false_correction() -> None:
    # Real error repaired
    ev_fix = evaluate_refinement_against_reference(
        before_text="Scotturbe 403 路公共汽车",
        proposed_text="Scotturb 403 路公共汽车",
        reference_text="Scotturb 403 路公共汽车",
        language="zh",
    )
    assert ev_fix["outcome"] == "improved"
    assert ev_fix["net_edits_reduced"] > 0
    assert ev_fix["false_correction"] is False

    # False correction on initially correct transcript
    ev_fc = evaluate_refinement_against_reference(
        before_text="Bonjour Esteban",
        proposed_text="Bonjour Estéban",
        reference_text="Bonjour Esteban",
        language="fr",
    )
    # Normalization removes accent differences OR if changed materially: check non-accent corruption
    ev_fc2 = evaluate_refinement_against_reference(
        before_text="Bonjour Esteban",
        proposed_text="Bonjour Tao",
        reference_text="Bonjour Esteban",
        language="fr",
    )
    assert ev_fc["false_correction"] is False or ev_fc2["false_correction"] is True
    assert ev_fc2["outcome"] == "false_correction"

    prop = PatchProposal(
        proposal_id="p1",
        case_id="case_1",
        language="fr",
        method="C1_contextual_audio",
        source_unit_ids=["c0_u0"],
        utterance_id="utt_0",
        source_start_s=1.0,
        source_end_s=2.5,
        before_text="Estebane",
        proposed_text="Esteban",
        changed=True,
        audio_verified=True,
        reasoning_content=None,
        rationale="test",
    )
    assert prop.status == "proposed_only"


def test_batched_text_refinement_profiles_screening_and_resume(tmp_path: Path) -> None:
    from nvidia_transcribe.refinement import (
        GlossaryConfig,
        RefinementConfig,
        load_glossary,
        refine_utterances_with_text_model,
        screen_refinement_edit,
    )

    gl_yaml = tmp_path / "series.yaml"
    gl_yaml.write_text("characters:\n  - Esteban\n  - Tao\nterms:\n  - Grand Condor\n", encoding="utf-8")
    gl = load_glossary(gl_yaml)
    assert "Esteban" in gl.characters
    assert "Grand Condor" in gl.terms

    # Screen rejects empty replacement, repetition loop, and wrong-language translation leak
    assert screen_refinement_edit(original_text="Bonjour", refined_text="", profile="balanced", source_language="fr") == (False, "empty_replacement")
    assert screen_refinement_edit(
        original_text="跳下垂直净空。",
        refined_text="Jump down vertical clearance.",
        profile="balanced",
        source_language="zh",
    ) == (False, "wrong_language_lost_cjk")
    assert screen_refinement_edit(
        original_text="Salut les amis.",
        refined_text="You are going to the market with the friends that were there.",
        profile="balanced",
        source_language="fr",
    ) == (False, "wrong_language_english_leak")
    # Expressive permits substantial non-English repair cleanly without artificial edit-distance block
    assert screen_refinement_edit(
        original_text="Tincio, Carlos, Diego, Zia et Tato.",
        refined_text="Le seizième siècle, Esteban, Zia et Tao découvrent les cités.",
        profile="expressive",
        source_language="fr",
    )[0] is True

    utts = [
        {"id": "utt-0", "speaker_id": "SPK_00", "source_text": "Tato, viens voir le Grand Confort !"},
        {"id": "utt-1", "speaker_id": "SPK_01", "source_text": "Pas maintenant, Estebane."},
    ]

    # Profile "off" makes zero client calls and retains raw_source_text == refined_text
    off_out, off_meta = refine_utterances_with_text_model(
        utts, source_language="fr", config=RefinementConfig(profile="off"), glossary=gl, text_client=object()
    )
    assert off_meta["changed_count"] == 0
    assert off_out[0]["refined_text"] == utts[0]["source_text"]

    class _StubRefClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *, messages, **kwargs):
            self.calls += 1
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "corrections": [
                                        {
                                            "utterance_id": "utt-0",
                                            "original_text": utts[0]["source_text"],
                                            "refined_text": "Tao, viens voir le Grand Condor !",
                                            "changed": True,
                                            "edit_type": "glossary_name",
                                            "rationale": "Tao & Grand Condor",
                                            "confidence": 0.94,
                                        },
                                        {
                                            "utterance_id": "utt-0",  # duplicate ignored safely
                                            "original_text": "bad",
                                            "refined_text": "duplicate should ignore",
                                            "changed": True,
                                            "edit_type": "dup",
                                            "rationale": "dup",
                                            "confidence": 0.1,
                                        },
                                        {
                                            "utterance_id": "utt-1",
                                            "original_text": utts[1]["source_text"],
                                            "refined_text": "Pas maintenant, Esteban.",
                                            "changed": True,
                                            "edit_type": "character_name",
                                            "rationale": "glossary name",
                                            "confidence": 0.96,
                                        },
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

    stub = _StubRefClient()
    ckpt = tmp_path / "ref.ckpt.json"
    out_bal, meta_bal = refine_utterances_with_text_model(
        utts,
        source_language="fr",
        config=RefinementConfig(profile="balanced", batch_size=4),
        glossary=gl,
        text_client=stub,
        checkpoint_path=ckpt,
    )
    assert stub.calls == 1
    assert meta_bal["changed_count"] == 2
    assert out_bal[0]["raw_source_text"] == utts[0]["source_text"]
    assert out_bal[0]["refined_text"] == "Tao, viens voir le Grand Condor !"
    assert out_bal[1]["refined_text"] == "Pas maintenant, Esteban."

    # Resuming same profile + glossary makes 0 additional API calls
    out_resumed, _ = refine_utterances_with_text_model(
        utts,
        source_language="fr",
        config=RefinementConfig(profile="balanced", batch_size=4),
        glossary=gl,
        text_client=stub,
        checkpoint_path=ckpt,
        resume=True,
    )
    assert stub.calls == 1
    assert out_resumed[1]["refined_text"] == "Pas maintenant, Esteban."


def test_retired_refinement_backend_is_rejected_before_inference():
    import pytest
    from nvidia_transcribe.refinement import RefinementConfig, refine_utterances_with_text_model

    with pytest.raises(ValueError, match="Unsupported refinement backend"):
        refine_utterances_with_text_model(
            [{"id": "u", "source_text": "Bonjour"}], source_language="fr",
            config=RefinementConfig(backend="swiftu", profile="balanced"),
            text_client=object(),
        )
