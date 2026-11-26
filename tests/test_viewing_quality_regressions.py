"""Focused CPU regressions for Dual-Language Viewing-Quality Remediation (Work Packages A-D)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nvidia_transcribe.cli import build_parser
from nvidia_transcribe.refinement import GlossaryConfig, RefinementConfig, refine_utterances_with_text_model
from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
from nvidia_transcribe.subtitles import (
    SubtitleConfig,
    assign_subtitle_lanes,
    build_display_cues,
    format_ass,
    render_subtitles,
)
from nvidia_transcribe.translation import (
    TranslationConfig,
    condense_high_cps_cues_with_text_model,
    translate_utterances,
)
from nvidia_transcribe.utterances import audit_and_reconstruct_contract, reconstruct_speaker_utterances


class _CallTrackingClient:
    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[dict] = []

    def complete(self, *, messages, max_tokens=4096, temperature=0.0, response_format=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "temperature": temperature})
        return self.responder(len(self.calls) - 1, messages)


def test_a1_untimed_cross_chunk_ordering_and_unit_preservation() -> None:
    # Chunk 1 untimed utterance must stay inside Chunk 1 timeline (before Chunk 2 at 180s),
    # NEVER thrown after 1000s via cidx * 1000.0, and source_start must remain None.
    units = [
        {"text": "Hello", "speaker": "0", "global_speaker_id": "SPK_0", "chunk_index": 0, "source_unit_index": 0, "valid_timing": True, "start": 10.0, "end": 12.0},
        {"text": "MissingTimingInChunk1", "speaker": "1", "global_speaker_id": "SPK_1", "chunk_index": 1, "source_unit_index": 0, "valid_timing": False, "start": None, "end": None, "alignment_status": "diarization_bounded_only"},
        {"text": "Chunk2Line", "speaker": "0", "global_speaker_id": "SPK_0", "chunk_index": 2, "source_unit_index": 0, "valid_timing": True, "start": 185.0, "end": 188.0},
        {"text": "LateEpisodeLine", "speaker": "0", "global_speaker_id": "SPK_0", "chunk_index": 11, "source_unit_index": 0, "valid_timing": True, "start": 995.0, "end": 999.0},
    ]
    utts = reconstruct_speaker_utterances(
        units,
        source_language="en",
        chunk_windows={0: (0.0, 95.0), 1: (85.0, 180.0), 2: (170.0, 265.0), 11: (980.0, 1080.0)},
    )
    texts = [u["source_text"] for u in utts]
    assert texts == ["Hello", "MissingTimingInChunk1", "Chunk2Line", "LateEpisodeLine"]
    assert utts[1]["source_start"] is None and utts[1]["source_end"] is None
    assert utts[1]["chunk_indices"] == [1]


def test_a2_existing_contract_reconstruction_and_classification() -> None:
    contract = {
        "schema_version": "nemotron-recording-contract/v1",
        "source": {"path": "/tmp/ep.webm", "duration_s": 1200.0},
        "source_language": "zh",
        "utterances": [
            {"id": "utt-0000", "speaker_id": "SPK_0", "chunk_indices": [0], "source_start": 15.0, "source_end": 18.0, "source_text": "我们开始吧。"},
            {"id": "utt-0001", "speaker_id": "SPK_0", "chunk_indices": [2], "source_start": 180.0, "source_end": 185.0, "source_text": "从没人看好逆袭成商业巨子，冲上星空榜。"},
            {"id": "utt-0002", "speaker_id": "SPK_0", "chunk_indices": [11], "source_start": 1005.0, "source_end": 1008.0, "source_text": "而你呢？"},
            {"id": "utt-0003", "speaker_id": "SPK_1", "chunk_indices": [1], "source_start": None, "source_end": None, "alignment_status": "diarization_bounded_only", "source_text": "从没人看好逆袭成商业巨子"},  # dup of utt-0001 across chunk 1/2
            {"id": "utt-0004", "speaker_id": "SPK_1", "chunk_indices": [11], "source_start": None, "source_end": None, "alignment_status": "diarization_bounded_only", "source_text": "我不计前嫌送你一场机缘，我不欠你的。"},  # independent omitted
            {"id": "utt-0005", "speaker_id": "SPK_2", "chunk_indices": [14], "source_start": None, "source_end": None, "alignment_status": "alignment_failed", "source_text": "就是你你你没有怎么怎么没有遇到过没有怎么怎么没有遇到过没有怎么怎么没有"},
        ],
    }
    migrated, audit = audit_and_reconstruct_contract(
        contract,
        chunk_windows={0: (0.0, 90.0), 1: (80.0, 175.0), 2: (165.0, 260.0), 11: (980.0, 1080.0), 14: (1180.0, 1200.0)},
    )
    texts = [u["source_text"] for u in migrated["utterances"]]
    # Original ID utt-0003 (Chunk 1) placed cleanly before Chunk 2 line, utt-0004 inside Chunk 11, loop at Chunk 14
    assert texts.index("从没人看好逆袭成商业巨子") < texts.index("从没人看好逆袭成商业巨子，冲上星空榜。")
    assert texts.index("我不计前嫌送你一场机缘，我不欠你的。") < texts.index("就是你你你没有怎么怎么没有遇到过没有怎么怎么没有遇到过没有怎么怎么没有")
    by_orig = {r["original_utterance_id"]: r for r in audit}
    assert by_orig["utt-0003"]["classification"] == "probable_duplicate"
    assert by_orig["utt-0004"]["classification"] == "plausible_independent_missing_timing"
    assert by_orig["utt-0005"]["classification"] == "likely_hallucination_or_loop"
    # Timestamps are never fabricated
    for u in migrated["utterances"]:
        if u["original_utterance_id"] in ("utt-0003", "utt-0004", "utt-0005"):
            assert u["source_start"] is None and u["source_end"] is None


def test_b_cache_correctness_invalidation_selective_reuse_and_rerender(tmp_path: Path) -> None:
    base_utts = [
        {"id": f"utt-{i:04d}", "speaker_id": "SPK_0", "source_start": i * 2.0, "source_end": i * 2.0 + 1.5, "source_text": f"Texte numéro {i}."}
        for i in range(6)
    ]

    def _ref_resp(_idx, msgs):
        # Return identity edits echoing target IDs cleanly from prompt
        lines = [ln for ln in msgs[1]["content"].splitlines() if ln.startswith("- [utt-")]
        in_targets = False
        corr = []
        for ln in msgs[1]["content"].splitlines():
            if ln.startswith("TARGET UTTERANCES"):
                in_targets = True
                continue
            if ln.startswith("READ-ONLY CONTEXT AFTER"):
                break
            if in_targets and ln.startswith("- ["):
                uid = ln.split("]")[0].split("[")[1]
                txt = ln.split("): ", 1)[1]
                corr.append({"utterance_id": uid, "original_text": txt, "refined_text": txt.replace("Texte", "Phrase"), "changed": True, "edit_type": "style", "rationale": "r", "confidence": 0.9})
        return {"choices": [{"message": {"content": json.dumps({"corrections": corr})}}]}

    def _tr_resp(_idx, msgs):
        tr = []
        in_targets = False
        for ln in msgs[1]["content"].splitlines():
            if ln.startswith("TARGET UTTERANCES"):
                in_targets = True
                continue
            if ln.startswith("READ-ONLY CONTEXT AFTER"):
                break
            if in_targets and ln.startswith("- ["):
                uid = ln.split("]")[0].split("[")[1]
                txt = ln.split("): ", 1)[1]
                tr.append({"utterance_id": uid, "translated_text": f"EN:{txt}", "grouped_with_prev": False, "notes": ""})
        return {"choices": [{"message": {"content": json.dumps({"translations": tr})}}]}

    ref_ckpt = tmp_path / "ref.ckpt.json"
    tr_ckpt = tmp_path / "tr.ckpt.json"
    rcfg = RefinementConfig(profile="balanced", batch_size=2, context_utterances_before=1, context_utterances_after=1)
    tcfg = TranslationConfig(style="natural", batch_size=2, context_utterances_before=1, context_utterances_after=1)

    # 1. Cold run: 3 batches each
    c1 = _CallTrackingClient(_ref_resp)
    out1, sum1 = refine_utterances_with_text_model(base_utts, source_language="fr", config=rcfg, text_client=c1, checkpoint_path=ref_ckpt)
    assert len(c1.calls) == 3

    tc1 = _CallTrackingClient(_tr_resp)
    tout1, tsum1 = translate_utterances(out1, source_language="fr", config=tcfg, text_client=tc1, checkpoint_path=tr_ckpt)
    assert len(tc1.calls) == 3

    # 2. Identical rerun: 100% cache hit -> 0 API calls
    c2 = _CallTrackingClient(_ref_resp)
    out2, sum2 = refine_utterances_with_text_model(base_utts, source_language="fr", config=rcfg, text_client=c2, checkpoint_path=ref_ckpt)
    assert len(c2.calls) == 0 and sum2["cache_hits"] == 6

    tc2 = _CallTrackingClient(_tr_resp)
    tout2, tsum2 = translate_utterances(out2, source_language="fr", config=tcfg, text_client=tc2, checkpoint_path=tr_ckpt)
    assert len(tc2.calls) == 0 and tsum2["cache_hits"] == 6

    # 3. Change utt-0001 text (keeping ID constant): batch 0 (target) + batch 1 (ctx_before) recompute, batch 2 stays cached!
    edited_utts = [{**u} for u in base_utts]
    edited_utts[1]["source_text"] = "Nouveau texte un !"
    c3 = _CallTrackingClient(_ref_resp)
    out3, sum3 = refine_utterances_with_text_model(edited_utts, source_language="fr", config=rcfg, text_client=c3, checkpoint_path=ref_ckpt)
    assert len(c3.calls) == 2
    assert out3[1]["refined_text"] == "Nouveau texte un !"
    assert sum3["cache_hits"] == 2

    tc3 = _CallTrackingClient(_tr_resp)
    tout3, tsum3 = translate_utterances(out3, source_language="fr", config=tcfg, text_client=tc3, checkpoint_path=tr_ckpt)
    assert tout3[1]["translated_text"] == "EN:Nouveau texte un !"
    assert tsum3["cache_hits"] == 2

    # 4. Changed glossary / style invalidates
    c4 = _CallTrackingClient(_ref_resp)
    _, sum4 = refine_utterances_with_text_model(edited_utts, source_language="fr", config=rcfg, glossary=GlossaryConfig(characters=("Esteban",)), text_client=c4, checkpoint_path=ref_ckpt)
    assert len(c4.calls) == 3

    tc4 = _CallTrackingClient(_tr_resp)
    _, tsum4 = translate_utterances(out3, source_language="fr", config=TranslationConfig(style="expressive", batch_size=2, context_utterances_before=1, context_utterances_after=1), text_client=tc4, checkpoint_path=tr_ckpt)
    assert len(tc4.calls) == 3

    # 5. Explicit --re-render performs ZERO inference calls
    out_dir = tmp_path / "ep_run"
    out_dir.mkdir()
    (out_dir / "translated.contract.json").write_text(json.dumps({"source": {"path": "/tmp/v.mp4"}, "utterances": tout3}), encoding="utf-8")
    rerender_client = _CallTrackingClient(lambda *_: (_ for _ in ()).throw(AssertionError("no inference allowed on --re-render")))
    res_rr = generate_subtitles(
        recording_contract_path=None,
        output_dir=out_dir,
        config=SubtitlePipelineConfig(re_render_only=True, subtitles=SubtitleConfig(ass_layout="raised_main")),
        text_client=rerender_client,
    )
    assert res_rr["refinement"]["status"] == "reused_translated_contract"
    assert len(rerender_client.calls) == 0


def test_c1_a3_translation_echo_dedup_and_mandarin_fragment_cleanup() -> None:
    utts = [
        {"id": "utt-0016", "speaker_id": "SPK_04", "source_start": 268.54, "source_end": 269.91, "source_text": "败由不得我随着幕后巨手终于升至台前"},
        {"id": "utt-0017", "speaker_id": "SPK_04", "source_start": 269.90, "source_end": 270.19, "source_text": "与掌握，得回。"},
        {"id": "utt-0018", "speaker_id": "SPK_01", "source_start": 300.0, "source_end": 301.0, "source_text": "嗯。"},  # genuine interjection
        {"id": "utt-0019", "speaker_id": "SPK_19", "source_start": 1163.32, "source_end": 1164.27, "source_text": "一定会再见的"},
        {"id": "utt-0020", "speaker_id": "SPK_18", "source_start": 1163.64, "source_end": 1164.11, "source_text": "的"},  # stray particle overlap
    ]
    stub = _CallTrackingClient(
        lambda _idx, _msgs: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "translations": [
                                    {"utterance_id": "utt-0016", "translated_text": "Failure is not up to me as the hand rises —and what I can grasp, what I can reclaim.", "grouped_with_prev": False, "notes": ""},
                                    {"utterance_id": "utt-0017", "translated_text": "—and what I can grasp, what I can reclaim.", "grouped_with_prev": True, "notes": "continuation"},
                                    {"utterance_id": "utt-0018", "translated_text": "Mm.", "grouped_with_prev": False, "notes": ""},
                                    {"utterance_id": "utt-0019", "translated_text": "We will meet again.", "grouped_with_prev": False, "notes": ""},
                                    {"utterance_id": "utt-0020", "translated_text": "", "grouped_with_prev": False, "notes": ""},
                                ]
                            }
                        )
                    }
                }
            ]
        }
    )
    out, summary = translate_utterances(utts, source_language="zh", config=TranslationConfig(batch_size=6), text_client=stub)
    # No repeated trailing clause on merged utt-0016!
    assert out[0]["translated_text"].count("what I can reclaim") == 1
    assert out[1]["merged_into"] == "utt-0016"
    # Genuine interjection '嗯。' -> 'Mm.' stays!
    assert out[2]["translated_text"] == "Mm." and not out[2].get("fragment_suppressed")
    # Stray '的' suppressed rather than leaking untranslated CJK particle!
    assert out[4].get("fragment_suppressed") is True
    assert out[4]["translated_text"] == ""
    assert summary["fragment_suppressed_count"] == 1

    # Subtitle renderer must not emit '的' via fallback_key!
    cues = build_display_cues(out)
    assert all("的" not in c.text for c in cues)


def test_c2_c4_d_positive_duration_slack_extension_third_lane_and_raised_main(tmp_path: Path) -> None:
    # Includes French cue-113-style rapid overlapping sequence, high-CPS trailing slack cue,
    # concurrent duplicate echo, and genuine 3-speaker overlap!
    utts = [
        {"id": "utt-0", "speaker_id": "SPK_00", "source_start": 10.0, "source_end": 11.2, "translated_text": "This is a rapid phrase that needs trailing silence to read calmly."},
        {"id": "utt-1", "speaker_id": "SPK_01", "source_start": 503.26, "source_end": 503.475, "translated_text": "fine."},
        {"id": "utt-2", "speaker_id": "SPK_02", "source_start": 503.88, "source_end": 504.095, "translated_text": "ogre."},
        {"id": "utt-3", "speaker_id": "SPK_00", "source_start": 600.0, "source_end": 604.0, "translated_text": "I am speaking first."},
        {"id": "utt-4", "speaker_id": "SPK_01", "source_start": 600.2, "source_end": 603.5, "translated_text": "I am speaking second."},
        {"id": "utt-5", "speaker_id": "SPK_02", "source_start": 600.4, "source_end": 603.0, "translated_text": "I am speaking third."},
        # Duplicate ASR echo across SPK_00 & SPK_03 simultaneously
        {"id": "utt-6", "speaker_id": "SPK_00", "source_start": 700.0, "source_end": 703.5, "translated_text": "The interior of the continent is still unexplored."},
        {"id": "utt-7", "speaker_id": "SPK_03", "source_start": 700.2, "source_end": 703.4, "translated_text": "The interior of the continent is still unexplored."},
    ]
    cfg = SubtitleConfig(ass_layout="raised_main", raised_main_margin_v=140, ass_max_lanes=3, ass_opaque_box=True)
    res = render_subtitles({"utterances": utts}, output_dir=tmp_path, episode_stem="EP166", config=cfg)
    srt = (tmp_path / "EP166.en.srt").read_text(encoding="utf-8")
    ass = (tmp_path / "EP166.en.ass").read_text(encoding="utf-8")

    assert res["stats"]["ass_overflow_cues"] == 0
    # Cue 0 extended past 1.2s into silence up to ~3.6s reducing CPS dramatically
    assert "I am speaking third." in ass and ",TopAlt," in ass
    # Raised-main ASS header has raised MarginV=140 and BorderStyle=4 opaque backdrop
    assert "Style: Main,DejaVu Sans,52,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,0,0,0,0,100,100,0,0,4,1.5,0.0,2,54,54,140,1" in ass
    # Two distinct source utterance IDs may be two simultaneous speakers;
    # identical translated wording alone cannot delete either source event.
    assert srt.count("The interior of the") == 2
    assert srt.count("still unexplored.") == 2
    assert ass.count("still unexplored.") == 2

    # Check CLI parser supports raised_main flags cleanly
    parser = build_parser()
    args = parser.parse_args(["generate-subtitles", "--recording-contract", "r.json", "--output-dir", "o", "--ass-layout", "raised_main", "--ass-margin-v", "135", "--condense-high-cps"])
    assert args.ass_layout == "raised_main" and args.ass_margin_v == 135 and args.condense_high_cps is True


def test_c3_selective_condensation_keeps_uncondensed_text(tmp_path: Path) -> None:
    utts = [
        {"id": "utt-0", "speaker_id": "SPK_00", "source_start": 10.0, "source_end": 12.0, "refined_text": "他一直在这没日没夜地练", "translated_text": "He has been over here training endlessly day and night without eating or drinking a single bit."},
        {"id": "utt-1", "speaker_id": "SPK_00", "source_start": 20.0, "source_end": 23.0, "refined_text": "走吧", "translated_text": "Let's go."},  # already slow ~3 cps -> skipped!
    ]
    client = _CallTrackingClient(
        lambda _idx, _msgs: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "utterance_id": "utt-0",
                                "condensed_text": "He's been training nonstop without food or water.",
                                "changed": True,
                                "rationale": "concise_idiom",
                            }
                        )
                    }
                }
            ]
        }
    )
    out, summary = condense_high_cps_cues_with_text_model(utts, source_language="zh", cps_threshold=25.0, text_client=client, checkpoint_path=tmp_path / "cond.ckpt.json")
    assert summary["candidate_count"] == 1
    assert summary["condensed_count"] == 1
    assert len(client.calls) == 1
    assert out[0]["translated_text"] == "He's been training nonstop without food or water."
    assert out[0]["uncondensed_translated_text"].startswith("He has been over here training endlessly")
    assert out[1]["translated_text"] == "Let's go."
