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
    translate_utterances,
)


class _CallTrackingClient:
    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[dict] = []

    def complete(self, *, messages, max_tokens=4096, temperature=0.0, response_format=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "temperature": temperature})
        return self.responder(len(self.calls) - 1, messages)


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
    args = parser.parse_args(["generate-subtitles", "--recording-contract", "r.json", "--output-dir", "o", "--ass-layout", "raised_main", "--ass-margin-v", "135"])
    assert args.ass_layout == "raised_main" and args.ass_margin_v == 135

