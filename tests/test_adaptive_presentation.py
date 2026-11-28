"""Adaptive placement invariants, independent of original episode content."""
from pathlib import Path
import ctypes.util
import json

import pytest

from nvidia_transcribe import presentation as p
from nvidia_transcribe.subtitles import SubtitleConfig, SubtitleCue, render_subtitles, format_ass
from scripts.evaluate_display_schedule import events, metrics


SOURCE = {"fingerprint_sha256":"a"*64,"duration_s":30.}


def observations(picture=None, bands=(), cuts=()):
    return {"schema":p.SCHEMA,"media_fingerprint_sha256":SOURCE["fingerprint_sha256"],
            "duration_s":30.,"frame_size":[1920,1080],"active_picture":picture,
            "samples":[{"time_s":10.,"active_picture":picture,"caption_candidates":list(bands)}],
            "known_scene_boundaries_s":list(cuts),"status":"sampled","caption_status":"unknown"}


def cue(order,start,end,text="A measured line.",raw_end=None):
    return SubtitleCue(str(order),start,end,text,"A",raw_start=start,
                       raw_end=end if raw_end is None else raw_end,display_order=order)


def fixed_measurement(monkeypatch):
    monkeypatch.setattr(p,"measure_blocks",
                        lambda cues,cfg: ({c.display_order:(-100,-40,100,40) for c in cues},1))


def test_sampled_geometry_and_caption_candidates_are_distinct():
    w,h = 200,100
    raw = bytearray(w*h)
    for y in range(12,88):
        raw[y*w:(y+1)*w] = bytes([100])*w
    for y in range(91,95):
        for x in range(65,135,3):
            raw[y*w+x]=255
    frame = p.observe_frame(bytes(raw),w,h)
    assert frame["active_picture"] == [0.,.12,1.,.88]
    assert frame["caption_candidates"]
    assert p.observe_frame(bytes(w*h),w,h)["active_picture"] is None


def test_observations_reject_cross_recording_and_invalid_bounds():
    doc = observations([0,.12,1,.87])
    assert p.admit_observations(doc,SOURCE) == doc
    with pytest.raises(ValueError,match="identity"):
        p.admit_observations(doc,{**SOURCE,"fingerprint_sha256":"b"*64})
    with pytest.raises(ValueError,match="region"):
        p.admit_observations({**doc,"active_picture":[0,.9,1,.2]},SOURCE)
    with pytest.raises(ValueError,match="scene"):
        p.admit_observations({**doc,"known_scene_boundaries_s":[31.]},SOURCE)


def test_usable_bottom_bar_not_invalidated_by_candidate_inside_picture(monkeypatch):
    fixed_measurement(monkeypatch)
    doc=observations([0,.12,1,.87],bands=[[.2,.74,.8,.79]])
    displayed,overflow,report=p.place_ass([cue(0,1,3)],SubtitleConfig(),doc)
    assert report["status"]=="measured" and not overflow
    assert displayed[0].ass_bounds[1] >= .87*1080
    assert displayed[0].ass_bounds[3] < 1080


def test_overlapping_dialogue_has_fixed_noncolliding_chronological_positions(monkeypatch):
    fixed_measurement(monkeypatch)
    doc=observations([0,.12,1,.87])
    a,b=cue(0,1,4,raw_end=2),cue(1,2.5,5)
    displayed,overflow,_=p.place_ass([a,b],SubtitleConfig(),doc)
    assert not overflow and len(displayed)==2
    assert a.ass_bounds[3] < b.ass_bounds[1]
    assert a.start==1 and b.start==2.5
    assert r"\an5\pos(" in format_ass(displayed)


def test_full_frame_and_unknown_reserve_caption_space(monkeypatch):
    fixed_measurement(monkeypatch)
    for doc in (None,observations([0,0,1,1],bands=[[.2,.89,.8,.95]])):
        shown,overflow,_=p.place_ass([cue(0,1,3)],SubtitleConfig(),doc)
        assert not overflow and shown[0].ass_bounds[3] <= int(.84*1080)


def test_known_scene_boundary_only_clamps_padding():
    a=cue(0,1,5,raw_end=2)
    b=cue(1,1,5,raw_end=4)
    p.constrain_scene_padding([a,b],observations(cuts=[3.]))
    assert a.end==3 and b.end==5


def test_manual_layout_bypasses_native_measurement(tmp_path,monkeypatch):
    def forbidden(*args):
        pytest.fail("Manual override entered adaptive measurement")
    monkeypatch.setattr(p,"measure_blocks",forbidden)
    contract={"source":SOURCE,"utterances":[{"id":"one","source_start":1.,"source_end":3.,
                                            "translated_text":"Keep the selected wording."}]}
    for layout,margin in (("standard",42),("raised_main",150)):
        res=render_subtitles(contract,output_dir=tmp_path/layout,episode_stem="sample",
                             config=SubtitleConfig(ass_layout=layout))
        ass=Path(res["outputs"]["ass"]).read_text()
        assert r"\pos(" not in ass and f",2,54,54,{margin},1" in ass
        assert res["presentation"]["status"]=="manual_override"


def test_native_measurement_or_explicit_unavailable_fallback(tmp_path):
    contract={"source":SOURCE,"utterances":[{"id":"one","source_start":1.,"source_end":3.,
                                            "translated_text":"Keep the selected wording."}]}
    res=render_subtitles(contract,output_dir=tmp_path,episode_stem="sample",
                         presentation_observations=observations([0,.12,1,.87]))
    if ctypes.util.find_library("ass"):
        assert res["presentation"]["status"]=="measured"
        row=res["cue_lineage"][0]
        assert row["ass_position"] and row["ass_bounds"]
        assert row["ass_bounds"][1] >= .87*1080
        assert r"\pos(" in Path(res["outputs"]["ass"]).read_text()
    else:
        assert res["presentation"]["status"]=="measurement_unavailable"
        assert res["presentation"]["effective_layout"]=="raised_main"


def test_metrics_use_exact_emitted_thresholds_and_ignore_position_tags(tmp_path):
    path=tmp_path/"test.ass"
    path.write_text(r"Dialogue: 0,0:00:01.00,0:00:02.20,Main,A,0,0,0,,{\an5\pos(960,950)}123456789012345678901234"+"\n")
    report=metrics(events(path,"ass"),"ass")
    assert report["above_20_cps"]==0 and report["under_1_2s"]==0
    assert report["max_cps"]==20.


def test_full_content_identity_and_confirmed_absence_are_distinct(monkeypatch):
    doc=observations([0,0,1,1])
    native={**SOURCE,"content_sha256":"b"*64}
    with pytest.raises(ValueError,match="identity"):
        p.admit_observations(doc,native)
    doc["media_content_sha256"]=native["content_sha256"]
    doc["caption_observations"]=[{"interval_s":[0.,5.],"status":"absent","regions":[],
                                  "evidence_sha256":"c"*64,"method":"synthetic caption-free control"}]
    p.admit_observations(doc,native)
    fixed_measurement(monkeypatch)
    shown,_,_=p.place_ass([cue(0,1,3),cue(1,10,12)],SubtitleConfig(),doc)
    assert shown[0].ass_bounds[3]>int(.84*1080)
    assert shown[1].ass_bounds[3]<=int(.84*1080)
