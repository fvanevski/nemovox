"""Selective independent acoustic recovery of still-missing/untimed dialogue.

Occurrence, approximate event timing and source wording are separate claims.
No raw observations, accepted timed turns or original recording are overwritten.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

from nvidia_transcribe.media import extract_wav_segment, validate_wav_file
from nvidia_transcribe.source_quality import _bounds, _digest, _lex, _overlap


MAX_TARGETS = 6
MAX_RUN_S = 30.0


def _matches(a, b):
    return [m for m in SequenceMatcher(None,a,b,autojunk=False).get_matching_blocks() if m.size]


def _anchor(a,b,language):
    minimum = 4 if language.startswith(("zh","ja")) else 3
    matches = _matches(a,b)
    return bool(matches and max(m.size for m in matches) >= minimum)


def _untimed_runs(rows):
    i = 0
    while i < len(rows):
        if _bounds(rows[i]):
            i += 1
            continue
        j = i+1
        while j < len(rows) and not _bounds(rows[j]):
            j += 1
        if i and j < len(rows):
            left,right = _bounds(rows[i-1]),_bounds(rows[j])
            if left[1] < right[0]:
                yield i,j,(left[1],right[0])
        i = j


def plan_recovery(recording, observations):
    rows = recording["utterances"]
    targets = []
    for i,j,bounds in _untimed_runs(rows):
        if bounds[1]-bounds[0] <= MAX_RUN_S:
            targets.append({"kind":"untimed","bounds":bounds,
                            "ids":[r["id"] for r in rows[i:j]]})
    # Require positive Nemotron speech evidence, not silence or an arbitrary
    # subtitle gap. Long gaps with no timed lexical evidence stay unresolved.
    candidates = []
    for obs in observations:
        for row in obs["utterances"]:
            bounds = _bounds(row)
            if not bounds or bounds[1]-bounds[0] < .65 or len(_lex(row["source_text"])) < 4:
                continue
            if any(_bounds(r) and _overlap(_bounds(r),bounds) > 0 for r in rows):
                continue
            if any(_overlap(t["bounds"],bounds) > 0 for t in targets):
                continue
            if any(_overlap(t["bounds"],bounds) > .5 * min(
                    t["bounds"][1]-t["bounds"][0],bounds[1]-bounds[0]) for t in candidates):
                continue
            candidates.append({"kind":"omission","bounds":bounds,"ids":[]})
    return (targets+candidates)[:MAX_TARGETS]


def _observed_rows(observation, bounds):
    a,b = bounds
    lo,hi = observation["crop"]
    rows = []
    for row in observation["rows"]:
        r = _bounds(row)
        if (r and a <= r[0] < r[1] <= b and r[0] > lo+.10
                and r[1] < hi-.10 and _lex(row["source_text"])):
            rows.append(row)
    if any(x["source_end"] > y["source_start"]+.05 for x,y in zip(rows,rows[1:])):
        return []
    return rows


def _sequence(rows):
    return tuple(w for r in rows for w in _lex(r["source_text"]))


def _associate_originals(original, candidate, language):
    """Match every original row to a positively observed ordered sequence.

    Contained duplicate fragments may map to one occurrence ONLY across
    distinct chunks. Unrelated or same-chunk repeated rows block the rewrite.
    """
    spans = []
    tokens = _sequence(candidate)
    for row in original:
        words = _lex(row["source_text"])
        blocks = _matches(words,tokens)
        if not words or not blocks or not _anchor(words,tokens,language):
            return None
        if sum(m.size for m in blocks)/len(words) < .55:
            return None
        start,end = blocks[0].b,blocks[-1].b+blocks[-1].size
        # A short repeated sequence elsewhere is not a unique association.
        longest = max(blocks,key=lambda m:m.size)
        key = words[longest.a:longest.a+longest.size]
        if sum(tokens[k:k+len(key)]==key for k in range(len(tokens)-len(key)+1)) != 1:
            return None
        for old_start,old_end,old in spans:
            if max(old_start,start) < min(old_end,end):
                contained = ((old_start <= start and end <= old_end)
                             or (start <= old_start and old_end <= end))
                chunks = set(row.get("chunk_indices") or [])
                other = set(old.get("chunk_indices") or [])
                if not contained or not chunks or not other or chunks & other:
                    return None
            elif start < old_end:
                return None
        spans.append((start,end,row))
    if not spans:
        return None
    # Choose maximal association spans, assigning actual Whisper segments
    # to the original coherent source events rather than guessing word times.
    maximal = [s for s in spans if not any(
        x[0]<=s[0] and s[1]<=x[1] and (x[0],x[1])!=(s[0],s[1]) for x in spans)]
    unique = []
    for s in sorted(maximal,key=lambda x:x[:2]):
        if not unique or s[:2]!=unique[-1][:2]:
            unique.append(s)
    grouped = []
    offset = 0
    for r in candidate:
        end = offset+len(_lex(r["source_text"]))
        overlaps = [max(0,min(end,b)-max(offset,a)) for a,b,_ in unique]
        best = max(overlaps,default=0)
        if not best or overlaps.count(best)!=1:
            return None
        index = overlaps.index(best)
        # A noisy interior segment is supported by the enclosing ordered
        # association. Do not require an exact anchor in every small
        # Whisper segment. Unmatched outer material is not admitted.
        lo,hi,_ = unique[index]
        if offset < lo-2 or end > hi+2:
            return None
        if grouped and index < grouped[-1][0]:
            return None
        if grouped and grouped[-1][0]==index:
            grouped[-1][1].append(r)
        else:
            grouped.append((index,[r]))
        offset = end
    if len(grouped)!=len(unique):
        return None
    for index,group in grouped:
        original_words = _lex(unique[index][2]["source_text"])
        observed_words = _sequence(group)
        matches = _matches(original_words,observed_words)
        if (not matches or sum(m.size for m in matches)/len(observed_words)<.55
                or not _anchor(original_words,observed_words,language)):
            return None
    return [g for _,g in grouped]


def _make_row(recording, group, origins, pair, evidence, chunks):
    a,b = group[0]["source_start"],group[-1]["source_end"]
    uid = "src-"+_digest([recording["source"]["fingerprint_sha256"],
                          [r["observation_id"] for r in group],a,b])[:24]
    language = recording["source_language"]
    text = ("".join(r["source_text"] for r in group) if language.startswith(("zh","ja"))
            else " ".join(r["source_text"] for r in group))
    return {"id":uid,"utterance_id":uid,"source_turn_id":uid,"source_turn_ids":[uid],
            "source_language":language,"source_text":text,"source_start":a,"source_end":b,
            "speaker_id":"SPK_UNK","speaker_identity_accepted":False,
            "source_words":[],"source_characters":[],"unit_count":0,
            "timed_unit_count":0,"untimed_unit_count":0,"timing_gaps":[],
            "chunk_indices":sorted(chunks),"alignment_status":"independent_event_bounds_approximate",
            "source_quality_status":"unresolved_wording_independent_recovery",
            "prior_source_hypotheses":origins,
            "timestamp_provenance":{"backend":"independent_asr_segment_estimates",
               "timing_status":"approximate","human_verified":False,"word_timing_verified":False,
               "audio_sha256":[p["audio_sha256"] for p in pair],
               "observation_ids":[r["observation_id"] for r in group],
               "nemotron_support":evidence}}


def apply_independent(recording, targets, observations, independent):
    rows = [dict(r) for r in recording["utterances"]]
    decisions = []
    lang = recording["source_language"]
    for index,t in enumerate(targets):
        pair = [o for o in independent if o["id"] in (f"challenge-{index}-0",f"challenge-{index}-1")]
        if len(pair)!=2 or pair[0]["audio_sha256"]==pair[1]["audio_sha256"] or pair[0]["crop"]==pair[1]["crop"]:
            continue
        a,b = t["bounds"]
        # Omission event may have narrower Whisper estimates; use the
        # actual enclosing gap, never padding across an accepted timed turn.
        bracket = (a,b)
        if t["kind"]=="omission":
            left = max([r["source_end"] for r in rows if _bounds(r) and r["source_end"]<=a]+[0.])
            right = min([r["source_start"] for r in rows if _bounds(r) and r["source_start"]>=b]
                        +[recording["source"]["duration_s"]])
            bracket = (max(left,a-1.),min(right,b+1.))
        first,second = [_observed_rows(o,bracket) for o in pair]
        if not first or not second:
            continue
        x,y = _sequence(first),_sequence(second)
        if (not x or not y or SequenceMatcher(None,x,y,autojunk=False).ratio()<.78
                or abs(first[0]["source_start"]-second[0]["source_start"])>1.25
                or abs(first[-1]["source_end"]-second[-1]["source_end"])>1.25):
            continue
        span=(second[0]["source_start"],second[-1]["source_end"])
        if any(_bounds(r) and _overlap(_bounds(r),span)>0 for r in rows):
            continue
        support = []
        chunks = set()
        for obs in observations:
            for r in obs["utterances"]:
                rb = _bounds(r)
                if rb and _overlap(rb,span)>.50*min(rb[1]-rb[0],span[1]-span[0]) and _anchor(y,_lex(r["source_text"]),lang):
                    support.append({"observation_id":r["observation_id"],"audio_sha256":obs["audio_sha256"]})
                    chunks.update(obs.get("chunks",[]))
        if t["kind"]=="untimed":
            ids=t["ids"]
            positions=[i for i,r in enumerate(rows) if r["id"] in ids]
            if len(positions)!=len(ids) or positions!=list(range(positions[0],positions[-1]+1)):
                continue
            originals=rows[positions[0]:positions[-1]+1]
            if any(_bounds(r) for r in originals):
                continue
            voices={r.get("speaker_id") for r in originals}
            if len(voices)>1:
                continue
            groups=_associate_originals(originals,second,lang)
            if not groups:
                continue
            chunks.update(c for r in originals for c in r.get("chunk_indices",[]))
            replacement=[_make_row(recording,g,originals,pair,support,chunks) for g in groups]
            rows[positions[0]:positions[-1]+1]=replacement
        else:
            # A competing timed or untimed lexical occurrence can be a
            # displaced duplicate. Do not insert in that situation.
            if (len({r["audio_sha256"] for r in support})<2 or
                    any(_anchor(y,_lex(r["source_text"]),lang)
                        and (not _bounds(r) or abs(r["source_start"]-a)<60) for r in rows)):
                continue
            # No merging distinct unverified speech turns.
            if len(first)!=1 or len(second)!=1:
                continue
            replacement=[_make_row(recording,second,[],pair,support,chunks)]
            pos=next((i for i,r in enumerate(rows) if _bounds(r) and r["source_start"]>span[0]),len(rows))
            rows[pos:pos]=replacement
        decisions.append({"status":"unresolved_wording_independent_recovery","kind":t["kind"],
                          "target":t,"window":span,"source_ids":[r["id"] for r in replacement],
                          "observation_ids":[r["observation_id"] for r in second]})
    return rows,decisions


def recover_independent_source(recording,observations,media,*,challenger_resource,output_dir):
    report={"status":"not_configured","targets":[],"decisions":[],"errors":[],
            "observations":[]}
    rows=recording["utterances"]
    if challenger_resource is None:
        return rows,report
    language=recording["source_language"]
    if not isinstance(language,str) or len(language.split("-")[0])!=2:
        report["status"]="unsupported_language"
        return rows,report
    targets=plan_recovery(recording,observations)
    report["targets"]=targets
    if not targets:
        report["status"]="no_targets"
        return rows,report
    with tempfile.TemporaryDirectory(prefix="independent-source-",dir=output_dir) as tmp:
        root=Path(tmp)
        crops=[]
        for i,t in enumerate(targets):
            a,b=t["bounds"]
            length=b-a
            # Different amounts of surrounding context, scaled to the
            # candidate occurrence/run instead of fixed episode crop sizes.
            for j,pad in enumerate((max(.5,length*.20),max(1.,length*.60))):
                lo,hi=max(0.,a-pad),min(media.duration_s,b+pad)
                path=root/f"challenge-{i}-{j}.wav"
                extract_wav_segment(media.path,start_s=lo,duration_s=hi-lo,out_wav=path,
                                    audio_stream_index=media.selected_audio_stream)
                duration,digest=validate_wav_file(path)
                if abs(duration-(hi-lo))>.1:
                    raise ValueError("Independent crop duration mismatch")
                crops.append({"id":path.stem,"path":str(path),"start":lo,"end":hi,"audio_sha256":digest})
        request={"checkpoint":challenger_resource.checkpoint_path,"language":language.split("-")[0],"crops":crops}
        try:
            proc=subprocess.run([challenger_resource.worker_python,
                str(Path(__file__).with_name("_whisper_challenge_worker.py"))],
                input=json.dumps(request),text=True,capture_output=True,timeout=900,
                env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})
            if proc.returncode:
                raise RuntimeError(f"Independent worker exited {proc.returncode}: {proc.stderr[-2000:]}")
            result=json.loads(proc.stdout)
            expected={c["id"]:(c["audio_sha256"],[c["start"],c["end"]]) for c in crops}
            actual={o["id"]:(o["audio_sha256"],o["crop"]) for o in result["observations"]}
            if expected!=actual or len(actual)!=len(result["observations"]) or result["device"]!="cpu":
                raise ValueError("Independent worker receipt mismatch")
            report["observations"]=result["observations"]
            report["resource"]={k:v for k,v in result.items() if k!="observations"}
            rows,report["decisions"]=apply_independent(recording,targets,observations,result["observations"])
            report["status"]="partial" if report["decisions"] else "unresolved"
        except (OSError,ValueError,KeyError,RuntimeError,subprocess.TimeoutExpired) as exc:
            report["status"]="error"
            report["errors"].append({"stage":"independent_asr","error":type(exc).__name__,"detail":str(exc)})
            return recording["utterances"],report
    return rows,report
