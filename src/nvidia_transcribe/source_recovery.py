"""Selective independent acoustic recovery of still-missing/untimed dialogue.

Occurrence, approximate event timing and source wording are separate claims.
No raw observations, accepted timed turns or original recording are overwritten.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

from nvidia_transcribe.media import extract_wav_segment, validate_wav_file
from nvidia_transcribe.source_quality import _bounds, _digest, _lex, _overlap
from nvidia_transcribe.source_occurrences import stable_id


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


def _independent_candidates(recording, observations):
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
    return targets+candidates


def plan_recovery(recording, observations):
    """Small standalone planner used by the independent resolver controls."""
    return _independent_candidates(recording,observations)[:MAX_TARGETS]


def independent_crops(bounds,duration):
    a,b = bounds
    length = b-a
    return [(max(0.,a-pad),min(duration,b+pad))
            for pad in (max(.5,length*.20),max(1.,length*.60))]


def schedule_independent(recording,observations,*,windows,policy):
    """Select independent work from already observed shared tasks.

    Specialized existing untimed/omission hypotheses must fit a task and
    positive acquisition coverage. Unmatched disputed tasks can request
    cross-model evidence without automatically authorizing replacement.
    """
    duration = recording["source"]["duration_s"]
    observed = [w for w in windows if w.get("state")=="observed"]
    proposals = []
    for task in observed:
        bounds = task["window"]
        task_observations = [o for o in observations if task["task_id"] in o.get("task_ids",[])]
        specialized = _independent_candidates(recording,task_observations)
        relevant = [t for t in specialized if _overlap(t["bounds"],bounds)>0
                    and t["bounds"][1]-t["bounds"][0]>=.65
                    and bounds[0]-.01<=t["bounds"][0]<t["bounds"][1]<=bounds[1]+.01]
        # Prefer an explicit original untimed run over unbound omission.
        candidates = sorted(relevant,key=lambda t:(t["kind"]!="untimed",t["bounds"]))
        if not candidates and (task.get("priority",4)<=2 or
                not any(o.get("utterances") for o in observations
                        if task["task_id"] in o.get("task_ids",[]))):
            candidates = [{"kind":"occurrence","bounds":bounds,
                           "ids":task.get("source_ids",[])}]
        for t in candidates:
            uid = stable_id("independent",[recording["source"]["fingerprint_sha256"],
                                           t["kind"],t["bounds"],t["ids"]])
            old = next((p for p in proposals if p["target_id"]==uid),None)
            if old:
                for key,items in (("task_ids",[task["task_id"]]),("evidence_ids",task["evidence_ids"]),
                                  ("chunks",task["chunks"])):
                    old[key] = sorted(set(old[key]+items))
                continue
            proposals.append({**t,"target_id":uid,"task_ids":[task["task_id"]],
                              "evidence_ids":list(task["evidence_ids"]),"chunks":list(task["chunks"])})
    admitted,deferred = [],[]
    audio_s = 0.
    # Priority must apply to the complete independent queue, not just
    # within each Nemotron task. A later tightly bracketed original run
    # must not be starved by earlier broad occurrence scans.
    task_rank={w["task_id"]:i for i,w in enumerate(observed)}
    task_priority={w["task_id"]:w.get("priority",4) for w in observed}
    proposals.sort(key=lambda t:(
        {"untimed":0,"omission":1,"occurrence":2}[t["kind"]],
        min(task_priority[uid] for uid in t["task_ids"]),
        min(task_rank[uid] for uid in t["task_ids"])))
    for t in proposals:
        crops = independent_crops(t["bounds"],duration)
        cost = sum(b-a for a,b in crops)
        reason = ("insufficient_distinct_context" if crops[0]==crops[1] else
                  "independent_target_budget" if len(admitted)>=policy.max_independent_targets else
                  "independent_audio_budget" if audio_s+cost>policy.max_independent_audio_s else None)
        if reason:
            deferred.append({**t,"state":"deferred","defer_reason":reason})
        else:
            admitted.append({**t,"state":"admitted","crops":crops})
            audio_s += cost
    return admitted,deferred


def _interior(row, observation):
    lo,hi=observation["crop"]
    a,b=_bounds(row)
    return a>lo+.10 and b<hi-.10


def _observed_rows(observation, bounds, *, allow_edge=False):
    a,b = bounds
    lo,hi = observation["crop"]
    rows = []
    for row in observation["rows"]:
        r = _bounds(row)
        if (r and a <= r[0] < r[1] <= b and lo <= r[0] < r[1] <= hi
                and (allow_edge or _interior(row,observation)) and _lex(row["source_text"])):
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
                # Cross-chunk membership is not proof of identity. Bound
                # occurrence rows need explicit shared observation lineage.
                if (row.get("source_unit_ids") or old.get("source_unit_ids")) and not (
                    set(row.get("source_unit_ids",[])) & set(old.get("source_unit_ids",[]))):
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
    associations=[]
    for index,group in grouped:
        lo,hi,_=unique[index]
        origins=[row for a,b,row in spans if lo<=a and b<=hi]
        associations.append((group,origins))
    return associations


def _associate_recoverable_run(original,candidate,language):
    """Recover a supported contiguous run while retaining low-information edges.

    Never drop weak rows, guess their times, or peel substantive conflicting
    rows to force a match. Interior ambiguous rows still need adjudication.
    """
    groups=_associate_originals(original,candidate,language)
    if groups:
        return groups,0,len(original)
    minimum=4 if language.startswith(("zh","ja")) else 3
    lo,hi=0,len(original)
    while lo<hi and len(_lex(original[lo]["source_text"]))<minimum:
        lo+=1
    while hi>lo and len(_lex(original[hi-1]["source_text"]))<minimum:
        hi-=1
    if lo==hi or (lo==0 and hi==len(original)):
        return None
    groups=_associate_originals(original[lo:hi],candidate,language)
    return (groups,lo,hi) if groups else None


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
            "source_occurrence_ids":[uid],
            "source_lineage":{"prior_candidate_source_ids":[r["id"] for r in origins],
                "prior_candidate_span_ids":sorted({sid for r in origins for sid in r.get("occurrence_span_ids",[])}),
                "acoustic_observation_ids":[r["observation_id"] for r in group],
                "source_word_timing_accepted":False},
            "timestamp_provenance":{"backend":"independent_asr_segment_estimates",
               "timing_status":"approximate","human_verified":False,"word_timing_verified":False,
               "audio_sha256":[p["audio_sha256"] for p in pair],
               "observation_ids":[r["observation_id"] for r in group],
               "timing_source_audio_sha256":pair[-1]["audio_sha256"],
               "nemotron_support":evidence}}


def apply_independent(recording, targets, observations, independent):
    rows = [dict(r) for r in recording["utterances"]]
    decisions = []
    lang = recording["source_language"]
    for index,t in enumerate(targets):
        if t["kind"] not in ("untimed","omission"):
            continue
        target_id = t.get("target_id",f"challenge-{index}")
        pair = [o for o in independent if o["id"] in (target_id+"-0",target_id+"-1")]
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
        # A boundary segment is a possibly truncated observation, not a
        # reliable timing authority. Retain its corroborative value only
        # when the complete lexical sequence agrees with an interior
        # observation from a separately extracted crop. Never use two
        # edge-only observations to supply event timing.
        first,second = [_observed_rows(o,bracket,allow_edge=True) for o in pair]
        if not first or not second:
            continue
        interior=[all(_interior(r,o) for r in seq)
                  for o,seq in zip(pair,(first,second))]
        edge_support=not all(interior)
        if not any(interior):
            continue
        if edge_support and _sequence(first)!=_sequence(second):
            continue
        if not interior[1]:
            pair.reverse()
            first,second=second,first
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
        residual_ids=[]
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
            association=_associate_recoverable_run(originals,second,lang)
            if not association:
                continue
            groups,lo,hi=association
            resolved=originals[lo:hi]
            residual=[*originals[:lo],*originals[hi:]]
            chunks.update(c for r in resolved for c in r.get("chunk_indices",[]))
            replacement=[]
            for group,associated in groups:
                new=_make_row(recording,group,resolved,pair,support,chunks)
                new["source_lineage"]["associated_prior_source_ids"]=[r["id"] for r in associated]
                if len(associated)==1:
                    original=associated[0]
                    uid=original["id"]
                    new.update(id=uid,utterance_id=uid,source_turn_id=uid,
                               source_turn_ids=[uid],source_occurrence_ids=[uid])
                    for key in ("source_unit_ids","recognized_turn_ids","occurrence_span_ids"):
                        new[key]=list(original.get(key,[]))
                replacement.append(new)
            if len({r["id"] for r in replacement})!=len(replacement):
                continue
            # Preserve unresolved original fragments in their original
            # relative order, with no fabricated duration or deletion.
            rows[positions[0]:positions[-1]+1]=[
                *[{**r,"source_quality_status":"unresolved_fragment_timing"} for r in originals[:lo]],
                *replacement,
                *[{**r,"source_quality_status":"unresolved_fragment_timing"} for r in originals[hi:]]]
            residual_ids=[r["id"] for r in residual]
        else:
            # A competing timed or untimed lexical occurrence can be a
            # displaced duplicate. Do not insert in that situation.
            if (len({r["audio_sha256"] for r in support})<2 or
                    any(_lex(r["source_text"])==y or
                        (_anchor(y,_lex(r["source_text"]),lang)
                         and (not _bounds(r) or abs(r["source_start"]-a)<60)) for r in rows)):
                continue
            # No merging distinct unverified speech turns.
            if len(first)!=1 or len(second)!=1:
                continue
            replacement=[_make_row(recording,second,[],pair,support,chunks)]
            pos=next((i for i,r in enumerate(rows) if _bounds(r) and r["source_start"]>span[0]),len(rows))
            rows[pos:pos]=replacement
        decisions.append({"status":"unresolved_wording_independent_recovery","kind":t["kind"],
                          "target":t,"window":span,"source_ids":[r["id"] for r in replacement],
                          "trigger_evidence_ids":list(t.get("evidence_ids",[])),
                          "recovery_task_ids":list(t.get("task_ids",[])),
                          "observation_ids":[r["observation_id"] for r in second],
                          "edge_observation_used_as_corroboration":edge_support,
                          "timing_source_audio_sha256":pair[-1]["audio_sha256"],
                          "residual_untimed_source_ids":residual_ids})
    return rows,decisions


def acquire_independent_evidence(recording,targets,media,*,challenger_resource,output_dir,policy):
    report={"status":"not_configured","targets":targets,"decisions":[],"errors":[],
            "observations":[],"audio_seconds":0.,"requests":0}
    if not targets:
        report["status"]="no_targets"
        return report
    if challenger_resource is None:
        return report
    language=recording["source_language"]
    if not isinstance(language,str) or len(language.split("-")[0])!=2:
        report["status"]="unsupported_language"
        return report
    started = time.monotonic()
    try:
        if (challenger_resource.adapter!="whisper_cpu" or challenger_resource.device!="cpu"
                or not Path(challenger_resource.worker_python).is_file()
                or not Path(challenger_resource.checkpoint_path).is_dir()):
            raise ValueError("Explicit cached CPU challenger resource unavailable")
        with tempfile.TemporaryDirectory(prefix="independent-source-",dir=output_dir) as tmp:
            root=Path(tmp)
            crops=[]
            for t in targets:
                for j,(lo,hi) in enumerate(t["crops"]):
                    path=root/(t["target_id"]+f"-{j}.wav")
                    extract_wav_segment(media.path,start_s=lo,duration_s=hi-lo,out_wav=path,
                                        audio_stream_index=media.selected_audio_stream)
                    duration,digest=validate_wav_file(path)
                    if abs(duration-(hi-lo))>.1:
                        raise ValueError("Independent crop duration mismatch")
                    crops.append({"id":path.stem,"path":str(path),"start":lo,"end":hi,"audio_sha256":digest})
            report["audio_seconds"]=round(sum(c["end"]-c["start"] for c in crops),6)
            if report["audio_seconds"]>policy.max_independent_audio_s+1e-6:
                raise ValueError("Independent audio budget exceeded")
            request={"checkpoint":challenger_resource.checkpoint_path,
                     "language":language.split("-")[0],"crops":crops}
            remaining=policy.max_independent_wall_s-(time.monotonic()-started)
            if remaining<=0:
                raise TimeoutError("Independent preparation exhausted wall budget")
            report["requests"]=len(crops)
            proc=subprocess.run([challenger_resource.worker_python,
                str(Path(__file__).with_name("_whisper_challenge_worker.py"))],
                input=json.dumps(request),text=True,capture_output=True,timeout=remaining,
                env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","CUDA_VISIBLE_DEVICES":""})
            if proc.returncode:
                raise RuntimeError(f"Independent worker exited {proc.returncode}: {proc.stderr[-2000:]}")
            result=json.loads(proc.stdout)
            expected={c["id"]:(c["audio_sha256"],[c["start"],c["end"]]) for c in crops}
            actual={o["id"]:(o["audio_sha256"],o["crop"]) for o in result["observations"]}
            if (expected!=actual or len(actual)!=len(result["observations"])
                    or result["device"]!="cpu" or result["backend"]!="faster-whisper"
                    or result["compute_type"]!="int8"):
                raise ValueError("Independent worker receipt mismatch")
            seen=set()
            for obs in result["observations"]:
                for row in obs["rows"]:
                    b=_bounds(row)
                    if (not b or not obs["crop"][0]<=b[0]<b[1]<=obs["crop"][1]+.01
                            or row["observation_id"] in seen):
                        raise ValueError("Invalid independent observation identity or bounds")
                    seen.add(row["observation_id"])
            report["observations"]=result["observations"]
            report["resource"]={**{k:v for k,v in result.items() if k!="observations"},
                "resource_name":challenger_resource.name,
                "resource_fingerprint":challenger_resource.fingerprint()}
            report["receipt_sha256"]=_digest(result)
            report["status"]="observed"
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.TimeoutExpired) as exc:
        report["status"]="error"
        report["errors"].append({"stage":"independent_asr","error":type(exc).__name__,"detail":str(exc)})
    report["wall_s"]=round(time.monotonic()-started,4)
    return report
