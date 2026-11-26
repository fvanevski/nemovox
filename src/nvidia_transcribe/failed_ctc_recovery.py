"""Event-level localization when a whole primary CTC operation fails.

Native generation is still an occurrence hypothesis. Independent segment
estimates can localize a unique source subspan, with positive native
recognition in a separate crop. Neither forced alignment nor a neighboring
timing bracket establishes occurrence identity. Never fabricate word times.
"""
from __future__ import annotations

from itertools import combinations

from nvidia_transcribe.source_occurrences import stable_id
from nvidia_transcribe.source_quality import _bounds, _lex, _contains, _overlap, _same_event_bounds
from nvidia_transcribe.utterances import join_multilingual_units


def failed_ctc_parents(ledger):
    exact={i["asr_observation_id"] for i in ledger.get("recognized_occurrence_inventories",[])
           if i.get("projection_status")=="exact"}
    by_parent={}
    for u in ledger.get("lexical_observations",[]):
        by_parent.setdefault(u.get("asr_observation_id"),[]).append(u)
    result={}
    for p in ledger.get("asr_observations",[]):
        units=by_parent.get(p["id"],[])
        if (p["id"] in exact and units and all(u.get("timing_reason")=="alignment_failed"
                and u.get("ctc_association_status")=="native_error"
                and u.get("ctc_start_s") is None and u.get("ctc_end_s") is None
                and u.get("raw_recognized_token_ids") for u in units)):
            result[p["chunk_index"]]=(p,units)
    return result


def _groups(observation,bounds,tolerance):
    rows=observation.get("rows",[])
    for i,first in enumerate(rows):
        if not _bounds(first):
            continue
        group=[]
        for r in rows[i:i+5]:
            rb=_bounds(r)
            if not rb or (group and not -.05<=rb[0]-group[-1]["source_end"]<=1.5):
                break
            group.append(r)
            span=(_bounds(first)[0],rb[1])
            if (bounds[0]<=sum(span)/2<=bounds[1]
                    and bounds[0]-tolerance<=span[0]<span[1]<=bounds[1]+tolerance
                    and span[1]-span[0]>=.65):
                words=tuple(w for x in group for w in _lex(x["source_text"]))
                crop=observation["crop"]
                if crop[0]-.01<=span[0]<span[1]<=crop[1]+.01:
                    yield (list(group),span,words,crop[0]+.1<span[0] and span[1]<crop[1]-.1)


def _source_units(row,failed):
    chunks=row.get("chunk_indices",[])
    if len(chunks)!=1 or chunks[0] not in failed or _bounds(row) or row.get("event_refs"):
        return None
    p,original=failed[chunks[0]]
    lookup={u["id"]:u for u in original}
    units=list(row.get("source_words",[]))+list(row.get("source_characters",[]))
    if (not units or any(u.get("valid_timing") or u.get("source_unit_id") not in lookup for u in units)):
        return None
    units.sort(key=lambda u:lookup[u["source_unit_id"]]["source_ordinal"])
    ids=[u["source_unit_id"] for u in units]
    if (len(set(ids))!=len(ids) or set(ids)!=set(row.get("source_unit_ids",[]))
            or any(lookup[uid].get("boundary_disposition")!="selected_unlocalized" for uid in ids)
            or len({lookup[uid]["recognized_turn_id"] for uid in ids})!=1
            or any(not _lex(u.get("text")) for u in units)
            or join_multilingual_units(units)!=row["source_text"]):
        return None
    return p,units


def _unit_slice(units,words):
    tokens=tuple(w for u in units for w in _lex(u["text"]))
    offsets=[0]
    for u in units:
        offsets.append(offsets[-1]+len(_lex(u["text"])))
    ends={offset:i for i,offset in enumerate(offsets)}
    matches=[(i,ends[offset+len(words)]) for i,offset in enumerate(offsets[:-1])
             if offset+len(words) in ends and tokens[offset:offset+len(words)]==words]
    return matches[0] if len(matches)==1 else None


def _native_support(native,words,bounds):
    result=[]
    for o in native:
        if (o.get("integrity",{}).get("status")!="complete"
                or o.get("integrity",{}).get("finish_reason")!="stop"
                or not o["crop"][0]+.1<bounds[0]<bounds[1]<o["crop"][1]-.1):
            continue
        raw=o.get("raw_response",{})
        content=((raw.get("choices") or [{}])[0].get("message") or {}).get("content")
        if isinstance(content,str) and _contains(_lex(content),words):
            result.append(o)
    return result


def recover_failed_ctc_events(recording,ledger,windows,observations,targets,independent,policy):
    """Use the existing observed-task/worker inventory; no additional model calls."""
    failed=failed_ctc_parents(ledger)
    rows=list(recording["utterances"])
    if not failed:
        return rows,[]
    sources={r["id"]:s for r in rows if (s:=_source_units(r,failed)) is not None}
    rowmap={r["id"]:r for r in rows}
    observed={w["task_id"]:w for w in windows if w.get("state")=="observed"}
    minimum=4 if recording["source_language"].startswith(("zh","ja")) else 3
    tolerance=policy.timing_tolerance_s
    proposals=[]
    for target in targets:
        task_ids=set(target["task_ids"]) & set(observed)
        if not task_ids:
            continue
        pair=[o for o in independent if o["id"] in
              (target["target_id"]+"-0",target["target_id"]+"-1")]
        if len(pair)!=2 or pair[0]["audio_sha256"]==pair[1]["audio_sha256"] or pair[0]["crop"]==pair[1]["crop"]:
            continue
        native=[o for o in observations if task_ids.intersection(o.get("task_ids",[]))]
        if not native:
            continue
        first,second=[list(_groups(o,target["bounds"],tolerance)) for o in pair]
        for a in first:
            if len(a[2])<minimum or len(set(a[2]))<3:
                continue
            matches=[b for b in second if b[2]==a[2] and (a[3] or b[3])
                     and _same_event_bounds(a[1],b[1],tolerance)]
            if len(matches)!=1:
                continue
            b=matches[0]
            if sum(x[2]==b[2] and _same_event_bounds(x[1],b[1],tolerance) for x in first)!=1:
                continue
            # A genuinely repeated source sequence cannot identify one origin
            # solely because one occurrence lies in a convenient crop.
            origins=[r for r in rows if _contains(_lex(r["source_text"]),a[2])]
            if len(origins)!=1 or origins[0]["id"] not in sources:
                continue
            row=origins[0]
            p,units=sources[row["id"]]
            loc=_unit_slice(units,a[2])
            if not loc or p["chunk_index"] not in target.get("chunks",[]):
                continue
            if not all(p["core_start_s"]<=x[1][0]<x[1][1]<=p["core_end_s"] for x in (a,b)):
                continue
            timing=b if b[3] else a
            span=timing[1]
            if any(_bounds(r) and _overlap(_bounds(r),span)>.08 for r in rows):
                continue
            support=_native_support(native,a[2],span)
            if not support:
                continue
            proposals.append({"origin":row["id"],"lo":loc[0],"hi":loc[1],"window":span,
                "observations":sorted({r["observation_id"] for g in (a,b) for r in g[0]}),
                "independent_audio_sha256":sorted({o["audio_sha256"] for o in pair}),
                "native_audio_sha256":sorted({o["audio_sha256"] for o in support}),
                "native_raw_response_sha256":sorted({o["raw_response_sha256"] for o in support
                                                     if o.get("raw_response_sha256")}),
                "native_primary_observation_id":p["id"],
                "trigger_evidence_ids":list(target.get("evidence_ids",[])),
                "recovery_task_ids":sorted(task_ids)})
    # Reject mutually incompatible mappings rather than electing by crop
    # order. Compatible nested subspans prefer the maximal supported span.
    bad=set()
    nested=set()
    for i,j in combinations(range(len(proposals)),2):
        a,b=proposals[i],proposals[j]
        same=a["origin"]==b["origin"]
        lex_overlap=same and max(a["lo"],b["lo"])<min(a["hi"],b["hi"])
        time_overlap=_overlap(a["window"],b["window"])>.08
        shared=set(a["observations"]) & set(b["observations"])
        if same and a["lo"]==b["lo"] and a["hi"]==b["hi"]:
            if _same_event_bounds(a["window"],b["window"],tolerance):
                # Equivalent evidence; retain deterministic first instance.
                nested.add(j)
            else:
                bad.update((i,j))
        elif lex_overlap:
            outer,inner,ii=(a,b,j) if a["lo"]<=b["lo"] and b["hi"]<=a["hi"] else (
                (b,a,i) if b["lo"]<=a["lo"] and a["hi"]<=b["hi"] else (None,None,None))
            if (outer is not None and outer["window"][0]-tolerance<=inner["window"][0]
                    and inner["window"][1]<=outer["window"][1]+tolerance and time_overlap):
                nested.add(ii)
            else:
                bad.update((i,j))
        elif time_overlap or shared or (same and
              (a["hi"]<=b["lo"] and a["window"][1]>b["window"][0]+.08
               or b["hi"]<=a["lo"] and b["window"][1]>a["window"][0]+.08)):
            bad.update((i,j))
        elif not same:
            ar,br=rowmap[a["origin"]],rowmap[b["origin"]]
            if set(ar.get("recognized_turn_ids",[])) & set(br.get("recognized_turn_ids",[])):
                order_a=rows.index(ar)
                order_b=rows.index(br)
                if (order_a<order_b and a["window"][0]>b["window"][0]
                        or order_b<order_a and b["window"][0]>a["window"][0]):
                    bad.update((i,j))
    selected=[p for i,p in enumerate(proposals) if i not in bad|nested]
    result,decisions=[],[]
    for row in rows:
        chosen=sorted((p for p in selected if p["origin"]==row["id"]),key=lambda p:p["lo"])
        if not chosen:
            result.append(row)
            continue
        _,units=sources[row["id"]]
        pieces=[]
        cursor=0
        for p in chosen:
            if cursor<p["lo"]:
                pieces.append((cursor,p["lo"],None))
            pieces.append((p["lo"],p["hi"],p))
            cursor=p["hi"]
        if cursor<len(units):
            pieces.append((cursor,len(units),None))
        for lo,hi,p in pieces:
            part=units[lo:hi]
            ids=[u["source_unit_id"] for u in part]
            uid=row["id"] if lo==0 else stable_id("utt",[row["id"],ids])
            new={**row,"id":uid,"utterance_id":uid,"source_turn_id":uid,
                 "source_turn_ids":[uid],"source_occurrence_ids":[uid],
                 "source_text":join_multilingual_units(part),"source_unit_ids":ids,
                 "occurrence_span_ids":sorted({u["occurrence_span_id"] for u in part
                                               if u.get("occurrence_span_id")}),
                 "source_words":[u for u in part if u.get("unit_type")!="character"],
                 "source_characters":[u for u in part if u.get("unit_type")=="character"],
                 "unit_count":len(part),"timed_unit_count":0,"untimed_unit_count":len(part),
                 "timing_gaps":[],"prior_source_hypotheses":[row],
                 "source_lineage":{"prior_source_ids":[row["id"]],
                                   "source_word_timing_accepted":False}}
            if p is None:
                new["source_quality_status"]="unresolved_fragment_timing"
                new["timestamp_provenance"]={"backend":"none","timing_status":"failed",
                    "unit_count":len(part),"timed_unit_count":0,"untimed_unit_count":len(part)}
            else:
                new.update(source_start=p["window"][0],source_end=p["window"][1],
                    speaker_id="SPK_UNK",speaker_identity_accepted=False,
                    local_speaker_id=None,raw_speaker_tag=None,source_words=[],source_characters=[],
                    unit_count=0,timed_unit_count=0,untimed_unit_count=0,
                    alignment_status="independent_event_bounds_approximate",
                    source_quality_status="unresolved_wording_independent_recovery",
                    timestamp_provenance={"backend":"independent_asr_segment_estimates",
                        "timing_status":"approximate","human_verified":False,"word_timing_verified":False,
                        "audio_sha256":p["independent_audio_sha256"],"observation_ids":p["observations"]})
                new["source_lineage"].update(acoustic_observation_ids=p["observations"],
                    native_audio_sha256=p["native_audio_sha256"],
                    native_raw_response_sha256=p["native_raw_response_sha256"],
                    native_primary_observation_id=p["native_primary_observation_id"])
                decisions.append({**p,"status":"timed_corroborated_failed_ctc_subspan",
                    "source_id":uid,"source_unit_ids":ids,"independent_observation_ids":p["observations"],
                    "source_truth_human_verified":False})
            result.append(new)
    if len({r["id"] for r in result})!=len(result):
        raise ValueError("Recovered source subspan ID collision")
    return result,decisions
