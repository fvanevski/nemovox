"""Event-level localization when a whole primary CTC operation fails.

Native generation is still an occurrence hypothesis. Independent segment
estimates can localize a unique source subspan, with positive native
recognition in a separate crop. Neither forced alignment nor a neighboring
timing bracket establishes occurrence identity. Never fabricate word times.
"""
from __future__ import annotations

from nvidia_transcribe.source_occurrences import stable_id
from nvidia_transcribe.source_quality import _bounds, _lex
from nvidia_transcribe.semantic_context import text_digest
from nvidia_transcribe.ordered_event_anchors import ordered_proposals, select_ordered_proposals, ambiguous_local_events
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


def recover_failed_ctc_events(recording,ledger,windows,observations,targets,independent,policy):
    """Use the existing observed-task/worker inventory; no additional model calls."""
    failed=failed_ctc_parents(ledger)
    rows=list(recording["utterances"])
    if not failed:
        return rows,[]
    sources={r["id"]:s for r in rows if (s:=_source_units(r,failed)) is not None}
    proposals=ordered_proposals(recording,ledger,sources,windows,observations,targets,independent,policy)
    selected=select_ordered_proposals(proposals,rows,policy.timing_tolerance_s)
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
                    source_quality_status="event_timing_recovered_original_wording",
                    wording_status="model_wording_corroborated" if p["wording_exact"] else "model_variants_unresolved",
                    timestamp_provenance={"backend":"independent_asr_segment_estimates",
                        "timing_status":"approximate","human_verified":False,"word_timing_verified":False,
                        "audio_sha256":p["independent_audio_sha256"],"observation_ids":p["observations"]})
                new["source_lineage"].update(acoustic_observation_ids=p["observations"],
                    native_audio_sha256=p["native_audio_sha256"],
                    native_raw_response_sha256=p["native_raw_response_sha256"],
                    native_primary_observation_id=p["native_primary_observation_id"],
                    localized_wording_evidence={"source_id":uid,"source_text_sha256":text_digest(new["source_text"]),
                        "source_unit_ids":ids,"status":"advisory_model_hypotheses","entity_identity_accepted":False,
                        "alternatives":[{**a,"source_unit_range":[a["source_unit_range"][0]-lo,a["source_unit_range"][1]-lo]}
                                        for a in p["wording_alternatives"]],
                        "constituent_event_evidence":p["constituent_event_evidence"]})
                decisions.append({**p,"status":"timed_corroborated_failed_ctc_subspan",
                    "source_id":uid,"source_unit_ids":ids,"independent_observation_ids":p["independent_observation_ids"],
                    "source_truth_human_verified":False})
            result.append(new)
    # An acoustic occurrence can be real while its original source-row
    # identity remains ambiguous. Render positively corroborated events
    # without assigning or consuming either competing primary hypothesis.
    for p in ambiguous_local_events(recording,sources,windows,observations,targets,independent,result,policy):
        lang=recording["source_language"]
        text=("".join if lang.startswith(("zh","ja")) else " ".join)(r["source_text"] for r in p["group"])
        uid=stable_id("utt",[recording["source"]["fingerprint_sha256"],p["observations"],p["window"]])
        candidate_ids=sorted(r["id"] for r in p["origins"])
        row={"id":uid,"utterance_id":uid,"source_turn_id":uid,"source_turn_ids":[uid],
             "source_occurrence_ids":[uid],"source_unit_ids":[],"recognized_turn_ids":[],"occurrence_span_ids":[],
             "source_language":lang,"source_text":text,"source_start":p["window"][0],"source_end":p["window"][1],
             "speaker_id":"SPK_UNK","speaker_identity_accepted":False,"local_speaker_id":None,"raw_speaker_tag":None,
             "source_words":[],"source_characters":[],"unit_count":0,"timed_unit_count":0,"untimed_unit_count":0,
             "timing_gaps":[],"event_refs":[],
             "chunk_indices":sorted({c for r in p["origins"] for c in r.get("chunk_indices",[])}
                 | {parent["chunk_index"] for parent,_ in sources.values()
                    if parent["core_start_s"]<=p["window"][0]<p["window"][1]<=parent["core_end_s"]}),
             "alignment_status":"independent_event_bounds_approximate",
             "source_quality_status":("unresolved_wording_ambiguous_origin_event" if p["ambiguous"]
                                      else "unresolved_wording_unbound_origin_event"),
             "prior_source_hypotheses":p["origins"],
             "source_lineage":{"prior_candidate_source_ids":candidate_ids,"primary_origin_accepted":False,
                "acoustic_observation_ids":p["observations"],"source_word_timing_accepted":False,
                "native_audio_sha256":p["native_audio_sha256"],"native_raw_response_sha256":p["native_raw_response_sha256"]},
             "timestamp_provenance":{"backend":"independent_asr_segment_estimates","timing_status":"approximate",
                "human_verified":False,"word_timing_verified":False,"audio_sha256":p["audio_sha256"],
                "observation_ids":p["observations"]}}
        insert=next((i for i,r in enumerate(result) if _bounds(r) and r["source_start"]>row["source_start"]),len(result))
        result.insert(insert,row)
        decisions.append({"status":("localized_event_ambiguous_primary_origin" if p["ambiguous"]
                                    else "localized_event_unbound_primary_origin"),"source_id":uid,
            "candidate_origin_ids":candidate_ids,"source_unit_ids":[],"window":p["window"],
            "observations":p["observations"],"independent_observation_ids":p["observations"],
            "audio_sha256":p["audio_sha256"],"native_audio_sha256":p["native_audio_sha256"],
            "native_raw_response_sha256":p["native_raw_response_sha256"],
            "trigger_evidence_ids":p["trigger_evidence_ids"],"recovery_task_ids":p["recovery_task_ids"],
            "primary_origin_accepted":False,"source_truth_human_verified":False})
    if len({r["id"] for r in result})!=len(result):
        raise ValueError("Recovered source subspan ID collision")
    return result,decisions
