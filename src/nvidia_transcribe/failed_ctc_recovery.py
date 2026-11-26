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
from nvidia_transcribe.ordered_event_anchors import ordered_proposals, select_ordered_proposals
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
    if len({r["id"] for r in result})!=len(result):
        raise ValueError("Recovered source subspan ID collision")
    return result,decisions
