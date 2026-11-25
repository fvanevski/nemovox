"""Occurrence-level resolution over one shared acoustic evidence inventory.

The existing strict event and ordered independent-run predicates remain
acceptance safeguards. Acquisition is completed before this stage. Decisions
bind actual evidence, not a text-only proposal, task proximity or a vote.
"""
from __future__ import annotations

from nvidia_transcribe.source_quality import (
    _bounds, _lex, _overlap, _same_event_bounds, _recover_untimed,
    select_source_events, _observation_groups,
)
from nvidia_transcribe.source_recovery import apply_independent


def cross_model_events(observations,targets,recovery,policy):
    """Exact, unique event agreement across two CPU crops and native ASR.

    Independent evidence can strengthen an occurrence hypothesis; no
    independent words become forced alignment or speaker truth.
    """
    events=[]
    independent=recovery.get("observations",[])
    for target in targets:
        tid=target["target_id"]
        pair=[o for o in independent if o["id"] in (tid+"-0",tid+"-1")]
        if (len(pair)!=2 or pair[0]["audio_sha256"]==pair[1]["audio_sha256"]
                or pair[0]["crop"]==pair[1]["crop"]):
            continue
        pairs=[]
        for a in pair[0]["rows"]:
            for b in pair[1]["rows"]:
                ab,bb=_bounds(a),_bounds(b)
                if (not ab or not bb or len(_lex(a["source_text"]))<2
                        or _lex(a["source_text"])!=_lex(b["source_text"])
                        or not _same_event_bounds(ab,bb,policy.timing_tolerance_s)
                        or not target["bounds"][0]<=sum(bb)/2<=target["bounds"][1]):
                    continue
                if not any(o["crop"][0]+.1 < x[0] < x[1] < o["crop"][1]-.1
                           for o,x in zip(pair,(ab,bb))):
                    continue
                pairs.append((a,b))
        for a,b in pairs:
            ids={a["observation_id"],b["observation_id"]}
            if sum(bool(ids & {x["observation_id"],y["observation_id"]}) for x,y in pairs)!=1:
                continue
            key=_lex(b["source_text"])
            support=[]
            ambiguous=False
            for obs in observations:
                if not set(obs.get("task_ids",[])).intersection(target["task_ids"]):
                    continue
                matches=[g for g in _observation_groups(obs)
                         if _lex(g["source_text"])==key
                         and _same_event_bounds(_bounds(g),_bounds(b),policy.timing_tolerance_s)]
                if len(matches)>1:
                    ambiguous=True
                    break
                if matches:
                    support.append((obs,matches[0]))
            if ambiguous or not support:
                continue
            # Reject incompatible concurrent lexical alternatives rather
            # than silently select one voice from a multi-voice collision.
            selected=support[0][1]
            if any(not _same_event_bounds(_bounds(selected),_bounds(g),policy.timing_tolerance_s)
                   for _,g in support):
                continue
            event={**selected,
                "supporting_observation_ids":sorted(ids|{uid for _,g in support for uid in g["member_ids"]}),
                "independent_observation_ids":sorted(ids),
                "independent_receipt_sha256":recovery.get("receipt_sha256"),
                "audio_sha256":sorted({o["audio_sha256"] for o in pair}|{o["audio_sha256"] for o,_ in support}),
                "raw_response_sha256":sorted({o["raw_response_sha256"] for o,_ in support}),
                "context_crops":[list(o["crop"]) for o in pair]+[list(o["crop"]) for o,_ in support],
                "chunks":target["chunks"],"trigger_evidence_ids":target["evidence_ids"],
                "recovery_task_ids":target["task_ids"]}
            events.append(event)
    return events


def consolidate_events(events,policy):
    """Reuse repeated observations without multiplying an event or crossing voices."""
    conflicts=[]
    blocked=set()
    for i,event in enumerate(events):
        for j,other in enumerate(events[:i]):
            shared=set(event["supporting_observation_ids"]) & set(other["supporting_observation_ids"])
            if shared and not (_lex(event["source_text"])==_lex(other["source_text"])
                    and _same_event_bounds(_bounds(event),_bounds(other),policy.timing_tolerance_s)):
                blocked.update((i,j))
                conflicts.append({"status":"unresolved_conflicting_observation_correspondence",
                    "observations":sorted(set(event["supporting_observation_ids"]+other["supporting_observation_ids"])),
                    "trigger_evidence_ids":sorted(set(event.get("trigger_evidence_ids",[])+other.get("trigger_evidence_ids",[]))),
                    "windows":[_bounds(event),_bounds(other)]})
    result=[]
    for i,event in enumerate(events):
        if i in blocked:
            continue
        hits=[e for e in result if _lex(e["source_text"])==_lex(event["source_text"])
              and _same_event_bounds(_bounds(e),_bounds(event),policy.timing_tolerance_s)
              and set(e["supporting_observation_ids"]).intersection(event["supporting_observation_ids"])]
        if len(hits)==1:
            e=hits[0]
            for key in ("supporting_observation_ids","audio_sha256","raw_response_sha256",
                        "trigger_evidence_ids","recovery_task_ids","independent_observation_ids","chunks"):
                e[key]=sorted(set(e.get(key,[])+event.get(key,[])))
            if event.get("independent_receipt_sha256"):
                e["independent_receipt_sha256"]=event["independent_receipt_sha256"]
        else:
            result.append({**event})
    return result,conflicts


def resolve_occurrences(recording,events,observations,targets,recovery,policy):
    cross=cross_model_events(observations,targets,recovery,policy)
    combined,conflicts=consolidate_events([*events,*cross],policy)
    timed,timing_decisions=_recover_untimed(recording,observations,policy)
    rows,decisions=select_source_events(timed,combined,policy)
    rows,independent_decisions=apply_independent(
        {**recording,"utterances":rows},targets,observations,recovery.get("observations",[]))
    recovery["decisions"]=independent_decisions
    recovery["cross_model_events"]=cross
    if recovery["status"]=="observed":
        recovery["status"]="partial" if independent_decisions or cross else "unresolved"
    return rows,[*conflicts,*timing_decisions,*decisions,*independent_decisions]


def resolution_receipts(recording,ledger,rows,decisions,policy):
    """Relate disputed inventory to actual canonical rows, never infer
    that all evidence in a scheduled window has been resolved."""
    units={u["id"]:u for u in ledger.get("lexical_observations",[])}
    results=[]
    for inv in ledger.get("recognized_occurrence_inventories",[]):
        for span in inv.get("locality",{}).get("spans",[]):
            ds=span.get("selection_dispositions",[])
            context=any(d["status"]=="context_candidate_unresolved" for d in ds)
            missing=span["locality_status"] in ("partial","unlocalized") and not span.get("post_alignment_usable_bounds_s")
            if not (context or missing or span["locality_status"]=="disputed"):
                continue
            sid=span["id"]
            refs=set(span["source_unit_ids"])
            receipt={"occurrence_span_id":sid,"source_unit_ids":sorted(refs),
                     "status":"unresolved","canonical_source_ids":[],
                     "acoustic_truth_accepted":False}
            # A unique, explicitly recovered whole origin row may establish
            # coarse timing while its wording remains uncertain.
            associated=[r for r in rows if refs<=set(r.get("source_unit_ids",[])) and _bounds(r)
                        and r.get("timestamp_provenance",{}).get("backend") in (
                            "separate_acoustic_contexts","independent_asr_segment_estimates")]
            if refs and len(associated)==1 and not context:
                row=associated[0]
                receipt["canonical_source_ids"]=[row["id"]]
                receipt["status"]=("timing_supported_wording_unresolved" if
                    str(row.get("source_quality_status","")).startswith("unresolved")
                    or row.get("source_semantic_resolution")=="partial"
                    else "represented_provisional")
            # Context-only source can be linked only by exact positive
            # lexical and temporal correspondence to a unique selected row,
            # not merely by a coalesced task or shared recognized turn.
            elif context and refs and all(uid in units for uid in span["source_unit_ids"]):
                # Source ledger retains normalized lexical text/order; use
                # the same lexical sequence rather than reconstruct timing.
                tokens=tuple(t for uid in span["source_unit_ids"] for t in _lex(units[uid]["text"]))
                from nvidia_transcribe.recovery_planner import interval
                candidate=interval(span.get("candidate_bounds_s"),recording["source"]["duration_s"])
                supported={d["source_id"] for d in decisions
                           if d.get("source_id") and sid in d.get("trigger_evidence_ids",[])
                           and d["status"] in ("already_represented","recovered_corroborated_turn",
                                               "timed_corroborated_turn","extended_corroborated_turn")}
                matches=[r for r in rows if r["id"] in supported and _lex(r["source_text"])==tokens
                         and candidate and _same_event_bounds(_bounds(r),candidate,policy.timing_tolerance_s)]
                if len(matches)==1:
                    receipt["canonical_source_ids"]=[matches[0]["id"]]
                    receipt["status"]="represented_provisional"
            results.append(receipt)
    return results
