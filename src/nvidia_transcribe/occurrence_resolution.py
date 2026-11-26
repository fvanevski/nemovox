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
                        or not all(target["bounds"][0]<=sum(x)/2<=target["bounds"][1]
                                   for x in (ab,bb))):
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
                         and all(_same_event_bounds(_bounds(g),_bounds(r),policy.timing_tolerance_s)
                                 for r in (a,b))
                         and target["bounds"][0]<=sum(_bounds(g))/2<=target["bounds"][1]]
                if len(matches)>1:
                    ambiguous=True
                    break
                if matches:
                    support.append((obs,matches[0]))
            if ambiguous or not support:
                continue
            # All selected native supports must agree on both boundaries.
            # Distinct concurrent hypotheses remain separate observations.
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
                "trigger_source_ids":target.get("ids",[]),
                "recovery_task_ids":target["task_ids"]}
            events.append(event)
    return events


def consolidate_events(events,policy):
    """Partition by shared observation membership, not transitive timing.

    Each observation belongs to at most one accepted event. A bridge may
    unite several equivalent candidates, but pairwise locality must hold
    across the entire connected component before anything is selected.
    """
    members=[set(event["supporting_observation_ids"]) for event in events]
    remaining=set(range(len(events)))
    result,conflicts=[],[]
    while remaining:
        first=min(remaining)
        remaining.remove(first)
        component=[first]
        support=set(members[first])
        while True:
            linked=sorted(i for i in remaining if members[i] & support)
            if not linked:
                break
            remaining.difference_update(linked)
            component.extend(linked)
            for i in linked:
                support.update(members[i])
        group=[events[i] for i in sorted(component)]
        coherent=all(_lex(a["source_text"])==_lex(b["source_text"])
                     and _same_event_bounds(_bounds(a),_bounds(b),policy.timing_tolerance_s)
                     for i,a in enumerate(group) for b in group[:i])
        if not coherent:
            conflicts.append({
                "status":"unresolved_conflicting_observation_correspondence",
                "observations":sorted(support),
                "trigger_evidence_ids":sorted({uid for e in group for uid in e.get("trigger_evidence_ids",[])}),
                "recovery_task_ids":sorted({uid for e in group for uid in e.get("recovery_task_ids",[])}),
                "windows":[_bounds(e) for e in group]})
            continue
        merged=dict(group[0])
        if len(group)>1:
            for key in ("supporting_observation_ids","audio_sha256","raw_response_sha256",
                        "trigger_evidence_ids","trigger_source_ids","recovery_task_ids",
                        "independent_observation_ids","chunks"):
                merged[key]=sorted({v for e in group for v in e.get(key,[])})
            merged["context_crops"]=[list(c) for c in sorted({
                tuple(c) for e in group for c in e.get("context_crops",[])})]
            receipts={e["independent_receipt_sha256"] for e in group if e.get("independent_receipt_sha256")}
            if len(receipts)>1:
                raise ValueError("Conflicting independent worker receipts in one acquisition")
            if receipts:
                merged["independent_receipt_sha256"]=next(iter(receipts))
        result.append(merged)
    return result,conflicts


def resolve_occurrences(recording,events,observations,targets,recovery,policy,*,ledger=None,windows=()):
    targeted=[]
    independent=recovery.get("observations",[])
    if ledger is not None:
        from nvidia_transcribe.targeted_occurrences import adjudicate_sequences, recover_unlocalized_events
        rows,first=adjudicate_sequences(recording,ledger,windows,observations,targets,independent)
        rows,second=recover_unlocalized_events({**recording,"utterances":rows},
                                              windows,observations,targets,independent)
        from nvidia_transcribe.failed_ctc_recovery import recover_failed_ctc_events
        rows,third=recover_failed_ctc_events({**recording,"utterances":rows},ledger,
            windows,observations,targets,independent,policy)
        targeted=first+second+third
        recording={**recording,"utterances":rows}
        # Special tasks are adjudicated atomically at sequence/event level.
        # Do not reuse their observations for individual fragment insertion.
        special={w["task_id"] for w in windows if w.get("priority",4)<0
                 or "failed_ctc_core" in w.get("reasons",[])}
        special.update(uid for d in third for uid in d["recovery_task_ids"])
        events=[e for e in events if not special.intersection(e.get("recovery_task_ids",[]))]
        observations=[o for o in observations if not special.intersection(o.get("task_ids",[]))]
        blocked={t["target_id"] for t in targets if special.intersection(t["task_ids"])}
        targets=[t for t in targets if t["target_id"] not in blocked]
        independent=[o for o in independent if o["id"].rsplit("-",1)[0] not in blocked]
    cross=cross_model_events(observations,targets,{**recovery,"observations":independent},policy)
    combined,conflicts=consolidate_events([*events,*cross],policy)
    timed,timing_decisions=_recover_untimed(recording,observations,policy)
    rows,decisions=select_source_events(timed,combined,policy)
    rows,independent_decisions=apply_independent(
        {**recording,"utterances":rows},targets,observations,independent)
    recovery["decisions"]=targeted+independent_decisions
    recovery["cross_model_events"]=cross
    if recovery["status"]=="observed":
        recovery["status"]="partial" if independent_decisions or cross or targeted else "unresolved"
    return rows,[*targeted,*conflicts,*timing_decisions,*decisions,*independent_decisions]


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
                    or row.get("wording_status")=="model_variants_unresolved"
                    or row.get("source_semantic_resolution")=="partial"
                    else "represented_provisional")
            elif refs and not context and any(
                    d["status"]=="timed_corroborated_failed_ctc_subspan"
                    and refs.intersection(d.get("source_unit_ids",[])) for d in decisions):
                relevant=[d for d in decisions if d["status"]=="timed_corroborated_failed_ctc_subspan"
                          and refs.intersection(d.get("source_unit_ids",[]))]
                covered=refs & {u for d in relevant for u in d["source_unit_ids"]}
                receipt["canonical_source_ids"]=sorted({d["source_id"] for d in relevant})
                receipt["supporting_observation_ids"]=sorted({u for d in relevant for u in d["observations"]})
                receipt["event_timing_supported_source_unit_ids"]=sorted(covered)
                receipt["status"]=("event_supported_word_timing_unresolved" if covered==refs
                                   else "partial_event_timing_supported")
            elif refs and not context and any(
                    d["status"]=="retired_displaced_context_hypothesis"
                    and refs<=set(d.get("source_unit_ids",[])) for d in decisions):
                d=next(d for d in decisions if d["status"]=="retired_displaced_context_hypothesis"
                       and refs<=set(d.get("source_unit_ids",[])))
                receipt["status"]="retired_displaced_context_hypothesis"
                receipt["canonical_source_ids"]=d["retained_source_ids"]
                receipt["supporting_observation_ids"]=d["observations"]
            elif refs and not context and missing:
                # Whole-event cross-model corroboration can support a
                # partially timed row without establishing individual
                # missing word times. Keep this span explicitly pending.
                supported={d["source_id"]:d for d in decisions
                           if d.get("source_id") and d["status"]=="already_represented"
                           and len(set(d.get("independent_observation_ids",[])))>=2}
                matches=[r for r in rows if r["id"] in supported
                         and refs<=set(r.get("source_unit_ids",[])) and _bounds(r)
                         and _same_event_bounds(_bounds(r),supported[r["id"]]["window"],
                                                policy.timing_tolerance_s)]
                if len(matches)==1:
                    row=matches[0]
                    receipt["canonical_source_ids"]=[row["id"]]
                    receipt["status"]="event_supported_word_timing_unresolved"
                    receipt["supporting_observation_ids"]=supported[row["id"]]["observations"]
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
