"""Recording-wide scheduling of explicit acoustic uncertainty.

Windows are listening proposals, never accepted event bounds. Keep evidence
identity, unresolved candidates and budget deferrals addressable; don't
enumerate every unobserved subtitle gap as speech.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

from nvidia_transcribe.source_occurrences import stable_id


def interval(value, duration):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    a, b = value
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
               and math.isfinite(x) for x in (a, b)):
        return None
    a, b = max(0., float(a)), min(duration, float(b))
    return (a, b) if a < b else None


def overlap(a, b):
    return max(0., min(a[1], b[1]) - max(a[0], b[0]))


def padded_window(bounds, duration, minimum=3., padding=.75):
    """Expand even a tiny positive bracket; no timing assignment follows."""
    a,b = bounds
    center = (a+b)/2
    half = max(minimum/2, (b-a)/2+padding)
    lo,hi = max(0., center-half),min(duration, center+half)
    if hi-lo < min(minimum,duration):
        lo = max(0., min(lo, duration-minimum))
        hi = min(duration, lo+minimum)
    return (round(lo,6), round(hi,6))


def _parts(bounds, duration, maximum):
    # Even partition avoids a millisecond remainder. All original coverage
    # survives; minimum useful listening duration is distinct from event time.
    a,b = bounds
    count = max(1, math.ceil((b-a)/maximum))
    width = (b-a)/count
    return [padded_window((a+i*width, a+(i+1)*width), duration,
                          minimum=min(3.,maximum), padding=0.)
            for i in range(count)]


def plan_recovery_windows(recording, ledger, request, advisory, policy):
    duration = float(recording["source"]["duration_s"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid recording duration")
    parents = {r["id"]: r for r in ledger.get("asr_observations", [])}
    by_chunk = {r["chunk_index"]: r for r in parents.values()}
    units = {u["id"]: u for u in ledger.get("lexical_observations", [])}
    proposals, unavailable = [], []
    media_id = recording["source"]["fingerprint_sha256"]
    decisions = {r["case_id"]: r.get("assessment", {}).get("decision")
                 for r in (advisory or {}).get("outcomes", [])
                 if r.get("status") == "advisory_only"}

    def add(uid, reason, priority, bounds, chunks=(), refs=(), source_ids=()):
        chunks = sorted(set(chunks))
        refs = sorted(set(refs))
        source_ids = sorted(set(source_ids))
        bounds = interval(bounds,duration)
        evidence = {"evidence_ids":[uid], "chunks":chunks, "reason":reason,
                    "reasons":[reason], "source_unit_ids":refs,
                    "source_ids":source_ids, "priority":priority}
        if bounds is None:
            unavailable.append({**evidence, "window":None,
                "task_id":stable_id("recovery", [media_id, uid, reason, None]),
                "state":"unlocalized", "defer_reason":"no_credible_listening_interval"})
            return
        for part in _parts(bounds,duration,policy.max_target_seconds):
            proposals.append({**evidence, "window":part})

    def parent_window(parent):
        return (parent.get("infer_start_s"),parent.get("infer_end_s"))

    # The occurrence ledger is the primary discovery authority. Usable
    # times and rejected native hypotheses are distinct, as are raw turns.
    for inv in ledger.get("recognized_occurrence_inventories", []):
        parent = parents.get(inv["asr_observation_id"], {})
        chunks = [parent["chunk_index"]] if "chunk_index" in parent else []
        locality = inv.get("locality", {})
        for span in locality.get("spans", []):
            ds = span.get("selection_dispositions", [])
            status = span["locality_status"]
            context = any(d["status"] == "context_candidate_unresolved" for d in ds)
            unresolved = status in ("partial","unlocalized") and not span.get("post_alignment_usable_bounds_s")
            if not (context or unresolved or status == "disputed"):
                continue
            reason = ("disputed_occurrence" if status == "disputed" else
                      "untimed_occurrence" if unresolved else "unselected_context")
            bounds = (interval(span.get("candidate_bounds_s"),duration)
                      or interval(span.get("usable_bounds_s"),duration)
                      or interval(parent_window(parent),duration))
            add(span["id"],reason,0 if status == "disputed" else 1 if unresolved else 2,
                padded_window(bounds,duration) if bounds else None,chunks,span["source_unit_ids"])
        for finding in locality.get("findings", []):
            # Keep widely separated competing hypotheses as separate
            # windows, not one enormous interval between them.
            bounds = [b for v in finding.get("candidate_windows_s", [])
                      if (b := interval(v,duration))]
            for b in bounds or [None]:
                add(finding["id"],finding["reason"],
                    1 if finding["reason"]=="intra_turn_order_conflict" else 2,
                    padded_window(b,duration) if b else None,chunks,finding.get("source_unit_ids", []))
        if inv.get("projection_status") != "exact":
            add(inv["asr_observation_id"],"unmapped_raw_projection",1,
                parent_window(parent),chunks,inv.get("working_source_unit_ids", []))
        unmatched = [o["id"] for o in inv.get("native_ctc_observations", [])
                     if not o.get("source_unit_id")]
        if unmatched:
            # Native-only lexical hypotheses are not new source words.
            add(inv["asr_observation_id"],"unassociated_native_observations",2,
                parent_window(parent),chunks)

    for case in request.get("cases", []):
        bounds = interval(case.get("context_window_s"),duration)
        uid = case["case_id"]
        priority = 3 if decisions.get(uid) == "retain_core" else 2
        chunks = case.get("source_chunks", [])
        if bounds:
            a,b = bounds
            add(uid,"boundary_conflict",priority,(max(0.,a-3.),min(duration,b+1.)),chunks)
        else:
            add(uid,"boundary_conflict",priority,None,chunks)
        old = case.get("left_lexical_correspondence",{}).get("left_words") or []
        times = [b for w in old if (b := interval((w.get("start_s"),w.get("end_s")),duration))]
        if times:
            competing = (min(b[0] for b in times),max(b[1] for b in times))
            if not bounds or not overlap(competing,bounds):
                add(uid+":competing_time","competing_timing_hypothesis",priority,
                    padded_window(competing,duration),chunks)

    for activity in ledger.get("diarization_activity_hypotheses", []):
        if not activity.get("possible_unrepresented_speech"):
            continue
        bounds = interval((activity.get("start_s"),activity.get("end_s")),duration)
        parent = parents.get(activity.get("asr_observation_id"),{})
        chunks = [parent["chunk_index"]] if "chunk_index" in parent else []
        if bounds and bounds[1]-bounds[0] >= .65:
            add(activity["id"],"activity_without_lexical_timing",4,
                padded_window(bounds,duration),chunks)

    # Stable semantic requests from a successor context stage use the same
    # scheduler. A request may nominate real source IDs, never dictate edits.
    rows = recording["utterances"]
    row_map = {r["id"]:r for r in rows}
    for case in request.get("occurrence_requests", []):
        ids = case.get("source_ids", [])
        if not ids or any(uid not in row_map for uid in ids):
            add(case["case_id"],"invalid_semantic_reference",3,None)
            continue
        chunks = [c for uid in ids for c in row_map[uid].get("chunk_indices", [])]
        bounds = interval(case.get("context_window_s"),duration)
        add(case["case_id"],"semantic_occurrence_request",2,bounds,chunks,source_ids=ids)

    def row_bounds(r):
        return interval((r.get("source_start"),r.get("source_end")),duration)

    i = 0
    while i < len(rows):
        if row_bounds(rows[i]):
            i += 1
            continue
        j = i+1
        while j < len(rows) and not row_bounds(rows[j]):
            j += 1
        run = rows[i:j]
        ids = [r["id"] for r in run]
        refs = [uid for r in run for uid in r.get("source_unit_ids", [])]
        chunks = sorted({c for r in run for c in r.get("chunk_indices", [])})
        left = row_bounds(rows[i-1]) if i else None
        right = row_bounds(rows[j]) if j < len(rows) else None
        bracket = interval((left[1],right[0]),duration) if left and right else None
        uid = stable_id("untimed",ids)
        if bracket:
            add(uid,"untimed_source_dialogue",1,padded_window(bracket,duration),chunks,refs,ids)
        else:
            windows = [b for c in chunks if c in by_chunk
                       if (b := interval(parent_window(by_chunk[c]),duration))]
            for b in windows or [None]:
                add(uid,"untimed_source_dialogue",1,b,chunks,refs,ids)
        i = j

    # Resolve stable source-row references from unit evidence, including
    # rows with numeric but disputed timing.
    for p in proposals:
        refs = set(p["source_unit_ids"])
        p["source_ids"] = sorted(set(p["source_ids"]) | {
            r["id"] for r in rows if refs.intersection(r.get("source_unit_ids", []))})

    # Coalesce only when the union fits the hard window bound. Retain ALL
    # contributing references and reasons; no coverage lost by containment.
    merged = []
    for p in sorted(proposals,key=lambda p:(p["priority"],p["window"][0],p["window"][1],p["evidence_ids"])):
        a,b = p["window"]
        match = next((w for w in merged
                      if overlap(w["window"],p["window"]) >= .5*min(b-a,w["window"][1]-w["window"][0])
                      and max(b,w["window"][1])-min(a,w["window"][0]) <= policy.max_target_seconds+1e-6),None)
        if match is None:
            merged.append(dict(p))
        else:
            match["window"] = (min(a,match["window"][0]),max(b,match["window"][1]))
            for key in ("evidence_ids","chunks","reasons","source_unit_ids","source_ids"):
                match[key] = sorted(set(match[key]+p[key]))
            if p["priority"] < match["priority"]:
                match["priority"],match["reason"] = p["priority"],p["reason"]

    # Fairness within each severity: round-robin across recording quarters.
    # One broad early failure cannot consume all equally urgent later work.
    ordered = []
    for priority in sorted({w["priority"] for w in merged}):
        buckets = defaultdict(deque)
        for w in sorted((w for w in merged if w["priority"]==priority),
                        key=lambda w:(w["window"][1]-w["window"][0],w["window"][0])):
            bucket = min(3,int(4*(sum(w["window"])/2)/duration))
            buckets[bucket].append(w)
        while any(buckets.values()):
            for bucket in sorted(buckets):
                if buckets[bucket]:
                    ordered.append(buckets[bucket].popleft())
    limit = min(policy.max_windows,max(6,math.ceil(duration/90.)))
    for i,w in enumerate(ordered):
        w["task_id"] = stable_id("recovery",[media_id,w["window"],w["evidence_ids"]])
        w["state"] = "admitted" if i < limit else "deferred"
        if i >= limit:
            w["defer_reason"] = "window_budget"
    # De-duplicate unlocalized receipts by stable task identity.
    unlocalized = list({w["task_id"]:w for w in unavailable}.values())
    return ordered[:limit],ordered[limit:]+unlocalized
