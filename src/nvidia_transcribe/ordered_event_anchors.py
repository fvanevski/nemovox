"""Localize immutable failed-CTC source units using ordered acoustic anchors.

Match lexical anchors, not ASR segment boundaries. Internal wording variants
remain hypotheses. Times come from actual observed word/segment boundaries.
"""
from difflib import SequenceMatcher
from itertools import combinations

from nvidia_transcribe.parser import parse_speaker_turns
from nvidia_transcribe.source_quality import _bounds, _lex, _overlap, _same_event_bounds


def _mapping(a, b):
    return {m.a+k: m.b+k for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
            for k in range(m.size)}


def _near(a, b, *, edges=True, fraction=.72):
    if not a or not b:
        return False
    m = _mapping(a, b)
    return bool(m and len(m)/len(a) >= fraction
                and (not edges or (0 in m and len(a)-1 in m))
                and max(m.values())-min(m.values())+1 <= len(a)*1.35
                and all(y-x <= 5 and m[y]-m[x] <= 5 for x,y in zip(sorted(m),sorted(m)[1:])))


def _atoms(row, native):
    """A multi-token word has one observed interval, not invented sub-times."""
    rb = _bounds(row)
    if not rb:
        return []
    key = "timed_lexical_units" if native else "words"
    supplied = row.get(key)
    if not supplied:
        # Segment-only independent observations still carry event boundaries.
        if native or supplied == []:
            return []
        supplied = [{"text": row["source_text"], "start": rb[0], "end": rb[1]}]
    atoms = []
    for w in supplied:
        tokens = _lex(w.get("lex") if native else w.get("text"))
        if not tokens:
            continue
        span = _bounds(w, "start", "end")
        if not span or span[0]<rb[0]-.10 or span[1]>rb[1]+.10:
            return []
        if atoms and span[0]<atoms[-1][2]-.10:
            return []
        atoms.extend((token, *span, i==0, i==len(tokens)-1, row["observation_id"])
                     for i,token in enumerate(tokens))
    if tuple(x[0] for x in atoms) != _lex(row["source_text"]):
        return []
    return atoms


def _streams(observation, native):
    """Never join separate native voices; independent contiguous segments may be joined."""
    group = []
    for row in sorted(observation.get("utterances" if native else "rows", []),
                      key=lambda r: _bounds(r) or (float("inf"), float("inf"))):
        atoms = _atoms(row, native)
        if not atoms or (group and (native or not -.10 <= atoms[0][1]-group[-1][2] <= 1.5)):
            if group:
                yield group
            group = []
        if atoms:
            group.extend(atoms)
        if native and group:
            yield group
            group = []
    if group:
        yield group


def _runs(words, atoms):
    m = _mapping(words, tuple(a[0] for a in atoms))
    group = []
    for i,j in sorted(m.items()):
        if group and (i-group[-1][0]>5 or j-group[-1][1]>5
                      or atoms[j][1]-atoms[group[-1][1]][2]>2):
            yield dict(group)
            group = []
        group.append((i,j))
    if group:
        yield dict(group)


def _raw_support(native, words, bounds):
    result = []
    for o in native:
        if not o["crop"][0]+.1<bounds[0]<bounds[1]<o["crop"][1]-.1:
            continue
        raw = o.get("raw_response", {})
        content = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content")
        if not isinstance(content, str):
            continue
        for turn in parse_speaker_turns(content):
            tokens = _lex(turn.text)
            if _near(words, tokens):
                m = _mapping(words, tokens)
                part = tokens[m[0]:m[len(words)-1]+1]
                result.append((o, part))
                break
    return result


def ordered_proposals(recording, ledger, sources, windows, observations, targets, independent, policy):
    """Return event proposals with exact source-unit boundaries and observed anchor receipts."""
    rows = recording["utterances"]
    observed = {w["task_id"]:w for w in windows if w.get("state")=="observed"}
    target_map = {t["target_id"]:t for t in targets if set(t["task_ids"]) & set(observed)}
    native = [o for o in observations if set(o.get("task_ids",[])) & set(observed)
              and o.get("integrity",{}).get("status")=="complete"
              and o.get("integrity",{}).get("finish_reason")=="stop"]
    streams = []
    for o in independent:
        target = target_map.get(o["id"].rsplit("-",1)[0])
        if target:
            streams.extend((False,o,atoms,set(target["task_ids"])) for atoms in _streams(o,False))
    for o in native:
        streams.extend((True,o,atoms,set(o["task_ids"])) for atoms in _streams(o,True))
    raw_turns = {}
    for u in ledger.get("lexical_observations",[]):
        raw_turns.setdefault((u.get("asr_observation_id"),u.get("recognized_turn_id")),[]).append(u)
    raw_sequences = [tuple(w for u in sorted(us,key=lambda x:x["source_ordinal"]) for w in _lex(u["text"]))
                     for us in raw_turns.values()]
    minimum = 4 if recording["source_language"].startswith(("zh","ja")) else 3
    tolerance = policy.timing_tolerance_s
    proposals = []
    for row in rows:
        if row["id"] not in sources:
            continue
        parent, units = sources[row["id"]]
        words = tuple(w for u in units for w in _lex(u["text"]))
        offsets = [0]
        for u in units:
            offsets.append(offsets[-1]+len(_lex(u["text"])))
        ends = {n:i for i,n in enumerate(offsets)}
        core = (parent["core_start_s"],parent["core_end_s"])
        maps = []
        for is_native,o,atoms,tasks in streams:
            if _overlap(o["crop"],core)<=0:
                continue
            maps.extend((is_native,o,atoms,tasks,m) for m in _runs(words,atoms) if len(m)>=minimum)
        for left,right in combinations(maps,2):
            an,ao,aa,at,am = left
            bn,bo,ba,bt,bm = right
            if (an and bn) or ao["audio_sha256"]==bo["audio_sha256"] or ao["crop"]==bo["crop"]:
                continue
            common = sorted(i for i in set(am)&set(bm)
                            if (abs(aa[am[i]][1]-ba[bm[i]][1])<=tolerance
                                and abs(aa[am[i]][2]-ba[bm[i]][2])<=tolerance))
            for k,i in enumerate(common):
                if i not in ends or not aa[am[i]][3] or not ba[bm[i]][3]:
                    continue
                for j in reversed(common[k+minimum-1:]):
                    if j+1 not in ends or not aa[am[j]][4] or not ba[bm[j]][4]:
                        continue
                    segment = words[i:j+1]
                    if len(set(segment))<3:
                        continue
                    anchors = [x for x in common if i<=x<=j]
                    if (len(anchors)/len(segment)<.72 or any(y-x>4 for x,y in zip(anchors,anchors[1:]))
                            or any(m[j]-m[i]+1>len(segment)*1.35 for m in (am,bm))):
                        continue
                    # Each boundary must be an observed whole atom in both
                    # streams. No duration assigned to unmatched internal words.
                    bounds = [(aa[am[i]][1],aa[am[j]][2]),(ba[bm[i]][1],ba[bm[j]][2])]
                    if not all(core[0]<=s<e<=core[1] and e-s>=.65 for s,e in bounds):
                        continue
                    if not _same_event_bounds(*bounds,tolerance):
                        continue
                    interior = [o["crop"][0]+.1<s<e<o["crop"][1]-.1 for o,(s,e) in zip((ao,bo),bounds)]
                    if not any(interior):
                        continue
                    span = bounds[0 if an else 1 if bn or interior[1] else 0]
                    if any(_bounds(r) and _overlap(_bounds(r),span)>.08 for r in rows):
                        continue
                    # Repeated raw hypotheses must stay unresolved, including
                    # near-identical variants in competing canonical rows.
                    count = sum(seq[k:k+len(segment)]==segment for seq in raw_sequences
                                for k in range(len(seq)-len(segment)+1))
                    if count!=1 or any(_near(segment,_lex(r["source_text"]),edges=False,fraction=.85)
                                       for r in rows if r["id"]!=row["id"]):
                        continue
                    support = _raw_support(native,segment,span)
                    if not support:
                        continue
                    all_tasks = at|bt|{t for o,_ in support for t in o["task_ids"]}
                    all_tasks &= set(observed)
                    alternatives = []
                    for nat,o,atoms,m in ((an,ao,aa,am),(bn,bo,ba,bm)):
                        part = atoms[m[i]:m[j]+1]
                        alternatives.append({"backend":"nemotron" if nat else "independent_asr",
                            "tokens":[a[0] for a in part],"audio_sha256":o["audio_sha256"],
                            "observation_ids":sorted({a[5] for a in part})})
                    for o,part in support:
                        alternatives.append({"backend":"nemotron","tokens":list(part),
                            "audio_sha256":o["audio_sha256"],"observation_ids":[o.get("raw_response_sha256") or o["audio_sha256"]]})
                    exact = all(tuple(a["tokens"])==segment for a in alternatives)
                    proposals.append({"origin":row["id"],"lo":ends[i],"hi":ends[j+1],"window":span,
                        "observations":sorted({v for a in alternatives for v in a["observation_ids"]}),
                        "independent_audio_sha256":sorted({o["audio_sha256"] for n,o in ((an,ao),(bn,bo)) if not n}),
                        "native_audio_sha256":sorted({o["audio_sha256"] for o,_ in support}),
                        "native_raw_response_sha256":sorted({o["raw_response_sha256"] for o,_ in support if o.get("raw_response_sha256")}),
                        "native_primary_observation_id":parent["id"],
                        "trigger_evidence_ids":sorted({v for t in all_tasks for v in observed[t]["evidence_ids"]}),
                        "recovery_task_ids":sorted(all_tasks),"wording_exact":exact,
                        "wording_alternatives":[{**a,"source_unit_range":[ends[i],ends[j+1]]} for a in alternatives],
                        "independent_observation_ids":sorted({v for a in alternatives if a["backend"]=="independent_asr" for v in a["observation_ids"]}),
                        "anchors":{x-i:[aa[am[x]][1],aa[am[x]][2],ba[bm[x]][1],ba[bm[x]][2]] for x in anchors},
                        "lexical_offset":i,"anchor_fraction":len(anchors)/len(segment)})
                    # Prefer maximal supported endpoint for this start and pair.
                    break
    return proposals


def select_ordered_proposals(proposals, rows, tolerance):
    """Reject incompatible locality, then maximize disjoint ordered coverage.

    Compatible overlapping alternatives are different extents of one event,
    not conflicts. Never choose between incompatible localities by coverage.
    """
    bad = set()
    rowmap = {r["id"]:r for r in rows}
    for i,j in combinations(range(len(proposals)),2):
        a,b = proposals[i],proposals[j]
        same = a["origin"]==b["origin"]
        lexical = same and max(a["lo"],b["lo"])<min(a["hi"],b["hi"])
        temporal = _overlap(a["window"],b["window"])>.10
        if lexical:
            am = {a["lexical_offset"]+int(k):v for k,v in a["anchors"].items()}
            bm = {b["lexical_offset"]+int(k):v for k,v in b["anchors"].items()}
            shared = set(am)&set(bm)
            # Different extents can be compatible only at actual shared anchors.
            if not shared or any(max(abs(x-y) for x,y in zip(am[k],bm[k]))>tolerance for k in shared):
                bad.update((i,j))
        elif same:
            if (temporal or a["hi"]<=b["lo"] and a["window"][0]>b["window"][0]
                    or b["hi"]<=a["lo"] and b["window"][0]>a["window"][0]):
                bad.update((i,j))
        else:
            ar,br = rowmap[a["origin"]],rowmap[b["origin"]]
            if temporal or (set(ar.get("recognized_turn_ids",[])) & set(br.get("recognized_turn_ids",[]))
                and (rows.index(ar)-rows.index(br))*(a["window"][0]-b["window"][0])<0):
                bad.update((i,j))
    selected = []
    for row in rows:
        available = [p for i,p in enumerate(proposals) if i not in bad and p["origin"]==row["id"]]
        available.sort(key=lambda p:(p["lo"],-p["hi"],-p["anchor_fraction"],p["window"]))
        # All incompatible pairs have already been rejected. Join only
        # overlapping source extents, with a shared observed lexical anchor;
        # use actual outer event boundaries and retain constituent receipts.
        while available:
            first = available.pop(0)
            chain = [first]
            end = first["hi"]
            while True:
                extensions = [p for p in available if p["lo"]<end<p["hi"]]
                if not extensions:
                    break
                nxt = max(extensions,key=lambda p:(p["hi"],p["anchor_fraction"],-p["lo"]))
                chain.append(nxt)
                end = nxt["hi"]
            available = [p for p in available if p["lo"]>=end]
            merged = dict(first)
            merged["hi"] = end
            merged["window"] = (first["window"][0],chain[-1]["window"][1])
            for key in ("observations","independent_observation_ids","independent_audio_sha256",
                        "native_audio_sha256","native_raw_response_sha256","trigger_evidence_ids","recovery_task_ids"):
                merged[key] = sorted({v for p in chain for v in p[key]})
            merged["wording_exact"] = all(p["wording_exact"] for p in chain)
            merged["wording_alternatives"] = [v for p in chain for v in p["wording_alternatives"]]
            merged["constituent_event_evidence"] = [{"source_unit_range":[p["lo"],p["hi"]],
                "window":p["window"],"observations":p["observations"],"anchors":p["anchors"],
                "lexical_offset":p["lexical_offset"],"anchor_fraction":p["anchor_fraction"]} for p in chain]
            # Top-level anchors describe the first constituent only; do not
            # imply that one pair observed the full union.
            for key in ("anchors","lexical_offset","anchor_fraction"):
                merged.pop(key)
            selected.append(merged)
    return selected
