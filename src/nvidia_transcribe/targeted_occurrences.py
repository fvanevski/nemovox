"""Targeted adjudication of context-displaced and unlocalized dialogue.

All accepted changes remain provisional model-supported hypotheses. The
immutable occurrence ledger and prior canonical rows remain traceable.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations
from nvidia_transcribe.source_quality import _bounds, _lex, _overlap, _contains
from nvidia_transcribe.source_occurrences import stable_id

def displaced_sequences(ledger, *, minimum_units=6):
    """Nominate exact recognized-order matches with disjoint CTC hypotheses.

    The shared crop must contain the earlier hypothesized words; the later
    projection must fall outside that shared crop. The match is a reason to
    listen, not a finding of echo or speaker identity.
    """
    if minimum_units<2:
        raise ValueError("Minimum sequence too short")
    parents={a["id"]:a for a in ledger.get("asr_observations",[])}
    exact={i["asr_observation_id"] for i in ledger.get(
        "recognized_occurrence_inventories",[]) if i.get("projection_status")=="exact"}
    by_chunk={}
    for u in ledger.get("lexical_observations",[]):
        if u["asr_observation_id"] in exact and u.get("raw_recognized_token_ids") and _lex(u["text"]):
            by_chunk.setdefault(u["chunk_index"],[]).append(u)
    for items in by_chunk.values():
        items.sort(key=lambda u:u["source_ordinal"])
    by_index={p["chunk_index"]:p for p in parents.values()}
    result=[]
    def bounds(u):
        a,b=u.get("ctc_start_s"),u.get("ctc_end_s")
        if isinstance(a,(int,float)) and isinstance(b,(int,float)) and a<b:
            return a,b
        return None
    for index,left in sorted(by_chunk.items()):
        right=by_chunk.get(index+1,[])
        if not right:
            continue
        a,b=by_index[index],by_index[index+1]
        shared=(max(a["infer_start_s"],b["infer_start_s"]),
                min(a["infer_end_s"],b["infer_end_s"]))
        if shared[0]>=shared[1]:
            continue
        lkeys=[_lex(u["text"]) for u in left]
        rkeys=[_lex(u["text"]) for u in right]
        for match in SequenceMatcher(None,lkeys,rkeys,autojunk=False).get_matching_blocks():
            run=[]
            def flush():
                if len(run)<minimum_units or len({_lex(x["text"]) for x,_ in run})<3:
                    return
                before,after=zip(*run)
                result.append({
                    "left_chunk":index,"right_chunk":index+1,
                    "shared_crop":shared,
                    "left_unit_ids":[u["id"] for u in before],
                    "right_unit_ids":[u["id"] for u in after],
                    "left_bounds":[bounds(u) for u in before],
                    "right_bounds":[bounds(u) for u in after],
                    "left_turn_ids":sorted({u["recognized_turn_id"] for u in before}),
                    "right_turn_ids":sorted({u["recognized_turn_id"] for u in after}),
                    "unit_count":len(run),"acoustic_truth_accepted":False,
                })
            for offset in range(match.size):
                x,y=left[match.a+offset],right[match.b+offset]
                xb,yb=bounds(x),bounds(y)
                eligible=(xb and yb and shared[0]<=xb[0]<xb[1]<=shared[1]
                    and shared[1]<=yb[0]<yb[1]<=b["core_end_s"]
                    and x.get("boundary_disposition")=="selected_provisional"
                    and y.get("boundary_disposition")=="selected_provisional")
                if eligible:
                    run.append((x,y))
                else:
                    flush()
                    run=[]
            flush()
    return result



def sequence_id(n):
    return stable_id("displaced-sequence",[n["left_unit_ids"],n["right_unit_ids"]])


def extent(bounds):
    return min(a for a,b in bounds),max(b for a,b in bounds)


def _tokens(rows):
    return tuple(w for r in rows for w in _lex(r["source_text"]))


def _ratio(a,b):
    return SequenceMatcher(None,a,b,autojunk=False).ratio()


def _longest(a,b):
    return SequenceMatcher(None,a,b,autojunk=False).find_longest_match().size


def _interior(obs,bounds):
    return obs["crop"][0]+.1<bounds[0]<bounds[1]<obs["crop"][1]-.1


def _ordered_groups(obs,key,bounds,maximum=5):
    """Actual adjacent segments only; no interpolation or bridging long gaps."""
    rows=sorted((r for r in obs[key] if _bounds(r) and
                 _overlap(_bounds(r),bounds)>0),key=lambda r:_bounds(r))
    for i in range(len(rows)):
        group=[]
        for row in rows[i:i+maximum]:
            if group and not -.1<=row["source_start"]-group[-1]["source_end"]<=1.5:
                break
            group.append(row)
            span=(group[0]["source_start"],group[-1]["source_end"])
            if _interior(obs,span):
                yield list(group),span,_tokens(group)


def _pairs(observations):
    return [(a,b) for a,b in combinations(observations,2)
            if a["audio_sha256"]!=b["audio_sha256"] and a["crop"]!=b["crop"]]


def _bound_observations(uid,reason,windows,observations,targets,independent):
    tasks={w["task_id"] for w in windows if w.get("state")=="observed"
           and uid in w["evidence_ids"] and reason in w["reasons"]}
    native=[o for o in observations if tasks.intersection(o.get("task_ids",[]))]
    ids={t["target_id"] for t in targets if tasks.intersection(t["task_ids"])}
    other=[o for o in independent if o["id"].rsplit("-",1)[0] in ids]
    return native,other,tasks


def _unique_match(obs,key,window,tokens,minimum=.80):
    matches=[(g,b,w) for g,b,w in _ordered_groups(obs,key,window)
             if _ratio(w,tokens)>=minimum and abs(b[0]-window[0])<=2
             and abs(b[1]-window[1])<=2]
    # Nested groups can differ by an interjection or arbitrary segmentation.
    # Prefer the highest lexical agreement, but not ambiguous equal scores.
    if not matches:
        return None
    matches.sort(key=lambda m:_ratio(m[2],tokens),reverse=True)
    best=matches[0]
    if any(_ratio(m[2],tokens)==_ratio(best[2],tokens) and m[1]!=best[1]
           for m in matches[1:]):
        return None
    return best


def _complete_rows(rows,ids):
    matched=[r for r in rows if ids.intersection(r.get("source_unit_ids",[]))]
    if (not matched or any(not set(r.get("source_unit_ids",[]))<=ids or not _bounds(r)
                           for r in matched)
            or {u for r in matched for u in r["source_unit_ids"]}!=ids):
        return []
    return sorted(matched,key=_bounds)


def adjudicate_sequences(recording,ledger,windows,observations,targets,independent):
    rows=[dict(r) for r in recording["utterances"]]
    decisions=[]
    for n in displaced_sequences(ledger):
        uid=sequence_id(n)
        earlier=_complete_rows(rows,set(n["left_unit_ids"]))
        later=_complete_rows(rows,set(n["right_unit_ids"]))
        if not earlier or not later or _tokens(earlier)!=_tokens(later):
            continue
        words=_tokens(earlier)
        old,new=extent(n["left_bounds"]),extent(n["right_bounds"])
        a,b,ta=_bound_observations(uid,"sequence_origin",windows,observations,targets,independent)
        c,d,tb=_bound_observations(uid,"sequence_displacement",windows,observations,targets,independent)
        arms=[o for o in c if tb.intersection(o.get("context_removed_task_ids",[]))]
        c=[o for o in c if o not in arms]
        # Earlier event must be positively present in both model families.
        positive=[(o,_unique_match(o,"utterances",old,words,.90)) for o in a]
        positive=[(o,g) for o,g in positive if g]
        corroboration=[(o,_unique_match(o,"rows",old,words)) for o in b]
        corroboration=[(o,g) for o,g in corroboration if g]
        # A later crop must exclude the earlier event, cover the entire
        # displaced region and contain competing speech at every disputed
        # row. Any substantial replay in raw or reconstructed text vetoes
        # retirement; lack of recognition alone is never sufficient.
        def competes(o,key):
            if not _interior(o,new) or _overlap(o["crop"],old)>0:
                return False
            observed=o[key]
            raw=o.get("raw_response",{})
            content=((raw.get("choices") or [{}])[0].get("message") or {}).get("content")
            lexical=_lex(content) if isinstance(content,str) else _tokens(observed)
            if _longest(words,lexical)>=6:
                return False
            for r in later:
                rb=_bounds(r)
                matching=[s for s in observed if _bounds(s) and _overlap(_bounds(s),rb)>0]
                if not matching:
                    return False
                w=_lex(r["source_text"])
                if len(w)>=3 and any(_contains(_lex(s["source_text"]),w) for s in matching):
                    return False
            return True
        native_competing=[o for o in c if competes(o,"utterances")]
        valid_arms=[o for o in arms if o.get("integrity",{}).get("status")=="complete"
                    and o.get("integrity",{}).get("finish_reason")=="stop"
                    and competes(o,"utterances")]
        independent_competing=[o for o in d if competes(o,"rows")]
        receipt={"sequence_id":uid,"source_ids":[r["id"] for r in later],
                 "retained_source_ids":[r["id"] for r in earlier],
                 "trigger_evidence_ids":[uid],"recovery_task_ids":sorted(ta|tb),
                 "earlier_bounds":old,"window":new,"acoustic_truth_accepted":False}
        # Do not selectively ignore an observation supporting a later repeat.
        if (not positive or not _pairs([o for o,g in corroboration])
                or not native_competing or not _pairs(independent_competing)
                or not valid_arms or len(valid_arms)!=len(arms)
                or len(native_competing)!=len(c) or len(independent_competing)!=len(d)):
            decisions.append({**receipt,"status":"unresolved_displaced_sequence"})
            continue
        support=[o for o,g in positive+corroboration]+native_competing+independent_competing+valid_arms
        receipt.update(status="retired_displaced_context_hypothesis",
                       observations=sorted({r["observation_id"] for o in support
                                            for r in o.get("utterances",o.get("rows",[]))}),
                       audio_sha256=sorted({o["audio_sha256"] for o in support}),
                       independent_observation_ids=sorted({o["id"] for o in b+d}),
                       source_unit_ids=n["right_unit_ids"],
                       context_removed_audio_sha256=sorted({o["audio_sha256"] for o in valid_arms}))
        # Retain the complete prior hypotheses, not their invalid timing as
        # if it were verified word timing at the earlier location.
        anchor=earlier[0]
        anchor["prior_source_hypotheses"]=[*anchor.get("prior_source_hypotheses",[]),*later]
        anchor["source_lineage"]={**anchor.get("source_lineage",{}),
            "displaced_context_resolutions":[*anchor.get("source_lineage",{}).get(
                "displaced_context_resolutions",[]),receipt]}
        remove={r["id"] for r in later}
        rows=[r for r in rows if r["id"] not in remove]
        decisions.append(receipt)
    return rows,decisions


def recover_unlocalized_events(recording,windows,observations,targets,independent):
    """Recover coherent original dialogue across a disputed neighbor bracket.

    Retain only actually observed event bounds. Require two native and two
    independent crops, ordered source association, no unsupported overlapping
    timed row, and no competing occurrence in the observed region.
    """
    rows=[dict(r) for r in recording["utterances"]]
    decisions=[]
    for task in windows:
        if task.get("state")!="observed" or "untimed_event_context" not in task.get("reasons",[]):
            continue
        origin_ids=set(task.get("source_ids",[]))
        originals=[r for r in rows if r["id"] in origin_ids and not _bounds(r)]
        if len(originals)<2:
            continue
        native=[o for o in observations if task["task_id"] in o.get("task_ids",[])]
        target_ids={t["target_id"] for t in targets if task["task_id"] in t["task_ids"]}
        other=[o for o in independent if o["id"].rsplit("-",1)[0] in target_ids]
        if not _pairs(native) or not _pairs(other):
            continue
        pos=rows.index(originals[0])
        previous=next((r for r in reversed(rows[:pos]) if _bounds(r)),None)
        candidates=[]
        for first in native:
            for g,b,words in _ordered_groups(first,"utterances",task["window"],maximum=1):
                if len(words)<8 or not task["window"][0]<=sum(b)/2<=task["window"][1]:
                    continue
                # Two complete localized native alternatives and two
                # independent alternatives, with differing segment boundaries.
                ns=[(o,_unique_match(o,"utterances",b,words)) for o in native]
                ns=[(o,x) for o,x in ns if x]
                ins=[(o,_unique_match(o,"rows",b,words)) for o in other]
                ins=[(o,x) for o,x in ins if x]
                if not _pairs([o for o,x in ns]) or not _pairs([o for o,x in ins]):
                    continue
                # An earlier recognized prefix can already be rendered. Its
                # use requires positive local corroboration in both families.
                prefix=()
                if previous and _bounds(previous)[1]<=b[0] and b[0]-_bounds(previous)[1]<=3:
                    pw=_lex(previous["source_text"])
                    if (any(_unique_match(o,"utterances",_bounds(previous),pw) for o in native)
                            and any(_unique_match(o,"rows",_bounds(previous),pw) for o in other)):
                        prefix=pw
                combined=prefix+words
                associated=[]
                last=0
                for row in originals:
                    w=_lex(row["source_text"])
                    blocks=[m for m in SequenceMatcher(None,w,combined,autojunk=False).get_matching_blocks() if m.size]
                    count=sum(m.size for m in blocks)
                    in_event=sum(max(0,m.b+m.size-max(m.b,len(prefix))) for m in blocks)
                    if (not w or count/len(w)<.8 or in_event<2 or not blocks
                            or blocks[0].b<last):
                        continue
                    associated.append(row)
                    last=blocks[-1].b+blocks[-1].size
                if len(associated)<2:
                    continue
                indices=[originals.index(r) for r in associated]
                if indices!=list(range(indices[0],indices[-1]+1)):
                    continue
                original_words=_tokens(associated)
                if _longest(original_words,words)<8 or _ratio(original_words,words)<.70:
                    continue
                # Absorb only a short fully represented competing prefix
                # from a different chunk. Preserve every other timed row.
                overlaps=[r for r in rows if _bounds(r) and _overlap(_bounds(r),b)>.08]
                if any(not (_bounds(r)[0]>=b[0]-.15 and _bounds(r)[1]<=b[1]+.15
                           and len(_lex(r["source_text"]))<=max(3,len(words)//4)
                           and set(r.get("chunk_indices",[])).isdisjoint(
                               {c for u in associated for c in u.get("chunk_indices",[])})
                           and any(_contains(x[2][0:max(3,len(words)//3)],_lex(r["source_text"]))
                                   for o,x in ns)) for r in overlaps):
                    continue
                candidates.append((g,b,words,associated,overlaps,ns,ins))
        if not candidates:
            continue
        candidates.sort(key=lambda x:(-len(x[3]),x[1]))
        g,b,words,associated,overlaps,ns,ins=candidates[0]
        if any(not {r["id"] for r in x[3]}<={r["id"] for r in associated}
               or abs(x[1][0]-b[0])>2 or abs(x[1][1]-b[1])>2
               or _ratio(x[2],words)<.8 for x in candidates[1:]):
            continue
        origins=associated+overlaps
        ids={r["id"] for r in origins}
        support=ns+ins
        observation_ids=sorted({r["observation_id"] for o,x in support for r in x[0]})
        audio=sorted({o["audio_sha256"] for o,x in support})
        uid=associated[0]["id"]
        row={**associated[0],"source_text":g[0]["source_text"],
             "source_start":b[0],"source_end":b[1],"speaker_id":"SPK_UNK",
             "speaker_identity_accepted":False,"source_turn_id":uid,"source_turn_ids":[uid],
             "source_occurrence_ids":[uid],"source_words":[],"source_characters":[],
             "unit_count":0,"timed_unit_count":0,"untimed_unit_count":0,"timing_gaps":[],
             "alignment_status":"corroborated_model_event_bounds",
             "source_quality_status":"unresolved_wording_recovered_event",
             "chunk_indices":sorted({c for r in origins for c in r.get("chunk_indices",[])}),
             "prior_source_hypotheses":origins,
             "timestamp_provenance":{"backend":"separate_acoustic_contexts",
                 "timing_status":"approximate","human_verified":False,"word_timing_verified":False,
                 "audio_sha256":audio,"observation_ids":observation_ids},
             "source_lineage":{"prior_source_ids":sorted(ids),"acoustic_observation_ids":observation_ids,
                 "source_word_timing_accepted":False,
                 "wording_alternatives":list(dict.fromkeys(" ".join(r["source_text"] for r in x[0])
                                                           for o,x in support))}}
        for key in ("source_unit_ids","recognized_turn_ids","occurrence_span_ids"):
            row[key]=sorted({v for r in origins for v in r.get(key,[])})
        rows=[r for r in rows if r["id"] not in ids]
        insert=next((i for i,r in enumerate(rows) if _bounds(r) and _bounds(r)[0]>b[0]),len(rows))
        rows.insert(insert,row)
        decisions.append({"status":"unresolved_wording_recovered_event","source_id":uid,
            "source_ids":sorted(ids),"window":b,"observations":observation_ids,
            "trigger_evidence_ids":task["evidence_ids"],"recovery_task_ids":[task["task_id"]],
            "independent_observation_ids":sorted({o["id"] for o,x in ins}),
            "audio_sha256":audio,"acoustic_truth_accepted":False})
    return rows,decisions
