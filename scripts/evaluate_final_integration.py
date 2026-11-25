"""Offline, matched final qualification against sealed prior observations.

No model calls; fixed reference events are corroborated model observations,
not human truth. Exact clip matching is distinguished from a registered
region comparison. Missing strict matches are not proof of omission.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from nvidia_transcribe.media import probe_media
from nvidia_transcribe.source_quality import _bounds, _digest, _lex, load_current_ledger
from scripts.evaluate_chunk_geometry import (
    REGIONS, compare_reference, comparison_metrics, read_authority, DIAGNOSTIC_SHA,
)

GEOMETRY_SHA="805ab0e43275171b16c0beebbc6c938d15ed5b055d4a6e5b342180f46d61eefe"
C2_SHA="6aeb764bf7e8368a71dc4fc1a8eeb4473831951c2c3723e8b990d7830863181c"
C2_MEDIA_SHA="f1f6984d0c35d8df782285d80ce29a478d859029c29fc438b0a24a8d6dd1bec6"
ORIGINAL_MEDIA="8d6e97fa0c3b0c674baa4e1864e72ada463dd4f448c7f9e3e6aacfb8249548e9"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def sealed(path, expected_sha):
    if sha(path)!=expected_sha:
        raise ValueError("Sealed archive identity mismatch")
    result={}
    with tarfile.open(path,"r:gz") as archive:
        root=None
        for m in archive:
            parts=PurePosixPath(m.name).parts
            if not parts or PurePosixPath(m.name).is_absolute() or ".." in parts:
                raise ValueError("Unsafe archive member")
            if root is None:
                root=parts[0]
            if parts[0]!=root:
                raise ValueError("Unexpected archive root")
            if m.isdir():
                continue
            if not m.isfile() or not 1<len(parts) or m.size>64*1024*1024:
                raise ValueError("Unexpected archive member")
            name="/".join(parts[1:])
            if name in result:
                raise ValueError("Duplicate archive member")
            result[name]=archive.extractfile(m).read()
    if "MANIFEST.sha256" in result:
        expected={}
        for line in result["MANIFEST.sha256"].decode().splitlines():
            digest,name=line.split("  ",1)
            if name in expected:
                raise ValueError("Duplicate manifest entry")
            expected[name]=digest
        if set(expected)!=set(result)-{"MANIFEST.sha256"}:
            raise ValueError("Archive manifest inventory mismatch")
        for name,digest in expected.items():
            if hashlib.sha256(result[name]).hexdigest()!=digest:
                raise ValueError("Archive payload mismatch: "+name)
    # Geometry archive has its own JSON manifest; whole archive SHA is the
    # already adjudicated exact authority (no files are extracted).
    return result


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def row_identity(row):
    return [row.get(k) for k in ("id","source_text","source_start","source_end",
                "speaker_id","source_unit_ids","occurrence_span_ids")]


def arm(rows, ledger, offset, *, strict=False):
    """Separate event-level support from the stricter original geometry
    reference criterion requiring no unresolved source-word locality.
    """
    disputed={s["id"] for inv in ledger["recognized_occurrence_inventories"]
              for s in inv["locality"]["spans"] if s["locality_status"]=="disputed"}
    all_rows=[]
    timed=[]
    for row in rows:
        r={**row,"observation_id":row["id"],"local_voice":row.get("speaker_id")}
        for field in ("source_start","source_end"):
            if r.get(field) is not None:
                r[field]+=offset
        all_rows.append(r)
        if (_bounds(r) and _lex(r["source_text"])
                and (not strict or (not row.get("untimed_unit_count",0)
                    and not disputed.intersection(row.get("occurrence_span_ids",[]))))):
            timed.append(r)
    return {"utterances":timed,"all_rows":all_rows,
            "raw_turn_lexemes":[_lex(turn["text"])
                for inv in ledger["recognized_occurrence_inventories"]
                for turn in inv["recognized_turns"]]}


def metrics(recording,ledger,translated,quality,metadata):
    rows=translated["utterances"]
    context=quality["recording_context"]
    independent=quality["independent_recovery"]
    scene=quality["scene_evidence"]
    review=translated["translation_review"]
    native_scene=[]
    for case in scene["decisions"]:
        for item in case.get("assessments",[]):
            native_scene.append(item["response"])
        if case.get("vision_response"):
            native_scene.append(case["vision_response"])
    return {
        "duration_s":recording["source"]["duration_s"],
        "original_rows":len(recording["utterances"]),"final_rows":len(rows),
        "original_wholly_untimed":sum(_bounds(r) is None for r in recording["utterances"]),
        "final_wholly_untimed":sum(_bounds(r) is None for r in rows),
        "original_untimed_units":sum(r.get("untimed_unit_count",0) for r in recording["utterances"]),
        "final_untimed_units":sum(r.get("untimed_unit_count",0) for r in rows),
        "primary_chunks":recording["processing"]["chunk_count"],
        "primary_effective_audio_s":round(sum(c["infer_end_s"]-c["infer_start_s"]
                         for c in recording["processing"]["chunks"]),3),
        "original_inventory_units":sum(len(i["working_source_unit_ids"])
                    for i in ledger["recognized_occurrence_inventories"]),
        "occurrence_dispositions":dict(Counter(r["status"] for r in quality["occurrence_resolutions"])),
        "source_resolution_decisions":dict(Counter(r["status"] for r in quality["decisions"])),
        "source_quality_status":quality["status"],
        "pending_occurrence_spans":len(quality["pending_occurrence_span_ids"]),
        "context":{"status":context["status"],"calls":context["calls"],
                   "eligible":len(context["eligible_source_ids"]),"covered":len(context["covered_source_ids"]),
                   "proposals":len(context["proposals"]),"rejected":len(context["rejected_proposals"]),
                   "usage":context["native_token_totals"],
                   "all_attempt_token_usage_reported":context["all_attempt_token_usage_reported"]},
        "context_dispositions":dict(Counter(x["status"] for x in quality["context_proposal_dispositions"])),
        "acoustic":quality["acoustic_acquisition"],
        "recovery_task_states":dict(Counter(x["state"] for x in quality["recovery_tasks"])),
        "independent":{k:independent.get(k) for k in ("status","requests","audio_seconds","wall_s","errors")},
        "scene":{"cases":len(scene["decisions"]),"deferred":len(scene["deferred_source_ids"]),
                 "outcomes":dict(Counter(x.get("outcome") for x in scene["decisions"])),
                 "errors":scene["errors"],"native_logical_calls":len(native_scene),
                 "native_attempts":sum(len(x.get("_client_request_diagnostics",{}).get("completion_attempts",[]))
                                       for x in native_scene),
                 "accepted_context_rows":sum(bool(r.get("semantic_context")) for r in rows)},
        "translation":translated["translation"],
        "review":{k:review.get(k) for k in ("status","batch_size","api_calls","comparative_calls","reviewed",
                    "changed","rejected_polarity","comparative_rejections","joined_passage_rejections",
                    "dependent_passage_rejections","unresolved_source_rejections","direction_grammar_rejections",
                    "cache_hits","elapsed_s","deferred_comparison_ids","errors")},
        "render":metadata["subtitles"] if metadata is not None else None,
        "render_status":"rendered" if metadata is not None else "not_rendered_review_error",
        "total_wall_time_s":metadata["total_wall_time_s"] if metadata is not None else None,
        "human_ground_truth_established":False,
    }


def compare_rows(prior, current):
    old={r["id"]:r for r in prior}
    new={r["id"]:r for r in current}
    if len(old)!=len(prior) or len(new)!=len(current):
        raise ValueError("Duplicate final occurrence identity")
    return {"removed_ids":sorted(old.keys()-new.keys()),"added_ids":sorted(new.keys()-old.keys()),
        "shared":[{"id":uid,"prior_source":row_identity(old[uid]),"current_source":row_identity(new[uid]),
                    "prior_primary":old[uid].get("machine_translated_text"),
                    "current_primary":new[uid].get("machine_translated_text"),
                    "prior_final":old[uid].get("translated_text"),"current_final":new[uid].get("translated_text")}
                  for uid in old if uid in new]}


def archived_diagnostic_translation(path, fingerprint):
    """Select only a unique translated contract from the exact archive.
    Multiple distinct matching contracts require explicit adjudication.
    """
    if sha(path)!=DIAGNOSTIC_SHA:
        raise ValueError("Diagnostic archive digest mismatch")
    candidates={}
    with tarfile.open(path,"r:gz") as archive:
        for m in archive:
            if not m.isfile() or not m.name.endswith(".json") or m.size>32*1024*1024:
                continue
            data=archive.extractfile(m).read()
            try:
                value=json.loads(data)
            except (ValueError,UnicodeError):
                continue
            if (isinstance(value,dict) and isinstance(value.get("source"),dict)
                    and value["source"].get("fingerprint_sha256")==fingerprint
                    and isinstance(value.get("utterances"),list)
                    and value["utterances"]
                    and isinstance(value.get("translation"),dict)
                    and isinstance(value.get("translation_review"),dict)
                    and all(isinstance(row,dict) and isinstance(row.get("translated_text"),str)
                            for row in value["utterances"])):
                key=hashlib.sha256(data).hexdigest()
                candidates.setdefault(key,{"path":m.name,"contract":value})
    inventory=[{"sha256":key,"member":v["path"]} for key,v in candidates.items()]
    if len(candidates)!=1:
        return {"status":"not_found" if not candidates else "ambiguous","inventory":inventory},None
    item=next(iter(candidates.values()))
    return {"status":"unique","inventory":inventory},item["contract"]


def regional_passages(rows):
    """Original-media coordinates, not row-ID alignment across versions."""
    regions={}
    for name,region,target,_ in REGIONS:
        lo,hi=region
        regions[name]=[{"source":row_identity(r),"primary_mt":r.get("machine_translated_text"),
                        "final":r.get("translated_text")}
                       for r in rows if _bounds(r) and _bounds(r)[0]<hi and _bounds(r)[1]>lo]
    return {"regions":regions,"wholly_untimed_count":sum(_bounds(r) is None for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",required=True,type=Path)
    parser.add_argument("--recording",type=Path)
    parser.add_argument("--media",required=True,type=Path)
    parser.add_argument("--geometry-archive",type=Path)
    parser.add_argument("--diagnostic-archive",type=Path)
    parser.add_argument("--region",choices=[name for name,*_ in REGIONS])
    parser.add_argument("--prior-semantic-archive",type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args()
    job=args.job
    path=args.recording or job/"asr"/"recording.contract.json"
    rec=read(path)
    ledger=load_current_ledger(rec,path)
    quality=read(job/"source-quality.json")
    canonical=read(job/"canonical-source.contract.json")
    translated=read(job/"translated.contract.json")
    metadata_path=job/"subtitles.metadata.json"
    if metadata_path.is_file():
        metadata=read(metadata_path)
    elif translated.get("translation_review",{}).get("status")=="completed_with_errors":
        metadata=None
    else:
        raise ValueError("Missing render metadata without explicit failed review")
    media=probe_media(args.media,audio_stream_index=rec["source"]["selected_audio_stream"])
    if (media.fingerprint_sha256!=rec["source"]["fingerprint_sha256"]
            or abs(media.duration_s-rec["source"]["duration_s"])>.01
            or quality["recording_sha256"]!=_digest(rec)
            or quality["media_fingerprint_sha256"]!=media.fingerprint_sha256
            or [row_identity(r) for r in canonical["utterances"]]!=
               [row_identity(r) for r in translated["utterances"]]):
        raise ValueError("Current recording/media/quality/translation lineage mismatch")
    full_sha=sha(args.media)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    root=args.output_dir
    summary={"media_sha256":full_sha,"recording_sha256":sha(path),
        "translated_sha256":sha(job/"translated.contract.json"),
        "current":metrics(rec,ledger,translated,quality,metadata)}
    if args.geometry_archive:
        members=sealed(args.geometry_archive,GEOMETRY_SHA)
        refs=json.loads(members["private-comparison.json"])
        names=list(refs)
        offset=0.
        if args.region:
            region=next(r for r in REGIONS if r[0]==args.region)
            offset,end=region[1]
            if abs(media.duration_s-(end-offset))>.05:
                raise ValueError("Registered region duration mismatch")
            names=[args.region]
            authority="exact_C2_clip" if args.region=="compression" and full_sha==C2_MEDIA_SHA else "operator_authenticated_registered_region"
        else:
            if media.fingerprint_sha256!=ORIGINAL_MEDIA or not args.diagnostic_archive:
                raise ValueError("Full French reference comparison requires original diagnostic archive/media")
            authority_data=read_authority(args.diagnostic_archive)
            if (authority_data["fingerprint_sha256"]!=media.fingerprint_sha256
                    or authority_data["selected_audio_stream"]!=media.selected_audio_stream
                    or abs(authority_data["duration_s"]-media.duration_s)>.01):
                raise ValueError("Original French reference media mismatch")
            authority="authenticated_original_full_episode"
        original_arm=arm(rec["utterances"],ledger,offset)
        final_arm=arm(canonical["utterances"],ledger,offset)
        strict_original=arm(rec["utterances"],ledger,offset,strict=True)
        comparisons={}
        for name in names:
            events=refs[name]["reference_events"]
            original=compare_reference(events,original_arm)
            final=compare_reference(events,final_arm)
            comparisons[name]={"original":original,"final":final,
                "strict_original":compare_reference(events,strict_original)}
        write(root/"private-reference-comparison.json",comparisons)
        summary["reference_comparison"]={"authority":authority,"offset_s":offset,
            "criterion":"event_level_positive_same_model_reference_not_human_truth",
            "regions":{name:{k:comparison_metrics(v) for k,v in values.items()}
                       for name,values in comparisons.items()}}
        if not args.region:
            receipt,prior=archived_diagnostic_translation(args.diagnostic_archive,media.fingerprint_sha256)
            summary["prior_diagnostic_translation"]=receipt
            if prior is not None:
                summary["prior_diagnostic_translation"].update(
                    rows=len(prior["utterances"]),
                    review={k:prior["translation_review"].get(k) for k in
                            ("api_calls","comparative_calls","reviewed","changed",
                             "comparative_rejections","elapsed_s")})
                write(root/"private-diagnostic-regions.json",{
                    "prior":regional_passages(prior["utterances"]),
                    "current":regional_passages(translated["utterances"]),
                    "matching":"same_original_media_intervals_not_ordinal_ids"})
    if args.prior_semantic_archive:
        members=sealed(args.prior_semantic_archive,C2_SHA)
        prefix="phase_b_job/"
        prior_rec=json.loads(members[prefix+"asr/recording.contract.json"])
        prior_ledger=json.loads(members[prefix+"asr/source-evidence.v1.json"])
        prior_quality=json.loads(members[prefix+"source-quality.json"])
        prior_translated=json.loads(members[prefix+"translated.contract.json"])
        prior_meta=json.loads(members[prefix+"subtitles.metadata.json"])
        if full_sha!=C2_MEDIA_SHA or media.fingerprint_sha256!=prior_rec["source"]["fingerprint_sha256"]:
            raise ValueError("Exact C2 matched comparison requires identical media bytes")
        if prior_quality["recording_sha256"]!=_digest(prior_rec):
            raise ValueError("Prior report binding mismatch")
        same_original=_digest(rec)==_digest(prior_rec) and _digest(ledger)==_digest(prior_ledger)
        same_mt=[(r["id"],r.get("machine_translated_text")) for r in translated["utterances"]]==[
            (r["id"],r.get("machine_translated_text")) for r in prior_translated["utterances"]]
        write(root/"private-prior-comparison.json",compare_rows(prior_translated["utterances"],translated["utterances"]))
        summary["prior_C2"]={"exact_media":True,"exact_original_recording_and_ledger":same_original,
            "identical_primary_MT_by_id":same_mt,"metrics":metrics(prior_rec,prior_ledger,prior_translated,prior_quality,prior_meta),
            "meaning_comparison_requires_qualitative_adjudication":True}
    write(root/"summary.json",summary)


if __name__=="__main__":
    main()
