"""Prepare immutable downstream qualification inputs from the accepted C3 archive.

No inference, no checkpoint relabeling, no arbitrary tar extraction. The new
review policy must compute its own decisions; only source and primary MT
are reused. The archived job remains untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from nvidia_transcribe.source_quality import _digest, load_current_ledger

ARCHIVE_SHA="49e31f742cae801672de81ea604144769d1e7e8f3114b6b4f5f83d07bcb2e1df"
CASES={
    "bounded":("phase_b_bounded_job","scratch/c2_original_asr"),
    "french":("phase_d_full_french_job","phase_d_full_french_job/asr"),
    "mandarin":("phase_e_multilingual_job","phase_e_multilingual_job/asr"),
}
SOURCE_FILES=("canonical-source.contract.json","source-quality.json","translation.checkpoint.json")
ORIGINAL_FILES=("recording.contract.json","manifest.json","source-evidence.v1.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_archive(path):
    with Path(path).open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=ARCHIVE_SHA:
            raise ValueError("C3 archive digest mismatch")
    members={}
    with tarfile.open(path,"r:gz") as archive:
        for item in archive:
            name=PurePosixPath(item.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe archive path")
            if item.isdir():
                continue
            if not item.isfile() or item.size>64*1024*1024 or str(name) in members:
                raise ValueError("Invalid or duplicate archive payload")
            members[str(name)]=archive.extractfile(item).read()
    manifest={}
    for line in members["MANIFEST.sha256"].decode().splitlines():
        sha,path=line.split("  ",1)
        key=str(PurePosixPath(path))
        if key in manifest or key not in members or digest(members[key])!=sha:
            raise ValueError("Invalid archive manifest payload: "+key)
        manifest[key]=sha
    extra=set(members)-set(manifest)-{"MANIFEST.sha256"}
    # The accepted archive contains twelve duplicated pre-resume frames.
    # Admit only those exact duplicate PNG bytes, never use them as inputs.
    if len(manifest)!=149 or len(extra)!=12 or any(
        not n.startswith("phase_b_bounded_job_pre_resume/") or not n.endswith(".png")
        or digest(members[n]) not in manifest.values() for n in extra):
        raise ValueError("Unexpected C3 archive inventory")
    return {n:members[n] for n in manifest}


def identity(rows):
    return [(r["id"],r["source_text"],r.get("source_start"),r.get("source_end"),
             r.get("speaker_id"),r.get("source_unit_ids"),r.get("occurrence_span_ids"),
             r.get("event_refs")) for r in rows]


def prepare(archive, case, output):
    members=read_archive(archive)
    prefix,original=CASES[case]
    get=lambda name:members[prefix+"/"+name]
    canonical=json.loads(get("canonical-source.contract.json"))
    quality=json.loads(get("source-quality.json"))
    prior=json.loads(get("translated.contract.json"))
    checkpoint=json.loads(get("source-quality.checkpoint.json"))
    rec=json.loads(members[original+"/recording.contract.json"])
    if (checkpoint["payload_sha256"]!=_digest(checkpoint["payload"])
            or checkpoint["payload"]!={"canonical":canonical,"report":quality}
            or quality["recording_sha256"]!=_digest(rec)
            or canonical["source"]!=rec["source"]
            or canonical["source_quality"]["recording_sha256"]!=quality["recording_sha256"]
            or identity(canonical["utterances"])!=identity(prior["utterances"])
            or len({r["id"] for r in prior["utterances"]})!=len(prior["utterances"])
            or any(not isinstance(r.get("machine_translated_text"),str)
                   or not r["machine_translated_text"].strip() for r in prior["utterances"])):
        raise ValueError("Retained source/primary MT authority mismatch")
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/"asr").mkdir()
    inputs={}
    for name in SOURCE_FILES:
        inputs[name]=get(name)
    for name in ORIGINAL_FILES:
        inputs["asr/"+name]=members[original+"/"+name]
    inputs["prior-translated.contract.json"]=get("translated.contract.json")
    for name,data in inputs.items():
        (output/name).write_bytes(data)
    load_current_ledger(rec,output/"asr/recording.contract.json")
    receipt={"case":case,"archive_sha256":ARCHIVE_SHA,"input_hashes":{k:digest(v) for k,v in inputs.items()},
        "original_recording_sha256":_digest(rec),"canonical_source_rows":len(canonical["utterances"]),
        "prior_review_status":prior["translation_review"]["status"],
        "media_path_hint":rec["source"]["path"],
        "source_language":rec["source_language"],"audio_stream":rec["source"]["selected_audio_stream"],
        "media_fingerprint_sha256":rec["source"]["fingerprint_sha256"],
        "primary_resource_fingerprint":prior["translation"]["resource_fingerprint"],
        "review_resource_fingerprint":prior["translation_review"]["resource_fingerprint"],
        "glossary_digest":prior["translation_review"]["glossary_digest"],
        "upstream_checkpoint_not_copied":True,"old_review_checkpoint_not_copied":True}
    (output/"retained-inputs.json").write_text(json.dumps(receipt,indent=2))
    return receipt


def verify(output):
    output=Path(output)
    read=lambda name:json.loads((output/name).read_bytes())
    receipt=read("retained-inputs.json")
    for name,sha in receipt["input_hashes"].items():
        if digest((output/name).read_bytes())!=sha:
            raise ValueError("Retained input changed: "+name)
    prior=read("prior-translated.contract.json")
    current=read("translated.contract.json")
    review=current["translation_review"]
    if (identity(prior["utterances"])!=identity(current["utterances"])
            or prior["source_quality"]!=current["source_quality"]
            or [r["machine_translated_text"] for r in prior["utterances"]]!=
               [r["machine_translated_text"] for r in current["utterances"]]
            or current["translation"]["resource_fingerprint"]!=receipt["primary_resource_fingerprint"]
            or current["translation"]["cache_hits"]!=len(current["utterances"])
            or review["resource_fingerprint"]!=receipt["review_resource_fingerprint"]
            or review["glossary_digest"]!=receipt["glossary_digest"]
            or review["status"] not in ("completed","completed_with_deferred_comparisons")):
        raise ValueError("Downstream identity, primary-cache or review admission mismatch")
    metadata=read("subtitles.metadata.json")
    files={}
    for extension in ("srt","ass"):
        paths=list(output.glob("*."+extension))
        if len(paths)!=1 or not paths[0].stat().st_size:
            raise ValueError("Missing or ambiguous rendered "+extension)
        files[extension]={"name":paths[0].name,"sha256":digest(paths[0].read_bytes())}
    diffs=[{"id":a["id"],"source_text":a["source_text"],
            "primary_mt":a["machine_translated_text"],"prior_final":a["translated_text"],
            "current_final":b["translated_text"],"current_proposal":b.get("translation_reviewed_text"),
            "current_status":b.get("translation_review_status"),
            "uncertainty":b.get("review_source_uncertainty"),
            "comparative":b.get("translation_comparative_review"),
            "joined_passage":b.get("translation_review_passage")}
           for a,b in zip(prior["utterances"],current["utterances"])]
    (output/"private-review-comparison.json").write_text(json.dumps(diffs,ensure_ascii=False,indent=2))
    summary={"case":receipt["case"],"rows":len(diffs),"retained_identity":"pass",
             "primary_cache_hits":current["translation"]["cache_hits"],
             "changed_final_rows":sum(a["prior_final"]!=a["current_final"] for a in diffs),
             "review":{k:v for k,v in review.items()
                       if k not in ("comparison_recoveries","native_attempts","errors")},
             "render":metadata["subtitles"],"files":files,
             "semantic_adjudication_required":True}
    (output/"completion-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    create=sub.add_parser("prepare")
    create.add_argument("--archive",type=Path,required=True)
    create.add_argument("--case",choices=CASES,required=True)
    create.add_argument("--output-dir",type=Path,required=True)
    check=sub.add_parser("verify")
    check.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    result=(prepare(args.archive,args.case,args.output_dir) if args.command=="prepare"
            else verify(args.output_dir))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
