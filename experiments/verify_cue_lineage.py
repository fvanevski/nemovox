"""Verify actual native cue lineage and selected English against retained output."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from nvidia_transcribe import connected_units


def main():
    p=argparse.ArgumentParser()
    p.add_argument("output_dir")
    args=p.parse_args()
    root=Path(args.output_dir)
    contract=json.loads((root/"translated.contract.json").read_text())
    metadata=json.loads((root/"subtitles.metadata.json").read_text())
    rows=contract["utterances"]
    connected_units.validate_selected_units(rows,require_complete=True)
    known={r["id"]:r for r in rows}
    grouped=defaultdict(list)
    errors=[]
    for c in metadata["cue_lineage"]:
        uid=c["occurrence_id"]
        if uid not in known:
            errors.append("unknown occurrence "+uid)
            continue
        r=known[uid]
        unit=r.get("translation_unit") or {}
        if c["actual_event_interval"]!=connected_units.interval(r):
            errors.append("actual event interval mismatch "+uid)
        if c["translation_unit_id"]!=unit.get("unit_id"):
            errors.append("unit mismatch "+uid)
        expected=[x["id"] for x in unit.get("constituents",[])] or [uid]
        if c["unit_occurrence_ids"]!=expected:
            errors.append("unit membership mismatch "+uid)
        grouped[uid].append(c)
    def norm(s):
        return re.sub(r"\s+"," ",s).strip()
    covered=len(grouped)
    for uid,r in known.items():
        if not grouped[uid]:
            errors.append("missing occurrence "+uid)
            continue
        # Order by raw allocation start/end where present; unrenderable
        # records remain explicit rather than disappearing from this audit.
        ordered=sorted(grouped[uid],key=lambda c: tuple(x if x is not None else float("inf")
                       for x in (c.get("display_allocation_interval") or [None,None])))
        actual=norm(" ".join(c["text"] for c in ordered))
        if actual!=norm(r["translated_text"]):
            errors.append("allocation text differs "+uid)
    result={"occurrences":len(rows),"lineage_records":len(metadata["cue_lineage"]),
            "covered":covered,"errors":errors,"status":"PASS" if not errors else "FAIL"}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__=="__main__":
    main()
