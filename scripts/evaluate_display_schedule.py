"""CPU-only comparison of emitted subtitles from a frozen selected contract.

Run from the source checkout under test using its explicit PYTHONPATH. This
script never opens media or an embedded subtitle track and never calls a model.
Full output and lineage are private; metrics.json contains no dialogue text.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import statistics
from pathlib import Path

from nvidia_transcribe import subtitles


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seconds(value):
    h, m, s = value.replace(",", ".").split(":")
    whole, _, frac = s.partition(".")
    return (int(h)*3600 + int(m)*60 + int(whole))*1000 + int(frac.ljust(3,"0"))


def events(path, fmt):
    result = []
    if fmt == "ass":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Dialogue: "):
                layer, start, end, style, _, _, _, _, _, text = line[10:].split(",", 9)
                # Strip only our generated positioning tag, not arbitrary dialogue.
                text = re.sub(r"^\{\\an5\\pos\(\d+,\d+\)\}", "", text)
                text = text.replace(r"\N", " ").replace(r"\{", "{").replace(r"\}", "}").replace(r"\\", "\\")
                result.append((seconds(start), seconds(end), style, len(text)))
    else:
        for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
            if not block:
                continue
            lines = block.splitlines()
            start, end = lines[1].split(" --> ")
            result.append((seconds(start), seconds(end), "SRT", len(" ".join(lines[2:]))))
    return result


def metrics(rows, fmt):
    invalid = sum(end <= start for start, end, _, _ in rows)
    valid = [(s, e, lane, n) for s, e, lane, n in rows if e > s]
    durations = [e-s for s, e, _, _ in valid]
    rates = [1000*n/(e-s) for s, e, _, n in valid]
    points = sorted({v for s, e, _, _ in valid for v in (s, e)})
    peak_count = peak_chars = peak_rate = overlap_s = same_lane_s = 0
    pressure_s = 0
    for left, right in zip(points, points[1:]):
        active = [(s, e, lane, n) for s, e, lane, n in valid if s <= left < e]
        count = len(active)
        chars = sum(n for _, _, _, n in active)
        rate = sum(1000*n/(e-s) for s, e, _, n in active)
        peak_count = max(peak_count, count)
        peak_chars = max(peak_chars, chars)
        peak_rate = max(peak_rate, rate)
        if count > 1:
            overlap_s += right-left
        if count > len({lane for _, _, lane, _ in active}):
            same_lane_s += right-left
        if count > 1 and (chars > 100 or rate > 32.4):
            pressure_s += right-left
    return {
        "events": len(rows), "invalid_duration": invalid,
        "chronological": all(a[0] <= b[0] for a, b in zip(rows, rows[1:])),
        "above_20_cps": sum(1000*n > 20*(e-s) for s,e,_,n in valid),
        "above_25_cps": sum(1000*n > 25*(e-s) for s,e,_,n in valid),
        "under_1s": sum(x < 1000 for x in durations),
        "under_1_2s": sum(x < 1200 for x in durations),
        "median_cps": statistics.median(rates) if rates else None,
        "max_cps": max(rates, default=None),
        "peak_simultaneous_events": peak_count,
        "peak_simultaneous_chars": peak_chars,
        "peak_sum_event_cps": peak_rate,
        "overlap_seconds": overlap_s/1000,
        "same_style_overlap_seconds": same_lane_s/1000 if fmt == "ass" else None,
        "high_simultaneous_pressure_seconds": pressure_s/1000,
    }


def lineage_metrics(lineage):
    displayed = [r for r in lineage if r["disposition"] == "displayed"
                 and r.get("display_interval") is not None]
    shifts = [r["display_interval"][0] - r["display_allocation_interval"][0]
              for r in displayed if r["display_allocation_interval"][0] is not None]
    constituent_shifts = [r["display_interval"][0] - r["effective_event_interval"][0]
                          for r in displayed if r.get("effective_event_interval")]
    return {
        "max_positive_constituent_onset_shift_s": max([0.,*constituent_shifts]),
        "max_negative_constituent_onset_shift_s": min([0.,*constituent_shifts]),
        "accounted_occurrences": len({r["occurrence_id"] for r in lineage}),
        "displayed_occurrences": len({r["occurrence_id"] for r in displayed}),
        "overflow_occurrences": len({r["occurrence_id"] for r in lineage
                                    if r["disposition"] == "overflow"}),
        "unrenderable_occurrences": len({r["occurrence_id"] for r in lineage
                                         if r["disposition"] == "unrenderable"}),
        "max_positive_allocation_onset_shift_s": max([0.0, *shifts]),
        "max_negative_allocation_onset_shift_s": min([0.0, *shifts]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="Authenticated SubtitleConfig JSON object")
    parser.add_argument("--observations", type=Path, help="Persisted recording-bound presentation observations")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = args.repo.resolve() / "src/nvidia_transcribe/subtitles.py"
    if Path(subtitles.__file__).resolve() != expected:
        raise SystemExit("Wrong renderer import: " + str(subtitles.__file__))
    source_hash = digest(args.contract)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    config = subtitles.SubtitleConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    args.output.mkdir()
    kwargs = ({"presentation_observations": json.loads(args.observations.read_text(encoding="utf-8"))}
              if args.observations else {})
    first = subtitles.render_subtitles(contract, output_dir=args.output / "first",
                                      episode_stem="comparison", config=config, **kwargs)
    second = subtitles.render_subtitles(contract, output_dir=args.output / "repeat",
                                       episode_stem="comparison", config=config, **kwargs)
    report = {"contract_sha256": source_hash, "renderer_sha256": digest(expected),
              "config": dataclasses.asdict(config), "config_sha256": digest(args.config),
              "stats": first["stats"], "presentation": first.get("presentation"),
              "observations_sha256": digest(args.observations) if args.observations else None, "formats": {},
              "ass_lineage": lineage_metrics(first["cue_lineage"]),
              "srt_lineage": (lineage_metrics(first["srt_cue_lineage"])
                              if "srt_cue_lineage" in first else None)}
    for fmt in ("ass", "srt"):
        first_path, repeat_path = Path(first["outputs"][fmt]), Path(second["outputs"][fmt])
        report["formats"][fmt] = {**metrics(events(first_path, fmt), fmt),
                                  "sha256": digest(first_path),
                                  "repeat_sha256": digest(repeat_path),
                                  "byte_reproducible": first_path.read_bytes() == repeat_path.read_bytes()}
    report["input_unchanged"] = source_hash == digest(args.contract)
    report["lineage_reproducible"] = (
        first["cue_lineage"] == second["cue_lineage"]
        and first.get("srt_cue_lineage") == second.get("srt_cue_lineage"))
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "private-lineage.json").write_text(
        json.dumps({"ass": first["cue_lineage"], "srt": first.get("srt_cue_lineage"),
                    "conflicts": first["conflicts"]}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not (report["input_unchanged"] and report["lineage_reproducible"]
            and all(f["byte_reproducible"] and f["invalid_duration"] == 0
                    for f in report["formats"].values())):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
