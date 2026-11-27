"""Native connected translation hypotheses and exact selected allocations.

A nomination is not a linguistic assertion. Joint Riva text is an alternative,
never canonical source. No synthetic timing is generated here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re

from nvidia_transcribe.semantic_context import current_row_context
from nvidia_transcribe.review_uncertainty import current_uncertainty

SCHEMA = "connected-translation/v1"
_TERMINAL = re.compile(r"""[.!?。！？…][\s"'”’」』]*$""")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def native(row):
    return (row.get("timestamp_provenance") or {}).get("backend") == "moss_events"


def uid(row):
    return str(row.get("utterance_id") or row["id"])


def interval(row):
    s, e = row.get("source_start"), row.get("source_end")
    if (type(s) in (int, float) and type(e) in (int, float)
            and math.isfinite(s) and math.isfinite(e) and 0 <= s < e):
        return [s, e]
    return None


def observation(row):
    return {"id": uid(row), "source_text": row["source_text"],
            "speaker_id": row.get("speaker_id"), "interval": interval(row),
            "native_event_index": (row.get("timestamp_provenance") or {}).get("native_event_index")}


def _normalized(text):
    return "".join(c.casefold() for c in text if c.isalnum())


def _bound(row):
    return bool(current_row_context(row) or current_uncertainty(row)
                or any(row.get(k) for k in ("semantic_context", "review_source_uncertainty",
                        "source_lineage", "source_caption_evidence", "source_semantic_resolution",
                        "source_quality_status")))


def hypothesis_text(group):
    # Flatten ONLY this competing Riva request, never the review source.
    def cjk(ch):
        return "\u3400" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff"
    text = group[0]["source_text"].strip()
    for row in group[1:]:
        other = row["source_text"].strip()
        text += ("" if cjk(text[-1]) and cjk(other[0]) else " ") + other
    return text


def _eligible(group, row, rows):
    members = [*group, row]
    prior = group[-1]
    if len(group) >= 3 or not all(native(r) for r in members):
        return False
    if not prior.get("speaker_id") or any(r.get("speaker_id") != row.get("speaker_id")
                                            for r in group):
        return False
    if any(not str(r.get("source_text") or "").strip() or _bound(r)
           or interval(r) is None for r in members):
        return False
    a, b = observation(prior), observation(row)
    if (type(a["native_event_index"]) is not int or type(b["native_event_index"]) is not int
            or b["native_event_index"] != a["native_event_index"] + 1):
        return False
    if _TERMINAL.search(prior["source_text"]):
        return False
    if _normalized(row["source_text"]) in {_normalized(r["source_text"]) for r in group}:
        return False
    start, end = interval(group[0])[0], interval(row)[1]
    gap = interval(row)[0] - interval(prior)[1]
    if not 0 <= gap <= 0.35 or end - start > 12 or len(hypothesis_text(members)) > 240:
        return False
    ids = {uid(r) for r in members}
    return not any(uid(r) not in ids and interval(r) is not None
                   and interval(r)[0] < end and interval(r)[1] > start for r in rows)


def nominate(rows):
    """One bounded non-overlapping partition of alternative requests."""
    if not rows or not all(native(r) for r in rows):
        return []
    if len({uid(r) for r in rows}) != len(rows):
        raise ValueError("Duplicate canonical occurrence IDs")
    groups = []
    for row in rows:
        if groups and _eligible(groups[-1], row, rows):
            groups[-1].append(row)
        else:
            groups.append([row])
    return [g for g in groups if len(g) > 1]


def request_record(group, examples):
    body = {"schema": SCHEMA, "constituents": [observation(r) for r in group],
            "joint_input": hypothesis_text(group), "context_examples": deepcopy(examples)}
    return {**body, "candidate_id": "joint-" + digest(body)}


def groups(rows):
    """Validate candidate membership against the entire ordered input."""
    if len({uid(r) for r in rows}) != len(rows):
        raise ValueError("Duplicate canonical occurrence IDs")
    result = []
    pos = 0
    while pos < len(rows):
        candidate = rows[pos].get("translation_unit_candidate")
        if candidate is None:
            result.append([pos])
            pos += 1
            continue
        if not isinstance(candidate, dict) or candidate.get("schema") != SCHEMA:
            raise ValueError("Invalid translation candidate schema")
        members = candidate.get("constituents")
        if not isinstance(members, list) or not 2 <= len(members) <= 3:
            raise ValueError("Invalid translation candidate membership")
        end = pos + len(members)
        chunk = rows[pos:end]
        if ([observation(r) for r in chunk] != members
                or any(r.get("translation_unit_candidate") != candidate for r in chunk)
                or not all(native(r) for r in chunk)):
            raise ValueError("Candidate source/membership mismatch")
        if not all(_eligible(chunk[:i], r, rows) for i, r in enumerate(chunk) if i):
            raise ValueError("Candidate violates protected boundary")
        examples = candidate.get("context_examples")
        if not isinstance(examples, list) or len(examples) > 4:
            raise ValueError("Invalid candidate examples")
        if any(candidate.get(k) != v for k, v in request_record(chunk, examples).items()):
            raise ValueError("Candidate identity mismatch")
        if (not isinstance(candidate.get("joint_mt"), str) or not candidate["joint_mt"].strip()
                or type(candidate.get("joint_input_tokens")) is not int
                or not 0 < candidate["joint_input_tokens"] <= 8192):
            raise ValueError("Missing candidate MT receipt")
        result.append(list(range(pos, end)))
        pos = end
    return result


def selected_unit(rows, *, jointly_adjudicated):
    body = {"schema": SCHEMA, "constituents": [observation(r) for r in rows],
            "allocation": [{"id": uid(r), "english": r["translated_text"]} for r in rows],
            "selection": "jointly_adjudicated" if jointly_adjudicated else "independent",
            "candidate_id": (rows[0]["translation_unit_candidate"]["candidate_id"]
                             if jointly_adjudicated else None)}
    return {**body, "unit_id": "linguistic-" + digest(body)}


def bind_selected_units(rows, partitions, selected):
    """Bind only exact accepted allocations after review safeguards run."""
    for block in partitions:
        if not all(uid(rows[i]) in selected for i in block):
            continue
        chunk = [rows[i] for i in block]
        if len(chunk) > 1 and any(r.get("translation_review_status") == "corrected" for r in chunk):
            record = selected_unit(chunk, jointly_adjudicated=True)
            for row in chunk:
                row["translation_unit"] = deepcopy(record)
        else:
            for row in chunk:
                row["translation_unit"] = selected_unit([row], jointly_adjudicated=False)


def validate_selected_units(rows, *, require_complete=False):
    """Check exact selected allocation before offline display."""
    pos = 0
    while pos < len(rows):
        row = rows[pos]
        record = row.get("translation_unit")
        if record is None:
            if require_complete and native(row) and not row.get("pre_overlay_translation_unit"):
                raise ValueError("Missing selected native translation allocation")
            pos += 1
            continue
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            raise ValueError("Invalid selected translation unit")
        members = record.get("constituents")
        if not isinstance(members, list) or not 1 <= len(members) <= 3:
            raise ValueError("Invalid selected unit membership")
        chunk = rows[pos:pos+len(members)]
        if ([observation(r) for r in chunk] != members
                or any(r.get("translation_unit") != record for r in chunk)):
            raise ValueError("Selected unit occurrence mismatch")
        joint = len(chunk) > 1
        if joint and (any(r.get("condensation_applied") for r in chunk)
                      or any((r.get("translation_unit_candidate") or {}).get("candidate_id")
                             != record.get("candidate_id") for r in chunk)):
            raise ValueError("Selected joint unit allocation invalidated")
        prior = [{**r, "translated_text": r.get("uncondensed_translated_text", r["translated_text"])
                  if r.get("condensation_applied") else r["translated_text"]} for r in chunk]
        if selected_unit(prior, jointly_adjudicated=joint) != record:
            raise ValueError("Selected translation allocation mismatch")
        if joint:
            accepted = [r for r in chunk if r.get("translation_review_status") == "corrected"]
            receipts = [r.get("translation_review_passage") for r in accepted]
            expected = [{"id": uid(r), "english": r["translated_text"].strip()} for r in chunk]
            if (not receipts or not isinstance(receipts[0], dict)
                    or not receipts[0].get("atomic_connected_allocation")
                    or receipts[0].get("proposal_allocation") != expected
                    or receipts[0].get("target_ids") != [uid(r) for r in chunk]
                    or receipts[0].get("verdict", {}).get("choice") != "proposal"
                    or any(r != receipts[0] for r in receipts)):
                raise ValueError("Missing exact atomic review allocation receipt")
        pos += len(chunk)
