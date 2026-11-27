"""Explicit operator-authored viewing overlays; never automatic recognition repair."""
from __future__ import annotations

from typing import Any

from nvidia_transcribe import connected_units


def apply_validation_overlay(
    contract: dict[str, Any],
    overlay: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply curated coarse-timing recovery, phrasing polish, and duplicate/artifact suppression.

    Preserves ``id``, ``original_utterance_id``, raw ASR timing (``raw_source_start`` /
    ``raw_source_end``), and prior English wording (``pre_overlay_translated_text``) so every
    override remains auditable and reversible without modifying raw ASR responses.
    """
    entries = list(overlay.get("utterances") or overlay.get("entries") or [])
    if not entries:
        return contract, {"status": "empty_overlay", "applied_count": 0}

    utts: list[dict[str, Any]] = [dict(u) for u in (contract.get("utterances") or [])]
    by_id: dict[str, int] = {}
    by_orig_id: dict[str, int] = {}
    for idx, u in enumerate(utts):
        if u.get("id"):
            by_id[str(u["id"])] = idx
        if u.get("utterance_id"):
            by_id[str(u["utterance_id"])] = idx
        if u.get("original_utterance_id"):
            by_orig_id[str(u["original_utterance_id"])] = idx

    recovered_timing = 0
    relocated_timing = 0
    edited_text = 0
    suppressed_count = 0
    skipped: list[str] = []

    for ent in entries:
        tid = str(ent.get("utterance_id") or ent.get("id") or "").strip()
        orig_id = str(ent.get("original_utterance_id") or "").strip()
        pos = by_id.get(tid)
        if pos is None and orig_id:
            pos = by_orig_id.get(orig_id)
        if pos is None:
            skipped.append(tid or orig_id or "unknown")
            continue

        u = utts[pos]
        expected_sub = str(ent.get("expected_source_substring") or "").strip()
        if expected_sub:
            src_hay = f"{u.get('source_text') or ''} {u.get('refined_text') or ''}"
            if expected_sub not in src_hay:
                raise ValueError(
                    f"Validation overlay mismatch for {tid or orig_id}: expected source substring "
                    f"{expected_sub!r} not found in {src_hay!r}"
                )

        u.setdefault("reconstructed_utterance_id", str(u.get("id") or tid))

        # 1. Suppress confirmed duplicate / hallucination artifacts
        if ent.get("suppress") or ent.get("suppressed_duplicate_of"):
            if u.get("translated_text") and "pre_overlay_translated_text" not in u:
                u["pre_overlay_translated_text"] = u["translated_text"]
            u["overlay_suppressed"] = True
            if ent.get("suppressed_duplicate_of"):
                u["suppressed_duplicate_of"] = str(ent["suppressed_duplicate_of"])
            if ent.get("reason"):
                u["overlay_reason"] = str(ent["reason"])
            u["translated_text"] = ""
            suppressed_count += 1
            continue

        # 2. Explicit operator timing correction
        new_s = ent.get("proposed_start_s") if "proposed_start_s" in ent else ent.get("source_start")
        new_e = ent.get("proposed_end_s") if "proposed_end_s" in ent else ent.get("source_end")
        if new_s is not None and new_e is not None:
            fs, fe = float(new_s), float(new_e)
            if fe <= fs or fs < 0.0:
                raise ValueError(f"Invalid overlay timing [{fs}, {fe}] for {tid or orig_id}")
            had_timing = u.get("source_start") is not None and u.get("source_end") is not None
            if had_timing and "raw_source_start" not in u:
                u["raw_source_start"] = u["source_start"]
                u["raw_source_end"] = u["source_end"]
            u["source_start"] = round(fs, 3)
            u["source_end"] = round(fe, 3)
            prov = dict(u.get("timestamp_provenance") or {})
            t_status = str(ent.get("timing_status") or ("relocated_reviewed" if had_timing else "coarse_recovered"))
            prov["timing_status"] = t_status
            prov["timing_source"] = "viewing_validation_overlay"
            if ent.get("evidence"):
                prov["evidence"] = list(ent["evidence"])
            u["timestamp_provenance"] = prov
            u["timing_status"] = t_status
            if had_timing:
                relocated_timing += 1
            else:
                recovered_timing += 1

        # 3. Localized English dialogue polish
        if ent.get("translated_text") is not None:
            new_en = " ".join(str(ent["translated_text"]).split()).strip()
            old_en = str(u.get("translated_text") or "").strip()
            if new_en != old_en:
                if old_en and "pre_overlay_translated_text" not in u:
                    u["pre_overlay_translated_text"] = old_en
                u["translated_text"] = new_en
                u["overlay_text_edited"] = True
                edited_text += 1

        if ent.get("reason"):
            u["overlay_reason"] = str(ent["reason"])

    # Explicit operator overrides are separate from model review authority.
    # Retain the pre-overlay decision, retiring affected complete units.
    invalidated = set()
    original_by_id = {str(row.get("utterance_id") or row.get("id")): row
                      for row in contract.get("utterances", [])}
    for row in utts:
        before = original_by_id.get(str(row.get("utterance_id") or row.get("id")), {})
        unit = row.get("translation_unit")
        if unit and any(row.get(k) != before.get(k) for k in (
                "source_start", "source_end", "translated_text", "overlay_suppressed",
                "suppressed_duplicate_of")):
            invalidated.add(unit["unit_id"])
    # Maintain chronological ordering for timed/recovered items while preserving IDs cleanly
    timed_pairs = []
    untimed_pairs = []
    for orig_idx, u in enumerate(utts):
        if u.get("source_start") is not None and u.get("source_end") is not None:
            timed_pairs.append(((0, float(u["source_start"]), float(u["source_end"]), orig_idx), u))
        else:
            untimed_pairs.append(((1, orig_idx), u))
    ordered_utts = [u for _, u in sorted(timed_pairs, key=lambda x: x[0])] + [
        u for _, u in sorted(untimed_pairs, key=lambda x: x[0])
    ]

    # An overlay can disrupt an untouched joint unit by moving a different
    # occurrence into its ordered membership or protected time envelope.
    # Evaluate the final sequence before retiring any complete allocation.
    positions = {str(row.get("utterance_id") or row.get("id")): i
                 for i, row in enumerate(ordered_utts)}
    intervals = [(i, connected_units.interval(row))
                 for i, row in enumerate(ordered_utts)]
    checked = set()
    for row in ordered_utts:
        unit = row.get("translation_unit")
        if not unit or unit["unit_id"] in invalidated or unit["unit_id"] in checked:
            continue
        checked.add(unit["unit_id"])
        members = unit["constituents"]
        if len(members) <= 1:
            continue
        member_positions = [positions.get(member["id"]) for member in members]
        first = member_positions[0]
        if first is None or member_positions != list(range(first, first + len(members))):
            invalidated.add(unit["unit_id"])
            continue
        # Untouched constituents retain their admitted intervals. Even if
        # adjacency survives, an unrelated overlapping event breaks the
        # original joint-candidate protection against intervening speech.
        start, end = members[0]["interval"][0], members[-1]["interval"][1]
        occupied = set(member_positions)
        if any(i not in occupied and bounds is not None
               and bounds[0] < end and bounds[1] > start for i, bounds in intervals):
            invalidated.add(unit["unit_id"])
    for row in ordered_utts:
        unit = row.get("translation_unit")
        if unit and unit["unit_id"] in invalidated:
            row["pre_overlay_translation_unit"] = row.pop("translation_unit")

    summary = {
        "status": "applied",
        "overlay_name": str(overlay.get("name") or "custom"),
        "recovered_timing_count": recovered_timing,
        "relocated_timing_count": relocated_timing,
        "edited_text_count": edited_text,
        "suppressed_count": suppressed_count,
        "skipped_ids": skipped,
        "invalidated_translation_units": sorted(invalidated),
    }
    return {**contract, "utterances": ordered_utts, "validation_overlay": summary}, summary
