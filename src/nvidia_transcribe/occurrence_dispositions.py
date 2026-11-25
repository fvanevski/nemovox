"""Finalize provisional boundary dispositions against the actual selected set."""
from __future__ import annotations


def finalize_boundary_dispositions(dispositions):
    """Keep direct evidence edges; never infer transitive occurrence identity.

    Earlier boundary passes may nominate a counterpart that a later pass
    suppresses or that core ownership never selects. Such an observation
    is unresolved, not a covered alternate. Selection and timing themselves
    are unchanged: recovery, not graph traversal, adjudicates occurrence.
    """
    by_id = {d["source_unit_id"]: d for d in dispositions}
    if len(by_id) != len(dispositions):
        raise ValueError("Duplicate source-unit disposition")
    selected = {uid for uid, d in by_id.items() if d["status"].startswith("selected_")}
    result = []
    for d in dispositions:
        refs = set(d["related_source_unit_ids"])
        if not refs <= by_id.keys() or d["source_unit_id"] in refs:
            raise ValueError("Invalid boundary correspondence reference")
        if d["status"] == "alternate_not_selected" and (not refs or not refs <= selected):
            result.append({
                **d,
                "status": "context_candidate_unresolved",
                "reason": "provisional_correspondence_without_selected_counterpart",
                "prior_selection_reason": d["reason"],
            })
        else:
            result.append(dict(d))
    return result
