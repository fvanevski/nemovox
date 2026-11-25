"""Acceptance classifier regression: evidence-only, never production overrides."""
import pytest

from scripts.replay_review_anchors import disposition_change

ANCHOR = "review_rejected_unresolved_source_preserved_machine_translation"
DEPENDENT = "review_rejected_dependent_passage_preserved_machine_translation"
REASON = "unresolved_source_name_or_number_change"


def row():
    return {"id": "u", "machine_translated_text": "That's not what Nadia thought.",
            "translated_text": "That's not what Nadia thought.",
            "translation_reviewed_text": "Nadia wouldn't have thought of that.",
            "translation_comparative_review": {"choice": "proposal"},
            "translation_review_status": ANCHOR,
            "translation_review_rejection_reason": REASON}


def test_status_only_reclassification_preserves_primary():
    a = row()
    b = dict(a, translation_review_status=DEPENDENT,
             translation_review_rejection_reason=None)
    assert disposition_change(a, b) == "status_only"
    assert disposition_change(a, a) is None


@pytest.mark.parametrize("old_status,old_reason", [(ANCHOR, REASON), (DEPENDENT, None)])
def test_approved_anchor_or_dependent_release(old_status, old_reason):
    a = dict(row(), translation_review_status=old_status,
             translation_review_rejection_reason=old_reason)
    b = dict(a, translated_text=a["translation_reviewed_text"],
             translation_review_status="corrected", translation_review_rejection_reason=None)
    assert disposition_change(a, b) == "text_changed"


@pytest.mark.parametrize("old_patch,new_patch", [
    ({"translation_review_rejection_reason": "unresolved_source_material_lexical_change"}, {}),
    ({"translation_review_status": "review_rejected_polarity_preserved_machine_translation"}, {}),
    ({"translation_review_status": "review_rejected_comparative_preserved_machine_translation"}, {}),
    ({}, {"translated_text": "Invented meaning."}),
    ({}, {"translation_review_status": "corrected"}),
    ({}, {"translation_review_rejection_reason": REASON}),
    ({}, {"translation_comparative_review": {"choice": "original"}}),
    ({}, {"machine_translated_text": "Different primary."}),
    ({}, {"translation_review_status": "review_error_preserved_machine_translation"}),
    ({}, {"translation_review_status": ANCHOR}),
])
def test_reclassification_fails_closed(old_patch, new_patch):
    a = dict(row(), **old_patch)
    b = dict(row(), translation_review_status=DEPENDENT,
             translation_review_rejection_reason=None)
    b.update(new_patch)
    with pytest.raises(ValueError, match="Unexpected changed disposition"):
        disposition_change(a, b)
