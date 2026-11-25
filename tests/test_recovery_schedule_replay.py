"""Regression for replay admission across JSON and Python representations."""
import json

import pytest

from scripts.replay_recovery_schedule import _same_admitted_bracket


@pytest.mark.parametrize("bounds,expected", [
    ((91.535, 93.57), True),
    ((91.535, 93.571), False),
    ((91.536, 93.57), False),
    ((93.57, 91.535), False),
    ((91.535,), False),
])
def test_archived_admission_bracket_roundtrip(bounds, expected):
    prior = json.loads(json.dumps([{"bounds": (91.535, 93.57)}]))
    now = [{"bounds": bounds}]
    assert isinstance(prior[0]["bounds"], list)
    assert _same_admitted_bracket(prior, now) is expected


def test_admission_bracket_requires_unique_targets():
    target = {"bounds": (91.535, 93.57)}
    assert not _same_admitted_bracket([], [target])
    assert not _same_admitted_bracket([target], [])
    assert not _same_admitted_bracket([target, target], [target])
    assert not _same_admitted_bracket([target], [target, target])
