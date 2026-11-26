import pytest

from libero_sketch.suites import ALL_SUITES, resolve_suites


def test_all_excludes_libero_90():
    assert ALL_SUITES == ["spatial", "object", "goal", "long"]
    assert resolve_suites("all") == ALL_SUITES


def test_libero_90_only_on_request():
    assert resolve_suites("90") == ["90"]          # the held-out validation set
    with pytest.raises(ValueError, match="Unknown suite"):
        resolve_suites("libero_90")
