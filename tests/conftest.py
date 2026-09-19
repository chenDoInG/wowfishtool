import os
import sys

# Let tests import fishing/audio_listener from the project root regardless of
# how pytest is invoked (bare `pytest`, `python -m pytest`, PyCharm's runner, ...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest


@pytest.fixture(autouse=True)
def _debug_snapshots_off(monkeypatch):
	"""The flag is switched on locally while collecting real captures; tests must not depend on
	it (or write into debug/). Tests that want the trace turn it on themselves."""
	import float_detector
	monkeypatch.setattr(float_detector, 'DEBUG_SNAPSHOTS', False)
