import os
import sys

# Let tests import fishing/audio_listener from the project root regardless of
# how pytest is invoked (bare `pytest`, `python -m pytest`, PyCharm's runner, ...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest


@pytest.fixture(autouse=True)
def _run_from_the_repo_root(monkeypatch):
	"""The bot's files are relative to the project root (var/fishing_float_*.png, debug/), so a test run from any
	other directory used to find no templates and fail 32 tests for a reason that had nothing to do with the code."""
	monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _debug_snapshots_off(monkeypatch, tmp_path_factory):
	"""The flag is switched on locally while collecting real captures; tests must not depend on
	it, and must never write into the real debug/ - that used to happen when a test turned the flag
	on without redirecting the snapshot folder and the code under test then fell back. Tests that want
	the trace turn the flag on themselves."""
	import float_detector
	monkeypatch.setattr(float_detector, 'DEBUG_SNAPSHOTS', False)
	monkeypatch.setattr(float_detector, 'DEBUG_SNAPSHOT_DIR', str(tmp_path_factory.mktemp('debug_snapshots')))
