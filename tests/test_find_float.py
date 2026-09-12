"""Regression test for find_float()'s template-matching logic.

Uses a frozen sample screenshot (not the live var/fishing_session.png, which gets
overwritten every time the bot actually runs) with a known, visually-verified float
position, so tuning FLOAT_MATCH_THRESHOLD / FLOAT_SEARCH_X_RANGE /
FLOAT_CLICK_X_OFFSET_RATIO later can't silently break detection without a test failing.
"""
import os

from fishing import find_float

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_screenshot.png')
EXPECTED_X, EXPECTED_Y = 1362.75, 660.5
TOLERANCE_PX = 20


def test_find_float_locates_the_known_float():
	place = find_float(FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - EXPECTED_X) <= TOLERANCE_PX
	assert abs(y - EXPECTED_Y) <= TOLERANCE_PX
