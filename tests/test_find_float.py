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

# A shoreline color-boundary line used to out-score the real float on pure grayscale
# template matching, landing the click on plain water. Regression fixture for that:
# the real float sits around (1117, 634), the false positive was at (1078, 789).
SHORELINE_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'shoreline_false_positive.png')
SHORELINE_EXPECTED_X, SHORELINE_EXPECTED_Y = 1117, 634
SHORELINE_FALSE_POSITIVE_X, SHORELINE_FALSE_POSITIVE_Y = 1078, 789

# A distant ship near the horizon had both a saturated warm (lit window) and cool (hull)
# color close together, passing the color check too, and out-scored the real float on
# grayscale correlation. Fixed by restricting the search band to where a nearby cast
# actually lands - well below where anything at/near the horizon (like a ship) sits.
SHIP_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'ship_false_positive.png')
SHIP_EXPECTED_X, SHIP_EXPECTED_Y = 1366, 851
SHIP_FALSE_POSITIVE_X, SHIP_FALSE_POSITIVE_Y = 700, 440

# Under overcast/dim lighting the blue feather's saturation drops well below what the
# cool color range required, so the float was rejected as colorless and never found.
DESATURATED_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'desaturated_feather.png')
DESATURATED_EXPECTED_X, DESATURATED_EXPECTED_Y = 1087, 796

# Backlit against a bright hazy sky, the feather's blue washed out to near the water's
# own saturation noise floor - too close to fix with a saturation threshold, so the
# cool-color requirement was dropped in favor of relying on the warm bobber base color
# (still strongly saturated here) plus the search-band restriction that already
# independently keeps distant objects like ships out.
BACKLIT_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'backlit_float.png')
BACKLIT_EXPECTED_X, BACKLIT_EXPECTED_Y = 1355, 688


def test_find_float_locates_the_known_float():
	place = find_float(FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - EXPECTED_X) <= TOLERANCE_PX
	assert abs(y - EXPECTED_Y) <= TOLERANCE_PX


def test_find_float_rejects_shoreline_false_positive():
	place = find_float(SHORELINE_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - SHORELINE_EXPECTED_X) <= TOLERANCE_PX
	assert abs(y - SHORELINE_EXPECTED_Y) <= TOLERANCE_PX
	# also explicitly guard against regressing back onto the old false-positive spot
	assert abs(x - SHORELINE_FALSE_POSITIVE_X) > TOLERANCE_PX or abs(y - SHORELINE_FALSE_POSITIVE_Y) > TOLERANCE_PX


def test_find_float_ignores_distant_ship():
	place = find_float(SHIP_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - SHIP_EXPECTED_X) <= TOLERANCE_PX
	assert abs(y - SHIP_EXPECTED_Y) <= TOLERANCE_PX
	assert abs(x - SHIP_FALSE_POSITIVE_X) > TOLERANCE_PX or abs(y - SHIP_FALSE_POSITIVE_Y) > TOLERANCE_PX


def test_find_float_detects_desaturated_feather():
	place = find_float(DESATURATED_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - DESATURATED_EXPECTED_X) <= TOLERANCE_PX
	assert abs(y - DESATURATED_EXPECTED_Y) <= TOLERANCE_PX


def test_find_float_detects_backlit_float():
	place = find_float(BACKLIT_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - BACKLIT_EXPECTED_X) <= TOLERANCE_PX
	assert abs(y - BACKLIT_EXPECTED_Y) <= TOLERANCE_PX
