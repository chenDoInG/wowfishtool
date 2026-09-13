"""Regression test for find_float()'s template-matching logic.

Uses frozen sample screenshots (not the live var/fishing_session.png, which gets
overwritten every time the bot actually runs) with known, visually-verified float
positions, so tuning the matching/color-gating constants in float_detector.py later
can't silently break detection without a test failing.
"""
import os

from float_detector import find_float

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

# Flat, calm Stormwind canal water gave every template a much weaker grayscale
# correlation than the choppy open-ocean scenes they were captured from - even a
# hand-picked crop known to contain the float peaked next to it, not on it. Fixed by
# adding a template captured from this exact scene rather than any matching-logic change.
STORMWIND_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'stormwind_canal.png')
STORMWIND_EXPECTED_X, STORMWIND_EXPECTED_Y = 1399, 576
# The wooden dock/boat railing visible in this same screenshot is locally more
# saturated than the canal water too, and used to win once the color check stopped
# requiring a specific hue - only excluded once large connected color blobs (a dock
# spans hundreds of pixels; the float's never has) got dropped before gating.
STORMWIND_FALSE_POSITIVE_X, STORMWIND_FALSE_POSITIVE_Y = 1304, 898

# A deep-blue dusk sea shifted the float's colors so far around the hue wheel (the red
# feather reading as magenta, ~160 hue, instead of its usual ~0-10) that no fixed hue
# range could find it at all, and the water itself was saturated enough that no fixed
# saturation floor worked either - fixed by gating on saturation relative to this
# scene's own water instead of a fixed absolute range.
DUSK_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'dusk_saturated_water.png')
DUSK_EXPECTED_X, DUSK_EXPECTED_Y = 1245, 695
DUSK_TOLERANCE_PX = 10   # the base's own hue is shifted too, so the click centroid falls
# back to FALLBACK_VERTICAL_BIAS (see float_detector.py) rather than a real color match

# An even more saturated sea plateaued right up through its own 99th percentile before
# jumping sharply at the float's outlier pixels - baselining off the 90th percentile
# left too little headroom below FLOAT_SATURATION_MARGIN to tell the two apart, so the
# float was rejected as just more water. Fixed by baselining off the 99.5th percentile
# instead, which sits on that plateau rather than already inside the jump.
EXTREME_DUSK_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'extreme_dusk_saturation.png')
EXTREME_DUSK_EXPECTED_X, EXTREME_DUSK_EXPECTED_Y = 1027, 708
EXTREME_DUSK_TOLERANCE_PX = 10   # same fallback caveat as the dusk case above


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


def test_find_float_detects_float_on_calm_canal_water():
	place = find_float(STORMWIND_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - STORMWIND_EXPECTED_X) <= TOLERANCE_PX
	assert abs(y - STORMWIND_EXPECTED_Y) <= TOLERANCE_PX
	assert abs(x - STORMWIND_FALSE_POSITIVE_X) > TOLERANCE_PX or abs(y - STORMWIND_FALSE_POSITIVE_Y) > TOLERANCE_PX


def test_find_float_detects_float_in_hue_shifted_dusk_water():
	place = find_float(DUSK_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - DUSK_EXPECTED_X) <= DUSK_TOLERANCE_PX
	assert abs(y - DUSK_EXPECTED_Y) <= DUSK_TOLERANCE_PX


def test_find_float_detects_float_in_extremely_saturated_water():
	place = find_float(EXTREME_DUSK_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - EXTREME_DUSK_EXPECTED_X) <= EXTREME_DUSK_TOLERANCE_PX
	assert abs(y - EXTREME_DUSK_EXPECTED_Y) <= EXTREME_DUSK_TOLERANCE_PX
