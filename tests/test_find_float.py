"""Regression test for find_float()'s template-matching logic.

Uses frozen sample screenshots (not the live var/fishing_session.png, which gets
overwritten every time the bot actually runs) with known, visually-verified float
positions, so tuning the matching/color-gating constants in float_detector.py later
can't silently break detection without a test failing.
"""
import os

import cv2
import numpy as np

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
EXTREME_DUSK_EXPECTED_X, EXTREME_DUSK_EXPECTED_Y = 1016.58, 670.76
EXTREME_DUSK_TOLERANCE_PX = 20   # which template ends up matching (and therefore the
# exact box the base color gets searched within) can flip between near-tied templates as
# new ones are added, so this can't be pinned as tightly as a single-template case could be

# A sunset scene where the sky's pink/purple tint carried into the water - the float
# itself was clearly visible and well inside the search band, but none of the templates
# captured up to that point matched it above FLOAT_MATCH_THRESHOLD at its real position
# (matches that did clear the threshold were all at other, wrong locations). Fixed by
# adding fishing_float_6.png, cropped from this exact scene.
SUNSET_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'sunset_purple_water.png')
SUNSET_EXPECTED_X, SUNSET_EXPECTED_Y = 1077, 434

# Water saturated enough that 17% of the search band's pixels sat at the HSV ceiling
# (255) meant the 99.5th-percentile baseline itself came out to 255 - adding
# FLOAT_SATURATION_MARGIN on top pushed the required threshold past the maximum
# representable saturation, so zero pixels anywhere could ever pass the color gate
# regardless of the float's own color. The float's template match still scored a
# confident 0.659 on grayscale alone. Fixed by skipping the color gate entirely when it
# would zero out the whole search area, trusting the template score alone instead.
CLIPPED_SATURATION_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'saturation_ceiling_clip.png')
CLIPPED_SATURATION_EXPECTED_X, CLIPPED_SATURATION_EXPECTED_Y = 914, 615

# A dark-night scene where the float itself renders dim, not just the water around it -
# its clearly-saturated feather pixels landed at value 39-61, mostly below the old fixed
# FLOAT_MIN_VALUE=60 floor, so the color gate rejected the float's own correct location
# (grayscale shape match found it fine, at a confident 0.753) as if it were colorless.
# Fixed by lowering the floor to 30 - see the comment on FLOAT_MIN_VALUE.
DIM_NIGHT_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'dim_night_float.png')
DIM_NIGHT_EXPECTED_X, DIM_NIGHT_EXPECTED_Y = 1243.67, 695.69
DIM_NIGHT_TOLERANCE_PX = 20

# A moonlit-choppy-water scene where the real float's own peak saturation (183) sat below
# the water's own baseline (218 before any margin), and it also has WoW's default UI
# player-frame cluster sitting in the search band twice (the always-on frame bottom-left,
# plus the "Modern" Edit Mode layout's duplicate bottom-right) - both far more saturated
# than the water and, once the real float lost the color gate, high-scoring enough to
# confidently win in the real float's place. Fixed in two parts: UI_EXCLUDE_REGIONS keeps
# either frame from ever being the answer, and the empty-color-mask fallback (see
# UNIFORM_DESAT below) recovers the real float once the gate is bypassed.
UI_FRAME_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'ui_frame_false_positive.png')
UI_FRAME_EXPECTED_X, UI_FRAME_EXPECTED_Y = 1161.16, 737.74
UI_FRAME_TOLERANCE_PX = 20
UI_FRAME_LEFT_FALSE_POSITIVE_X, UI_FRAME_LEFT_FALSE_POSITIVE_Y = 742, 1061
UI_FRAME_RIGHT_FALSE_POSITIVE_X, UI_FRAME_RIGHT_FALSE_POSITIVE_Y = 1698, 1099

# The same moonlit-choppy-water scene as above, but this time literally nothing in the
# whole search band (outside the excluded UI frames) clears the color gate - not the
# float, not any other water pixel either: baseline 219 (threshold 239) against the
# float's own peak of 184. Same failure shape as the saturation-ceiling-clip case (the
# gate can't discriminate anything in this scene) even though the threshold never
# numerically exceeds 255. Fixed by falling back to the grayscale shape match alone
# whenever the color mask ends up completely empty, not just when the threshold clips.
UNIFORM_DESAT_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'uniformly_desaturated_water.png')
UNIFORM_DESAT_EXPECTED_X, UNIFORM_DESAT_EXPECTED_Y = 1365.30, 695.93
UNIFORM_DESAT_TOLERANCE_PX = 20

# A dark-night scene where literally every cast fell back to the geometric-center click
# point: the base's own hue (~15-21) sat squarely inside the daylight FLOAT_BASE_COLOR_RANGES
# window, but its saturation/value (25-131/34-106) fell well under that range's 80/100
# floors, so it never registered as base-colored despite being clearly visible once
# brightened for inspection. Confirmed against 8 real fallback captures from the same
# session - 3 of 8 recovered a real color match with the added range below, the rest still
# fall back for an unrelated reason (a weak/offset shape match not padding enough of the
# base into the searched region at all, not a color range problem). Fixed by adding a
# third, dimmer range with the same hue window as the daylight base - see the comment on
# FLOAT_BASE_COLOR_RANGES.
DIM_NIGHT_BASE_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'dim_night_base_color.png')
DIM_NIGHT_BASE_EXPECTED_X, DIM_NIGHT_BASE_EXPECTED_Y = 1355.61, 695.46


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


def test_find_float_detects_float_in_sunset_tinted_water():
	place = find_float(SUNSET_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - SUNSET_EXPECTED_X) <= TOLERANCE_PX
	assert abs(y - SUNSET_EXPECTED_Y) <= TOLERANCE_PX


def test_find_float_detects_float_when_water_saturation_clips_the_color_gate():
	place = find_float(CLIPPED_SATURATION_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - CLIPPED_SATURATION_EXPECTED_X) <= TOLERANCE_PX
	assert abs(y - CLIPPED_SATURATION_EXPECTED_Y) <= TOLERANCE_PX


def test_find_float_detects_dim_float_in_dark_night_scene():
	place = find_float(DIM_NIGHT_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - DIM_NIGHT_EXPECTED_X) <= DIM_NIGHT_TOLERANCE_PX
	assert abs(y - DIM_NIGHT_EXPECTED_Y) <= DIM_NIGHT_TOLERANCE_PX


def test_find_float_ignores_default_ui_player_frames():
	place = find_float(UI_FRAME_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - UI_FRAME_EXPECTED_X) <= UI_FRAME_TOLERANCE_PX
	assert abs(y - UI_FRAME_EXPECTED_Y) <= UI_FRAME_TOLERANCE_PX
	# also explicitly guard against regressing back onto either UI frame
	assert abs(x - UI_FRAME_LEFT_FALSE_POSITIVE_X) > TOLERANCE_PX or abs(y - UI_FRAME_LEFT_FALSE_POSITIVE_Y) > TOLERANCE_PX
	assert abs(x - UI_FRAME_RIGHT_FALSE_POSITIVE_X) > TOLERANCE_PX or abs(y - UI_FRAME_RIGHT_FALSE_POSITIVE_Y) > TOLERANCE_PX


def test_find_float_detects_float_when_whole_scene_is_too_desaturated_for_the_gate():
	place = find_float(UNIFORM_DESAT_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - UNIFORM_DESAT_EXPECTED_X) <= UNIFORM_DESAT_TOLERANCE_PX
	assert abs(y - UNIFORM_DESAT_EXPECTED_Y) <= UNIFORM_DESAT_TOLERANCE_PX


def test_find_float_detects_base_color_of_a_dark_night_float():
	place = find_float(DIM_NIGHT_BASE_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - DIM_NIGHT_BASE_EXPECTED_X) <= TOLERANCE_PX
	assert abs(y - DIM_NIGHT_BASE_EXPECTED_Y) <= TOLERANCE_PX


def test_find_float_rejects_colorless_frame(tmp_path):
	# A flat grey frame - standing in for WoW's disconnect/login/character-select
	# screens, which have no saturated pixels at all. The color gate must stay active
	# here (low water_baseline, nowhere near the 255 ceiling) and correctly reject
	# whatever the grayscale-only template correlation happens to score on plain grey -
	# guards against the saturation-ceiling fix above (color_gate_active in find_float())
	# ever being broadened into skipping the gate on any empty mask, not just the
	# specific "threshold exceeds 255" case it's meant for.
	blank_frame = np.full((1050, 1893, 3), 120, dtype=np.uint8)
	frame_path = tmp_path / 'colorless_frame.png'
	cv2.imwrite(str(frame_path), blank_frame)

	assert find_float(str(frame_path)) is None
