"""Regression test for find_float()'s template-matching logic.

Uses frozen sample screenshots (not the live var/fishing_session.png, which gets
overwritten every time the bot actually runs) with known, visually-verified float
positions, so tuning the matching/color-gating constants in float_detector.py later
can't silently break detection without a test failing.
"""
import os

import cv2
import numpy as np

from float_detector import FLOAT_MAX_BLOB_SIZE, _adaptive_color_mask, _drop_large_blobs, find_float

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

# A heavily orange-tinted dusk sea (logged live as a "not found" miss, not a bad-click
# fallback) where the water's own 99.5th-percentile saturation already sat at 190, so the
# adaptive threshold (210) was higher than the float's own base ever reached in this
# lighting (measured 70-140) - the gate rejected the float's true position outright. The
# "gate found nothing anywhere" escape hatch didn't fire here, unlike UNIFORM_DESAT/
# CLIPPED_SATURATION above: unrelated scattered pixels elsewhere in the search band (not
# shaped anything like the float) happened to clear that same threshold, keeping the mask
# non-empty and the gate looking "active" even though it still couldn't discriminate the
# float from water. Confirmed real: grayscale alone scored 0.53-0.7 at the float's actual
# position (comfortably within the confirmed-real range noted on FLOAT_MATCH_THRESHOLD),
# while every gated candidate topped out at 0.24-0.34. Fixed by also falling back to the
# grayscale-only match when the gate accepts nothing above FLOAT_MATCH_THRESHOLD anywhere,
# not just when its mask is completely empty.
WARM_DUSK_GATE_MISS_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'warm_dusk_gate_miss.png')
WARM_DUSK_GATE_MISS_EXPECTED_X, WARM_DUSK_GATE_MISS_EXPECTED_Y = 1438, 800

# A different warm-dusk capture (same session, box-location matching worked fine here -
# grayscale scored 0.594 right on the float) where the water's own color fell inside the
# base's daylight FLOAT_BASE_COLOR_RANGES over a wide contiguous area, fusing the float's
# own base into one 224x157 blob with the surrounding water. _drop_large_blobs correctly
# dropped that fused blob (it clears FLOAT_MAX_BLOB_SIZE), but the scattered few-pixel
# fragments left over elsewhere in the padded region still summed past
# the summed-pixel floor, centroiding to a point ~85px from the float - confidently
# wrong rather than correctly falling back. Fixed by also requiring the single largest
# surviving component to look like a real blob - see FLOAT_MIN_BASE_BLOB_AREA.
CLICK_NOISE_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'warm_water_gate_click_noise.png')
CLICK_NOISE_EXPECTED_X, CLICK_NOISE_EXPECTED_Y = 1388, 765
CLICK_NOISE_TOLERANCE_PX = 40   # the fix only restores the existing geometric-center
# fallback (falling back is still correct here - the base's true color is indistinguishable
# from the water over a wide area, so there's no real color match to be had), which is
# intentionally looser than a real color-matched click - see FALLBACK_VERTICAL_BIAS


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


def test_find_float_detects_float_when_gate_rejects_it_but_other_pixels_keep_it_active():
	place = find_float(WARM_DUSK_GATE_MISS_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - WARM_DUSK_GATE_MISS_EXPECTED_X) <= TOLERANCE_PX
	assert abs(y - WARM_DUSK_GATE_MISS_EXPECTED_Y) <= TOLERANCE_PX


def test_find_float_ignores_scattered_noise_when_water_matches_base_color():
	place = find_float(CLICK_NOISE_FIXTURE_PATH)

	assert place is not None
	x, y = place
	assert abs(x - CLICK_NOISE_EXPECTED_X) <= CLICK_NOISE_TOLERANCE_PX
	assert abs(y - CLICK_NOISE_EXPECTED_Y) <= CLICK_NOISE_TOLERANCE_PX


def test_adaptive_color_mask_does_not_drop_large_blobs_itself():
	"""_adaptive_color_mask() must hand back the raw saturation mask, not one that's
	already had oversized blobs dropped - find_float() needs to blank UI_EXCLUDE_REGIONS
	first and only drop oversized blobs after (see the next test for why), which only
	works if this function doesn't do that dropping internally beforehand."""
	# Canvas sized so the blob below stays under the 99.5th-percentile baseline's own
	# 0.5% slice of the image - otherwise the blob would skew the adaptive threshold
	# itself high enough to swallow its own pixels, which is a real effect (see
	# FLOAT_SATURATION_BASELINE_PERCENTILE) but not what this test is checking.
	hsv = np.zeros((800, 1400, 3), dtype=np.uint8)
	hsv[:, :, 1] = 30
	hsv[:, :, 2] = 150
	hsv[100:120, 50:300, 1] = 200   # 250px wide - well over FLOAT_MAX_BLOB_SIZE

	mask = _adaptive_color_mask(hsv)

	assert int((mask[100:120, 50:300] > 0).sum()) == 250 * 20


def test_find_float_keeps_a_float_touching_a_ui_frame_blob():
	"""A float that merely touches a UI-frame-sized saturated blob (rather than landing
	inside UI_EXCLUDE_REGIONS itself) must not have its own, individually-small color
	evidence wiped out because _drop_large_blobs saw one merged oversized blob before UI
	exclusion got a chance to separate them. Reproduced live: a 40x10 float-sized patch
	lost all its color support once an adjacent 170x10 UI-sized patch pushed their
	combined bounding box over FLOAT_MAX_BLOB_SIZE, even though each alone was
	comfortably under it - fixed by blanking the UI area before dropping large blobs,
	not after (see the comment in find_float())."""
	w, h = 1280, 593
	hsv = np.zeros((h, w, 3), dtype=np.uint8)
	hsv[:, :, 1] = 30
	hsv[:, :, 2] = 150
	hsv[200:210, 100:140, 1] = 200   # float-sized patch: 40x10
	hsv[200:210, 140:310, 1] = 200   # UI-sized patch touching it: 170x10

	mask = _adaptive_color_mask(hsv)
	# Simulate find_float()'s actual composition order: blank the UI area first, only
	# then drop oversized blobs.
	mask[195:215, 140:310] = 0
	mask = _drop_large_blobs(mask, FLOAT_MAX_BLOB_SIZE)

	assert int((mask[200:210, 100:140] > 0).sum()) == 40 * 10


def test_find_float_rejects_colorless_frame(tmp_path):
	# A flat grey frame - standing in for WoW's disconnect/login/character-select
	# screens, which have no saturated pixels at all. The gate finds nothing here, so
	# find_float() falls back to the ungated grayscale score (see _pick_match()) - which
	# must itself stay under FLOAT_MATCH_THRESHOLD on plain grey, or this frame would
	# read as a float.
	blank_frame = np.full((1050, 1893, 3), 120, dtype=np.uint8)
	frame_path = tmp_path / 'colorless_frame.png'
	cv2.imwrite(str(frame_path), blank_frame)

	assert find_float(str(frame_path)) is None


def _frame_with_float_at(center_x, center_y, tmp_path):
	"""Flat 2560x1410 grey frame with a real float template pasted so its center lands
	at (center_x, center_y), saved to disk; returns the path."""
	template = cv2.imread('var/fishing_float_1.png')
	th, tw = template.shape[:2]
	frame = np.full((1410, 2560, 3), 120, dtype=np.uint8)
	x0, y0 = center_x - tw // 2, center_y - th // 2
	frame[y0:y0 + th, x0:x0 + tw] = template
	path = tmp_path / 'frame.png'
	cv2.imwrite(str(path), frame)
	return str(path)


def test_find_float_ignores_a_perfect_match_inside_a_ui_exclude_region(tmp_path):
	"""UI_EXCLUDE_REGIONS must hold on the ungated grayscale fallback too, not just in the
	color mask: a pixel-exact template match sitting inside the player-frame region
	scores ~1.0 ungated, and must still not be returned."""
	assert find_float(_frame_with_float_at(790, 1070, tmp_path)) is None


def test_find_float_still_finds_the_same_match_outside_ui_exclude_regions(tmp_path):
	"""Control for the test above: the identical pasted float, moved out of every UI
	region, is found."""
	assert find_float(_frame_with_float_at(1280, 800, tmp_path)) is not None
