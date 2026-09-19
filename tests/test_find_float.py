"""Regression tests for find_float()'s template-matching logic.

Uses frozen sample screenshots (not the live var/fishing_session.png, which gets
overwritten every time the bot actually runs) with known, visually-verified float
positions, so tuning the matching/color-gating constants in float_detector.py later
can't silently break detection without a test failing.
"""
import os

import cv2
import numpy as np
import pytest

from float_detector import (FLOAT_MAX_BLOB_SIZE, FLOAT_MIN_TEXTURE, _adaptive_color_mask, _box_texture, _build_color_mask,
								_drop_large_blobs, _Match, _pick_match, find_float)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
TOLERANCE_PX = 20

# (fixture, expected (x, y), tolerance, spots that must NOT be returned, what it guards)
FLOAT_CASES = [
	('ship_false_positive.png', (1366, 851), TOLERANCE_PX, [(700, 440)],
	 'a distant ship near the horizon has saturated warm+cool colors and out-scored the '
	 'float; fixed by restricting the search band to where a nearby cast lands (the '
	 'same open-sea capture also covers the old shoreline color-boundary false positive)'),
	('desaturated_feather.png', (1087, 796), TOLERANCE_PX, [],
	 'overcast lighting drops the blue feather well below any fixed saturation floor'),
	('backlit_float.png', (1355, 688), TOLERANCE_PX, [],
	 'backlit against a hazy sky the feather washes out to water level - relies on the '
	 'warm base color instead'),
	('stormwind_canal.png', (1399, 576), TOLERANCE_PX, [(1304, 898)],
	 'flat canal water weakens every template (fixed with a scene-specific template); the '
	 'dock/railing is locally more saturated and must be dropped as a large blob'),
	('dusk_saturated_water.png', (1245, 695), 10, [],
	 'deep-blue dusk sea shifts the float hues around the wheel - needs the adaptive '
	 'saturation gate; the base hue is shifted too, so the click falls back to '
	 'FALLBACK_VERTICAL_BIAS, hence the tighter tolerance'),
	('sunset_purple_water.png', (1077, 434), TOLERANCE_PX, [],
	 'pink/purple sunset water; no template matched until fishing_float_6.png was added'),
	('saturation_ceiling_clip.png', (914, 615), TOLERANCE_PX, [],
	 'water so saturated the gate threshold exceeds 255 - the gate passes nothing, so '
	 'the ungated grayscale match must carry it (score 0.659)'),
	('dim_night_float.png', (1243.67, 695.69), TOLERANCE_PX, [],
	 'dark night: the float itself renders dim (feather value 39-61), below the old '
	 'FLOAT_MIN_VALUE=60'),
	('dim_night_base_color.png', (1355.61, 695.46), TOLERANCE_PX, [],
	 'dark night base: hue inside the daylight range but S/V far below its floors, so '
	 'every cast fell back to the geometric center; needs the dim base color range'),
	('warm_dusk_gate_miss.png', (1438, 800), TOLERANCE_PX, [],
	 'orange dusk: the gate rejects the float\'s true position while unrelated scattered '
	 'pixels keep it "active" - grayscale alone scored 0.53-0.7 at the float, gated '
	 'candidates only 0.24-0.34'),
	('warm_water_gate_click_noise.png', (1388, 765), 40, [],
	 'warm water fuses the base into one giant blob that gets dropped; leftover fragments '
	 'summed to a confident-but-wrong centroid ~85px off. Correct answer is the '
	 'geometric-center fallback, hence the looser tolerance (see FALLBACK_VERTICAL_BIAS)'),
]


@pytest.mark.parametrize('fixture, expected, tolerance, forbidden, why', FLOAT_CASES,
						 ids=[case[0].removesuffix('.png') for case in FLOAT_CASES])
def test_find_float_locates_the_float(fixture, expected, tolerance, forbidden, why):
	place = find_float(os.path.join(FIXTURE_DIR, fixture))

	assert place is not None, why
	x, y = place
	assert abs(x - expected[0]) <= tolerance, why
	assert abs(y - expected[1]) <= tolerance, why
	for false_x, false_y in forbidden:
		assert abs(x - false_x) > TOLERANCE_PX or abs(y - false_y) > TOLERANCE_PX, why


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


def test_find_float_finds_a_float_low_in_the_search_band(tmp_path):
	"""Real casts have landed as low as 77% of the window's height; the band's lower edge
	must stay padded past that (FLOAT_SEARCH_Y_RANGE)."""
	assert find_float(_frame_with_float_at(1280, int(1410 * 0.77), tmp_path)) is not None


def test_find_float_ignores_a_match_up_near_the_horizon(tmp_path):
	"""A perfect template match well above where a nearby cast can land (where a distant
	ship would sit) must be outside the search band."""
	assert find_float(_frame_with_float_at(1280, int(1410 * 0.30), tmp_path)) is None


def _water_hsv(h=800, w=1400, saturation=30, value=150):
	"""Uniform water-like HSV canvas, sized so a few thousand saturated pixels stay under
	the 99.5th-percentile baseline's own 0.5% slice (see FLOAT_SATURATION_BASELINE_PERCENTILE)."""
	hsv = np.zeros((h, w, 3), dtype=np.uint8)
	hsv[:, :, 0] = 100
	hsv[:, :, 1] = saturation
	hsv[:, :, 2] = value
	return hsv


def test_adaptive_color_mask_admits_a_dim_but_saturated_float():
	"""A dark-night float's feather renders saturated but dim (value 39-61) against water
	that never exceeds ~46 - it must clear the gate, i.e. FLOAT_MIN_VALUE stays low."""
	hsv = _water_hsv(saturation=30, value=30)
	hsv[300:310, 400:440, 1] = 200
	hsv[300:310, 400:440, 2] = 45

	mask = _adaptive_color_mask(hsv)

	assert int((mask[300:310, 400:440] > 0).sum()) == 40 * 10


def test_adaptive_color_mask_baselines_off_the_top_of_a_saturation_plateau():
	"""A very saturated sea plateaus well above its median right up through its own 99th
	percentile before the float's outliers jump above it. The baseline must sit on that
	plateau (99.5th percentile), or the plateau itself passes the gate as 'float-colored'."""
	hsv = _water_hsv(saturation=30)
	hsv[:70, :, 1] = 200             # 70/800 = 8.75% of rows: the plateau, all "just water"
	hsv[300:310, 400:440, 1] = 250   # the float's own outlier pixels, well above it

	mask = _adaptive_color_mask(hsv)

	assert int((mask[:70] > 0).sum()) == 0
	assert int((mask[300:310, 400:440] > 0).sum()) == 40 * 10


def test_color_mask_drops_a_large_saturated_structure_but_keeps_the_float():
	"""A dock/hull-sized saturated blob (over FLOAT_MAX_BLOB_SIZE) must not survive as float
	color evidence, while a float-sized blob elsewhere in the same band does. Checked at the
	mask level: on a whole frame, the ungated grayscale fallback would rescue the result
	either way, hiding a broken blob filter."""
	hsv = _water_hsv()
	hsv[100:110, 50:300, 1] = 200    # 250x10 structure
	hsv[400:410, 800:840, 1] = 200   # 40x10 float-sized patch
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	mask = _build_color_mask(bgr, [], (0, 0, hsv.shape[1], hsv.shape[0]))

	assert int((mask[100:110, 50:300] > 0).sum()) == 0
	assert int((mask[400:410, 800:840] > 0).sum()) > 0


@pytest.mark.parametrize('x_ratio', [0.28, 0.72])
def test_find_float_finds_a_float_near_the_search_bands_side_edges(tmp_path, x_ratio):
	"""Real casts land off-center; the band's horizontal range (FLOAT_SEARCH_X_RANGE) must
	stay wide enough to keep them."""
	assert find_float(_frame_with_float_at(int(2560 * x_ratio), int(1410 * 0.55), tmp_path)) is not None


@pytest.mark.parametrize('x_ratio', [0.15, 0.85])
def test_find_float_ignores_a_match_outside_the_search_band_sides(tmp_path, x_ratio):
	assert find_float(_frame_with_float_at(int(2560 * x_ratio), int(1410 * 0.55), tmp_path)) is None


def _match(score):
	return _Match(score, (0, 0), (10, 10), 'template.png')


def test_pick_match_rejects_a_score_in_the_gap_below_the_real_float_floor():
	"""Confirmed-real matches score 0.439+; a real miss scored 0.330 (a stray water-texture
	match). Both the gated and the ungated candidate at that level must read as not found,
	not get clicked and cost a whole listen() timeout."""
	assert _pick_match(_match(0.33), _match(0.33)) is None
	assert _pick_match(_match(0.44), _match(0.44)) is not None


def test_pick_match_falls_back_to_ungated_only_when_the_gated_one_is_too_weak():
	gated_weak, raw_strong = _match(0.20), _match(0.60)
	assert _pick_match(gated_weak, raw_strong) is raw_strong
	# a gated candidate that clears the threshold wins even if the ungated one scores higher
	gated_ok, raw_higher = _match(0.50), _match(0.90)
	assert _pick_match(gated_ok, raw_higher) is gated_ok
	assert _pick_match(None, None) is None


# backlit_float.png's UI regions were blacked out in the original capture. A synthetic
# WoW-style player frame (gold-bordered portrait, made-up name "Testchar", solid green
# health bar, solid blue mana bar) is painted into each UI_EXCLUDE_REGIONS box, bottom-left
# and bottom-right, so the exclusion - and the water baseline's handling of UI pixels - has
# something real to act on. The float's position is unchanged from the black-region original.
# One fixture is enough: the other blacked-out captures only need to keep finding their float.
UI_FRAMES_FIXTURE = os.path.join(FIXTURE_DIR, 'backlit_float.png')


def test_synthetic_ui_player_frames_really_are_a_distractor(monkeypatch):
	"""With UI exclusion off, the painted frames must actually win the click - otherwise the
	FLOAT_CASES rows would pass whether or not the exclusion works."""
	monkeypatch.setattr('float_detector.UI_EXCLUDE_REGIONS', ())

	x, y = find_float(UI_FRAMES_FIXTURE)

	assert y > 1000   # down in the UI band, not on the water


def test_adaptive_color_mask_baseline_ignores_ui_boxes():
	"""A player frame's solid bars are far more saturated than water. Counted into the water
	baseline they lift the 99.5th percentile to their own level, the threshold clears the
	float, and the whole gate goes blind - which real, un-blacked-out UI frames did to a
	backlit capture before the baseline learned to skip UI_EXCLUDE_REGIONS."""
	hsv = _water_hsv(saturation=30, value=150)
	hsv[300:310, 400:440, 1] = 100                   # the float's own pixels, above the water
	hsv[700:790, 100:400, 1] = 230                   # a UI frame: ~2.4% of the canvas
	ui_box = (100, 700, 400, 790)

	with_ui = _adaptive_color_mask(hsv)
	without_ui = _adaptive_color_mask(hsv, [ui_box])

	assert int((with_ui[300:310, 400:440] > 0).sum()) == 0            # gate blinded by the frame
	assert int((without_ui[300:310, 400:440] > 0).sum()) == 40 * 10   # float survives


@pytest.mark.parametrize('shape', [(1, 1), (3, 3), (60, 100), (200, 300), (300, 400)])
def test_find_float_treats_a_screenshot_too_small_for_the_templates_as_not_found(tmp_path, shape):
	"""A minimized/shrunk window gives a search band smaller than the templates (or no band at
	all). That must read as "not found", not raise and take the whole fishing loop down with it."""
	path = tmp_path / 'tiny.png'
	cv2.imwrite(str(path), np.random.default_rng(0).integers(0, 255, (*shape, 3), dtype=np.uint8))

	assert find_float(str(path)) is None


def test_find_float_says_so_when_there_are_no_templates(tmp_path, monkeypatch, capsys):
	monkeypatch.setattr('float_detector.FLOAT_TEMPLATE_GLOB', str(tmp_path / 'nothing_*.png'))

	assert find_float(os.path.join(FIXTURE_DIR, 'dim_night_float.png')) is None
	assert 'No usable float templates' in capsys.readouterr().out


def _debug_lines(capsys):
	return [line for line in capsys.readouterr().out.splitlines() if line.startswith('[float] ')]


def test_no_trace_lines_unless_debug_is_on(capsys):
	find_float(os.path.join(FIXTURE_DIR, 'dim_night_float.png'))

	assert _debug_lines(capsys) == []


def test_debug_traces_every_step_of_a_color_matched_detection(capsys, monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(tmp_path))

	find_float(os.path.join(FIXTURE_DIR, 'dim_night_float.png'))

	lines = '\n'.join(_debug_lines(capsys))
	for step in ('start:', 'ui: excluding 2 box(es)', 'gate: water baseline', 'gate: color mask after blanking', 'templates:',
				 'match: var/fishing_float_1.png', 'pick: color-gated match', 'click: base color found', 'click: using the base color centroid', 'end: click point'):
		assert step in lines, step


def test_debug_says_why_the_click_point_fell_back(capsys, monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(tmp_path))

	find_float(os.path.join(FIXTURE_DIR, 'warm_water_gate_click_noise.png'))

	lines = '\n'.join(_debug_lines(capsys))
	assert 'click: base color not found (largest blob' in lines
	assert "click: using the matched box's center" in lines
	assert os.listdir(tmp_path)   # the annotated fallback snapshot still gets saved


def test_debug_says_why_nothing_was_found(capsys, monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	path = tmp_path / 'flat.png'
	cv2.imwrite(str(path), np.full((1050, 1893, 3), 120, dtype=np.uint8))

	assert find_float(str(path)) is None

	lines = '\n'.join(_debug_lines(capsys))
	assert 'pick: nothing above threshold' in lines
	assert 'end: float not found' in lines


def test_debug_says_when_the_screenshot_has_no_search_band(capsys, monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	path = tmp_path / 'tiny.png'
	cv2.imwrite(str(path), np.zeros((1, 1, 3), dtype=np.uint8))

	assert find_float(str(path)) is None

	assert 'search band is empty' in '\n'.join(_debug_lines(capsys))


def test_debug_says_no_position_passed_the_color_gate_instead_of_a_fake_location(capsys, monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(tmp_path))
	frame = cv2.imread(os.path.join(FIXTURE_DIR, 'dim_night_float.png'))
	frame[695 - 90:695 + 90, 1243 - 90:1243 + 90] = frame[695 - 90:695 + 90, 1243 - 270:1243 - 90]   # erase the float: clone water over it
	path = tmp_path / 'no_float.png'
	cv2.imwrite(str(path), frame)

	find_float(str(path))

	lines = '\n'.join(_debug_lines(capsys))
	assert 'no position had enough float-colored pixels' in lines
	assert 'with color gate -1.0' not in lines


def test_debug_says_when_ui_exclusion_removed_a_templates_best_spot(capsys, monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)

	find_float(_frame_with_float_at(790, 1070, tmp_path))   # a perfect match sitting inside a UI box

	assert 'UI exclusion removed its best spot' in '\n'.join(_debug_lines(capsys))


def test_debug_lists_the_blobs_the_gate_dropped(capsys, monkeypatch):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	hsv = _water_hsv()
	hsv[100:110, 50:300, 1] = 200   # 250x10 structure, over FLOAT_MAX_BLOB_SIZE
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	_build_color_mask(bgr, [], (0, 0, hsv.shape[1], hsv.shape[0]))

	assert 'after dropping 1 blob(s) over 200px [(250, 10)]' in '\n'.join(_debug_lines(capsys))


@pytest.mark.parametrize('fixture', ['no_float_dark_water_1.png', 'no_float_dark_water_2.png'])
def test_find_float_finds_nothing_on_an_empty_stretch_of_flat_dark_water(fixture):
	"""Real frames with no float in the water at all (only the search band's water is kept, the
	rest blacked out). The grayscale fallback used to "find" one on nearly featureless water,
	scoring 0.697 and 0.599; a window that flat can't be a float (FLOAT_MIN_TEXTURE)."""
	assert find_float(os.path.join(FIXTURE_DIR, fixture)) is None


def test_box_texture_is_the_standard_deviation_of_the_window_anchored_at_each_pixel():
	"""texture[y, x] must describe the same box matchTemplate's result[y, x] scores."""
	gray = np.random.default_rng(0).integers(0, 255, (120, 200), dtype=np.uint8)
	tw, th = 33, 21

	texture = _box_texture(gray, (tw, th))

	for y, x in [(0, 0), (7, 11), (50, 90), (120 - th, 200 - tw)]:
		assert abs(texture[y, x] - gray[y:y + th, x:x + tw].std()) < 1e-6


def test_find_float_still_finds_the_real_float_when_its_surroundings_are_flat(tmp_path):
	"""A float pasted onto perfectly flat water has flat surroundings but the float itself is
	textured - the floor must only remove windows with no float in them."""
	template = cv2.imread('var/fishing_float_1.png')
	assert template.std() > FLOAT_MIN_TEXTURE   # the premise: the float itself is well above the floor

	assert find_float(_frame_with_float_at(1280, 800, tmp_path)) is not None


def test_debug_says_when_the_texture_floor_removed_a_templates_best_spot(capsys, monkeypatch):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)

	find_float(os.path.join(FIXTURE_DIR, 'no_float_dark_water_1.png'))

	lines = '\n'.join(_debug_lines(capsys))
	assert 'texture floor (std ' in lines and 'nearly featureless window' in lines
	assert 'end: float not found' in lines
