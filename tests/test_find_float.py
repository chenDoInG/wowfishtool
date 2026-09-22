"""Regression tests for find_float()'s template-matching logic.

Uses frozen sample screenshots (not the live var/fishing_session.png, which gets
overwritten every time the bot actually runs) with known, visually-verified float
positions, so tuning the matching/color-gating constants in float_detector.py later
can't silently break detection without a test failing.
"""
import math
import os

import cv2
import numpy as np
import pytest

from float_detector import (EVIDENCE_AGREEMENT, EVIDENCE_BASE, EVIDENCE_GATE, FALLBACK_POINT, FLOAT_MAX_BLOB_SIZE, FLOAT_MIN_TEXTURE, Candidate,
								_adaptive_color_mask, _agreeing_candidate, _base_click_point, _base_color_mask, _box_texture, _corroborating_evidence, _build_color_mask, _decide,
								_band_base_range, _drop_large_blobs, _click_point, _find_base, _load_templates, _template_base_anchor, _normalization_scale, _search_bounds,
								find_float, find_float_detailed, save_notfound_snapshot)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
TOLERANCE_PX = 20

# (fixture, expected (x, y), tolerance, spots that must NOT be returned, what it guards)
FLOAT_CASES = [
	('ship_false_positive.png', (1366, 851), TOLERANCE_PX, [(700, 440)],
	 'a distant ship near the horizon has saturated warm+cool colors and out-scored the '
	 'float; fixed by restricting the search band to where a nearby cast lands (the '
	 'same open-sea capture also covers the old shoreline color-boundary false positive)'),
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
	('sunset_purple_water.png', (1077, 434), 10, [],
	 'pink/purple sunset water; no template matched until fishing_float_6.png was added'),
	('saturation_ceiling_clip.png', (914, 615), TOLERANCE_PX, [],
	 'water so saturated the gate threshold exceeds 255 - the gate passes nothing, so '
	 'the ungated grayscale match must carry it (score 0.659)'),
	('small_window_float.png', (418, 316), 12, [],
	 'WoW window shrunk to 919x524: the float is ~25 px against templates of 50-140 px and was not found '
	 '5 times in 14; matching on a copy resized to the reference width finds it (see FLOAT_REFERENCE_WIDTH). '
	 'Tolerance is in this small window\'s own pixels'),
	('small_float_on_pale_water.png', (1218, 853), 20, [(1688, 598)],
	 'dusk water with pale wave crests and a small, dark float (cast far out): by shape alone the best window was '
	 'a wave crest at (1688, 598), 7 times in 81 casts - the float\'s base is the only base-colored spot in the band'),
	('daytime_waves_float.png', (1155, 932), 30, [(887, 607)],
	 'daytime sea whose water is as saturated as the cliffs (gate blind, threshold 199) and whose hue is the base range: '
	 'color evidence pointed at a cliff (0.41) while the real float scored 0.55-0.59 on three templates that agree on it'),
	('night_water_float_at_band_top.png', (1167, 569), 10, [],
	 'night water, a far cast landing at 38% of the height: the old band edge cut off the top of the feather (0.37 here, '
	 'not found in another cast at 0.24-0.33). The base is also paler than the daylight range assumed (S 44-113)'),
	('warm_dusk_afterimage.png', (1299, 880), 30, [(1222, 798)],
	 'warm dusk water sharing the base hue (base color covers 86-91% of the band): the previous cast\'s float lingers as a '
	 'featherless base 100 px from the new float and out-scored it (13 of 140 casts); the feather is outside the base color '
	 'ranges while the water and the afterimage are inside, so only gate pixels outside them count'),
	('dim_night_base_color.png', (1355.61, 695.46), TOLERANCE_PX, [],
	 'dark night base: hue inside the daylight range but S/V far below its floors, so '
	 'every cast fell back to the geometric center; needs the dim base color range'),
	('warm_dusk_gate_miss.png', (1438, 800), TOLERANCE_PX, [],
	 'orange dusk: the gate rejects the float\'s true position while unrelated scattered '
	 'pixels keep it "active" - grayscale alone scored 0.53-0.7 at the float, gated '
	 'candidates only 0.24-0.34'),
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
	# find_float() has no color evidence to work with (see _decide()) and the frame
	# must read as not found, not as a float.
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

	mask = _build_color_mask(hsv, [])

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


def _candidate(score, evidence):
	return Candidate(score, (0, 0), (10, 10), 'template.png', evidence)


def test_decide_rejects_a_score_in_the_gap_below_the_real_float_floor():
	"""Confirmed-real matches score 0.375+ (0.30 for a float still fading in); a real miss scored 0.330 (a stray
	water-texture match). Both kinds of evidence at that level must read as not found, not get clicked and
	cost a whole listen() timeout."""
	assert _decide({EVIDENCE_GATE: _candidate(0.33, EVIDENCE_GATE), EVIDENCE_BASE: _candidate(0.33, EVIDENCE_BASE)}) is None
	assert _decide({EVIDENCE_GATE: _candidate(0.44, EVIDENCE_GATE), EVIDENCE_BASE: _candidate(0.44, EVIDENCE_BASE)}) is not None


def test_decide_falls_back_to_base_colored_windows_only_when_the_gate_is_too_weak():
	gate_weak, base_strong = _candidate(0.20, EVIDENCE_GATE), _candidate(0.60, EVIDENCE_BASE)
	assert _decide({EVIDENCE_GATE: gate_weak, EVIDENCE_BASE: base_strong}) is base_strong
	# a gated candidate that clears the threshold wins even if the base-colored one scores higher
	gate_ok, base_higher = _candidate(0.50, EVIDENCE_GATE), _candidate(0.90, EVIDENCE_BASE)
	assert _decide({EVIDENCE_GATE: gate_ok, EVIDENCE_BASE: base_higher}) is gate_ok
	assert _decide({EVIDENCE_GATE: None, EVIDENCE_BASE: None}) is None
	assert _decide({}) is None


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

	assert find_float(os.path.join(FIXTURE_DIR, 'dim_night_base_color.png')) is None
	assert 'No usable float templates' in capsys.readouterr().out


def _debug_lines(capsys):
	return [line for line in capsys.readouterr().out.splitlines() if line.startswith('[float] ')]


def test_no_trace_lines_unless_debug_is_on(capsys):
	find_float(os.path.join(FIXTURE_DIR, 'dim_night_base_color.png'))

	assert _debug_lines(capsys) == []


def test_debug_traces_every_step_of_a_color_matched_detection(capsys, monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(tmp_path))

	find_float(os.path.join(FIXTURE_DIR, 'dim_night_base_color.png'))

	lines = '\n'.join(_debug_lines(capsys))
	for step in ('start:', 'ui: excluding 2 box(es)', 'gate: water baseline', 'gate: color mask after blanking', 'templates:',
				 'match: var/fishing_float_1.png', 'pick: color-gated match', 'click: base color found', 'click: using the base color centroid', 'end: click point'):
		assert step in lines, step


def test_debug_says_why_the_click_point_fell_back(capsys, monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(tmp_path))

	find_float(os.path.join(FIXTURE_DIR, 'dusk_saturated_water.png'))

	lines = '\n'.join(_debug_lines(capsys))
	assert 'click: base color not found (largest blob' in lines
	assert "click: using the default position" in lines   # this fixture's template is mostly water, so it has no base anchor of its own
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

	find_float(os.path.join(FIXTURE_DIR, 'no_float_dark_water.png'))   # nothing float-colored anywhere in this water

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

	_build_color_mask(hsv, [])

	assert 'after dropping 1 blob(s) over 200px [(250, 10)]' in '\n'.join(_debug_lines(capsys))


@pytest.mark.parametrize('fixture', ['no_float_dark_water.png', 'no_float_dusk_water.png'])
def test_find_float_finds_nothing_on_an_empty_stretch_of_dark_water(fixture):
	"""Real frames with no float in the water at all (only the search band's water is kept, the
	rest blacked out). The grayscale fallback used to "find" one on nearly featureless water
	(dark_water: 0.599) and on finely rippled dusk water (dusk_water, windows with std
	3.9-4.1, next to floating creature-name labels); windows that flat can't be a float
	(FLOAT_MIN_TEXTURE)."""
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

	find_float(os.path.join(FIXTURE_DIR, 'no_float_dark_water.png'))

	lines = '\n'.join(_debug_lines(capsys))
	assert 'texture floor (std ' in lines and 'nearly featureless window' in lines
	assert 'end: float not found' in lines


def _resized_fixture(name, factor, tmp_path):
	img = cv2.imread(os.path.join(FIXTURE_DIR, name))
	img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)
	path = tmp_path / 'resized.png'
	cv2.imwrite(str(path), img)
	return str(path)


@pytest.mark.parametrize('factor', [0.4, 0.75, 1.8])
def test_find_float_reports_the_position_in_the_screenshots_own_pixels_at_any_window_size(tmp_path, factor):
	"""The same capture shrunk or enlarged to a different window size must still find the float,
	and the answer must be in that screenshot's pixels (the bot moves the mouse by it)."""
	place = find_float(_resized_fixture('dusk_saturated_water.png', factor, tmp_path))

	assert place is not None
	assert abs(place[0] - 1245 * factor) <= 12 * max(factor, 1)
	assert abs(place[1] - 695 * factor) <= 12 * max(factor, 1)


@pytest.mark.parametrize('width, scale', [(2556, 1), (2560, 1), (3000, 1), (2100, 1), (2000, 2556 / 2000), (1893, 2556 / 1893), (919, 2556 / 919), (3840, 2556 / 3840)])
def test_normalization_scale(width, scale):
	"""Wide-but-normal windows are left alone; small and very large ones are matched on a resized copy."""
	assert _normalization_scale(width) == pytest.approx(scale)


def test_normalization_scale_never_blows_a_tiny_screenshot_up_into_a_huge_one():
	assert _normalization_scale(300) == 1


def test_debug_says_when_it_matches_on_a_resized_copy(capsys, monkeypatch):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)

	find_float(os.path.join(FIXTURE_DIR, 'small_window_float.png'))

	lines = '\n'.join(_debug_lines(capsys))
	assert 'scale: 919px wide' in lines and 'resized by 2.78' in lines


def _hsv_patch(h, s, v, size=12, canvas=(60, 80), background=(115, 60, 40)):
	"""A canvas of `background` HSV with a size x size patch of (h, s, v) in the middle."""
	hsv = np.zeros((*canvas, 3), dtype=np.uint8)
	hsv[:, :] = background
	top, left = (canvas[0] - size) // 2, (canvas[1] - size) // 2
	hsv[top:top + size, left:left + size] = (h, s, v)
	return hsv


def test_click_point_finds_a_dark_desaturated_green_base():
	"""The base measured on 68 real floats from a dusk-to-night session: H 52-94, S 27-90, V 19-57.
	None of the older ranges reached it, so every cast fell back to the box-center click."""
	point = _base_click_point(_hsv_patch(66, 46, 38))

	assert point is not None
	assert abs(point[0] - 40) <= 2 and abs(point[1] - 30) <= 2


def test_click_point_ignores_pale_blue_water_that_the_fallback_used_to_pick():
	"""Regions where the fallback picked sky-reflection water instead of a float have hue 100-120."""
	assert _base_click_point(_hsv_patch(110, 50, 200)) is None


def test_a_float_with_no_base_color_anywhere_is_not_found_on_shape_alone(capsys, monkeypatch, tmp_path):
	"""Shape alone used to be enough once the color gate was blind, and it picked pale wave crests. The same
	frame with all color removed keeps the float's shape but has no base-colored window anywhere."""
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	gray = cv2.cvtColor(cv2.imread(os.path.join(FIXTURE_DIR, 'small_float_on_pale_water.png')), cv2.COLOR_BGR2GRAY)
	path = tmp_path / 'colorless.png'
	cv2.imwrite(str(path), cv2.merge([gray, gray, gray]))

	assert find_float(str(path)) is None

	lines = '\n'.join(_debug_lines(capsys))
	assert 'base color: 0 px' in lines
	assert 'no position had enough base-colored pixels' in lines
	assert 'end: float not found' in lines


def test_debug_reports_the_base_color_pixels_and_the_windows_that_have_it(capsys, monkeypatch):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)

	find_float(os.path.join(FIXTURE_DIR, 'small_float_on_pale_water.png'))

	lines = '\n'.join(_debug_lines(capsys))
	assert 'using the shape match among base-colored windows' in lines
	assert 'with base color ' in lines


def test_find_float_detailed_says_which_evidence_decided(tmp_path):
	"""The caller can tell a float backed by relative-saturation evidence from one that only the base color vouches for."""
	gated = find_float_detailed(os.path.join(FIXTURE_DIR, 'backlit_float.png'))
	base_colored = find_float_detailed(os.path.join(FIXTURE_DIR, 'small_float_on_pale_water.png'))

	assert gated.confidence == EVIDENCE_GATE
	assert base_colored.confidence == EVIDENCE_BASE and base_colored.on_base_color
	assert base_colored.template.endswith('.png') and base_colored.score > 0.35


def test_find_float_is_the_point_of_find_float_detailed():
	path = os.path.join(FIXTURE_DIR, 'small_float_on_pale_water.png')

	assert find_float(path) == find_float_detailed(path).point
	assert find_float_detailed(os.path.join(FIXTURE_DIR, 'no_float_dark_water.png')) is None


def test_an_absurdly_thin_screenshot_is_not_found_instead_of_raising(tmp_path):
	"""100000x10 would resize to a zero-height image; cv2.resize raised and took the fishing loop down with it."""
	path = tmp_path / 'thin.png'
	cv2.imwrite(str(path), np.zeros((10, 100000, 3), dtype=np.uint8))

	assert find_float(str(path)) is None


def _shape_best(score, x, y, w=65, h=65, name='t.png'):
	return Candidate(score, (x, y), (w, h), name, EVIDENCE_AGREEMENT)


def test_agreeing_candidate_needs_several_templates_at_the_same_place():
	three = [_shape_best(0.55, 100, 100, 133, 97), _shape_best(0.59, 130, 110), _shape_best(0.40, 120, 105, 73, 58)]
	assert _agreeing_candidate(three).score == 0.59   # the best member of the group
	assert _agreeing_candidate(three[:2]) is None     # two templates are not enough


def test_agreement_needs_a_share_of_the_templates_not_just_three_when_there_are_many():
	"""Adding templates must raise the bar: 3 of 10 agreeing is easy by chance, 4 of 10 (40%) is the rule."""
	def group_of(n, size):
		return [_shape_best(0.55, 100 + i * 5, 100, name='t%d.png' % i) for i in range(size)] \
			+ [_shape_best(0.55, 100 + 200 * (i + 1), 100 + 150 * i, name='far%d.png' % i) for i in range(n - size)]
	assert _agreeing_candidate(group_of(7, 3)) is not None    # 3 of 7: ceil(0.4 * 7) = 3, unchanged from the fixed rule
	assert _agreeing_candidate(group_of(10, 3)) is None       # 3 of 10 is under 40%
	assert _agreeing_candidate(group_of(10, 4)) is not None
	assert _agreeing_candidate(group_of(5, 3)) is not None    # never fewer than 3, never more than 40% of what there is


def test_agreeing_candidate_ignores_templates_that_disagree_or_score_too_low():
	spread = [_shape_best(0.55, 100, 100), _shape_best(0.59, 400, 100), _shape_best(0.50, 100, 400)]
	assert _agreeing_candidate(spread) is None
	weak = [_shape_best(0.30, 100, 100), _shape_best(0.30, 110, 100), _shape_best(0.30, 120, 100)]
	assert _agreeing_candidate(weak) is None


def test_decide_ranks_the_gate_then_specific_base_color_then_template_agreement():
	"""The gate is hue-agnostic; base color that is specific to a small blob has a physical signal behind it; template
	agreement is shape only, and on rippled water 3 templates occasionally agree on a wave crest."""
	gate, base, agree = _candidate(0.50, EVIDENCE_GATE), _candidate(0.45, EVIDENCE_BASE), _candidate(0.90, EVIDENCE_AGREEMENT)
	assert _decide({EVIDENCE_GATE: gate, EVIDENCE_BASE: base, EVIDENCE_AGREEMENT: agree}) is gate
	assert _decide({EVIDENCE_GATE: None, EVIDENCE_BASE: base, EVIDENCE_AGREEMENT: agree}) is base
	assert _decide({EVIDENCE_GATE: None, EVIDENCE_BASE: None, EVIDENCE_AGREEMENT: agree}) is agree


def test_base_color_steps_aside_when_the_whole_scene_shares_its_hue():
	"""A daytime sea and a warm dusk fall 84-88% inside the base ranges; where the color is informative it is 0-5%."""
	flooded = _hsv_patch(66, 46, 38, size=150, canvas=(200, 200))               # 56% of the band inside the ranges, blob under the size cap
	specific = _hsv_patch(66, 46, 38, size=14, canvas=(200, 200))               # a bobber-sized blob on other water

	assert int(np.count_nonzero(_base_color_mask(flooded, []))) == 0
	assert int(np.count_nonzero(_base_color_mask(specific, []))) > 0


def test_a_scene_that_shares_the_base_hue_falls_back_to_template_agreement_even_with_noise():
	"""warm_dusk_gate_miss plus noise blinds the gate, and its water is inside the base ranges (84%). Choosing among
	base-colored windows one by one picked a spot 548 px away; the kind of evidence has to step aside as a whole."""
	frame = cv2.imread(os.path.join(FIXTURE_DIR, 'warm_dusk_gate_miss.png')).astype(int)
	noisy = np.clip(frame + np.random.default_rng(0).normal(0, 6, frame.shape), 0, 255).astype(np.uint8)
	path = os.path.join(FIXTURE_DIR, '..', 'noisy_warm_dusk.tmp.png')
	cv2.imwrite(path, noisy)
	try:
		detection = find_float_detailed(path)
	finally:
		os.remove(path)

	assert detection is not None and detection.confidence == EVIDENCE_AGREEMENT
	assert abs(detection.point[0] - 1438) <= 45 and abs(detection.point[1] - 800) <= 45


def test_a_base_blob_that_fills_the_region_is_scenery_not_a_base():
	"""The daytime sea and its cliffs share the base's hue: 'the base' was 76-90% of the click region. A real base is 1-4%."""
	assert _base_click_point(_hsv_patch(66, 46, 38, size=55, canvas=(60, 60))) is None
	assert _base_click_point(_hsv_patch(66, 46, 38, size=12, canvas=(60, 60))) is not None


def test_debug_traces_the_agreement_between_templates(capsys, monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(tmp_path))

	detection = find_float_detailed(os.path.join(FIXTURE_DIR, 'daytime_waves_float.png'))

	lines = '\n'.join(_debug_lines(capsys))
	assert detection.confidence == EVIDENCE_AGREEMENT
	assert 'agreement: 3 templates agree near' in lines
	assert 'using the window the templates agree on' in lines
	assert 'click: base color found' in lines   # the union of ranges floods here; the range that fits the lighting does not


def _placed(score, x, y, evidence):
	return Candidate(score, (x, y), (65, 65), 'template.png', evidence)


def test_corroborating_evidence_counts_other_kinds_that_land_on_the_same_place():
	chosen = _placed(0.50, 100, 100, EVIDENCE_GATE)
	candidates = {EVIDENCE_GATE: chosen, EVIDENCE_BASE: _placed(0.45, 120, 110, EVIDENCE_BASE),
				  EVIDENCE_AGREEMENT: _placed(0.60, 400, 100, EVIDENCE_AGREEMENT)}

	assert _corroborating_evidence(chosen, candidates) == (EVIDENCE_BASE,)   # agreement points elsewhere, so it does not count


def test_corroborating_evidence_ignores_kinds_below_the_threshold_or_missing():
	chosen = _placed(0.50, 100, 100, EVIDENCE_BASE)
	candidates = {EVIDENCE_GATE: _placed(0.20, 100, 100, EVIDENCE_GATE), EVIDENCE_BASE: chosen, EVIDENCE_AGREEMENT: None}

	assert _corroborating_evidence(chosen, candidates) == ()


def test_real_floats_are_usually_corroborated_and_the_hard_scenes_are_flagged():
	"""9 of 11 measured floats had a second independent kind of evidence at the same place; the two that did not are
	the hardest scenes (dusk water with a faint float, a daytime sea whose water shares the base's hue)."""
	ship = find_float_detailed(os.path.join(FIXTURE_DIR, 'ship_false_positive.png'))
	pale = find_float_detailed(os.path.join(FIXTURE_DIR, 'small_float_on_pale_water.png'))

	assert ship.corroborated and set(ship.corroborated_by) == {EVIDENCE_BASE, EVIDENCE_AGREEMENT}
	assert not pale.corroborated and pale.corroborated_by == ()


def _snapshot_names(directory):
	return sorted(os.path.splitext(name)[0].split('_')[0] for name in os.listdir(directory))


def test_a_low_confidence_pick_saves_a_snapshot_and_says_so(capsys, monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(tmp_path))

	find_float(os.path.join(FIXTURE_DIR, 'small_float_on_pale_water.png'))

	assert _snapshot_names(tmp_path) == ['lowconf']
	out = capsys.readouterr().out
	assert 'NOT corroborated by any other evidence' in out and 'Low confidence' in out


def test_an_uncorroborated_pick_scoring_at_or_above_the_lowconf_score_saves_nothing(monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(tmp_path))
	detection = find_float_detailed(os.path.join(FIXTURE_DIR, 'small_float_on_pale_water.png'))
	assert not detection.corroborated
	for name in os.listdir(tmp_path):
		os.remove(os.path.join(tmp_path, name))

	monkeypatch.setattr('float_detector.FLOAT_LOWCONF_SCORE', detection.score)   # the pick now scores exactly at the line
	find_float(os.path.join(FIXTURE_DIR, 'small_float_on_pale_water.png'))

	assert os.listdir(tmp_path) == []


def test_a_corroborated_pick_on_the_base_color_saves_nothing(monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(tmp_path))

	find_float(os.path.join(FIXTURE_DIR, 'ship_false_positive.png'))

	assert os.listdir(tmp_path) == []


def test_a_fallback_click_saves_one_fallback_snapshot_even_when_it_is_also_uncorroborated(monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(tmp_path))

	monkeypatch.setattr('float_detector._base_click_point', lambda hsv_region: None)   # force the box-center click
	find_float(os.path.join(FIXTURE_DIR, 'daytime_waves_float.png'))   # and nothing corroborates the pick

	assert _snapshot_names(tmp_path) == ['fallback']


def test_snapshots_saved_within_one_second_do_not_overwrite_each_other(monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(tmp_path))

	for _ in range(3):
		find_float(os.path.join(FIXTURE_DIR, 'small_float_on_pale_water.png'))

	assert len(os.listdir(tmp_path)) == 3


def test_an_unwritable_snapshot_folder_loses_the_picture_not_the_detection(capsys, monkeypatch, tmp_path):
	"""A debugging aid must never take detection down with it: the folder cannot be created under a plain file."""
	blocker = tmp_path / 'blocker'
	blocker.write_text('x')
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(blocker / 'debug'))

	place = find_float(os.path.join(FIXTURE_DIR, 'small_float_on_pale_water.png'))   # a low-confidence pick, so it tries to save

	assert place is not None
	assert 'Could not save the debug snapshot' in capsys.readouterr().out


def test_the_click_finds_the_base_where_one_range_floods_with_the_water():
	"""Daytime waves: one base range matches all the water, so the union is a single oversized blob. Each range is
	judged on its own, so the base is still found and the click lands on it instead of the box center."""
	detection = find_float_detailed(os.path.join(FIXTURE_DIR, 'daytime_waves_float.png'))

	assert detection.on_base_color
	assert math.hypot(detection.point[0] - 1155, detection.point[1] - 932) < 10


def test_a_float_at_the_top_of_the_band_is_matched_whole_not_half_cut_off():
	"""The band starts at 36% of the height: a far cast lands just below 38%, where the old edge cut the feather off."""
	detection = find_float_detailed(os.path.join(FIXTURE_DIR, 'night_water_float_at_band_top.png'))

	assert detection.score > 0.6   # 0.77 with the whole float in the band, 0.37 with the top cut off


def test_a_pale_night_base_is_still_found_by_color():
	"""A night-lit base has a saturation of 44-113; the daylight range must reach that low or the click falls back."""
	detection = find_float_detailed(os.path.join(FIXTURE_DIR, 'night_water_float_at_band_top.png'))

	assert detection.on_base_color


def _click_scene(blob_x):
	"""A 200x200 water image with one cream blob (a bobber base's color) at blob_x, and a template box at (100, 60), 40x40."""
	bgr = np.full((200, 200, 3), (90, 80, 70), dtype=np.uint8)
	base = cv2.cvtColor(np.full((1, 1, 3), (20, 100, 180), dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
	bgr[80:100, blob_x:blob_x + 12] = base
	candidate = Candidate(0.6, (100, 60), (40, 40), 'var/fishing_float_5.png', EVIDENCE_GATE)
	return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), candidate


def test_the_click_does_not_take_a_base_colored_spot_left_of_the_box():
	"""The feather points left, so a click on the water beside the box misses; a spot to the right can be the base."""
	bgr, hsv, candidate = _click_scene(blob_x=82)   # x 82-93, inside the 0.3-of-the-width padding that used to reach left of the box

	_, on_base = _click_point(bgr, hsv, candidate, (0, 0, 200, 200))

	assert not on_base


def test_the_click_does_take_a_base_that_hangs_right_of_the_box():
	bgr, hsv, candidate = _click_scene(blob_x=145)   # right of the box (x 100-140) but inside the padding

	(x, _), on_base = _click_point(bgr, hsv, candidate, (0, 0, 200, 200))

	assert on_base and 145 <= x <= 157


def test_a_template_with_a_solid_base_knows_where_it_sits_and_a_watery_one_does_not():
	x, y = _template_base_anchor('var/fishing_float_1.png')

	assert (0.65, 0.6) < (x, y) < (0.85, 0.8)   # measured 0.74, 0.69 here and 0.75-0.76, 0.63-0.65 on four live frames
	assert _template_base_anchor('var/fishing_float_5.png') == FALLBACK_POINT


def test_a_click_that_cannot_find_the_base_uses_the_templates_own_base_position(monkeypatch):
	monkeypatch.setattr('float_detector._base_click_point', lambda hsv_region: None)

	point = find_float(os.path.join(FIXTURE_DIR, 'night_water_float_at_band_top.png'))

	assert math.hypot(point[0] - 1167, point[1] - 569) < 5   # 14.6 px off with the fixed (0.5, 0.65) of the box


def test_where_the_base_color_floods_the_scene_the_gate_keeps_only_pixels_outside_it(capsys, monkeypatch, tmp_path):
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOTS', True)
	monkeypatch.setattr('float_detector.DEBUG_SNAPSHOT_DIR', str(tmp_path))

	find_float(os.path.join(FIXTURE_DIR, 'warm_dusk_afterimage.png'))

	assert 'keeping only gate pixels outside its ranges' in '\n'.join(_debug_lines(capsys))


def test_the_gate_is_left_alone_where_the_base_color_is_a_minority():
	"""On a scene the base ranges do not flood (a cool sea) the gate mask must not be narrowed."""
	hsv = np.zeros((60, 80, 3), dtype=np.uint8)
	hsv[:, :] = (105, 60, 90)        # water outside every base range
	hsv[20:24, 30:35] = (100, 200, 200)   # a saturated float-colored patch, under the baseline's 0.5% tail
	_, share = _band_base_range(hsv, [])

	assert share == 0 and np.count_nonzero(_build_color_mask(hsv, [])) > 0


def test_the_canal_wall_reaching_into_the_top_of_the_band_is_not_taken_for_the_float():
	"""A night canal: the dark float scores under the threshold while the stone wall in the band's top-right corner scored
	0.50 with the first template, so 29 of 196 casts clicked the same spot on the wall (with the band starting at 34%)."""
	point = find_float(os.path.join(FIXTURE_DIR, 'canal_wall_at_band_top.png'))

	assert point is None or not (point[0] >= 1600 and point[1] < 620)


def test_the_notfound_snapshot_draws_the_search_band_and_leaves_the_rest_of_the_frame_alone(tmp_path):
	source = os.path.join(FIXTURE_DIR, 'dim_night_base_color.png')
	dest = str(tmp_path / 'notfound.png')

	assert save_notfound_snapshot(source, dest)

	original, drawn = cv2.imread(source), cv2.imread(dest)
	assert drawn.shape == original.shape
	h, w = original.shape[:2]
	x0, y0, x1, y1 = _search_bounds(w, h)
	middle = (x0 + x1) // 2
	assert tuple(drawn[y0, middle]) == (0, 255, 255) and tuple(drawn[y1, middle]) == (0, 255, 255)   # the band's top and bottom edges
	assert np.array_equal(drawn[:y0 - 40], original[:y0 - 40])   # above the band (and its label) nothing is drawn
	assert np.array_equal(drawn[y1 + 5:], original[y1 + 5:])   # nor below it


def test_the_notfound_snapshot_reports_an_unreadable_screenshot_instead_of_raising(tmp_path):
	assert save_notfound_snapshot(str(tmp_path / 'missing.png'), str(tmp_path / 'out.png')) is False


# ---- latent risks flagged by review, pinned as tests rather than fixed (see the review for the tradeoffs)

def test_find_base_picks_the_first_qualifying_range_not_the_biggest_blob():
	"""_find_base tries each FLOAT_BASE_COLOR_RANGES range on its own and returns on the first one that clears the
	blob-size checks - it does not compare blob sizes across ranges. Here a small patch qualifies under range 0
	(daylight warm tan) and a far bigger, more confident patch qualifies under range 2 (daylight hue dimmed by
	night lighting); the small early one still wins. Pinning this so the order-dependence is visible rather than
	silent - if _find_base is changed to prefer the biggest qualifying blob instead, update this assertion."""
	canvas = (100, 100)
	hsv = np.zeros((*canvas, 3), dtype=np.uint8)
	hsv[:, :] = (115, 60, 40)   # background outside all four ranges
	hsv[10:16, 10:16] = (20, 150, 180)     # 6x6 = 36px, range 0 only
	hsv[50:70, 50:70] = (20, 30, 60)       # 20x20 = 400px, range 2 only - much bigger, but checked later

	point, area = _find_base(hsv)

	assert area == 36   # range 0's small blob, not range 2's 400px one
	assert abs(point[0] - 12.5) <= 2 and abs(point[1] - 12.5) <= 2


def test_the_agreement_share_was_calibrated_on_the_current_template_set():
	"""FLOAT_MIN_AGREEING_SHARE (0.4) and the 55%/31%/14% recall-at-N-agreeing figures in its comment were measured
	against the 7 templates in var/ at the time of writing; `needed` scales with however many templates are loaded
	(see _agreeing_candidate), so adding templates does not break the mechanism, but it can leave 0.4 itself
	uncalibrated for a template set that looks very different from the measured one. This is a tripwire, not a
	correctness check: when it fails, re-read the README's '补一个新的鱼漂模板' section and consider re-measuring
	recall at a few agreeing-template counts before trusting template-agreement on the new set."""
	count = len(_load_templates())

	assert count == 7, str(count) + ' templates found, not the 7 FLOAT_MIN_AGREEING_SHARE was calibrated on'
