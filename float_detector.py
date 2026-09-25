"""Find the fishing float in a screenshot of the WoW window.

Pipeline, in the order find_float_detailed() runs it:

1. normalize   Resize to a reference width when the window is much smaller or larger than the ones the
               templates were cropped from.
2. evidence    Maps of "float-colored" pixels over the search band: much more saturated than this scene's
               water (any hue), and inside the fixed HSV ranges of the bobber's base.
3. candidates  Grayscale template correlation. Windows on a UI element or without texture are excluded; per
               kind of evidence, each template contributes its best supported window. A third kind is
               several templates agreeing on one place.
4. decide      The strongest kind of evidence whose best candidate clears FLOAT_MATCH_THRESHOLD wins.
               Shape alone never decides: on water it scores 0.4-0.7 almost anywhere.
5. click point The base's color centroid near the match, else the template's own base position in the box.

Coordinates are in the original screenshot's pixels. Comments say what a constant is for and what it was
measured on; the history is in the commit messages and README.
"""
import glob
import json
import math
import os
import time
from typing import Dict, NamedTuple, Optional

import cv2
import numpy as np

# ------------------------------------------------------------------------------ configuration

FLOAT_TEMPLATE_GLOB = 'var/fishing_float_*.png'

# Lowest correlation that counts as a float. Real floats scored 0.30 (fading in) to 0.9; the real miss that must
# stay a miss scored 0.33. A weak match becomes "not found" (the caller retries) rather than a wasted click.
FLOAT_MATCH_THRESHOLD = 0.35

# An uncorroborated match scoring below this is worth a 'lowconf' debug snapshot; at or above it a lone signal on the
# base color is routine on a dark night canal (real floats there sat at 0.57-0.69 without a second kind of evidence),
# while the faint float on pale water (0.51) is the kind of pick worth a look.
FLOAT_LOWCONF_SCORE = 0.55

# Search band, as fractions of the window. Real detections fell between 39% and 77% of the height, below the
# horizon and above the character; padded, because a float cropped out of the band cannot be recovered later.
# The top was 0.38 until a far cast landed at 38.6%, its upper half cut off (0.24-0.33, not found); 0.34 found it, but at
# night the stone wall along a canal reaches into a band that high and out-scored the dark float (0.50 against under 0.35):
# 29 of 196 casts in one session clicked the same spot on the wall. 0.36 keeps the cut-off float (its top sits at 37.9%)
# and drops the wall, while 0.30 already lets a warm dusk's scenery win (warm_dusk_gate_miss).
FLOAT_SEARCH_X_RANGE = (0.25, 0.75)
FLOAT_SEARCH_Y_RANGE = (0.36, 0.80)

# Screen elements inside the band that are never the float: the player frame (bottom-left) and the copy WoW's
# default modern layout adds (bottom-right). Fractions of the window, measured on a 2560x1410 capture, padded.
# Only the default for a UI that has never been calibrated - see UI_REGIONS_PATH below.
UI_EXCLUDE_REGIONS = (
	((0.26, 0.36), (0.72, 0.80)),
	((0.62, 0.76), (0.71, 0.83)),
)
# ui_calibrate.py writes real, auto-detected regions here (window fractions, same shape as UI_EXCLUDE_REGIONS
# above) for a UI layout that does not match the hardcoded default - moved frames, a non-default Edit Mode
# layout, or an addon that reskins the unit frame entirely. Not committed (var/ is gitignored, per-user data).
UI_REGIONS_PATH = 'var/ui_regions.json'


def _ui_exclude_regions():
	"""UI_EXCLUDE_REGIONS, or the calibrated regions from UI_REGIONS_PATH if that file exists. A missing or
	unreadable file falls back to the hardcoded default rather than taking detection down - calibration is
	optional, not required to run at all."""
	if not os.path.exists(UI_REGIONS_PATH):
		return UI_EXCLUDE_REGIONS
	try:
		with open(UI_REGIONS_PATH) as f:
			return json.load(f)
	except (OSError, ValueError) as error:
		print('Could not read ' + UI_REGIONS_PATH + ', using the default UI_EXCLUDE_REGIONS: ' + str(error))
		return UI_EXCLUDE_REGIONS

# Evidence 1, relative saturation: a pixel counts when it is more saturated than the scene's water by a margin.
# The baseline is the 99.5th percentile (a saturated sea plateaus up to its 99th) and leaves the UI frames out
# (~2% of the band, they would drag it up and blind the gate).
# Where the base color floods the band (FLOAT_MAX_BASE_SCENE_SHARE) only gate pixels outside its ranges count: see _build_color_mask.
FLOAT_SATURATION_BASELINE_PERCENTILE = 99.5
FLOAT_SATURATION_MARGIN = 20
FLOAT_MIN_VALUE = 30         # near-black pixels have unstable saturation; the dimmest real float pixel was 39
FLOAT_MIN_COLOR_PIXELS = 15  # evidence pixels a window needs to be supported by that evidence
FLOAT_MAX_BLOB_SIZE = 200    # a connected evidence blob wider or taller than this is scenery (dock, hull)

# Evidence 2, the base's own color: one narrow range per lighting (a wide one bleeds into water).
# The dusk-to-night green base was measured on 68 real floats (H 52-94, S 27-90, V 19-57).
FLOAT_BASE_COLOR_RANGES = (
	((10, 45, 100), (35, 255, 255)),   # daylight warm tan/yellow; a night-lit base is paler (S 44-113)
	((40, 30, 120), (75, 90, 255)),    # dusk/night-tinted, desaturated
	((10, 15, 25), (35, 140, 110)),    # the daylight hue dimmed by night lighting
	((35, 25, 25), (95, 120, 200)),    # dark, desaturated green
)
FLOAT_MIN_BASE_BLOB_AREA = 30      # the click point needs one solid base blob this big, not scattered flecks
FLOAT_MAX_BASE_BLOB_SHARE = 0.5    # ...and it must not fill the click region: a real base is 1-9% of it, scenery 77-95%
# Evidence 2's candidate window (see _base_color_mask) additionally requires a round-ish blob, not a fragment of
# scenery clipped by the same color range (a sunset's reflection on the water shares the base's hue at dusk) - the
# click point search (_find_base) does not use these, see its docstring for why. Measured on the 2026-09-23
# Stormwind Harbor session: two confirmed real bases filled 0.50-0.57 of their box (aspect, long side over short,
# 1.45-1.64); the reflection fragment behind a real miss that day filled 0.43 of a 79x11 box (aspect 7.18 - caught
# by the aspect check below); a diagonal ripple fragment clipped to a near-square box passed that but filled only
# 0.05-0.17 of it (caught by the extent check - a rotated sliver looks square end-to-end but is mostly empty).
FLOAT_MIN_BASE_BLOB_EXTENT = 0.35  # area / bounding-box area
FLOAT_MAX_BASE_BLOB_ASPECT = 2.5   # long side / short side of the bounding box
# Base color is ignored as evidence when more than this share of the band is inside its ranges: the water or terrain
# shares the hue and says nothing about where the float is. Measured on 14 scenes: 0-4.8% where informative, 21.7-88.2% where not.
FLOAT_MAX_BASE_SCENE_SHARE = 0.15

# Windows with a grayscale standard deviation below this are not floats: featureless water still correlates at 0.6-0.7.
# Empty water measured 0.2-4.1, real floats 6.0+. A float still fading in after the cast (3.0-4.7) is excluded until
# it has fully appeared, so the caller's retry finds it. The floor sits in a gap that is thin on both sides.
FLOAT_MIN_TEXTURE = 6

# Evidence 3, agreement: a real float is matched by several differently sized templates at the same place, a rippled
# surface matches each somewhere else. Needs no color, so it covers a daytime sea whose water shares the base's hue
# (real float 0.55-0.59 with three templates agreeing, while the color "evidence" pointed at a cliff). On all 11 real-float
# images the largest agreeing group sits on the float; on empty water it reached 3 in 1 of 20 synthetic frames, so it
# ranks below base color.
FLOAT_MIN_AGREEING_TEMPLATES = 3
# ...and at least this share of the templates in play, so adding templates raises the bar. With the 7 templates measured
# this is also 3 (ceil(0.4 * 7)); fewer templates rarely agree (6: 55% of real floats found, 5: 31%, 4: 14%).
FLOAT_MIN_AGREEING_SHARE = 0.4
FLOAT_AGREEMENT_RADIUS = 45

# Click point: look for the base in the box padded to the right and down (never up: the feather is there; never left:
# the feather points left and a click on the water beside it misses, while a click on the float itself, feather included,
# catches - 2 of 4 warm-dusk clicks 5-15 px left of the box missed, the 2 inside it caught). The base hangs below and to
# the right of a tight template box (one fixture's is 0.2 of the box width past it). Else click the template's own base
# position, or FALLBACK_POINT when its base is too small to locate.
CLICK_SEARCH_PADDING_TOP_RATIO = 0
CLICK_SEARCH_PADDING_BOTTOM_RATIO = 0.5
CLICK_SEARCH_PADDING_LEFT_RATIO = 0
CLICK_SEARCH_PADDING_RIGHT_RATIO = 0.3
FALLBACK_VERTICAL_BIAS = 0.65
FALLBACK_POINT = (0.5, FALLBACK_VERTICAL_BIAS)   # (x, y) as fractions of the box, for a template whose own base cannot be located
# The base must fill this share of the template to trust its position there. Re-measured on the current 7 templates
# with _find_base itself: 10.14%, 4.90%, 11.79%, 12.14% clear this floor and use their own base position; 0.89%,
# 1.05%, 1.13% (mostly water, only a few base pixels whose centroid would be noise) fall under it and use
# FALLBACK_POINT instead - 3 of 7, not the 1 an earlier, never-accurate version of this comment implied.
FLOAT_MIN_TEMPLATE_BASE_SHARE = 0.03

# Templates are crops from ~2560 px wide windows and only match a float of about that size. A screenshot outside
# FLOAT_UNSCALED_WIDTH_RATIOS of the reference width is matched on a resized copy (a 919 px window showed the float at
# ~25 px against templates of 50-140 px). FLOAT_MAX_UPSCALE keeps a tiny screenshot from becoming huge.
FLOAT_REFERENCE_WIDTH = 2556
FLOAT_UNSCALED_WIDTH_RATIOS = (0.8, 1.4)
FLOAT_MAX_UPSCALE = 4

# Debugging: a trace line per step, and an annotated screenshot in DEBUG_SNAPSHOT_DIR when the click falls back to the
# box or the pick is on base color that no second kind of evidence backs. Keep False when committing.
DEBUG_SNAPSHOTS = True
DEBUG_TRACE = False   # the [float] trace lines only; the snapshots above stay on. Only matters while DEBUG_SNAPSHOTS is on
DEBUG_SNAPSHOT_DIR = 'debug'   # not var/, which holds the bot's real runtime data

# ------------------------------------------------------------------------------ result types

# The kinds of evidence, strongest first. Agreement ranks last: no independent physical signal is behind it.
EVIDENCE_GATE = 'color-gated'
EVIDENCE_BASE = 'base-colored'
EVIDENCE_AGREEMENT = 'template-agreement'
EVIDENCE_PRIORITY = (EVIDENCE_GATE, EVIDENCE_BASE, EVIDENCE_AGREEMENT)


class Candidate(NamedTuple):
	score: float
	loc: tuple       # top-left, in search-band coordinates
	size: tuple      # (width, height) of the template
	template: str
	evidence: str    # the kind of evidence backing the window


class Detection(NamedTuple):
	point: tuple        # (x, y) to click, in the screenshot's own pixels
	confidence: str     # the evidence that decided: one of EVIDENCE_PRIORITY, strongest first
	score: float
	template: str
	on_base_color: bool  # the click point is the base's color centroid, not the match's geometric center
	corroborated_by: tuple = ()   # the other kinds of evidence that independently point at the same place

	@property
	def corroborated(self):
		"""Whether a second, independent kind of evidence backs the place: 9 of 11 real floats, 0 of 4 measured false positives."""
		return len(self.corroborated_by) > 0


# ------------------------------------------------------------------------------ helpers

def _center(candidate: Candidate):
	return candidate.loc[0] + candidate.size[0] / 2, candidate.loc[1] + candidate.size[1] / 2


def _same_place(a: Candidate, b: Candidate):
	"""Whether two candidates' centers lie within FLOAT_AGREEMENT_RADIUS px of each other."""
	(ax, ay), (bx, by) = _center(a), _center(b)
	return (ax - bx) ** 2 + (ay - by) ** 2 <= FLOAT_AGREEMENT_RADIUS ** 2


def _debug(message):
	"""Trace line, printed only while DEBUG_SNAPSHOTS and DEBUG_TRACE are on."""
	if DEBUG_SNAPSHOTS and DEBUG_TRACE:
		print('[float] ' + message)


def _box_texture(gray: np.ndarray, window_size):
	"""Grayscale standard deviation of the box anchored at each pixel's top-left (the box result[y, x] scores)."""
	g = gray.astype(np.float64)
	mean = cv2.boxFilter(g, cv2.CV_64F, window_size, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
	mean_of_squares = cv2.boxFilter(g * g, cv2.CV_64F, window_size, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
	return np.sqrt(np.maximum(mean_of_squares - mean * mean, 0))


def _box_density(mask: np.ndarray, window_size):
	"""Count of nonzero `mask` pixels in the box anchored at each pixel's top-left."""
	# mask pixels are 0 or 255, so the unnormalized box sum is 255x the actual pixel count
	density = cv2.boxFilter(mask, cv2.CV_32F, window_size, normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
	return density / 255.0


def _drop_large_blobs_stats(mask: np.ndarray, max_size: int):
	"""(mask without components wider or taller than `max_size`, area of the biggest one kept or 0,
	[(width, height)] of those dropped)."""
	_, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
	widths, heights = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
	keep = (widths <= max_size) & (heights <= max_size)
	keep[0] = False   # label 0 is the background
	dropped = [(int(w), int(h)) for w, h in zip(widths[1:], heights[1:]) if w > max_size or h > max_size]
	cleaned = (keep[labels] * 255).astype(np.uint8)
	return cleaned, int(stats[keep, cv2.CC_STAT_AREA].max(initial=0)), dropped


def _drop_large_blobs(mask: np.ndarray, max_size: int):
	"""Zero out connected components wider or taller than `max_size`."""
	return _drop_large_blobs_stats(mask, max_size)[0]


def _compact_blob_stats(mask: np.ndarray, max_size: int, min_extent: float, max_aspect: float, min_area: int = 0):
	"""(mask keeping only components that are also round-ish - fill at least `min_extent` of their bounding box and
	are no more elongated than `max_aspect` - area of the biggest one kept or 0). See FLOAT_MIN_BASE_BLOB_EXTENT for
	what these catch that the size cap alone does not. `min_area` additionally drops small compact specks (a cluster
	of them can otherwise still sum past FLOAT_MIN_COLOR_PIXELS in _find_candidates's per-window density check, even
	though none of them is big enough on its own to be a base - see _base_color_mask)."""
	_, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
	widths, heights, areas = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
	extent = areas / np.maximum(widths * heights, 1)
	aspect = np.maximum(widths, heights) / np.maximum(np.minimum(widths, heights), 1)
	keep = (widths <= max_size) & (heights <= max_size) & (extent >= min_extent) & (aspect <= max_aspect) & (areas >= min_area)
	keep[0] = False   # label 0 is the background
	cleaned = (keep[labels] * 255).astype(np.uint8)
	return cleaned, int(areas[keep].max(initial=0))


def _blank_ui(mask: np.ndarray, ui_boxes):
	"""Zero the UI boxes of a band-sized mask: their bars and gold border are colorful but not a float."""
	for bx0, by0, bx1, by1 in ui_boxes:
		mask[by0:by1, bx0:bx1] = 0


def _search_bounds(w: int, h: int):
	"""(x0, y0, x1, y1) of the search band in screenshot pixels."""
	return (int(w * FLOAT_SEARCH_X_RANGE[0]), int(h * FLOAT_SEARCH_Y_RANGE[0]),
			int(w * FLOAT_SEARCH_X_RANGE[1]), int(h * FLOAT_SEARCH_Y_RANGE[1]))


def _ui_boxes(w: int, h: int, bounds):
	"""The UI exclude regions (see _ui_exclude_regions) as boxes in band coordinates, clipped to the band."""
	search_x0, search_y0, search_x1, search_y1 = bounds
	boxes = []
	for (x_range, y_range) in _ui_exclude_regions():
		x0 = max(0, int(w * x_range[0]) - search_x0)
		x1 = min(search_x1 - search_x0, int(w * x_range[1]) - search_x0)
		y0 = max(0, int(h * y_range[0]) - search_y0)
		y1 = min(search_y1 - search_y0, int(h * y_range[1]) - search_y0)
		if x1 > x0 and y1 > y0:
			boxes.append((x0, y0, x1, y1))
	return boxes


def _normalization_scale(width: int):
	"""Factor to resize a `width` px screenshot by before matching, or 1."""
	low, high = FLOAT_UNSCALED_WIDTH_RATIOS
	ratio = width / FLOAT_REFERENCE_WIDTH
	if low <= ratio <= high:
		return 1
	scale = FLOAT_REFERENCE_WIDTH / width
	return scale if scale <= FLOAT_MAX_UPSCALE else 1


def _load_templates():
	"""[(path, grayscale image)] for every template on disk."""
	templates = []
	for path in sorted(glob.glob(FLOAT_TEMPLATE_GLOB)):
		template = cv2.imread(path, 0)
		if template is not None:
			templates.append((path, template))
	return templates


# ------------------------------------------------------------------------------ 2. evidence

def _adaptive_color_mask(hsv_region: np.ndarray, ui_boxes=()):
	"""Pixels distinctly more saturated than this scene's water, whatever the hue. `ui_boxes` are left out of the
	baseline. May be empty (a colorless frame, or water so saturated that baseline + margin exceeds 255)."""
	saturation = hsv_region[:, :, 1]
	value = hsv_region[:, :, 2]
	water_pixels = np.ones(saturation.shape, dtype=bool)
	for bx0, by0, bx1, by1 in ui_boxes:
		water_pixels[by0:by1, bx0:bx1] = False
	if not water_pixels.any():
		water_pixels[:] = True
	water_baseline = np.percentile(saturation[water_pixels], FLOAT_SATURATION_BASELINE_PERCENTILE)
	threshold = water_baseline + FLOAT_SATURATION_MARGIN
	mask = ((saturation > threshold) & (value > FLOAT_MIN_VALUE)).astype(np.uint8) * 255
	if DEBUG_SNAPSHOTS:
		_debug('gate: water baseline (S p' + str(FLOAT_SATURATION_BASELINE_PERCENTILE) + ', UI boxes skipped) = ' + str(round(float(water_baseline), 1))
			+ ', threshold = ' + str(round(float(threshold), 1)) + (' (above 255: nothing can pass)' if threshold > 255 else '')
			+ ', float-colored pixels = ' + str(int(np.count_nonzero(mask))))
	return mask


def _band_base_range(hsv_band: np.ndarray, ui_boxes):
	"""The base color ranges' pixels over the band, UI blanked, and the share of the band they cover."""
	base_range = _base_range_mask(hsv_band)
	_blank_ui(base_range, ui_boxes)
	return base_range, np.count_nonzero(base_range) / base_range.size


def _build_color_mask(hsv_band: np.ndarray, ui_boxes, base_range=None):
	"""Evidence 1: the saturation mask with UI blanked, then oversized blobs dropped. In this order: a float touching
	a UI frame would otherwise merge with it into a blob too big to keep.

	Where the base color floods the scene (FLOAT_MAX_BASE_SCENE_SHARE) the water, and the previous cast's float still
	fading on it, are inside its ranges while the feather is not: only gate pixels outside the ranges are kept, so a
	window with a base but no feather - that afterimage - is not supported. Measured on 140 warm-dusk casts: 13 picks
	sat on the afterimage, all 13 moved onto the real float; no fixture changed."""
	color_mask = _adaptive_color_mask(hsv_band, ui_boxes)
	_blank_ui(color_mask, ui_boxes)
	base_range, scene_share = _band_base_range(hsv_band, ui_boxes) if base_range is None else (base_range, np.count_nonzero(base_range) / base_range.size)
	if scene_share > FLOAT_MAX_BASE_SCENE_SHARE:
		before = int(np.count_nonzero(color_mask))
		outside_ranges = cv2.dilate(cv2.bitwise_not(base_range), np.ones((3, 3), dtype=np.uint8))   # a pixel of slack for the edges
		color_mask = cv2.bitwise_and(color_mask, outside_ranges)
		_debug('gate: base color covers ' + str(round(100 * scene_share)) + '% of the band - keeping only gate pixels outside its ranges (the feather, not the water or an afterimage): '
			+ str(before) + ' px -> ' + str(int(np.count_nonzero(color_mask))) + ' px')
	after_ui = int(np.count_nonzero(color_mask))
	color_mask, _, dropped = _drop_large_blobs_stats(color_mask, FLOAT_MAX_BLOB_SIZE)
	if DEBUG_SNAPSHOTS:
		_debug('gate: color mask after blanking ' + str(len(ui_boxes)) + ' UI box(es) = ' + str(after_ui) + ' px, after dropping '
			+ str(len(dropped)) + ' blob(s) over ' + str(FLOAT_MAX_BLOB_SIZE) + 'px' + (' ' + str(dropped[:5]) if dropped else '')
			+ ' = ' + str(int(np.count_nonzero(color_mask))) + ' px')
	return color_mask


def _base_range_mask(hsv: np.ndarray):
	"""Pixels inside any FLOAT_BASE_COLOR_RANGES range, specks removed."""
	mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
	for lo, hi in FLOAT_BASE_COLOR_RANGES:
		mask |= cv2.inRange(hsv, lo, hi)
	return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))


def _base_color_mask(hsv_band: np.ndarray, ui_boxes, base_range=None):
	"""Evidence 2: base-colored pixels with UI blanked, oversized blobs dropped, and small or non-round ones (scenery
	clipped by the same range - see FLOAT_MIN_BASE_BLOB_EXTENT) dropped too. Empty when the scene floods the ranges
	(FLOAT_MAX_BASE_SCENE_SHARE): filtering windows one by one would only leave the wrong ones."""
	mask, scene_share = _band_base_range(hsv_band, ui_boxes) if base_range is None else (base_range.copy(), np.count_nonzero(base_range) / base_range.size)
	if scene_share > FLOAT_MAX_BASE_SCENE_SHARE:
		_debug('base color: ' + str(round(100 * scene_share)) + '% of the band is inside the base color ranges - the scene shares the hue, ignoring it as evidence')
		return np.zeros_like(mask)
	mask, _ = _compact_blob_stats(mask, FLOAT_MAX_BLOB_SIZE, FLOAT_MIN_BASE_BLOB_EXTENT, FLOAT_MAX_BASE_BLOB_ASPECT, FLOAT_MIN_BASE_BLOB_AREA)
	_debug('base color: ' + str(int(np.count_nonzero(mask))) + ' px in the band inside the base color ranges, round enough to trust')
	return mask


# ------------------------------------------------------------------------------ 3. candidates

def _find_candidates(search_gray: np.ndarray, evidence_masks: Dict[str, np.ndarray], ui_boxes):
	"""The best match per kind of evidence: {evidence: Candidate or None}. Windows on UI or without texture are
	excluded; then, per evidence, the best window among those it supports. (Taking the best window overall first
	would lose a float that is not the best-correlating spot, which on water it usually is not.)"""
	best: Dict[str, Optional[Candidate]] = {evidence: None for evidence in EVIDENCE_PRIORITY}
	shape_bests = []   # each template's best window by shape alone, for the agreement evidence
	densities = {}   # (evidence, window size) -> density map; same-size templates share one
	textures = {}    # window size -> texture map
	templates = _load_templates()
	_debug('templates: ' + str(len(templates)) + ' loaded (scores below are after UI and texture exclusion), search band ' + str(search_gray.shape[1]) + 'x' + str(search_gray.shape[0]))
	if not templates:
		print('No usable float templates matching ' + FLOAT_TEMPLATE_GLOB + ' - nothing to match against')
	for template_path, template in templates:
		th, tw = template.shape[:2]
		if th > search_gray.shape[0] or tw > search_gray.shape[1]:
			# matchTemplate would raise
			_debug('match: ' + template_path + ' (' + str(tw) + 'x' + str(th) + ') skipped - larger than the search band')
			continue
		result = cv2.matchTemplate(search_gray, template, cv2.TM_CCOEFF_NORMED)
		rh, rw = result.shape
		_, before_val, _, before_loc = cv2.minMaxLoc(result)

		# result[y, x] scores the window anchored at (x, y); its center is (x + tw/2, y + th/2)
		for bx0, by0, bx1, by1 in ui_boxes:
			result[max(0, by0 - th // 2):max(0, by1 - th // 2), max(0, bx0 - tw // 2):max(0, bx1 - tw // 2)] = -1
		_, after_ui_val, _, after_ui_loc = cv2.minMaxLoc(result)
		if after_ui_loc != before_loc:
			_debug('match: ' + template_path + ' UI exclusion removed its best spot ' + str(before_loc) + ' (score ' + str(round(before_val, 3))
				+ ', window centered in a UI box); next best is ' + str(after_ui_loc) + ' (' + str(round(after_ui_val, 3)) + ')')
		if (tw, th) not in textures:
			textures[(tw, th)] = _box_texture(search_gray, (tw, th))
		result[textures[(tw, th)][:rh, :rw] < FLOAT_MIN_TEXTURE] = -1

		_, shape_val, _, shape_loc = cv2.minMaxLoc(result)
		shape_bests.append(Candidate(shape_val, shape_loc, (tw, th), template_path, EVIDENCE_AGREEMENT))
		if shape_loc != after_ui_loc:
			_debug('match: ' + template_path + ' texture floor (std ' + str(FLOAT_MIN_TEXTURE) + ') removed its best spot ' + str(after_ui_loc)
				+ ' (score ' + str(round(after_ui_val, 3)) + ', a nearly featureless window); next best is ' + str(shape_loc) + ' (' + str(round(shape_val, 3)) + ')')

		line = 'match: ' + template_path + ' (' + str(tw) + 'x' + str(th) + ') shape only ' + str(round(shape_val, 3)) + ' at ' + str(shape_loc)
		for evidence, label, empty in ((EVIDENCE_GATE, 'with color gate', 'float-colored'), (EVIDENCE_BASE, 'with base color', 'base-colored')):
			if (evidence, tw, th) not in densities:
				densities[(evidence, tw, th)] = _box_density(evidence_masks[evidence], (tw, th))
			supported = result.copy()
			supported[densities[(evidence, tw, th)][:rh, :rw] < FLOAT_MIN_COLOR_PIXELS] = -1
			_, val, _, loc = cv2.minMaxLoc(supported)
			if best[evidence] is None or val > best[evidence].score:
				best[evidence] = Candidate(val, loc, (tw, th), template_path, evidence)
			line += ', ' + label + ' ' + (str(round(val, 3)) + ' at ' + str(loc) if val > -1 else 'no position had enough ' + empty + ' pixels')
		_debug(line)
	best[EVIDENCE_AGREEMENT] = _agreeing_candidate(shape_bests)
	return best


def _agreeing_candidate(shape_bests):
	"""The best member of the largest group of templates whose best windows agree on a place, or None when the group
	has fewer than FLOAT_MIN_AGREEING_TEMPLATES templates or FLOAT_MIN_AGREEING_SHARE of those matched."""
	group_of = [[m for m in shape_bests if m.score > FLOAT_MATCH_THRESHOLD and c.score > FLOAT_MATCH_THRESHOLD and _same_place(m, c)]
				for c in shape_bests]
	group = max(group_of, key=lambda g: (len(g), max((m.score for m in g), default=-1)), default=[])
	needed = max(FLOAT_MIN_AGREEING_TEMPLATES, math.ceil(round(FLOAT_MIN_AGREEING_SHARE * len(shape_bests), 9)))
	if len(group) < needed:
		_debug('agreement: no ' + str(needed) + ' of ' + str(len(shape_bests)) + ' templates agree on a place (largest group ' + str(len(group)) + ')')
		return None
	winner = max(group, key=lambda m: m.score)
	_debug('agreement: ' + str(len(group)) + ' templates agree near ' + str(tuple(round(v) for v in _center(winner))) + ' - best ' + winner.template
		+ ' ' + str(round(winner.score, 3)))
	return winner


# ------------------------------------------------------------------------------ 4. decide

def _decide(candidates: Dict[str, Optional[Candidate]]) -> Optional[Candidate]:
	"""The candidate of the strongest kind of evidence that clears FLOAT_MATCH_THRESHOLD, or None.

	The saturation gate can be blind for reasons unrelated to the float (clipped saturation, a moonlit sea, a warm
	dusk); base color covers those scenes. Shape alone is not evidence: on rippled dusk water the best shape match
	was a pale wave crest in 7 of 81 casts while the real float scored lower elsewhere."""
	for evidence in EVIDENCE_PRIORITY:
		candidate = candidates.get(evidence)
		if candidate is not None and candidate.score > FLOAT_MATCH_THRESHOLD:
			_debug('pick: ' + {EVIDENCE_GATE: 'color-gated match ',
							   EVIDENCE_BASE: 'gate found nothing above ' + str(FLOAT_MATCH_THRESHOLD)
							   + ' - using the shape match among base-colored windows ',
							   EVIDENCE_AGREEMENT: 'no color evidence - using the window the templates agree on '}[evidence]
				+ candidate.template + ' score ' + str(round(candidate.score, 3)) + ' at ' + str(candidate.loc))
			return candidate
	scores = ', '.join(evidence + ' ' + (str(round(candidates[evidence].score, 3)) if candidates.get(evidence) and candidates[evidence].score > -1 else 'none')
					   for evidence in EVIDENCE_PRIORITY)
	_debug('pick: nothing above threshold ' + str(FLOAT_MATCH_THRESHOLD) + ' (' + scores + ') - not found')
	return None


def _corroborating_evidence(chosen: Candidate, candidates: Dict[str, Optional[Candidate]]):
	"""The other kinds of evidence whose best candidate clears the threshold within FLOAT_AGREEMENT_RADIUS px of the
	chosen one: independent signals landing on the same place are far stronger than the same signals scattered.

	A density-at-the-chosen-spot variant (crediting a kind of evidence even when its own best-scoring window was
	elsewhere) was tried and reverted: on the 2026-09-23 Stormwind Harbor session it correctly caught 9 real floats
	that this check alone misses, but it also credited 3 real misses, and no measurable property of the evidence
	there (raw density, largest single blob in the window, the click point's own on_base_color check, scene-wide
	base color share) could tell those two groups apart - ambient same-hued water in that scene is common enough
	that "some base-colored material is nearby" carries little information by itself."""
	return tuple(evidence for evidence in EVIDENCE_PRIORITY
				 if evidence != chosen.evidence and candidates.get(evidence) is not None
				 and candidates[evidence].score > FLOAT_MATCH_THRESHOLD and _same_place(candidates[evidence], chosen))


# ------------------------------------------------------------------------------ 5. click point

def _find_base(hsv_region: np.ndarray):
	"""((x, y) centroid of the base's color in `hsv_region`, area of its largest blob), or (None, [why each range failed]).
	Each range is tried on its own: the union floods when one range matches the water, fusing base and water into a blob
	dropped as scenery. A blob qualifies when it clears FLOAT_MIN_BASE_BLOB_AREA (scattered flecks are what water sharing
	the hue leaves) and does not fill the region (that is a surface, not a base).

	Deliberately not compactness-filtered like _base_color_mask: on warm_dusk_afterimage.png the real base itself renders
	as several small elongated fragments (the same color range's edge cuts through it under that lighting), and their
	combined centroid - the moments below are taken over every surviving blob in the range, not just the biggest - lands
	on the real base; requiring each fragment to be round on its own throws that away and was tried and made it worse."""
	rejections = []
	for lo, hi in FLOAT_BASE_COLOR_RANGES:
		mask = cv2.morphologyEx(cv2.inRange(hsv_region, lo, hi), cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
		mask, largest_component, _ = _drop_large_blobs_stats(mask, FLOAT_MAX_BLOB_SIZE)
		if largest_component < FLOAT_MIN_BASE_BLOB_AREA:
			rejections.append(str(largest_component) + ' px')
		elif largest_component > FLOAT_MAX_BASE_BLOB_SHARE * mask.size:
			rejections.append(str(largest_component) + ' px = scenery')
		else:
			moments = cv2.moments(mask, binaryImage=True)
			return (moments['m10'] / moments['m00'], moments['m01'] / moments['m00']), largest_component
	return None, rejections


def _base_click_point(hsv_region: np.ndarray):
	"""Centroid of the base's color in `hsv_region` (region coordinates), or None without one solid blob to trust."""
	point, detail = _find_base(hsv_region)
	if point is None:
		_debug('click: base color not found (largest blob per range: ' + ', '.join(detail) + '; need ' + str(FLOAT_MIN_BASE_BLOB_AREA) + ' and under ' + str(round(100 * FLOAT_MAX_BASE_BLOB_SHARE)) + '% of the region)')
		return None
	_debug('click: base color found (largest blob ' + str(detail) + ' px)')
	return point


def _template_base_anchor(template_path: str):
	"""Where the base sits inside this template, as (x, y) fractions of its size: the anchor for a click that cannot
	find the base by color. FALLBACK_POINT when the template's base is too small to locate."""
	template = cv2.imread(template_path)
	if template is None:
		return FALLBACK_POINT
	h, w = template.shape[:2]
	point, detail = _find_base(cv2.cvtColor(template, cv2.COLOR_BGR2HSV))
	if point is None or detail < FLOAT_MIN_TEMPLATE_BASE_SHARE * w * h:
		return FALLBACK_POINT
	return point[0] / w, point[1] / h


def _save_debug_snapshot(img_bgr: np.ndarray, box_tl, box_size, click_point, prefix: str, label: str):
	"""If DEBUG_SNAPSHOTS is on, save the screenshot with the matched box, click point and `label` drawn on it.
	Returns the path, or None."""
	if not DEBUG_SNAPSHOTS:
		return None
	tw, th = box_size
	annotated = img_bgr.copy()
	cv2.rectangle(annotated, box_tl, (box_tl[0] + tw, box_tl[1] + th), (0, 255, 0), 2)
	cv2.circle(annotated, (int(click_point[0]), int(click_point[1])), 6, (0, 0, 255), -1)
	cv2.putText(annotated, label, (box_tl[0], max(20, box_tl[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
	path = os.path.join(DEBUG_SNAPSHOT_DIR, prefix + '_' + str(int(time.time() * 1000)) + '.png')   # ms: no overwrites within a second
	try:
		os.makedirs(DEBUG_SNAPSHOT_DIR, exist_ok=True)
		if not cv2.imwrite(path, annotated):
			raise OSError('cv2.imwrite returned False')
	except (OSError, cv2.error) as error:
		# a debugging aid must never take detection down
		print('Could not save the debug snapshot into ' + DEBUG_SNAPSHOT_DIR + ': ' + str(error))
		return None
	return path


def save_notfound_snapshot(screenshot_path, dest_path):
	"""Write the screenshot to `dest_path` with the search band drawn on it, so a float that sits outside the band (or under a
	UI box) can be told from one the matching simply missed: the yellow box is where the float is looked for, the blue boxes
	inside it are the screen elements excluded. Returns True when written, False when the screenshot cannot be read or
	written (a debugging aid must never take the caller down)."""
	img_bgr = cv2.imread(screenshot_path)
	if img_bgr is None:
		return False
	h, w = img_bgr.shape[:2]
	bounds = _search_bounds(w, h)
	x0, y0, x1, y1 = bounds
	for bx0, by0, bx1, by1 in _ui_boxes(w, h, bounds):
		cv2.rectangle(img_bgr, (x0 + bx0, y0 + by0), (x0 + bx1, y0 + by1), (255, 128, 0), 2)
	cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (0, 255, 255), 2)
	cv2.putText(img_bgr, 'search band (float outside it is not looked for)', (x0, max(20, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
	try:
		return bool(cv2.imwrite(dest_path, img_bgr))
	except cv2.error:
		return False


def _click_point(img_bgr: np.ndarray, hsv: np.ndarray, candidate: Candidate, bounds):
	"""((x, y), on_base_color): the base's color centroid near the match, else the template's own base position in the box."""
	h, w = img_bgr.shape[:2]
	tw, th = candidate.size
	tl = (candidate.loc[0] + bounds[0], candidate.loc[1] + bounds[1])   # back in full-image coordinates

	# a tight template can leave little of the base inside the box, so search a box padded right and down (see CLICK_SEARCH_PADDING_*)
	x0, y0 = max(0, tl[0] - int(tw * CLICK_SEARCH_PADDING_LEFT_RATIO)), max(0, tl[1] - int(th * CLICK_SEARCH_PADDING_TOP_RATIO))
	x1, y1 = min(w, tl[0] + tw + int(tw * CLICK_SEARCH_PADDING_RIGHT_RATIO)), min(h, tl[1] + th + int(th * CLICK_SEARCH_PADDING_BOTTOM_RATIO))
	_debug('click: searching for the base color in region (' + str(x0) + ',' + str(y0) + ')-(' + str(x1) + ',' + str(y1) + ')')
	point = _base_click_point(hsv[y0:y1, x0:x1])
	if point is not None:
		_debug('click: using the base color centroid ' + str((round(x0 + point[0], 1), round(y0 + point[1], 1))))
		return (x0 + point[0], y0 + point[1]), True

	anchor = _template_base_anchor(candidate.template)
	fallback_point = (tl[0] + tw * anchor[0], tl[1] + th * anchor[1])
	_debug('click: using ' + ('the default position' if anchor == FALLBACK_POINT else candidate.template + '\'s base position') + ' ' + str((round(anchor[0], 2), round(anchor[1], 2)))
		+ ' of the matched box: ' + str((round(fallback_point[0], 1), round(fallback_point[1], 1))))
	return fallback_point, False


# ------------------------------------------------------------------------------ entry points

def find_float_detailed(screenshot_path) -> Optional[Detection]:
	"""The float in the screenshot with the evidence behind it, or None if not found."""
	img_bgr = cv2.imread(screenshot_path)
	if img_bgr is None:
		# e.g. caught mid-write: treat it like "not found"
		print('Could not read screenshot: ' + screenshot_path)
		return None
	h, w = img_bgr.shape[:2]
	scale = _normalization_scale(w)
	if scale != 1 and min(round(w * scale), round(h * scale)) < 1:
		_debug('end: a ' + str(w) + 'x' + str(h) + ' screenshot would resize to nothing - not found')
		return None   # absurd aspect ratio; cv2.resize would raise
	if scale != 1:
		_debug('scale: ' + str(w) + 'px wide is ' + str(round(w / FLOAT_REFERENCE_WIDTH, 2)) + 'x the reference width ' + str(FLOAT_REFERENCE_WIDTH)
			+ ' - matching on a copy resized by ' + str(round(scale, 2)))
		img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
		h, w = img_bgr.shape[:2]
	bounds = _search_bounds(w, h)
	x0, y0, x1, y1 = bounds
	_debug('start: ' + screenshot_path + ' is ' + str(w) + 'x' + str(h) + ', search band x ' + str(x0) + '-' + str(x1) + ' y ' + str(y0) + '-' + str(y1))
	if x1 <= x0 or y1 <= y0:
		_debug('end: the search band is empty (degenerate screenshot) - not found')
		return None   # e.g. a minimized window: nothing to search

	hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
	hsv_band = hsv[y0:y1, x0:x1]
	search_gray = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
	ui_boxes = _ui_boxes(w, h, bounds)
	_debug('ui: excluding ' + str(len(ui_boxes)) + ' box(es) inside the band: ' + str(ui_boxes))

	base_range, _ = _band_base_range(hsv_band, ui_boxes)
	evidence_masks = {EVIDENCE_GATE: _build_color_mask(hsv_band, ui_boxes, base_range), EVIDENCE_BASE: _base_color_mask(hsv_band, ui_boxes, base_range)}
	candidates = _find_candidates(search_gray, evidence_masks, ui_boxes)
	candidate = _decide(candidates)
	if candidate is None:
		_debug('end: float not found')
		return None

	point, on_base_color = _click_point(img_bgr, hsv, candidate, bounds)
	corroborated_by = _corroborating_evidence(candidate, candidates)
	summary = candidate.evidence + ', ' + ('corroborated by ' + ' + '.join(corroborated_by) if corroborated_by else 'NOT corroborated by any other evidence')
	print('Matched ' + candidate.template + ' (score ' + str(round(candidate.score, 3)) + ') - ' + summary)

	# mistakes hide in fallback clicks and in weak matches no second evidence backs: keep a picture of each for review
	box_tl = (candidate.loc[0] + bounds[0], candidate.loc[1] + bounds[1])
	label = candidate.evidence + ' ' + str(round(candidate.score, 2)) + (' corroborated' if corroborated_by else ' UNCORROBORATED')
	if not on_base_color:
		path = _save_debug_snapshot(img_bgr, box_tl, candidate.size, point, 'fallback', label)
		if path:
			print('Click point fell back to the matched box\'s center - saved ' + path + ' for review')
	elif not corroborated_by and candidate.score < FLOAT_LOWCONF_SCORE:
		path = _save_debug_snapshot(img_bgr, box_tl, candidate.size, point, 'lowconf', label)
		if path:
			print('Low confidence (' + summary + ') - saved ' + path + ' for review')

	if scale != 1:
		point = (point[0] / scale, point[1] / scale)   # back to the original screenshot's pixels
	_debug('end: click point ' + str((round(point[0], 1), round(point[1], 1))) + ' - ' + summary)
	return Detection(point, candidate.evidence, candidate.score, candidate.template, on_base_color, corroborated_by)


def find_float(screenshot_path):
	"""Pixel (x, y) to click for the float in the screenshot, or None if not found."""
	detection = find_float_detailed(screenshot_path)
	return None if detection is None else detection.point
