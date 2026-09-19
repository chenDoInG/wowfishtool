"""Find the fishing float in a screenshot of the WoW window.

The pipeline, in the order find_float_detailed() runs it:

1. normalize   Resize the screenshot to a reference width when the window is much smaller or larger
               than the ones the templates were cropped from, so the float has the templates' size.
2. evidence    Two independent maps of "float-colored" pixels over the search band: pixels much more
               saturated than this scene's water (works for any hue), and pixels inside the fixed
               HSV ranges of the bobber's base (works when the water is too saturated for the first).
3. candidates  Grayscale template correlation for every template. Windows on a screen element or
               with no texture are excluded outright. For each kind of evidence, each template
               contributes its best window among those the evidence supports.
4. decide      The strongest kind of evidence whose best candidate clears FLOAT_MATCH_THRESHOLD wins.
               Shape alone never decides: on water it scores 0.4-0.7 almost anywhere.
5. click point The base's color centroid near the match, else the match's geometric center.

Coordinates are reported in the original screenshot's pixels. The history behind each number is in
the commit messages and README; the comments here say what a constant is for and what it was measured on.
"""
import glob
import os
import time
from typing import Dict, NamedTuple, Optional

import cv2
import numpy as np

# ------------------------------------------------------------------------------ configuration

FLOAT_TEMPLATE_GLOB = 'var/fishing_float_*.png'

# Lowest correlation that counts as a float. Confirmed real floats scored 0.30 (still fading in) to
# 0.9; the real miss that must stay a miss scored 0.33. Deliberately in the gap: a weak match becomes
# "not found" (the caller retries) instead of a click that wastes a whole bite timeout.
FLOAT_MATCH_THRESHOLD = 0.35

# Where a cast lands, as fractions of the window. Every real detection so far fell between 45% and 77%
# of the height, below the horizon (a distant ship) and above the character; the top and bottom edges are
# padded past that because cropping the float out of the band cannot be recovered by any later stage.
FLOAT_SEARCH_X_RANGE = (0.25, 0.75)
FLOAT_SEARCH_Y_RANGE = (0.38, 0.80)

# Fixed screen elements inside the band that must never be taken for the float: the always-on player frame
# (bottom-left) and the duplicate WoW's default "Modern" layout adds (bottom-right). Fractions of the window,
# measured on a 2560x1410 capture and padded.
UI_EXCLUDE_REGIONS = (
	((0.26, 0.36), (0.72, 0.80)),
	((0.62, 0.76), (0.71, 0.83)),
)

# Evidence 1, relative saturation: a pixel counts when it is more saturated than the scene's water by a margin,
# whatever the hue. The baseline is the 99.5th percentile (a saturated sea plateaus up to its 99th, so a lower
# percentile leaves no headroom); the UI frames are left out of it because two of them (~2% of the band) would
# drag it up to their own saturation and blind the gate.
FLOAT_SATURATION_BASELINE_PERCENTILE = 99.5
FLOAT_SATURATION_MARGIN = 20
FLOAT_MIN_VALUE = 30        # near-black pixels have unstable saturation; the dimmest real float pixel seen was 39
FLOAT_MIN_COLOR_PIXELS = 15  # evidence pixels a window needs before that evidence supports it (both kinds)
FLOAT_MAX_BLOB_SIZE = 200    # a connected evidence blob wider or taller than this is scenery (a dock, a hull), not a float

# Evidence 2, the base's own color. One range per lighting; each is kept narrow because a wide one bleeds into water.
# The dusk-to-night green base was measured on 68 real floats (H 52-94, S 27-90, V 19-57): no other range reaches it.
FLOAT_BASE_COLOR_RANGES = (
	((10, 80, 100), (35, 255, 255)),   # daylight warm tan/yellow
	((40, 30, 120), (75, 90, 255)),    # dusk/night-tinted, desaturated
	((10, 15, 25), (35, 140, 110)),    # the daylight hue dimmed by night lighting
	((35, 25, 25), (95, 120, 200)),    # dark, desaturated green
)
FLOAT_MIN_BASE_BLOB_AREA = 30        # the click point needs one solid base blob this big, not scattered flecks

# Grayscale windows flatter than this cannot be a float: featureless water still correlates at 0.6-0.7. Empty water
# measured 0.2-4.1, real float windows 6.0 and up (faint, fading-in floats sit at the bottom of that range).
FLOAT_MIN_TEXTURE = 6

# Click point: look for the base in the match's box padded down and sideways (never up: the feather is there),
# else click FALLBACK_VERTICAL_BIAS of the way down the box - the base sits below the feather that pulls the box's
# center up. The bias was measured against two manually verified scenes.
CLICK_SEARCH_PADDING_TOP_RATIO = 0
CLICK_SEARCH_PADDING_BOTTOM_RATIO = 0.5
CLICK_SEARCH_PADDING_X_RATIO = 0.3
FALLBACK_VERTICAL_BIAS = 0.65

# Templates are crops from ~2560 px wide windows, so they only match a float of about that apparent size. A screenshot
# outside FLOAT_UNSCALED_WIDTH_RATIOS of the reference width is matched on a resized copy (a real 919 px window showed
# the float at ~25 px against templates of 50-140 px). FLOAT_MAX_UPSCALE keeps a tiny screenshot from becoming huge.
FLOAT_REFERENCE_WIDTH = 2556
FLOAT_UNSCALED_WIDTH_RATIOS = (0.8, 1.4)
FLOAT_MAX_UPSCALE = 4

# Debugging: trace lines for every step, and an annotated screenshot in DEBUG_SNAPSHOT_DIR whenever the click point falls
# back to the box center. Off by default so a normal run prints and writes nothing extra.
DEBUG_SNAPSHOTS = False
DEBUG_SNAPSHOT_DIR = 'debug'   # kept out of var/, which holds the bot's real runtime data

# ------------------------------------------------------------------------------ result types

# The kinds of color evidence, strongest first. A window is "supported" by a kind when it holds at least
# FLOAT_MIN_COLOR_PIXELS pixels of it.
EVIDENCE_GATE = 'color-gated'
EVIDENCE_BASE = 'base-colored'
EVIDENCE_PRIORITY = (EVIDENCE_GATE, EVIDENCE_BASE)


class Candidate(NamedTuple):
	score: float
	loc: tuple       # top-left, in search-band coordinates
	size: tuple      # (width, height) of the template
	template: str
	evidence: str    # the kind of color evidence backing the window


class Detection(NamedTuple):
	point: tuple        # (x, y) to click, in the screenshot's own pixels
	confidence: str     # the evidence that decided: EVIDENCE_GATE (strong) or EVIDENCE_BASE
	score: float
	template: str
	on_base_color: bool  # the click point is the base's color centroid, not the match's geometric center


# ------------------------------------------------------------------------------ helpers

def _debug(message):
	"""Trace line for one step, printed only while DEBUG_SNAPSHOTS is on. Callers that would do real work
	just to build the message check DEBUG_SNAPSHOTS first."""
	if DEBUG_SNAPSHOTS:
		print('[float] ' + message)


def _box_texture(gray: np.ndarray, window_size):
	"""Per-pixel grayscale standard deviation of the window_size box anchored at that pixel's top-left,
	i.e. texture[y, x] covers the same box matchTemplate's result[y, x] scores."""
	g = gray.astype(np.float64)
	mean = cv2.boxFilter(g, cv2.CV_64F, window_size, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
	mean_of_squares = cv2.boxFilter(g * g, cv2.CV_64F, window_size, anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
	return np.sqrt(np.maximum(mean_of_squares - mean * mean, 0))


def _box_density(mask: np.ndarray, window_size):
	"""Per-pixel count of nonzero `mask` pixels in the window_size box anchored at that pixel's top-left."""
	# mask pixels are 0 or 255, so the unnormalized box sum is 255x the actual pixel count
	density = cv2.boxFilter(mask, cv2.CV_32F, window_size, normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
	return density / 255.0


def _drop_small_blobs_stats(mask: np.ndarray, max_size: int):
	"""(cleaned, largest_area): `mask` without the connected components wider or taller than `max_size`,
	plus the pixel area of the biggest component that survived (0 if none did)."""
	_, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
	keep = (stats[:, cv2.CC_STAT_WIDTH] <= max_size) & (stats[:, cv2.CC_STAT_HEIGHT] <= max_size)
	keep[0] = False   # label 0 is the background
	cleaned = (keep[labels] * 255).astype(np.uint8)
	return cleaned, int(stats[keep, cv2.CC_STAT_AREA].max(initial=0))


def _drop_large_blobs(mask: np.ndarray, max_size: int):
	"""Zero out connected components wider or taller than `max_size` - structures, not the float's small blob."""
	return _drop_small_blobs_stats(mask, max_size)[0]


def _blank_ui(mask: np.ndarray, ui_boxes):
	"""Zero the player-frame boxes of a band-sized mask: their saturated bars and gold border are colored
	evidence in every sense except being a float."""
	for bx0, by0, bx1, by1 in ui_boxes:
		mask[by0:by1, bx0:bx1] = 0


def _search_bounds(w: int, h: int):
	"""(x0, y0, x1, y1) of the search band in full-screenshot pixel coordinates."""
	return (int(w * FLOAT_SEARCH_X_RANGE[0]), int(h * FLOAT_SEARCH_Y_RANGE[0]),
			int(w * FLOAT_SEARCH_X_RANGE[1]), int(h * FLOAT_SEARCH_Y_RANGE[1]))


def _ui_boxes(w: int, h: int, bounds):
	"""UI_EXCLUDE_REGIONS as (x0, y0, x1, y1) boxes in search-band coordinates, clipped to the band."""
	search_x0, search_y0, search_x1, search_y1 = bounds
	boxes = []
	for (x_range, y_range) in UI_EXCLUDE_REGIONS:
		x0 = max(0, int(w * x_range[0]) - search_x0)
		x1 = min(search_x1 - search_x0, int(w * x_range[1]) - search_x0)
		y0 = max(0, int(h * y_range[0]) - search_y0)
		y1 = min(search_y1 - search_y0, int(h * y_range[1]) - search_y0)
		if x1 > x0 and y1 > y0:
			boxes.append((x0, y0, x1, y1))
	return boxes


def _normalization_scale(width: int):
	"""Factor to resize a screenshot `width` px wide by before matching, or 1 to leave it alone."""
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
	"""Pixels distinctly more saturated than this scene's own water, whatever the hue.

	The mask can legitimately be empty - a colorless frame (login screen), or water so saturated that
	baseline + margin exceeds the HSV ceiling - and then this evidence simply supports no window.
	`ui_boxes` (band coordinates) are left out of the water baseline. Oversized blobs are NOT dropped here;
	see _build_color_mask()."""
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


def _build_color_mask(hsv_band: np.ndarray, ui_boxes):
	"""Evidence 1 over the band: the adaptive saturation mask, UI frames blanked, oversized blobs dropped.
	The order matters: a float touching a UI frame's saturated pixels would otherwise merge with it into one
	blob too big to keep, and its own evidence would be dropped as collateral."""
	color_mask = _adaptive_color_mask(hsv_band, ui_boxes)
	_blank_ui(color_mask, ui_boxes)
	if DEBUG_SNAPSHOTS:
		after_ui = int(np.count_nonzero(color_mask))
		_, _, stats, _ = cv2.connectedComponentsWithStats(color_mask, connectivity=8)
		dropped = [(int(w), int(h)) for w, h in zip(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
				   if w > FLOAT_MAX_BLOB_SIZE or h > FLOAT_MAX_BLOB_SIZE]
	color_mask = _drop_large_blobs(color_mask, FLOAT_MAX_BLOB_SIZE)
	if DEBUG_SNAPSHOTS:
		_debug('gate: color mask after blanking ' + str(len(ui_boxes)) + ' UI box(es) = ' + str(after_ui) + ' px, after dropping '
			+ str(len(dropped)) + ' blob(s) over ' + str(FLOAT_MAX_BLOB_SIZE) + 'px' + (' ' + str(dropped[:5]) if dropped else '')
			+ ' = ' + str(int(np.count_nonzero(color_mask))) + ' px')
	return color_mask


def _base_range_mask(hsv: np.ndarray):
	"""Pixels inside any FLOAT_BASE_COLOR_RANGES range (single-pixel specks removed). The one definition of
	"base-colored", shared by the evidence map and the click point."""
	mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
	for lo, hi in FLOAT_BASE_COLOR_RANGES:
		mask |= cv2.inRange(hsv, lo, hi)
	return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))


def _base_color_mask(hsv_band: np.ndarray, ui_boxes):
	"""Evidence 2 over the band: base-colored pixels, UI frames blanked, oversized blobs (water or scenery
	that shares the hue) dropped."""
	mask = _base_range_mask(hsv_band)
	_blank_ui(mask, ui_boxes)
	mask = _drop_large_blobs(mask, FLOAT_MAX_BLOB_SIZE)
	_debug('base color: ' + str(int(np.count_nonzero(mask))) + ' px in the band inside the base color ranges')
	return mask


# ------------------------------------------------------------------------------ 3. candidates

def _find_candidates(search_gray: np.ndarray, evidence_masks: Dict[str, np.ndarray], ui_boxes):
	"""The best template match per kind of evidence: {evidence: Candidate or None}.

	Each template is correlated over the whole band, then windows centered on a screen element or with no
	texture are excluded. Per kind of evidence, the best window among those it supports is that template's
	contender. (Picking the best window overall and asking afterwards would lose a real float that is not the
	best-correlating spot, which on water it usually is not.)"""
	best: Dict[str, Optional[Candidate]] = {evidence: None for evidence in EVIDENCE_PRIORITY}
	densities = {}   # (evidence, window size) -> density map; same-size templates share one
	textures = {}    # window size -> texture map
	templates = _load_templates()
	_debug('templates: ' + str(len(templates)) + ' loaded (scores below are after UI and texture exclusion), search band ' + str(search_gray.shape[1]) + 'x' + str(search_gray.shape[0]))
	if not templates:
		print('No usable float templates matching ' + FLOAT_TEMPLATE_GLOB + ' - nothing to match against')
	for template_path, template in templates:
		th, tw = template.shape[:2]
		if th > search_gray.shape[0] or tw > search_gray.shape[1]:
			# a window shrunk so far that the band is smaller than the template: matchTemplate would raise
			_debug('match: ' + template_path + ' (' + str(tw) + 'x' + str(th) + ') skipped - larger than the search band')
			continue
		result = cv2.matchTemplate(search_gray, template, cv2.TM_CCOEFF_NORMED)
		rh, rw = result.shape
		if DEBUG_SNAPSHOTS:
			_, before_val, _, before_loc = cv2.minMaxLoc(result)

		# result[y, x] scores the window anchored at (x, y); its center is (x + tw/2, y + th/2)
		for bx0, by0, bx1, by1 in ui_boxes:
			result[max(0, by0 - th // 2):max(0, by1 - th // 2), max(0, bx0 - tw // 2):max(0, bx1 - tw // 2)] = -1
		if DEBUG_SNAPSHOTS:
			_, after_ui_val, _, after_ui_loc = cv2.minMaxLoc(result)
			if after_ui_loc != before_loc:
				_debug('match: ' + template_path + ' UI exclusion removed its best spot ' + str(before_loc) + ' (score ' + str(round(before_val, 3))
					+ ', window centered in a UI box); next best is ' + str(after_ui_loc) + ' (' + str(round(after_ui_val, 3)) + ')')
		if (tw, th) not in textures:
			textures[(tw, th)] = _box_texture(search_gray, (tw, th))
		result[textures[(tw, th)][:rh, :rw] < FLOAT_MIN_TEXTURE] = -1

		_, shape_val, _, shape_loc = cv2.minMaxLoc(result)   # reported only: shape alone never decides
		if DEBUG_SNAPSHOTS and shape_loc != after_ui_loc:
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
	return best


# ------------------------------------------------------------------------------ 4. decide

def _decide(candidates: Dict[str, Optional[Candidate]]) -> Optional[Candidate]:
	"""The candidate of the strongest kind of evidence that clears FLOAT_MATCH_THRESHOLD, or None.

	The relative-saturation gate can be blind for reasons unrelated to the float - saturation clipped at the
	ceiling, a moonlit sea whose water outshines it, a warm dusk that lifts the threshold above the float - and
	then every gated candidate scores poorly even though the shape match is confident. The base-color evidence
	covers those scenes. Shape alone is not evidence: on rippled dusk water the best window by shape was a pale
	wave crest in 7 of 81 casts while the small, dark, real float sat elsewhere in the frame with a lower score."""
	for evidence in EVIDENCE_PRIORITY:
		candidate = candidates.get(evidence)
		if candidate is not None and candidate.score > FLOAT_MATCH_THRESHOLD:
			_debug('pick: ' + (evidence + ' match ' if evidence == EVIDENCE_GATE else 'gate found nothing above ' + str(FLOAT_MATCH_THRESHOLD)
				+ ' - using the shape match among base-colored windows ') + candidate.template + ' score ' + str(round(candidate.score, 3)) + ' at ' + str(candidate.loc))
			return candidate
	scores = ', '.join(evidence + ' ' + (str(round(candidates[evidence].score, 3)) if candidates.get(evidence) else 'none') for evidence in EVIDENCE_PRIORITY)
	_debug('pick: nothing above threshold ' + str(FLOAT_MATCH_THRESHOLD) + ' (' + scores + ') - not found')
	return None


# ------------------------------------------------------------------------------ 5. click point

def _base_click_point(hsv_region: np.ndarray):
	"""Centroid of the bobber base's color within `hsv_region` (region coordinates), or None when there is no
	single solid base-colored blob to trust - scattered flecks that sum to a plausible total are what water
	sharing the base's hue leaves behind, so the biggest component alone must clear the floor."""
	mask, largest_component = _drop_small_blobs_stats(_base_range_mask(hsv_region), FLOAT_MAX_BLOB_SIZE)
	if largest_component < FLOAT_MIN_BASE_BLOB_AREA:
		_debug('click: base color not found (largest blob ' + str(largest_component) + ' px, need ' + str(FLOAT_MIN_BASE_BLOB_AREA) + ')')
		return None
	moments = cv2.moments(mask, binaryImage=True)
	_debug('click: base color found (largest blob ' + str(largest_component) + ' px, ' + str(int(moments['m00'])) + ' px total)')
	return moments['m10'] / moments['m00'], moments['m01'] / moments['m00']


def _save_fallback_debug_snapshot(img_bgr: np.ndarray, box_tl, box_size, click_point):
	"""If DEBUG_SNAPSHOTS is on, save the screenshot with the matched box and click point drawn on it, so a click
	that fell back to the geometric center can be checked after the fact. No-op otherwise."""
	if not DEBUG_SNAPSHOTS:
		return
	os.makedirs(DEBUG_SNAPSHOT_DIR, exist_ok=True)
	tw, th = box_size
	annotated = img_bgr.copy()
	cv2.rectangle(annotated, box_tl, (box_tl[0] + tw, box_tl[1] + th), (0, 255, 0), 2)
	cv2.circle(annotated, (int(click_point[0]), int(click_point[1])), 6, (0, 0, 255), -1)
	path = os.path.join(DEBUG_SNAPSHOT_DIR, 'fallback_' + str(int(time.time())) + '.png')
	cv2.imwrite(path, annotated)
	print('Click point fell back to the matched box\'s center - saved ' + path + ' for review')


def _click_point(img_bgr: np.ndarray, hsv: np.ndarray, candidate: Candidate, bounds):
	"""((x, y), on_base_color) in the image's pixels: the base's color centroid near the match, else the
	match's geometric center."""
	h, w = img_bgr.shape[:2]
	tw, th = candidate.size
	tl = (candidate.loc[0] + bounds[0], candidate.loc[1] + bounds[1])   # back in full-image coordinates

	# A tight template can leave the box mostly feather with little of the base inside it, so the base is
	# searched in the box padded beyond it (down and sideways only).
	x0, y0 = max(0, tl[0] - int(tw * CLICK_SEARCH_PADDING_X_RATIO)), max(0, tl[1] - int(th * CLICK_SEARCH_PADDING_TOP_RATIO))
	x1, y1 = min(w, tl[0] + tw + int(tw * CLICK_SEARCH_PADDING_X_RATIO)), min(h, tl[1] + th + int(th * CLICK_SEARCH_PADDING_BOTTOM_RATIO))
	_debug('click: searching for the base color in region (' + str(x0) + ',' + str(y0) + ')-(' + str(x1) + ',' + str(y1) + ')')
	point = _base_click_point(hsv[y0:y1, x0:x1])
	if point is not None:
		_debug('click: using the base color centroid ' + str((round(x0 + point[0], 1), round(y0 + point[1], 1))))
		return (x0 + point[0], y0 + point[1]), True

	fallback_point = (tl[0] + tw / 2, tl[1] + th * FALLBACK_VERTICAL_BIAS)
	_debug('click: using the matched box\'s center, ' + str(FALLBACK_VERTICAL_BIAS) + ' of the way down: ' + str((round(fallback_point[0], 1), round(fallback_point[1], 1))))
	_save_fallback_debug_snapshot(img_bgr, tl, candidate.size, fallback_point)
	return fallback_point, False


# ------------------------------------------------------------------------------ entry points

def find_float_detailed(screenshot_path) -> Optional[Detection]:
	"""The float in the screenshot with the evidence behind it, or None if not found."""
	img_bgr = cv2.imread(screenshot_path)
	if img_bgr is None:
		# e.g. a screenshot caught mid-write by a concurrent reader - treat it like "not found"
		print('Could not read screenshot: ' + screenshot_path)
		return None
	h, w = img_bgr.shape[:2]
	scale = _normalization_scale(w)
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

	evidence_masks = {EVIDENCE_GATE: _build_color_mask(hsv_band, ui_boxes), EVIDENCE_BASE: _base_color_mask(hsv_band, ui_boxes)}
	candidate = _decide(_find_candidates(search_gray, evidence_masks, ui_boxes))
	if candidate is None:
		_debug('end: float not found')
		return None

	point, on_base_color = _click_point(img_bgr, hsv, candidate, bounds)
	print('Matched ' + candidate.template + ' (score ' + str(round(candidate.score, 3)) + ')')
	if scale != 1:
		point = (point[0] / scale, point[1] / scale)   # back to the original screenshot's pixels
	_debug('end: click point ' + str((round(point[0], 1), round(point[1], 1))))
	return Detection(point, candidate.evidence, candidate.score, candidate.template, on_base_color)


def find_float(screenshot_path):
	"""Pixel (x, y) to click for the float in the screenshot, or None if not found."""
	detection = find_float_detailed(screenshot_path)
	return None if detection is None else detection.point
