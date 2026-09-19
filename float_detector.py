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
import math
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
FLOAT_MAX_BASE_BLOB_SHARE = 0.5      # the click point's base blob must not fill its search region: a real base is 1-9% of it, scenery 77-95%
FLOAT_MAX_BASE_SCENE_SHARE = 0.15    # base color is ignored as evidence when this much of the whole band falls inside its ranges:
# then the water or terrain shares its hue and the ranges say nothing about where the float is. Measured on 14 scenes: 0-4.8%
# where it is informative, 21.7-88.2% where it is not (a daytime sea, a warm dusk, teal shallows).

# Grayscale windows flatter than this are not treated as a float: featureless water still correlates at 0.6-0.7.
# Windows on empty water measured a standard deviation of 0.2-4.1, real float windows 6.0 and up. The exception is a
# float still fading in after the cast (std 3.0-4.7 on 3 of 5 measured): it is excluded until it has fully appeared,
# so the caller's retry finds it a moment later. The floor sits in the gap and is thin on both sides.
FLOAT_MIN_TEXTURE = 6

# Evidence 3, agreement between templates: a real float is matched by several differently sized templates at the same
# place, while a rippled surface usually matches each of them somewhere else. FLOAT_MIN_AGREEING_TEMPLATES templates whose
# best windows lie within FLOAT_AGREEMENT_RADIUS px of each other, each clearing FLOAT_MATCH_THRESHOLD, back a candidate with
# no color at all - what a daytime sea whose water shares the base's hue leaves, where the real float (0.55-0.59, three
# templates agreeing) had no color evidence and the "evidence" pointed at a cliff (0.41). Measured: on all 11 real-float
# images the largest agreeing group sits on the float; on empty water it reached 3 in 1 of 20 synthetic frames, so it
# ranks below specific base color.
FLOAT_MIN_AGREEING_TEMPLATES = 3
FLOAT_MIN_AGREEING_SHARE = 0.4   # ... and at least this share of the templates in play, so that adding templates (the README tells you to)
# raises the bar instead of making agreement easier. With the 7 templates measured this is the same 3 (ceil(0.4 * 7)); at 4 of 7 it would
# have found only 18% of the real floats, and with fewer templates agreement almost never fires (6: 55%, 5: 31%, 4: 14%).
FLOAT_AGREEMENT_RADIUS = 45

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

# The kinds of evidence, strongest first. A window is "supported" by a color kind when it holds at least
# FLOAT_MIN_COLOR_PIXELS pixels of it; EVIDENCE_AGREEMENT needs no color (see FLOAT_MIN_AGREEING_TEMPLATES) and ranks
# last: it is the one with no independent physical signal behind it.
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
		"""Whether a second, independent kind of evidence backs the place. Real floats usually have one (9 of 11
		measured); every false positive measured had none. A pick without one is low confidence."""
		return len(self.corroborated_by) > 0


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
	"""Evidence 2 over the band: base-colored pixels, UI frames blanked, oversized blobs dropped. Empty when the
	scene floods the ranges (FLOAT_MAX_BASE_SCENE_SHARE): a color that holds nearly everywhere locates nothing, and
	filtering windows one by one would only leave the wrong ones - so the whole kind steps aside."""
	mask = _base_range_mask(hsv_band)
	_blank_ui(mask, ui_boxes)
	scene_share = np.count_nonzero(mask) / mask.size
	if scene_share > FLOAT_MAX_BASE_SCENE_SHARE:
		_debug('base color: ' + str(round(100 * scene_share)) + '% of the band is inside the base color ranges - the scene shares the hue, ignoring it as evidence')
		return np.zeros_like(mask)
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
	shape_bests = []   # each template's best window by shape alone (after exclusions), for the agreement evidence
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

		_, shape_val, _, shape_loc = cv2.minMaxLoc(result)
		shape_bests.append(Candidate(shape_val, shape_loc, (tw, th), template_path, EVIDENCE_AGREEMENT))
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
	best[EVIDENCE_AGREEMENT] = _agreeing_candidate(shape_bests)
	return best


def _agreeing_candidate(shape_bests):
	"""The best-scoring member of the largest group of templates whose best windows agree on where the float is,
	or None when no group is big enough: at least FLOAT_MIN_AGREEING_TEMPLATES templates and at least
	FLOAT_MIN_AGREEING_SHARE of all the templates that were matched."""
	def center(c):
		return c.loc[0] + c.size[0] / 2, c.loc[1] + c.size[1] / 2
	group_of = []
	for c in shape_bests:
		cx, cy = center(c)
		group_of.append([m for m in shape_bests if m.score > FLOAT_MATCH_THRESHOLD and c.score > FLOAT_MATCH_THRESHOLD
						 and (center(m)[0] - cx) ** 2 + (center(m)[1] - cy) ** 2 <= FLOAT_AGREEMENT_RADIUS ** 2])
	group = max(group_of, key=lambda g: (len(g), max((m.score for m in g), default=-1)), default=[])
	needed = max(FLOAT_MIN_AGREEING_TEMPLATES, math.ceil(round(FLOAT_MIN_AGREEING_SHARE * len(shape_bests), 9)))
	if len(group) < needed:
		_debug('agreement: no ' + str(needed) + ' of ' + str(len(shape_bests)) + ' templates agree on a place (largest group ' + str(len(group)) + ')')
		return None
	winner = max(group, key=lambda m: m.score)
	_debug('agreement: ' + str(len(group)) + ' templates agree near ' + str(tuple(round(v) for v in center(winner))) + ' - best ' + winner.template
		+ ' ' + str(round(winner.score, 3)))
	return winner


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
	"""The other kinds of evidence whose best candidate clears the threshold within FLOAT_AGREEMENT_RADIUS px of
	the chosen one: independent signals (relative saturation, base color, agreement between templates) that land
	on the same place are far stronger than the same three kinds scattered over the band."""
	def center(c):
		return c.loc[0] + c.size[0] / 2, c.loc[1] + c.size[1] / 2
	cx, cy = center(chosen)
	return tuple(evidence for evidence in EVIDENCE_PRIORITY
				 if evidence != chosen.evidence and candidates.get(evidence) is not None
				 and candidates[evidence].score > FLOAT_MATCH_THRESHOLD
				 and (center(candidates[evidence])[0] - cx) ** 2 + (center(candidates[evidence])[1] - cy) ** 2 <= FLOAT_AGREEMENT_RADIUS ** 2)


# ------------------------------------------------------------------------------ 5. click point

def _base_click_point(hsv_region: np.ndarray):
	"""Centroid of the bobber base's color within `hsv_region` (region coordinates), or None when there is no
	single solid base-colored blob to trust. Each FLOAT_BASE_COLOR_RANGES range is tried on its own, in order: the
	union floods when one range matches the water (it fuses base and water into a blob that is dropped as scenery),
	while the range that fits this lighting alone leaves the base as one compact blob. A blob qualifies when it clears
	the floor (scattered flecks that sum to a plausible total are what water sharing the hue leaves behind) and does
	not fill the region (that is a surface, not a base)."""
	rejections = []
	for lo, hi in FLOAT_BASE_COLOR_RANGES:
		mask = cv2.morphologyEx(cv2.inRange(hsv_region, lo, hi), cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
		mask, largest_component = _drop_small_blobs_stats(mask, FLOAT_MAX_BLOB_SIZE)
		if largest_component < FLOAT_MIN_BASE_BLOB_AREA:
			rejections.append(str(largest_component) + ' px')
		elif largest_component > FLOAT_MAX_BASE_BLOB_SHARE * mask.size:
			rejections.append(str(largest_component) + ' px = scenery')
		else:
			moments = cv2.moments(mask, binaryImage=True)
			_debug('click: base color found (largest blob ' + str(largest_component) + ' px, ' + str(int(moments['m00'])) + ' px total)')
			return moments['m10'] / moments['m00'], moments['m01'] / moments['m00']
	_debug('click: base color not found (largest blob per range: ' + ', '.join(rejections) + '; need ' + str(FLOAT_MIN_BASE_BLOB_AREA) + ' and under ' + str(round(100 * FLOAT_MAX_BASE_BLOB_SHARE)) + '% of the region)')
	return None


def _save_debug_snapshot(img_bgr: np.ndarray, box_tl, box_size, click_point, prefix: str, label: str):
	"""If DEBUG_SNAPSHOTS is on, save the screenshot into DEBUG_SNAPSHOT_DIR with the matched box, the click point and
	`label` drawn on it, so a pick worth a second look can be checked after the fact. Returns the path, or None."""
	if not DEBUG_SNAPSHOTS:
		return None
	tw, th = box_size
	annotated = img_bgr.copy()
	cv2.rectangle(annotated, box_tl, (box_tl[0] + tw, box_tl[1] + th), (0, 255, 0), 2)
	cv2.circle(annotated, (int(click_point[0]), int(click_point[1])), 6, (0, 0, 255), -1)
	cv2.putText(annotated, label, (box_tl[0], max(20, box_tl[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
	path = os.path.join(DEBUG_SNAPSHOT_DIR, prefix + '_' + str(int(time.time() * 1000)) + '.png')   # ms: two saves in one second must not overwrite each other
	try:
		os.makedirs(DEBUG_SNAPSHOT_DIR, exist_ok=True)
		if not cv2.imwrite(path, annotated):
			raise OSError('cv2.imwrite returned False')
	except (OSError, cv2.error) as error:
		# a debugging aid must never take detection down: an unwritable folder or a full disk just loses the picture
		print('Could not save the debug snapshot into ' + DEBUG_SNAPSHOT_DIR + ': ' + str(error))
		return None
	return path


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
	if scale != 1 and min(round(w * scale), round(h * scale)) < 1:
		_debug('end: a ' + str(w) + 'x' + str(h) + ' screenshot would resize to nothing - not found')
		return None   # an absurd aspect ratio (nothing a game window can produce); cv2.resize would raise
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
	candidates = _find_candidates(search_gray, evidence_masks, ui_boxes)
	candidate = _decide(candidates)
	if candidate is None:
		_debug('end: float not found')
		return None

	point, on_base_color = _click_point(img_bgr, hsv, candidate, bounds)
	corroborated_by = _corroborating_evidence(candidate, candidates)
	summary = candidate.evidence + ', ' + ('corroborated by ' + ' + '.join(corroborated_by) if corroborated_by else 'NOT corroborated by any other evidence')
	print('Matched ' + candidate.template + ' (score ' + str(round(candidate.score, 3)) + ') - ' + summary)

	# Picks that leave the best evidence behind are where the mistakes are: a click that fell back to the box center, or
	# a match no second kind of evidence backs. Keep a picture of each so they can be reviewed without a live log.
	box_tl = (candidate.loc[0] + bounds[0], candidate.loc[1] + bounds[1])
	label = candidate.evidence + ' ' + str(round(candidate.score, 2)) + (' corroborated' if corroborated_by else ' UNCORROBORATED')
	if not on_base_color:
		path = _save_debug_snapshot(img_bgr, box_tl, candidate.size, point, 'fallback', label)
		if path:
			print('Click point fell back to the matched box\'s center - saved ' + path + ' for review')
	elif not corroborated_by:
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
