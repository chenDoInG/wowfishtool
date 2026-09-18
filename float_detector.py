import glob
import os
import time

import cv2
import numpy as np

FLOAT_TEMPLATE_GLOB = 'var/fishing_float_*.png'

# Every confirmed-real match logged so far has scored 0.439 or higher (even in the
# hardest lighting fixtures). A real miss - a hazy scene where the color gate found zero
# density at the float's own true position, so only a stray water-texture match survived
# the gate at all - scored 0.330: comfortably below that floor. Sitting the threshold in
# the gap between the two means a match this weak now gets treated like "not found"
# (a quick retry) instead of confidently committing to a wrong location and then wasting
# a full listen() timeout waiting for a bite that was never coming.
FLOAT_MATCH_THRESHOLD = 0.35

# The float always lands in the water close to the character, which - because a nearby
# point on the water is lower in the view than a distant one - puts it in the lower part
# of the window, well below the horizon and anything far off on it (a distant ship, say).
# Restricting the search to this band keeps that class of thing, and repetitive
# water-ripple texture elsewhere on screen, from out-scoring the real float. Every
# confirmed-real detection logged so far has landed between 45% and 77% of the window's
# height; a nearby ship is far more likely to sit above that than a closer cast is to
# land below it, hence the lower bound sitting much closer to the observed range.
#
# The upper end of that range varies more than expected across different window/camera
# setups - one real session consistently landed casts around 74-77%, well past the 72%
# this used to stop at. That's not a color or shape problem at all: cropping the float
# out of the search band before matching even starts means no amount of tuning the color
# gate or templates can recover it, since the pixels just aren't in the region being
# analyzed. Pad the upper bound well past every real detection logged so far instead of
# exactly to it, so the next new camera setup doesn't clip the same way.
FLOAT_SEARCH_X_RANGE = (0.25, 0.75)
FLOAT_SEARCH_Y_RANGE = (0.38, 0.80)

# Player-frame clusters (portrait, name, health/mana bars) sit at fixed screen spots that
# land inside the band above - and they're exactly the kind of thing the adaptive color
# gate is meant to let through: a gold-bordered icon plus solid green/blue bars is
# comfortably more saturated than any water. On a real capture, once a genuinely
# dim/hard-to-see float got rejected by the color gate (see FLOAT_MIN_VALUE /
# FLOAT_SATURATION_MARGIN above), one of these frames was the next-highest-scoring thing
# around and won outright, confidently above FLOAT_MATCH_THRESHOLD - the bot would have
# clicked a UI element instead of the float. A screen element should never be mistaken
# for the float regardless of how the color gate happens to score elsewhere, so both are
# excluded outright rather than left to compete:
#   - the default always-on player frame, bottom-left, present in every capture
#     regardless of Edit Mode layout;
#   - WoW's current default "Modern" Edit Mode layout additionally plants a duplicate of
#     that same cluster near the bottom-right.
# Both measured off a real 2560x1410 capture; padded on every side since exact placement
# can drift a little with a different resolution or UI scale.
UI_EXCLUDE_REGIONS = (
	((0.26, 0.36), (0.72, 0.80)),
	((0.62, 0.76), (0.71, 0.83)),
)

# Water/shoreline edges, ships, and even the water itself can carry a strong, fixed
# hue - WoW tints its lighting per-zone/time-of-day (grey overcast, blue dusk, warm
# afternoon, ...), and that tint shifts wherever a fixed HSV hue/saturation range
# expects to find the float's colors to be, breaking any one fixed range sooner or
# later. What holds regardless of tint: the float's bobber+feather are always far more
# saturated than the water immediately around them, even when that water is itself
# fairly saturated (e.g. a deep blue dusk sea). So instead of a fixed color range,
# measure how saturated *this* scene's water actually is and require a pixel to clear
# that baseline by a margin to count as part of the float - adapts to the scene
# instead of needing yet another hardcoded range for the next new lighting condition.
# The baseline itself needs to sit close to the water's actual ceiling, not just above
# its typical/median pixel: a very saturated sea can plateau hard right up to its own
# 99th-plus percentile (still just water) before jumping sharply at the float's outlier
# pixels, so a lower percentile like 90 sits on that plateau and leaves too thin a
# margin to separate the two in that case.
FLOAT_SATURATION_BASELINE_PERCENTILE = 99.5
FLOAT_SATURATION_MARGIN = 20

# Guards against near-black pixels, whose saturation is numerically unstable (a tiny
# absolute BGR difference swings the max-min/max ratio wildly), registering as
# float-colored noise. Used to sit at 60, which was fine everywhere it got tested until
# a real dark-night scene: the float itself renders dim there, its own clearly-saturated
# feather pixels landing at value 39-61 - mostly *below* the old floor - while the
# water around it never exceeds ~46 even at its own 99th percentile. So brightness can't
# discriminate float from water in that scene at all (only saturation still can); the
# floor's remaining job is purely rejecting actual near-black noise, which this scene's
# water sits comfortably above (10th percentile 30). 30 admits the dimmest confirmed-real
# float pixel logged so far with a 9-unit margin and was re-verified against every other
# fixture logged before it.
FLOAT_MIN_VALUE = 30
FLOAT_MIN_COLOR_PIXELS = 15

# A big saturated structure - a dock, a ship's hull - can clear the margin above too,
# since it's a real, consistently-colored object rather than water noise. What it
# never has is the float's small footprint: real detections have topped out around a
# 46x14px blob, while a dock spans hundreds of pixels. Drop any connected blob of
# "float-colored" pixels bigger than this in either dimension before gating on density.
FLOAT_MAX_BLOB_SIZE = 200

# The adaptive check above answers "is the float here at all" and is deliberately
# hue-agnostic, but that also means it usually keys on the feather (its colors read as
# more saturated than the base's yellow/tan against most water) rather than the base -
# no good for clicking. The base's actual hue range is narrower and more predictable
# than "whatever is more saturated than the water", so click positioning still uses it
# directly; on the rare scene where none of these ranges find enough of it, the caller
# falls back to the matched window's geometric center rather than failing outright.
#
# More than one range exists because ambient lighting tints the base's color along with
# everything else - a dusk/night zone can shift it from its usual warm tan (hue ~10-35,
# strongly saturated) to a desaturated yellow-green (hue ~40-75, only weakly saturated).
# Each range is kept narrow and scene-specific rather than widening one range to cover
# both, since a wide range risks bleeding into water that happens to sit in the gap
# between them in some other scene (e.g. one daylight fixture's water itself reads at
# roughly hue 47 - right where a single, wider range would have to pass through).
FLOAT_BASE_COLOR_RANGES = (
	((10, 80, 100), (35, 255, 255)),   # normal daylight warm tan/yellow base
	((40, 30, 120), (75, 90, 255)),    # dusk/night-tinted, desaturated base
	((10, 15, 25), (35, 140, 110)),    # same warm hue as the daylight base, but dark - a
	# real dark-night capture measured its base at hue ~15-21 (squarely inside the
	# daylight range above) yet saturation/value only 25-131/34-106 - both well under
	# that range's 80/100 floors, so every cast in that session fell back to the
	# geometric-center click point instead of a real color match, even though the float
	# was clearly visible. Same hue window as the daylight range (this is that same base
	# color, just dimmed by night lighting, not a different tint), with S/V floors
	# lowered to admit it.
)
FLOAT_MIN_BASE_COLOR_PIXELS = 15

# A warm-enough dusk sea can put the *water itself* inside the daylight base range above
# over a wide, contiguous area - not just the float. _drop_large_blobs correctly strips
# that giant float+water blob out (its bounding box clears FLOAT_MAX_BLOB_SIZE), but what
# survives is then just scattered, disconnected leftover flecks that happen to also fall
# in range elsewhere in the padded region - real noise, unrelated to the float - which can
# still sum past FLOAT_MIN_BASE_COLOR_PIXELS across enough of them and produce a confident-
# looking but wrong centroid, real duller (2026-09-18 live miss: base+water fused into one
# 224x157 blob, correctly dropped; the leftover noise still summed to 68px across several
# fragments, none bigger than 23px, and centroided ~85px from the float's real position).
# Every confirmed-real detection logged so far has its single largest surviving connected
# component at 39px or more (down to a mask that's just one 39px blob with nothing else);
# the confirmed-noise case above topped out at 23px. Requiring the largest component alone
# (not the sum across all of them) to clear a floor between those two sits away from both
# without touching the sum-based centroid math real detections already rely on.
FLOAT_MIN_BASE_BLOB_AREA = 30

# How far beyond the matched template's own box to look for the base's color - see the
# comment where this is used in find_float(). Only padding downward/sideways, never
# upward: the base always sits at or below the matched box in every fixture that's
# needed padding at all, and the float's own feather/bobber sit above the base within
# the box already, so padding upward only ever risks reaching into whatever backdrop
# happens to be above the float (the Stormwind fixture has a dock up there) without
# ever helping find the base.
CLICK_SEARCH_PADDING_TOP_RATIO = 0
CLICK_SEARCH_PADDING_BOTTOM_RATIO = 0.5
CLICK_SEARCH_PADDING_X_RATIO = 0.3

# Where the base sits vertically within the matched box, as a fraction of its height -
# used only when _float_click_point() can't find the base's own color at all (a scene
# where the base blends into the water too closely for any fixed range to separate, e.g.
# max graphics quality rendering the same dusk tint far more strongly onto every object).
# The two known scenes that fall back to this were both measured against their real,
# manually-verified float position: the base sits at ~65% of the box's height, not 50% -
# the feather it's attached to occupies the upper portion, pulling the box's own vertical
# center up past the base. Horizontal centering is left alone since both scenes' real
# position landed exactly on the box's horizontal center already.
FALLBACK_VERTICAL_BIAS = 0.65

# Off by default so a normal run never touches disk for this - flip to True (fishing.py
# does this for you when its own DEBUG_SNAPSHOTS is set, see there) to save an annotated
# screenshot into DEBUG_SNAPSHOT_DIR any time the click point falls back to the matched
# box's geometric center (see the comment where this is used in find_float()), for
# reviewing after the fact instead of only when a bad catch happens to get noticed.
DEBUG_SNAPSHOTS = True
DEBUG_SNAPSHOT_DIR = 'debug'   # kept out of var/, which holds the bot's real runtime data


def _box_density(mask: np.ndarray, window_size):
	"""Per-pixel count of nonzero `mask` pixels in a window_size box anchored at that
	pixel's top-left, i.e. density[y, x] covers the same box matchTemplate's
	result[y, x] scores."""
	# mask pixels are 0 or 255, so the unnormalized box sum is 255x the actual pixel count
	density = cv2.boxFilter(mask, cv2.CV_32F, window_size, normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
	return density / 255.0


def _drop_large_blobs(mask: np.ndarray, max_size: int):
	"""Zero out connected components of `mask` wider or taller than `max_size` - real
	structures (a dock, a ship's hull) rather than the float's small bobber+feather."""
	_, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
	cleaned = mask.copy()
	for i in range(1, stats.shape[0]):
		blob_w, blob_h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
		if blob_w > max_size or blob_h > max_size:
			cleaned[labels == i] = 0
	return cleaned


def _adaptive_color_mask(hsv_region: np.ndarray):
	"""Pixels distinctly more saturated than this scene's own water, regardless of what
	hue that happens to be - see the comment on FLOAT_SATURATION_MARGIN above.

	Returns (mask, gate_reliable). In a scene saturated enough that its own 99.5th-
	percentile pixel already sits at the HSV ceiling (255), water_baseline +
	FLOAT_SATURATION_MARGIN exceeds the max representable saturation - no pixel,
	however float-colored, could ever pass the gate, regardless of the float's own
	color. gate_reliable is False only in that specific degenerate case (confirmed on a
	real miss: 17% of the search area sat at S==255, threshold came out above 255, and
	the float's own template match still scored a confident 0.659) - not whenever the
	mask merely happens to be empty, which also happens on ordinary colorless frames
	(disconnect/login/character-select screens) where the gate is working correctly and
	rejecting a real non-float scene."""
	saturation = hsv_region[:, :, 1]
	value = hsv_region[:, :, 2]
	water_baseline = np.percentile(saturation, FLOAT_SATURATION_BASELINE_PERCENTILE)
	threshold = water_baseline + FLOAT_SATURATION_MARGIN
	mask = ((saturation > threshold) & (value > FLOAT_MIN_VALUE)).astype(np.uint8) * 255
	return _drop_large_blobs(mask, FLOAT_MAX_BLOB_SIZE), threshold <= 255


def _float_click_point(bgr_region: np.ndarray):
	"""Pixel-coordinate centroid of the bobber base's color within `bgr_region`, or None
	if there aren't enough matching pixels to trust it (caller falls back to the
	geometric center in that case)."""
	hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
	mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
	for lo, hi in FLOAT_BASE_COLOR_RANGES:
		mask |= cv2.inRange(hsv, lo, hi)
	# The padded search region below can reach into a same-hued background structure
	# (e.g. the wooden dock in the Stormwind fixture) - drop any blob too big to be the
	# float's own base before centroiding, same rationale as _drop_large_blobs above.
	mask = _drop_large_blobs(mask, FLOAT_MAX_BLOB_SIZE)
	moments = cv2.moments(mask, binaryImage=True)
	if moments['m00'] < FLOAT_MIN_BASE_COLOR_PIXELS:
		return None
	# A scene where the water shares the base's color range can leave behind scattered
	# noise fragments that individually mean nothing but sum past the floor above - see
	# FLOAT_MIN_BASE_BLOB_AREA. A real base is one solid blob, so require whichever single
	# component is biggest to look like one, not just the total across however many there
	# are; the sum-based centroid below is unchanged for every scene where that holds.
	_, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
	largest_component = max((stats[i, cv2.CC_STAT_AREA] for i in range(1, stats.shape[0])), default=0)
	if largest_component < FLOAT_MIN_BASE_BLOB_AREA:
		return None
	return moments['m10'] / moments['m00'], moments['m01'] / moments['m00']


def _save_fallback_debug_snapshot(img_bgr: np.ndarray, box_tl, box_size, click_point):
	"""If DEBUG_SNAPSHOTS is on, save a copy of the screenshot (with the matched box and
	click point drawn on it) into DEBUG_SNAPSHOT_DIR, so a click that had to fall back to
	the geometric center (see find_float()) can be checked after the fact instead of only
	when the user happens to notice a bad catch. No-op otherwise."""
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


def find_float(screenshot_path):
	# Tried masked template matching (matchTemplate(..., mask=...) so the background
	# water can't affect the score) to get one "universal" background-free template
	# instead of several lighting-specific ones - scores looked great (0.97+) but
	# locations were wildly wrong (100-500px off), since a sparse color-only mask
	# throws away the float's shape and just matches any similarly-colored blob. Not
	# worth revisiting without a much more careful mask.
	img_bgr = cv2.imread(screenshot_path)
	if img_bgr is None:
		# e.g. a screenshot caught mid-write by a concurrent reader - not worth
		# crashing the whole bot over, so log it and treat it like "not found".
		print('Could not read screenshot: ' + screenshot_path)
		return None
	# noinspection PyTypeChecker
	img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	h, w = img_gray.shape[:2]

	search_x0, search_x1 = int(w * FLOAT_SEARCH_X_RANGE[0]), int(w * FLOAT_SEARCH_X_RANGE[1])
	search_y0, search_y1 = int(h * FLOAT_SEARCH_Y_RANGE[0]), int(h * FLOAT_SEARCH_Y_RANGE[1])
	search_area_gray: np.ndarray = img_gray[search_y0:search_y1, search_x0:search_x1]
	search_area_bgr: np.ndarray = img_bgr[search_y0:search_y1, search_x0:search_x1]
	search_area_hsv: np.ndarray = cv2.cvtColor(search_area_bgr, cv2.COLOR_BGR2HSV)
	color_mask, color_gate_active = _adaptive_color_mask(search_area_hsv)

	# Blank out each known UI player-frame's fixed spot (see UI_EXCLUDE_REGIONS above)
	# within the search area's own local coordinates, clipped to whatever part of it
	# actually falls inside the search band, so no candidate window anchored there can
	# ever read as float-colored.
	for (x_range, y_range) in UI_EXCLUDE_REGIONS:
		ui_x0 = max(0, int(w * x_range[0]) - search_x0)
		ui_x1 = min(search_x1 - search_x0, int(w * x_range[1]) - search_x0)
		ui_y0 = max(0, int(h * y_range[0]) - search_y0)
		ui_y1 = min(search_y1 - search_y0, int(h * y_range[1]) - search_y0)
		if ui_x1 > ui_x0 and ui_y1 > ui_y0:
			color_mask[ui_y0:ui_y1, ui_x0:ui_x1] = 0

	# A moonlit/choppy-water scene can push the water's own 99.5th-percentile saturation
	# high enough that literally nothing outside the excluded UI frames ever clears
	# threshold - not just the float, everything, including water pixels that would
	# ordinarily anchor a false positive elsewhere. Confirmed on a real capture: baseline
	# 219 (threshold 239) against the float's own peak of 184, and zero pixels passing
	# anywhere else in the whole search band either. Same failure shape as the
	# saturation-ceiling-clip case above (the gate ends up unable to discriminate
	# anything in this specific scene) even though the threshold itself never numerically
	# exceeds 255 - so it gets the same treatment: stop trusting a gate that is
	# demonstrably not separating float from water here, and fall back to the grayscale
	# shape match alone, same as the >255 case already does.
	if color_gate_active and not np.any(color_mask):
		color_gate_active = False

	best_val = 0
	best_loc = None
	best_size = None
	best_template = None
	# Tracked alongside the gated result (see below) rather than only computed on demand,
	# so picking it never costs a second pass over the templates.
	raw_best_val = 0
	raw_best_loc = None
	raw_best_size = None
	raw_best_template = None
	for template_path in sorted(glob.glob(FLOAT_TEMPLATE_GLOB)):
		template = cv2.imread(template_path, 0)
		if template is None:
			continue
		th, tw = template.shape[:2]
		result = cv2.matchTemplate(search_area_gray, template, cv2.TM_CCOEFF_NORMED)

		_, raw_max_val, _, raw_max_loc = cv2.minMaxLoc(result)
		if raw_max_val > raw_best_val:
			raw_best_val, raw_best_loc, raw_best_size, raw_best_template = raw_max_val, raw_max_loc, (tw, th), template_path

		# Water/shoreline edges can score just as well as the real float on pure grayscale
		# correlation, but the water is never as saturated/colorful as the float's bobber
		# base - rule out any position that doesn't have enough of that color nearby
		# before picking the best-scoring one.
		if color_gate_active:
			rh, rw = result.shape
			color_density = _box_density(color_mask, (tw, th))[:rh, :rw]
			result[color_density < FLOAT_MIN_COLOR_PIXELS] = -1

		_, max_val, _, max_loc = cv2.minMaxLoc(result)
		if max_val > best_val:
			best_val, best_loc, best_size, best_template = max_val, max_loc, (tw, th), template_path

	# A scene can be saturated enough overall that *some* pixel far from the float clears
	# the adaptive threshold (so the "gate found nothing anywhere" escape hatch above never
	# fires) while the float's own, comparatively duller base in this same lighting never
	# does - the gate still zeroes out the float's true position, same practical effect as
	# a fully broken gate, just not visible as a globally empty mask. Confirmed on a real
	# capture (see the warm_dusk_gate_miss fixture): the float's own position scored 0.53-
	# 0.7 ungated - solidly within the confirmed-real range noted on FLOAT_MATCH_THRESHOLD
	# above - yet every gated candidate topped out at 0.24-0.34 because whatever scattered
	# pixels elsewhere kept the gate "active" didn't line up with anything float-shaped.
	# Same remedy as the fully-empty-mask case: stop trusting a gate that isn't separating
	# float from water here either, and fall back to the grayscale shape match alone.
	if color_gate_active and best_val <= FLOAT_MATCH_THRESHOLD and raw_best_val > FLOAT_MATCH_THRESHOLD:
		best_val, best_loc, best_size, best_template = raw_best_val, raw_best_loc, raw_best_size, raw_best_template

	if best_val <= FLOAT_MATCH_THRESHOLD or best_loc is None or best_template is None or best_size is None:
		return None

	print('Matched ' + best_template + ' (score ' + str(round(best_val, 3)) + ')')

	tw, th = best_size
	tl = (best_loc[0] + search_x0, best_loc[1] + search_y0)   # top-left, back in full-screenshot coordinates

	# The tighter templates match a smaller, more exact silhouette, so a slightly
	# imperfect grayscale alignment can leave the matched box mostly containing the
	# feather with little or none of the base actually inside it. Pad the region the
	# click point is searched in beyond the exact matched box so the base is still
	# reachable even when the match itself is a bit off.
	pad_x = int(tw * CLICK_SEARCH_PADDING_X_RATIO)
	pad_top = int(th * CLICK_SEARCH_PADDING_TOP_RATIO)
	pad_bottom = int(th * CLICK_SEARCH_PADDING_BOTTOM_RATIO)
	click_x0, click_y0 = max(0, tl[0] - pad_x), max(0, tl[1] - pad_top)
	click_x1, click_y1 = min(w, tl[0] + tw + pad_x), min(h, tl[1] + th + pad_bottom)
	matched_region: np.ndarray = img_bgr[click_y0:click_y1, click_x0:click_x1]
	click_point = _float_click_point(matched_region)
	if click_point is not None:
		return click_x0 + click_point[0], click_y0 + click_point[1]

	fallback_point = (tl[0] + tw / 2, tl[1] + th * FALLBACK_VERTICAL_BIAS)
	_save_fallback_debug_snapshot(img_bgr, tl, best_size, fallback_point)
	return fallback_point
