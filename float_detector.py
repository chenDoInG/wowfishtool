import glob

import cv2
import numpy as np

FLOAT_TEMPLATE_GLOB = 'var/fishing_float_*.png'
FLOAT_MATCH_THRESHOLD = 0.3

# The float always lands in the water close to the character, which - because a nearby
# point on the water is lower in the view than a distant one - puts it in the lower part
# of the window, well below the horizon and anything far off on it (a distant ship, say).
# Restricting the search to this band keeps that class of thing, and repetitive
# water-ripple texture elsewhere on screen, from out-scoring the real float. Every
# confirmed-real detection logged so far has landed between 45% and 60% of the window's
# height; a nearby ship is far more likely to sit above that than a closer cast is to
# land below it, hence the lower bound sitting much closer to the observed range.
FLOAT_SEARCH_X_RANGE = (0.25, 0.75)
FLOAT_SEARCH_Y_RANGE = (0.38, 0.72)

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
FLOAT_MIN_VALUE = 60   # ignore dark/shadowed pixels regardless of saturation
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
)
FLOAT_MIN_BASE_COLOR_PIXELS = 15

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
	hue that happens to be - see the comment on FLOAT_SATURATION_MARGIN above."""
	saturation = hsv_region[:, :, 1]
	value = hsv_region[:, :, 2]
	water_baseline = np.percentile(saturation, FLOAT_SATURATION_BASELINE_PERCENTILE)
	threshold = water_baseline + FLOAT_SATURATION_MARGIN
	mask = ((saturation > threshold) & (value > FLOAT_MIN_VALUE)).astype(np.uint8) * 255
	return _drop_large_blobs(mask, FLOAT_MAX_BLOB_SIZE)


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
	return moments['m10'] / moments['m00'], moments['m01'] / moments['m00']


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
	color_mask = _adaptive_color_mask(search_area_hsv)

	best_val = 0
	best_loc = None
	best_size = None
	best_template = None
	for template_path in sorted(glob.glob(FLOAT_TEMPLATE_GLOB)):
		template = cv2.imread(template_path, 0)
		if template is None:
			continue
		th, tw = template.shape[:2]
		result = cv2.matchTemplate(search_area_gray, template, cv2.TM_CCOEFF_NORMED)

		# Water/shoreline edges can score just as well as the real float on pure grayscale
		# correlation, but the water is never as saturated/colorful as the float's bobber
		# base - rule out any position that doesn't have enough of that color nearby
		# before picking the best-scoring one.
		rh, rw = result.shape
		color_density = _box_density(color_mask, (tw, th))[:rh, :rw]
		result[color_density < FLOAT_MIN_COLOR_PIXELS] = -1

		_, max_val, _, max_loc = cv2.minMaxLoc(result)
		if max_val > best_val:
			best_val, best_loc, best_size, best_template = max_val, max_loc, (tw, th), template_path

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
	return tl[0] + tw / 2, tl[1] + th * FALLBACK_VERTICAL_BIAS
