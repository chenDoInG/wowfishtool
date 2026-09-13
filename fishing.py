import glob
import threading
import time

import cv2
import numpy as np
import psutil
import pyautogui
import pygetwindow as gw
import pyscreenshot as ImageGrab
from pynput import keyboard

from audio_listener import listen

dev = False

WOW_PROCESS_NAMES = ["Wow.exe", "World of Warcraft", "World of Warcraft Classic"]

SCREENSHOT_PATH = 'var/fishing_session.png'
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
# directly; on the rare scene where this range doesn't find enough of it, the caller
# falls back to the matched window's geometric center rather than failing outright.
FLOAT_BASE_COLOR_RANGE = ((10, 80, 100), (35, 255, 255))
FLOAT_MIN_BASE_COLOR_PIXELS = 15

CAST_KEY = '1'   # fishing rod's action bar slot
BAIT_KEY = '2'   # in-game macro that re-lures the fishing pole
BAIT_REAPPLY_INTERVAL_SECONDS = 10 * 60 + 15   # a little past the lure's actual duration, so it never gets reapplied while the old one still has time left

game_window_bbox = None   # (left, top, right, bottom) of the WoW window in absolute screen coords
last_bait_time = None     # time.time() of the last bait application, or None if not yet applied

fishing_active = threading.Event()   # set while the fishing loop should be running
stop_requested = threading.Event()   # set when F11 asks the current session to stop


def on_key_press(key):
	if key == keyboard.Key.f10:
		if not fishing_active.is_set():
			print('F10 pressed: starting fishing loop')
			stop_requested.clear()
			fishing_active.set()
	elif key == keyboard.Key.f11:
		if fishing_active.is_set():
			print('F11 pressed: stopping fishing loop')
			stop_requested.set()


def start_hotkey_listener():
	listener = keyboard.Listener(on_press=on_key_press)
	listener.daemon = True
	listener.start()
	return listener


def is_wow_running():
	for pid in psutil.pids():
		try:
			name = psutil.Process(pid).name()
		except (psutil.NoSuchProcess, psutil.AccessDenied):
			continue
		if any(w in name for w in WOW_PROCESS_NAMES):
			print(name)
			return True
	return False


def check_process():
	print('Checking WoW is running')
	running = is_wow_running()
	if not running and not dev:
		print('WoW is not running')
		exit()
	print('WoW is running')
	return running


def locate_game_window():
	"""Refresh game_window_bbox from the WoW window's current position/size.

	Called before every cast, not just once at startup, since the window can be
	moved or resized mid-session - screenshotting a stale bbox then just captures
	whatever else is now sitting at that old location on screen.
	"""
	global game_window_bbox
	previous_bbox = game_window_bbox
	titles = [t for t in gw.getAllTitles() if t and ('魔兽世界' in t or 'warcraft' in t.lower())]
	if titles:
		title = titles[0]
		left, top, width, height = gw.getWindowGeometry(title)
		game_window_bbox = (int(left), int(top), int(left + width), int(top + height))
		if game_window_bbox != previous_bbox:
			print('Found WoW window "' + title + '" at ' + str(game_window_bbox))
	else:
		print('Could not find the WoW window, falling back to capturing the whole screen')
		screen = ImageGrab.grab()
		game_window_bbox = (0, 0, screen.size[0], screen.size[1])


def maybe_reapply_bait():
	"""Press BAIT_KEY (bound in-game to a macro that re-lures the fishing pole) the
	first time this runs and again every BAIT_REAPPLY_INTERVAL_SECONDS after that."""
	global last_bait_time
	if last_bait_time is not None and time.time() - last_bait_time < BAIT_REAPPLY_INTERVAL_SECONDS:
		return
	print('Applying fishing lure')
	pyautogui.press(BAIT_KEY)
	last_bait_time = time.time()
	stop_requested.wait(2)


def send_float():
	print('Sending float')
	pyautogui.press(CAST_KEY)
	# The cast animation takes about this long before the bobber actually lands and
	# settles in the water - screenshotting any earlier catches it still mid-air/still
	# animating in, which find_float() then just fails to match. If it starts missing
	# the float often (check the "Xs since cast" logged on a successful find), this
	# needs to go back up rather than lower.
	stop_requested.wait(1.5)


def jump():
	stop_requested.wait(1)


def locate_float():
	"""Screenshot the game window and return the (x, y) position of the float in it, or None."""
	locate_game_window()
	screenshot_path = SCREENSHOT_PATH if not dev else 'var/fishing_session_' + str(int(time.time())) + '.png'
	ImageGrab.grab(game_window_bbox).save(screenshot_path)
	return find_float(screenshot_path)


def _box_density(mask, window_size):
	"""Per-pixel count of nonzero `mask` pixels in a window_size box anchored at that
	pixel's top-left, i.e. density[y, x] covers the same box matchTemplate's
	result[y, x] scores."""
	# mask pixels are 0 or 255, so the unnormalized box sum is 255x the actual pixel count
	density = cv2.boxFilter(mask, cv2.CV_32F, window_size, normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
	return density / 255.0


def _drop_large_blobs(mask, max_size):
	"""Zero out connected components of `mask` wider or taller than `max_size` - real
	structures (a dock, a ship's hull) rather than the float's small bobber+feather."""
	_, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
	cleaned = mask.copy()
	for i in range(1, stats.shape[0]):
		blob_w, blob_h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
		if blob_w > max_size or blob_h > max_size:
			cleaned[labels == i] = 0
	return cleaned


def _adaptive_color_mask(hsv_region):
	"""Pixels distinctly more saturated than this scene's own water, regardless of what
	hue that happens to be - see the comment on FLOAT_SATURATION_MARGIN above."""
	saturation = hsv_region[:, :, 1]
	value = hsv_region[:, :, 2]
	water_baseline = np.percentile(saturation, FLOAT_SATURATION_BASELINE_PERCENTILE)
	threshold = water_baseline + FLOAT_SATURATION_MARGIN
	mask = ((saturation > threshold) & (value > FLOAT_MIN_VALUE)).astype(np.uint8) * 255
	return _drop_large_blobs(mask, FLOAT_MAX_BLOB_SIZE)


def _float_click_point(bgr_region):
	"""Pixel-coordinate centroid of the bobber base's color within `bgr_region`, or None
	if there aren't enough matching pixels to trust it (caller falls back to the
	geometric center in that case)."""
	hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, FLOAT_BASE_COLOR_RANGE[0], FLOAT_BASE_COLOR_RANGE[1])
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
	img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	h, w = img_gray.shape[:2]

	search_x0, search_x1 = int(w * FLOAT_SEARCH_X_RANGE[0]), int(w * FLOAT_SEARCH_X_RANGE[1])
	search_y0, search_y1 = int(h * FLOAT_SEARCH_Y_RANGE[0]), int(h * FLOAT_SEARCH_Y_RANGE[1])
	search_area_gray = img_gray[search_y0:search_y1, search_x0:search_x1]
	search_area_bgr = img_bgr[search_y0:search_y1, search_x0:search_x1]
	search_area_hsv = cv2.cvtColor(search_area_bgr, cv2.COLOR_BGR2HSV)
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

	if best_val <= FLOAT_MATCH_THRESHOLD or best_loc is None:
		return None

	print('Matched ' + best_template + ' (score ' + str(round(best_val, 3)) + ')')

	tw, th = best_size
	tl = (best_loc[0] + search_x0, best_loc[1] + search_y0)   # top-left, back in full-screenshot coordinates
	matched_region = img_bgr[tl[1]:tl[1] + th, tl[0]:tl[0] + tw]
	click_point = _float_click_point(matched_region)
	if click_point is not None:
		return tl[0] + click_point[0], tl[1] + click_point[1]
	return tl[0] + tw / 2, tl[1] + th / 2


def move_mouse(place, duration=0.3, quiet=False, elapsed_since_cast=None):
	x, y = place
	if not quiet:
		msg = "Moving cursor to float at " + str(place)
		if elapsed_since_cast is not None:
			msg += " (%.1fs since cast)" % elapsed_since_cast
		print(msg)
	offset_x, offset_y = game_window_bbox[0], game_window_bbox[1]
	pyautogui.moveTo(offset_x + x, offset_y + y, duration=duration)


def snatch(place):
	print('Snatching!')
	# The mouse has usually been sitting still on the float for up to 20s while
	# listen() waited. WoW seems to only refresh what object is under the cursor
	# when it sees a real mouse-move event, and the float's bobbing animation can
	# drift it out from under a cursor that hasn't moved in a while, so a click
	# right now can land on plain water even though the cursor looks right on it.
	# Re-issuing the move immediately before clicking forces a fresh pick.
	move_mouse(place, duration=0, quiet=True)
	pyautogui.click(button='right')


def fish_once():
	"""Cast, wait for a bite and try to catch it. Returns True if a fish was caught."""
	maybe_reapply_bait()
	if stop_requested.is_set():
		return False

	cast_time = time.time()
	send_float()
	if stop_requested.is_set():
		return False

	place = locate_float()
	if not place:
		print('Float was not found, retrying in 3 seconds')
		stop_requested.wait(3)
		place = locate_float()
		if not place:
			print('Still can\'t find float, giving up on this cast')
			jump()
			return False

	move_mouse(place, elapsed_since_cast=time.time() - cast_time)
	if not listen(stop_event=stop_requested):
		print('Didn\'t hear a bite, trying again')
		jump()
		return False
	if stop_requested.is_set():
		return False

	snatch(place)
	stop_requested.wait(1)
	return True


def main():
	if check_process() and not dev:
		print("Waiting 2 seconds, so you can switch to WoW")
		time.sleep(2)

	locate_game_window()
	start_hotkey_listener()
	print('Press F10 to start fishing, F11 to stop')

	while not dev:
		fishing_active.wait()
		if stop_requested.is_set():
			fishing_active.clear()
			continue

		catched = 0
		while fishing_active.is_set() and not stop_requested.is_set():
			if fish_once():
				catched += 1

		print('catched ' + str(catched))
		fishing_active.clear()
		print('Stopped. Press F10 to start again.')


if __name__ == '__main__':
	main()
