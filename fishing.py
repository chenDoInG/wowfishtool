import glob
import threading
import time

import cv2
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

# Water/shoreline edges, and other objects like passing ships, occasionally out-score
# the real float on pure grayscale template matching. The float's orange/yellow bobber
# base is reliably strongly saturated in every lighting condition seen so far - plain
# water never has it - so requiring a minimum here rules that class of false positive
# out regardless of how the water happens to look. The feather's blue used to be
# required too, but under backlit/overcast lighting its saturation can wash out to
# near the water's own noise floor, making it unreliable as a second, independent
# check; the distant-ship case that check also caught is instead handled by
# FLOAT_SEARCH_Y_RANGE excluding anything near the horizon.
FLOAT_WARM_COLOR_RANGE = ((0, 90, 90), (30, 255, 255))
FLOAT_MIN_WARM_PIXELS = 15

# The feather sticks out from the bobber base at an angle that differs per template
# (and isn't fixed relative to the template's bounding box), so a fixed click-offset
# ratio doesn't generalize across templates - one template's correct offset visibly
# overshoots past the bobber on another. Instead, click the centroid of just the
# base's narrower yellow/orange color range within the matched region of the actual
# screenshot, which finds the real bobber regardless of which template matched.
# The lower bound is set above the red feather tip's hue (~0-8) so a feather that
# happens to be larger/brighter than the base in a given screenshot doesn't pull the
# centroid off the base and onto the feather's tip, while still being low enough to
# pick up the base's own reddish-orange edge when the match window is a bit off-center
# and the purer yellow-orange center of the base falls outside it.
FLOAT_BASE_COLOR_RANGE = ((10, 80, 100), (35, 255, 255))
FLOAT_MIN_BASE_COLOR_PIXELS = 15

game_window_bbox = None   # (left, top, right, bottom) of the WoW window in absolute screen coords

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


def send_float():
	print('Sending float')
	pyautogui.press('1')
	time.sleep(2)


def jump():
	time.sleep(1)


def locate_float():
	"""Screenshot the game window and return the (x, y) position of the float in it, or None."""
	locate_game_window()
	screenshot_path = SCREENSHOT_PATH if not dev else 'var/fishing_session_' + str(int(time.time())) + '.png'
	ImageGrab.grab(game_window_bbox).save(screenshot_path)
	return find_float(screenshot_path)


def _color_range_density(bgr_region, window_size, color_range):
	"""Per-pixel count of pixels matching `color_range` in a window_size box anchored at
	that pixel's top-left, i.e. density[y, x] covers the same box matchTemplate's
	result[y, x] scores."""
	hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, color_range[0], color_range[1])
	# mask pixels are 0 or 255, so the unnormalized box sum is 255x the actual pixel count
	density = cv2.boxFilter(mask, cv2.CV_32F, window_size, normalize=False, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
	return density / 255.0


def _float_base_color_centroid(bgr_region):
	"""Pixel-coordinate centroid of the bobber base's color within `bgr_region`, or None
	if there aren't enough matching pixels to trust (falls back to the geometric center)."""
	hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, FLOAT_BASE_COLOR_RANGE[0], FLOAT_BASE_COLOR_RANGE[1])
	moments = cv2.moments(mask, binaryImage=True)
	if moments['m00'] < FLOAT_MIN_BASE_COLOR_PIXELS:
		return None
	return moments['m10'] / moments['m00'], moments['m01'] / moments['m00']


def find_float(screenshot_path):
	# todo: maybe make some universal float without background?
	img_bgr = cv2.imread(screenshot_path)
	img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	h, w = img_gray.shape[:2]

	search_x0, search_x1 = int(w * FLOAT_SEARCH_X_RANGE[0]), int(w * FLOAT_SEARCH_X_RANGE[1])
	search_y0, search_y1 = int(h * FLOAT_SEARCH_Y_RANGE[0]), int(h * FLOAT_SEARCH_Y_RANGE[1])
	search_area_gray = img_gray[search_y0:search_y1, search_x0:search_x1]
	search_area_bgr = img_bgr[search_y0:search_y1, search_x0:search_x1]

	best_val = 0
	best_loc = None
	best_size = None
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
		warm_density = _color_range_density(search_area_bgr, (tw, th), FLOAT_WARM_COLOR_RANGE)[:rh, :rw]
		result[warm_density < FLOAT_MIN_WARM_PIXELS] = -1

		_, max_val, _, max_loc = cv2.minMaxLoc(result)
		if max_val > best_val:
			best_val, best_loc, best_size = max_val, max_loc, (tw, th)

	if best_val <= FLOAT_MATCH_THRESHOLD or best_loc is None:
		return None

	tw, th = best_size
	tl = (best_loc[0] + search_x0, best_loc[1] + search_y0)   # top-left, back in full-screenshot coordinates
	matched_region = img_bgr[tl[1]:tl[1] + th, tl[0]:tl[0] + tw]
	base_center = _float_base_color_centroid(matched_region)
	if base_center is not None:
		return tl[0] + base_center[0], tl[1] + base_center[1]
	return tl[0] + tw / 2, tl[1] + th / 2


def move_mouse(place, duration=0.3, quiet=False):
	x, y = place[0], place[1]
	if not quiet:
		print("Moving cursor to float at " + str(place))
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
	send_float()
	if stop_requested.is_set():
		return False

	place = locate_float()
	if not place:
		print('Float was not found, retrying in 3 seconds')
		time.sleep(3)
		place = locate_float()
		if not place:
			print('Still can\'t find float, giving up on this cast')
			jump()
			return False

	move_mouse(place)
	if not listen(stop_event=stop_requested):
		print('Didn\'t hear a bite, trying again')
		jump()
		return False
	if stop_requested.is_set():
		return False

	snatch(place)
	time.sleep(1)
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
