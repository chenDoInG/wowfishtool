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

# The float always lands in the water roughly in front of the character, which is this
# central band of the window. Restricting the search to it keeps repetitive water-ripple
# texture elsewhere on screen from occasionally out-scoring the real float.
FLOAT_SEARCH_X_RANGE = (0.25, 0.75)
FLOAT_SEARCH_Y_RANGE = (0.20, 0.60)

# The template's bounding box is pulled left of the actual bobber because the feather
# sticks out to its left; nudge the click point right by this fraction of the matched
# template's width to land on the bobber instead of its edge.
FLOAT_CLICK_X_OFFSET_RATIO = 0.25

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
	global game_window_bbox
	titles = [t for t in gw.getAllTitles() if t and ('魔兽世界' in t or 'warcraft' in t.lower())]
	if titles:
		title = titles[0]
		left, top, width, height = gw.getWindowGeometry(title)
		game_window_bbox = (int(left), int(top), int(left + width), int(top + height))
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
	screenshot_path = SCREENSHOT_PATH if not dev else 'var/fishing_session_' + str(int(time.time())) + '.png'
	ImageGrab.grab(game_window_bbox).save(screenshot_path)
	return find_float(screenshot_path)


def find_float(screenshot_path):
	# todo: maybe make some universal float without background?
	img_rgb = cv2.imread(screenshot_path)
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	h, w = img_gray.shape[:2]

	search_x0, search_x1 = int(w * FLOAT_SEARCH_X_RANGE[0]), int(w * FLOAT_SEARCH_X_RANGE[1])
	search_y0, search_y1 = int(h * FLOAT_SEARCH_Y_RANGE[0]), int(h * FLOAT_SEARCH_Y_RANGE[1])
	search_area = img_gray[search_y0:search_y1, search_x0:search_x1]

	best_val = 0
	best_loc = None
	best_size = None
	for template_path in sorted(glob.glob(FLOAT_TEMPLATE_GLOB)):
		template = cv2.imread(template_path, 0)
		if template is None:
			continue
		th, tw = template.shape[:2]
		result = cv2.matchTemplate(search_area, template, cv2.TM_CCOEFF_NORMED)
		_, max_val, _, max_loc = cv2.minMaxLoc(result)
		if max_val > best_val:
			best_val, best_loc, best_size = max_val, max_loc, (tw, th)

	if best_val <= FLOAT_MATCH_THRESHOLD or best_loc is None:
		return None

	tw, th = best_size
	tl = (best_loc[0] + search_x0, best_loc[1] + search_y0)   # top-left, back in full-screenshot coordinates
	center_x = tl[0] + tw / 2 + tw * FLOAT_CLICK_X_OFFSET_RATIO
	center_y = tl[1] + th / 2
	return center_x, center_y


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
