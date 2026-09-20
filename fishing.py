import os
import shutil
import sys
import threading
import time

import cv2
import numpy as np
import psutil
import pyautogui
import pygetwindow as gw
import pyscreenshot as ImageGrab
from pynput import keyboard

import float_detector
from audio_listener import listen
from float_detector import find_float
from login_screen import is_login_screen

dev = False

# What psutil's Process.name() returns differs by OS: on Windows it's the executable's
# short filename (e.g. "WowClassic.exe" - confirmed against a real Windows report that
# "Wow.exe"/"World of Warcraft"/"World of Warcraft Classic" weren't matching), on macOS
# it's the actual binary's full name, spaces and all (confirmed live here: psutil reports
# it as 'World of Warcraft Classic' - retail would presumably be 'World of Warcraft').
# "world of warcraft" alone covers both macOS cases since Classic's name contains it as a
# substring. Matching is case-insensitive (see is_wow_running()).
WOW_PROCESS_NAMES = ["wow.exe", "wowclassic.exe", "wow-64.exe", "wowclassic-64.exe", "world of warcraft"]

SCREENSHOT_PATH = 'var/fishing_session.png'

CAST_KEY = '1'   # fishing rod's action bar slot
BAIT_KEY = '2'   # in-game macro that re-lures the fishing pole
CAST_SETTLE_SECONDS = 2.5   # wait after casting before the screenshot: the bobber must land and settle. Raised from 2 to see whether the previous float's afterimage (seen in debug frames) has faded by then - not yet verified
BAIT_REAPPLY_INTERVAL_SECONDS = 10 * 60 + 15   # a little past the lure's actual duration, so it never gets reapplied while the old one still has time left
MAX_CONSECUTIVE_MISSES = 10   # this many fish_once() calls in a row without a catch means something's actually wrong (window moved, wrong zone, game state stuck) rather than just bad luck - stop instead of grinding uselessly

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
		if any(w in name.lower() for w in WOW_PROCESS_NAMES):
			print(name)
			return True
	return False


def check_process():
	print('Checking WoW is running')
	running = is_wow_running()
	if not running and not dev:
		print('WoW is not running')
		sys.exit()
	print('WoW is running')
	return running


def _window_geometry(title):
	"""(left, top, width, height) of the window with this exact title.

	pygetwindow 0.0.9 (still the only version ever published, as of writing) is an
	openly-unfinished library whose macOS and Windows backends expose completely
	different, non-overlapping functions for this: macOS only has getWindowGeometry(),
	Windows only has getWindowsWithTitle() (confirmed against both a real Windows
	traceback and this module's own source per platform - there's no single function
	name that works on both), so this has to branch instead of picking one.

	sys.platform is 'win32' on Windows regardless of 32/64-bit - there's no separate
	'win64' value, that's not a gap here. Checked explicitly (rather than treating
	anything-not-Windows as macOS) since pygetwindow itself only supports these two
	platforms anyway - failing clearly here beats silently trying the wrong branch's
	function and getting a confusing AttributeError instead."""
	if sys.platform == 'darwin':
		return gw.getWindowGeometry(title)
	elif sys.platform == 'win32':
		window = gw.getWindowsWithTitle(title)[0]
		return window.left, window.top, window.width, window.height
	raise NotImplementedError('_window_geometry() has no implementation for sys.platform=' + sys.platform)


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
		left, top, width, height = _window_geometry(title)
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
	stop_requested.wait(CAST_SETTLE_SECONDS)


def locate_float():
	"""Screenshot the game window and return the (x, y) position of the float in it, or None."""
	locate_game_window()
	screenshot_path = SCREENSHOT_PATH if not dev else 'var/fishing_session_' + str(int(time.time())) + '.png'
	ImageGrab.grab(game_window_bbox).save(screenshot_path)
	return find_float(screenshot_path)


def on_login_page():
	"""Whether the game window shows the login or character-select page: what a disconnect, being logged in elsewhere,
	a server kick and a frozen account all end on, whatever the dialog says."""
	locate_game_window()
	screenshot = ImageGrab.grab(game_window_bbox)
	return is_login_screen(cv2.cvtColor(np.array(screenshot.convert('RGB')), cv2.COLOR_RGB2BGR))


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
	print('Snatching at ' + time.strftime('%H:%M:%S') + '!')
	# The mouse has usually been sitting still on the float for up to 20s while
	# listen() waited. WoW seems to only refresh what object is under the cursor
	# when it sees a real mouse-move event, and the float's bobbing animation can
	# drift it out from under a cursor that hasn't moved in a while, so a click
	# right now can land on plain water even though the cursor looks right on it.
	# Re-issuing the move immediately before clicking forces a fresh pick.
	move_mouse(place, duration=0, quiet=True)
	pyautogui.click(button='right')


def try_recover_from_disconnect():
	"""Best-effort recovery for the case where MAX_CONSECUTIVE_MISSES was actually caused
	by getting disconnected (the "已从服务器断开" dialog), not a detection problem: press
	Enter three times, a couple seconds apart, on the guess that Enter activates whatever
	button the disconnect dialog and the login/realm/character-select screens after it
	leave focused. No visual confirmation this actually worked - it's a cheap, harmless
	guess to try before giving up for good, not a real reconnect flow.

	PARTIALLY VERIFIED: a real disconnect confirmed the first two presses do get through
	the disconnect dialog and login/realm-select screens - a debug_notfound snapshot
	caught it sitting on the character-select screen (the right character highlighted,
	"进入魔兽世界" button visible) afterward. But the gap between the 2nd and 3rd press
	wasn't enough for character-select to actually finish loading/settle before the 3rd
	Enter fired, so it never actually entered the world - widened twice now (2s -> 5s ->
	10s); still unverified whether it's enough yet. Same for the final wait (entering the
	world triggers its own loading screen) - widened once (8s -> 18s), also unverified.

	A plain detection miss (no disconnect at all, character still in-world and fishing
	normally) can also fire MAX_CONSECUTIVE_MISSES - Enter with no dialog open just opens
	the chat box, and every key fish_once() presses after that (bait, cast) gets typed into
	it as chat text instead of doing anything. A trailing Escape was tried here as a blind
	cleanup step for that case, but recovery_*.png snapshots caught it doing the opposite
	in the *successful*-recovery case: with nothing left open (character already back
	in-world), Escape doesn't no-op - it opens the main game menu (WoW's Escape is a
	toggle: closes whatever's focused, or opens the game menu if nothing is). That menu
	then sits there blocking every subsequent cast/click until someone closes it by hand,
	which is worse than the chat-box case it was meant to fix - that one still self-resolves
	via the same give-up-after-10-more-misses path this whole recovery does when it fails,
	so dropping the trailing Escape doesn't actually make the miss-not-disconnect case any
	worse, while it stops the success case from getting sabotaged. Removed."""
	print('Trying Enter x3 in case this is a disconnect, not a detection problem')
	pyautogui.press('enter')
	stop_requested.wait(2)
	pyautogui.press('enter')
	# Character-select needs to finish loading (and the right character be selected)
	# before Enter there activates "进入魔兽世界" - shorter waits used here before gave up
	# too early, leaving it sitting on this screen instead of actually entering the world.
	stop_requested.wait(10)
	pyautogui.press('enter')
	# If this really was a disconnect, this third Enter is the one that re-enters the
	# world, which triggers a loading screen - give that more room than the 2s gap
	# above before anything else tries to act on the (still loading) game window.
	stop_requested.wait(18)
	if float_detector.DEBUG_SNAPSHOTS:
		# Whether the wait above was actually long enough is otherwise only checkable by
		# happening to be watching the screen live when it matters - save what the game
		# window looks like right after recovery so it can be checked after the fact.
		os.makedirs(float_detector.DEBUG_SNAPSHOT_DIR, exist_ok=True)
		recovery_path = os.path.join(float_detector.DEBUG_SNAPSHOT_DIR, 'recovery_' + str(int(time.time())) + '.png')
		ImageGrab.grab(game_window_bbox).save(recovery_path)
		print('Saved ' + recovery_path + ' to check whether recovery actually got back in-world')


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
		print('Float was not found, retrying in 0.5 seconds')
		stop_requested.wait(0.5)
		place = locate_float()
		if not place:
			if float_detector.DEBUG_SNAPSHOTS:
				# SCREENSHOT_PATH gets overwritten by the next cast, so preserve this
				# one under a unique name before it's gone.
				os.makedirs(float_detector.DEBUG_SNAPSHOT_DIR, exist_ok=True)
				debug_path = os.path.join(float_detector.DEBUG_SNAPSHOT_DIR, 'notfound_' + str(int(time.time())) + '.png')
				shutil.copy(SCREENSHOT_PATH, debug_path)
				print('Still can\'t find float, giving up on this cast - saved ' + debug_path + ' for review')
			else:
				print('Still can\'t find float, giving up on this cast')
			return False

	move_mouse(place, elapsed_since_cast=time.time() - cast_time)
	if not listen(stop_event=stop_requested):
		print('Didn\'t hear a bite, trying again')
		return False
	if stop_requested.is_set():
		return False

	snatch(place)
	# Give the client a moment to actually register the catch before the next
	# cast - the give-up paths above skip this since there's nothing to let
	# settle, they just go straight to recasting.
	stop_requested.wait(1)
	return True


def run_session():
	"""Fish until F11, or until the misses say something is wrong. Returns how many fish were caught."""
	caught = 0
	consecutive_misses = 0
	tried_recovery = False
	while fishing_active.is_set() and not stop_requested.is_set():
		if fish_once():
			caught += 1
			consecutive_misses = 0
			tried_recovery = False
		elif stop_requested.is_set():
			break   # F11 ended this cast early; that is not a miss
		else:
			consecutive_misses += 1
			if consecutive_misses >= MAX_CONSECUTIVE_MISSES:
				if not tried_recovery:
					print(str(consecutive_misses) + ' misses in a row - checking whether the game has disconnected')
					if not on_login_page():
						print('The game is not on the login screen, so this is not a disconnect - stopping')
						break
					print('On the login screen - trying to get back in')
					try_recover_from_disconnect()
					consecutive_misses = 0
					tried_recovery = True
				else:
					print(str(MAX_CONSECUTIVE_MISSES) + ' misses in a row even after trying to recover - stopping')
					break
	return caught


def main():
	if check_process() and not dev:
		print("Waiting 2 seconds, so you can switch to WoW")
		time.sleep(2)

	locate_game_window()
	start_hotkey_listener()
	print('Press F10 to start fishing, F11 to stop')

	while not dev:
		fishing_active.wait()

		caught = run_session()
		print('caught ' + str(caught))
		fishing_active.clear()
		print('Stopped. Press F10 to start again.')


if __name__ == '__main__':
	main()
