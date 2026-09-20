"""Tell whether the screenshot is WoW's login or character-select page instead of the game world.

A disconnect, being logged in elsewhere, a server kick and a frozen account all end on that page (only the dialog text
differs), or one Enter further on the character-select screen: the bot can find nothing to fish there and pressing
keys does nothing. Each page is recognised by two of its own fixed controls at the bottom of the screen, both of which
must be found (the classic-era login page: the game logo at the top-left and the exit button; a test-server login page:
the Blizzard logo and the exit button; character select: the enter-world button and the create-character button). Stopping a real fishing session by mistake costs more than noticing the page a few casts later,
and the game world has none of these controls. Nothing here depends on a logo that only some clients show.
"""
import glob
import os

import cv2
import numpy as np

# (template file, the corner or edge of the screen it hangs from, width and height in pixels of the width-normalised screenshot to search there)
PAGES = {
	'classic login page': (
		('var/login_screen_logo_classic.png', 'top-left', 350, 200),
		('var/login_screen_button_exit_classic.png', 'bottom-right', 400, 160),
	),
	'login page': (
		('var/login_screen_logo_blizzard.png', 'bottom-center', 350, 160),
		('var/login_screen_button_exit.png', 'bottom-right', 400, 160),
	),
	'character select': (
		('var/login_screen_button_enter_world.png', 'bottom-center', 350, 160),
		('var/login_screen_button_create_character.png', 'bottom-right', 400, 160),
	),
}
REFERENCE_WIDTH = 1278          # the templates were cropped from a window this wide; bigger screenshots are shrunk to it
TEMPLATE_SCALES = (0.8, 0.9, 1.0, 1.1, 1.25)   # the page's UI scales with the window height, not only its width
MIN_LOGO_SCORE = 0.7


def _search_area(img: np.ndarray, anchor: str, width: int, height: int):
	h, w = img.shape[:2]
	if anchor == 'top-left':
		return img[:height, :width]
	rows = slice(max(0, h - height), h)
	if anchor == 'bottom-right':
		return img[rows, max(0, w - width):]
	return img[rows, max(0, (w - width) // 2):(w + width) // 2]   # bottom-center


def _best_score(area: np.ndarray, template: np.ndarray) -> float:
	best = -1.0
	for scale in TEMPLATE_SCALES:
		scaled = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
		if scaled.shape[0] > area.shape[0] or scaled.shape[1] > area.shape[1]:
			continue
		best = max(best, float(cv2.minMaxLoc(cv2.matchTemplate(area, scaled, cv2.TM_CCOEFF_NORMED))[1]))
	return best


def is_login_screen(img_bgr: np.ndarray) -> bool:
	"""True when both controls of either page are found. False when a template is missing, so a checkout without them just never stops early."""
	h, w = img_bgr.shape[:2]
	if w < 1 or h < 1:
		return False
	scale = REFERENCE_WIDTH / w
	if round(w * scale) < 1 or round(h * scale) < 1 or scale > 8:
		return False
	if scale != 1:
		img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
	gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

	def found(path, anchor, width, height):
		template = cv2.imread(path, 0)
		return template is not None and _best_score(_search_area(gray, anchor, width, height), template) >= MIN_LOGO_SCORE

	return any(all(found(*control) for control in controls) for controls in PAGES.values())
