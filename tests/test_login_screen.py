import glob
import os

import cv2
import numpy as np
import pytest

import login_screen
from login_screen import is_login_screen

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


@pytest.fixture(scope='module')
def login():
	return cv2.imread(os.path.join(FIXTURE_DIR, 'login_screen.png'))


@pytest.fixture(scope='module')
def classic_login():
	"""The classic-era client's login page after a disconnect (the "reconnect" dialog)."""
	return cv2.imread(os.path.join(FIXTURE_DIR, 'login_screen_classic.png'))


def test_the_classic_login_page_is_recognised(classic_login):
	assert is_login_screen(classic_login)


@pytest.mark.parametrize('factor', [0.6, 1.3])
def test_the_classic_login_page_is_recognised_at_other_window_sizes(classic_login, factor):
	resized = cv2.resize(classic_login, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)

	assert is_login_screen(resized)


def test_the_login_page_is_recognised_whatever_dialog_is_on_it(login):
	"""The fixture carries the 'logged in elsewhere' dialog; a disconnect, a kick or a frozen account put a different
	text on the same page, and the two logos the check looks at are outside the dialog."""
	assert is_login_screen(login)


@pytest.mark.parametrize('factor', [0.4, 0.6, 0.8, 1.3])
def test_the_login_page_is_recognised_at_other_window_sizes(login, factor):
	resized = cv2.resize(login, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)

	assert is_login_screen(resized)


def test_a_full_game_world_frame_is_not_the_login_page():
	world = cv2.imread(os.path.join(os.path.dirname(__file__), '..', 'docs', 'float_safe_zone.png'))

	assert not is_login_screen(world)


@pytest.mark.parametrize('path', sorted(glob.glob(os.path.join(FIXTURE_DIR, '*.png'))))
def test_no_other_fixture_is_the_login_page(path):
	if os.path.basename(path) in ('login_screen.png', 'login_screen_classic.png'):
		pytest.skip('one of the pages itself')

	assert not is_login_screen(cv2.imread(path))


@pytest.mark.parametrize('page, fixture_name', [('classic login page', 'classic_login'), ('login page', 'login')])
def test_every_control_of_a_page_is_needed_so_a_real_session_is_never_stopped_on_one_lookalike(request, page, fixture_name):
	image = request.getfixturevalue(fixture_name)
	assert is_login_screen(image)

	for _, anchor, width, height in login_screen.PAGES[page]:
		without_one = image.copy()
		login_screen._search_area(without_one, anchor, width, height)[:] = 0   # the search area is a view into the copy

		assert not is_login_screen(without_one), 'still recognised without its ' + anchor + ' control'


def test_a_blank_or_degenerate_screenshot_is_not_the_login_page():
	assert not is_login_screen(np.zeros((1408, 2556, 3), dtype=np.uint8))
	assert not is_login_screen(np.zeros((1, 1, 3), dtype=np.uint8))
	assert not is_login_screen(np.zeros((10, 100000, 3), dtype=np.uint8))


def test_a_checkout_without_the_logo_templates_never_reports_the_login_page(login, monkeypatch):
	monkeypatch.setattr(login_screen, 'PAGES', {'gone': (('var/no_such_logo.png', 'top-left', 250, 210),)})

	assert not is_login_screen(login)
