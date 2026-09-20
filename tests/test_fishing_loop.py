import pytest

import fishing

LIMIT = fishing.MAX_CONSECUTIVE_MISSES


def _run(monkeypatch, casts, on_login_page):
	"""Run fishing.run_session() with fish_once() replaying `casts` (True = caught) and the game on the login page or not;
	returns (caught, casts made, recoveries tried, login page looks)."""
	script = iter(casts)
	calls = {'casts': 0, 'recoveries': 0, 'looks': 0}

	def fake_fish_once():
		calls['casts'] += 1
		return next(script)

	def fake_on_login_page():
		calls['looks'] += 1
		return on_login_page

	def fake_recovery():
		calls['recoveries'] += 1

	monkeypatch.setattr(fishing, 'fish_once', fake_fish_once)
	monkeypatch.setattr(fishing, 'on_login_page', fake_on_login_page)
	monkeypatch.setattr(fishing, 'try_recover_from_disconnect', fake_recovery)
	fishing.stop_requested.clear()
	fishing.fishing_active.set()
	try:
		caught = fishing.run_session()
	finally:
		fishing.fishing_active.clear()
	return caught, calls['casts'], calls['recoveries'], calls['looks']


def test_ten_misses_in_a_row_and_no_login_page_means_stop_without_pressing_enter(monkeypatch):
	"""The game is still connected, so Enter would only open the chat box: it is not a disconnect."""
	assert _run(monkeypatch, [False] * 50, on_login_page=False) == (0, LIMIT, 0, 1)


def test_ten_misses_in_a_row_on_the_login_page_tries_to_get_back_in_once_then_stops(monkeypatch):
	assert _run(monkeypatch, [False] * 50, on_login_page=True) == (0, 2 * LIMIT, 1, 1)   # ten, one recovery, ten more, stop


def test_the_login_page_is_not_looked_at_while_casts_keep_catching_something(monkeypatch):
	casts = ([False] * (LIMIT - 1) + [True]) * 3 + [False] * LIMIT   # never ten misses in a row until the end
	caught, made, recoveries, looks = _run(monkeypatch, casts, on_login_page=False)

	assert (caught, made, recoveries, looks) == (3, len(casts), 0, 1)   # the one look is the final run of ten


def test_a_catch_gives_the_recovery_back(monkeypatch):
	casts = [False] * LIMIT + [True] + [False] * 50
	assert _run(monkeypatch, casts, on_login_page=True) == (1, LIMIT + 1 + 2 * LIMIT, 2, 2)


def test_f11_during_a_cast_is_not_counted_as_a_miss(monkeypatch):
	casts = {'made': 0}

	def fake_fish_once():
		casts['made'] += 1
		if casts['made'] == 3:
			fishing.stop_requested.set()   # F11 lands during the third cast, which returns False
		return False

	looked = []
	monkeypatch.setattr(fishing, 'fish_once', fake_fish_once)
	monkeypatch.setattr(fishing, 'on_login_page', lambda: looked.append(1) or True)
	monkeypatch.setattr(fishing, 'MAX_CONSECUTIVE_MISSES', 3)
	fishing.stop_requested.clear()
	fishing.fishing_active.set()
	try:
		fishing.run_session()
	finally:
		fishing.fishing_active.clear()
		fishing.stop_requested.clear()

	assert looked == []   # the third miss came with F11 pressed: the session just ends
