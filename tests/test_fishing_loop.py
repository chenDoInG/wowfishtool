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
		return True

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


def test_a_recovery_that_leaves_the_game_on_the_login_page_stops_at_once(monkeypatch):
	casts = iter([False] * 50)
	calls = {'casts': 0}

	def fake_fish_once():
		calls['casts'] += 1
		return next(casts)

	monkeypatch.setattr(fishing, 'fish_once', fake_fish_once)
	monkeypatch.setattr(fishing, 'on_login_page', lambda: True)
	monkeypatch.setattr(fishing, 'try_recover_from_disconnect', lambda: False)
	fishing.stop_requested.clear()
	fishing.fishing_active.set()
	try:
		fishing.run_session()
	finally:
		fishing.fishing_active.clear()

	assert calls['casts'] == LIMIT   # no ten more casts on a page that cannot fish


# ---- the recovery itself: Enter, look, Enter, look ...

def _recover(monkeypatch, on_login_page_answers, stop_after_looks=None):
	"""Run try_recover_from_disconnect() with the login-page check answering from a list (the last answer repeats);
	returns (result, Enter presses, waits, screen looks)."""
	from types import SimpleNamespace
	presses, waits, looks = [], [], []
	answers = list(on_login_page_answers)
	stopped = {'flag': False}

	def fake_wait(seconds):
		waits.append(seconds)
		return stopped['flag']

	def fake_on_login_page():
		looks.append(1)
		if stop_after_looks is not None and len(looks) >= stop_after_looks:
			stopped['flag'] = True
		return answers[min(len(looks) - 1, len(answers) - 1)]

	monkeypatch.setattr(fishing.pyautogui, 'press', lambda key: presses.append(key))
	monkeypatch.setattr(fishing, 'stop_requested', SimpleNamespace(wait=fake_wait, is_set=lambda: stopped['flag']))
	monkeypatch.setattr(fishing, 'on_login_page', fake_on_login_page)
	return fishing.try_recover_from_disconnect(), presses, waits, len(looks)


def test_recovery_follows_the_pages_one_enter_at_a_time_and_stops_pressing_once_in_the_world(monkeypatch):
	# login page after the 1st press (dialog dismissed), character select after the 2nd, in the world (twice) after the 3rd
	result, presses, waits, looks = _recover(monkeypatch, [True, True, False, False])

	assert result is True
	assert presses == ['enter'] * 3
	assert waits == [2, 10, 18, fishing.RECOVERY_CONFIRM_SECONDS]
	assert looks == 4


def test_a_loading_screen_that_turns_out_to_be_the_login_page_again_keeps_pressing(monkeypatch):
	# not the login page at the first look, back on it at the confirming look, then really in the world
	result, presses, waits, looks = _recover(monkeypatch, [False, True, False, False])

	assert result is True and presses == ['enter'] * 2


def test_recovery_gives_up_after_every_round_when_the_login_page_never_goes_away(monkeypatch):
	result, presses, waits, looks = _recover(monkeypatch, [True])

	assert result is False
	assert len(presses) == len(fishing.RECOVERY_WAITS) and waits == list(fishing.RECOVERY_WAITS)


def test_f11_during_the_recovery_ends_it_without_more_presses(monkeypatch):
	result, presses, waits, looks = _recover(monkeypatch, [True], stop_after_looks=1)

	assert result is False and len(presses) == 1
