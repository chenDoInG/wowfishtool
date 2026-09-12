"""Regression tests for audio_listener's bite-detection decision logic.

is_bite() is pure (no PyAudio/stream I/O), so it can be tested with synthetic RMS
data without a real microphone, game, or BlackHole loopback device.
"""
from audio_listener import is_bite


def test_quiet_window_is_not_a_bite():
	assert is_bite([1, 2, 3, 2, 1], threshold=15) is False


def test_a_brief_spike_is_not_a_bite():
	# A real bite is sustained over the whole ~1s window, unlike a single loud chunk
	# (e.g. a random click/pop) surrounded by otherwise quiet audio. Window length
	# mirrors a real ~1s window at 48000Hz / 1024-sample chunks (~44 chunks).
	window = [3] * 43 + [300]
	assert is_bite(window, threshold=15) is False


def test_sustained_loud_window_is_a_bite():
	assert is_bite([50, 60, 55, 58, 52], threshold=15) is True


def test_empty_window_is_not_a_bite():
	assert is_bite([], threshold=15) is False
