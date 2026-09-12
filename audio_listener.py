import pyaudio
import audioop
import math
import time
from collections import deque

LOOPBACK_NAME_HINT = 'blackhole'


def find_loopback_device_index(p, name_hint=LOOPBACK_NAME_HINT):
	"""Return the device index of an input device whose name contains `name_hint`, or None."""
	for i in range(p.get_device_count()):
		info = p.get_device_info_by_index(i)
		if name_hint.lower() in str(info.get('name', '')).lower() and info.get('maxInputChannels', 0) > 0:
			return i
	return None


def is_bite(rms_window, threshold):
	"""Return True if the average RMS across the window indicates a sustained bite sound.

	A real bite sound is sustained over a second or two, unlike a brief noise spike, so
	this checks the whole window's average rather than triggering off any single loud chunk.
	"""
	if not rms_window:
		return False
	return sum(rms_window) / len(rms_window) > threshold


def listen(threshold=15, rate=None, channels=1, silence_limit_seconds=1, timeout_seconds=23, stop_event=None, device_index=None):
	"""Listen for the fishing bite sound and return True once one is heard.

	By default this listens on the BlackHole loopback device (game audio routed through it),
	falling back to the system default input device if BlackHole isn't found. Pass
	`device_index` to force a specific device.

	Gives up and returns False after `timeout_seconds` with no bite sound, or as soon as
	`stop_event` (a threading.Event) is set, if one is passed in. The default padded a
	couple seconds past the server's ~19-20s bite window to cover the time locate_float()
	spends screenshotting and matching before this even starts listening.
	"""
	print('Listening for the fishing bite sound...')
	CHUNK = 1024  # CHUNKS of bytes to read each time from mic

	p = pyaudio.PyAudio()
	if device_index is None:
		device_index = find_loopback_device_index(p)
	if device_index is not None:
		device_info = p.get_device_info_by_index(device_index)
		if rate is None:
			rate = int(device_info['defaultSampleRate'])
		channels = min(channels, int(device_info['maxInputChannels']) or channels)
	if rate is None:
		rate = 18000

	stream = p.open(format=pyaudio.paInt16,
	                channels=channels,
	                rate=rate,
	                input=True,
	                input_device_index=device_index,
	                frames_per_buffer=CHUNK)

	rel = rate / CHUNK
	slid_win = deque(maxlen=math.ceil(silence_limit_seconds * rel))
	success = False
	max_avg_seen = 0
	listening_start_time = time.time()
	while True:
		if stop_event is not None and stop_event.is_set():
			break
		try:
			cur_data = stream.read(CHUNK, exception_on_overflow=False)
			rms = audioop.rms(cur_data, 2)
			slid_win.append(rms)
			if len(slid_win) == slid_win.maxlen:
				avg = sum(slid_win) / len(slid_win)
				max_avg_seen = max(max_avg_seen, avg)
				if is_bite(slid_win, threshold):
					print('Heard a bite! (avg level ' + str(round(avg)) + ', threshold ' + str(threshold) + ')')
					success = True
					break
			if time.time() - listening_start_time > timeout_seconds:
				print('No bite after ' + str(timeout_seconds) + 's (peak avg level seen: ' + str(round(max_avg_seen)) + ', threshold ' + str(threshold) + ')')
				break
		except IOError:
			break

	stream.close()
	p.terminate()
	return success
