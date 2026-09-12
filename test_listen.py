"""Manual test for audio_listener.listen(). Run this directly in PyCharm.

This needs a real microphone and a human making noise, so it's not an automated
pass/fail test - it prints what it's doing so you can judge the result yourself.
"""
import time

import pyaudio
import audioop

from audio_listener import listen, find_loopback_device_index


def list_devices():
	p = pyaudio.PyAudio()
	print('Available audio devices:')
	for i in range(p.get_device_count()):
		info = p.get_device_info_by_index(i)
		print('  ' + str(i) + ': ' + str(info.get('name')) +
		      ' (in=' + str(info.get('maxInputChannels')) +
		      ' out=' + str(info.get('maxOutputChannels')) +
		      ' rate=' + str(info.get('defaultSampleRate')) + ')')
	p.terminate()


def monitor_levels(seconds=8, device_index=None, rate=None, channels=1):
	"""Print live amplitude from `device_index` so you can see real numbers while the game plays."""
	CHUNK = 1024
	p = pyaudio.PyAudio()
	if device_index is not None:
		info = p.get_device_info_by_index(device_index)
		if rate is None:
			rate = int(info['defaultSampleRate'])
		channels = min(channels, int(info['maxInputChannels']) or channels)
	if rate is None:
		rate = 18000
	stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
	                input_device_index=device_index, frames_per_buffer=CHUNK)
	print('Monitoring for ' + str(seconds) + 's on device ' + str(device_index) + '...')
	t0 = time.time()
	peak = 0
	while time.time() - t0 < seconds:
		data = stream.read(CHUNK, exception_on_overflow=False)
		v = audioop.rms(data, 2)
		peak = max(peak, v)
		print(round(v, 1))
	stream.close()
	p.terminate()
	print('Peak amplitude seen: ' + str(round(peak, 1)))
	return peak


if __name__ == '__main__':
	list_devices()
	print()

	p = pyaudio.PyAudio()
	loopback_index = find_loopback_device_index(p)
	p.terminate()

	if loopback_index is None:
		print('No BlackHole loopback device found - falling back to default input device.')
		print('Make sure WoW\'s output is routed to BlackHole (or a Multi-Output Device')
		print('containing BlackHole) in macOS Sound settings / Audio MIDI Setup.')
	else:
		print('Using loopback device index ' + str(loopback_index) + ' for capture.')
	print()

	print('=== Step 1: live amplitude monitor ===')
	print('Watch the numbers below while a bite sound plays in-game (or clap near the mic')
	print('if using the default device). If they stay near 0, audio isn\'t reaching this')
	print('device (routing / permissions). If they spike, use that peak to pick THRESHOLD.')
	peak = monitor_levels(seconds=8, device_index=loopback_index)
	print()

	print('=== Step 2: run the real listen() bite-detection function ===')
	print('Default THRESHOLD=1200. It waits up to 20s for a sound to cross it.')
	result = listen(threshold=1200, device_index=loopback_index)
	print('listen() returned: ' + str(result))
