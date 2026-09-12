"""Manual test: does a synthetic right-click actually reach other apps?

Run this from the same program you run fishing.py from (PyCharm). It moves the
mouse to the current position and right-clicks after a short delay - put your
cursor over the desktop or an app that shows a right-click context menu (e.g.
Finder desktop, or a text editor) before the countdown ends, and see whether a
context menu actually pops up. If nothing happens, the accessibility
permission for PyCharm is the problem, not the game or the fishing.py logic.
"""
import time

import pyautogui

if __name__ == '__main__':
	print('Move your mouse over the desktop or a window that shows a right-click menu.')
	for i in range(5, 0, -1):
		print(str(i) + '...')
		time.sleep(1)
	x, y = pyautogui.position()
	print('Right-clicking at current position: ' + str((x, y)))
	pyautogui.click(button='right')
	print('Did a context menu appear? If not, PyCharm needs Accessibility permission.')
