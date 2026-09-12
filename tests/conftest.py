import os
import sys

# Let tests import fishing/audio_listener from the project root regardless of
# how pytest is invoked (bare `pytest`, `python -m pytest`, PyCharm's runner, ...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
