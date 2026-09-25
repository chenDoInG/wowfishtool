"""Regression tests for ui_calibrate.py's bar-detection logic.

Uses synthetic HSV canvases (no real screenshot needed) for the mechanism itself; the real-screenshot
end-to-end check (does it actually find a real player frame) is a manual step - see the module docstring.
"""
import json
import os

import cv2
import numpy as np
import pytest

import ui_calibrate
from ui_calibrate import calibrate, find_ui_regions


def _canvas(w=1000, h=700):
    bgr = np.full((h, w, 3), (42, 42, 50), dtype=np.uint8)   # HSV (0, 40, 50): outside both bar color ranges
    return bgr


def _paint_hsv(bgr, y0, y1, x0, x1, hsv_color):
    patch = np.full((y1 - y0, x1 - x0, 3), hsv_color, dtype=np.uint8)
    bgr[y0:y1, x0:x1] = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)


def test_finds_a_paired_health_and_resource_bar():
    """A wide green bar directly above a wide blue bar of a similar width, at a similar x, is trusted as a
    unit frame - the shape and pairing real health/resource bars have."""
    bgr = _canvas()
    _paint_hsv(bgr, 300, 320, 200, 350, (50, 200, 150))   # health: 150x20, aspect 7.5
    _paint_hsv(bgr, 322, 338, 200, 350, (110, 200, 150))  # resource: right below it, same width

    regions = find_ui_regions(bgr)

    assert len(regions) == 1
    x0, y0, x1, y1 = regions[0]
    assert x0 < 200 and x1 > 350   # padded outward, not just the bars' own bounding box
    assert y0 < 300 and y1 > 338


@pytest.mark.parametrize('resource_hue', [4, 175, 27], ids=['rage_low_hue', 'rage_high_hue', 'energy'])
def test_finds_a_health_bar_paired_with_a_non_mana_resource_bar(resource_hue):
    """A warrior's rage bar and a rogue's energy bar are not blue - RESOURCE_HSV_RANGES must catch those too,
    not just mana, or those classes could never calibrate their own frame."""
    bgr = _canvas()
    _paint_hsv(bgr, 300, 320, 200, 350, (50, 200, 150))
    _paint_hsv(bgr, 322, 338, 200, 350, (resource_hue, 200, 150))

    assert len(find_ui_regions(bgr)) == 1


def test_finds_two_separate_frames_mirrored_left_and_right():
    """A player frame (portrait conventionally to the bars' left) and a target frame (WoW's default mirrors
    this, portrait to the right) must both be found - the padding is symmetric left/right for exactly this."""
    bgr = _canvas(w=1600)
    _paint_hsv(bgr, 300, 320, 200, 350, (50, 200, 150))
    _paint_hsv(bgr, 322, 338, 200, 350, (110, 200, 150))
    _paint_hsv(bgr, 300, 320, 1200, 1350, (50, 200, 150))
    _paint_hsv(bgr, 322, 338, 1200, 1350, (110, 200, 150))

    regions = find_ui_regions(bgr)

    assert len(regions) == 2


def test_ignores_a_green_bar_with_no_resource_bar_nearby():
    """The fishing cast's own progress bar is exactly this shape (long, green, alone, centered) - it must
    not be mistaken for a health bar just because nothing else is close enough to disqualify it by shape."""
    bgr = _canvas()
    _paint_hsv(bgr, 500, 520, 300, 700, (50, 200, 150))   # centered: (300+700)/2 == canvas width / 2

    assert find_ui_regions(bgr) == []


def test_accepts_a_lone_health_bar_away_from_center_with_no_resource_bar():
    """A rage/energy bar reading 0 renders no fill at all - confirmed on a real screenshot where a warrior's
    own frame was missed for exactly this reason. Off-center, a lone health bar is still trusted, unlike the
    centered fishing cast bar in test_ignores_a_green_bar_with_no_resource_bar_nearby."""
    bgr = _canvas()
    _paint_hsv(bgr, 300, 320, 100, 250, (50, 200, 150))   # x 100-250, well left of center on a 1000-wide canvas

    regions = find_ui_regions(bgr)

    assert len(regions) == 1
    x0, y0, x1, y1 = regions[0]
    assert x0 < 100 and x1 > 250 and y0 < 300


def test_ignores_a_short_lone_health_bar_like_a_nameplate():
    """A crowd of other players each floats a small health bar over their head (a nameplate) - measured on a
    real screenshot these were 9-11px tall against 23-24px for the player's own and a target's real frame.
    Off-center alone is not enough to trust a lone bar; it must also be tall enough to be a real frame's own."""
    bgr = _canvas()
    _paint_hsv(bgr, 300, 310, 100, 250, (50, 200, 150))   # 10px tall, off-center - nameplate-sized

    assert find_ui_regions(bgr) == []


def test_ignores_a_round_green_blob():
    """A round patch (grass, a green creature) is not bar-shaped and must not be mistaken for a health bar
    even if a blue patch happens to sit right below it."""
    bgr = _canvas()
    _paint_hsv(bgr, 300, 340, 300, 340, (50, 200, 150))   # 40x40, aspect 1 - not a bar
    _paint_hsv(bgr, 342, 358, 260, 380, (110, 200, 150))

    assert find_ui_regions(bgr) == []


def test_calibrate_writes_the_regions_and_a_preview(tmp_path, monkeypatch):
    bgr = _canvas()
    _paint_hsv(bgr, 300, 320, 200, 350, (50, 200, 150))
    _paint_hsv(bgr, 322, 338, 200, 350, (110, 200, 150))
    screenshot_path = str(tmp_path / 'shot.png')
    cv2.imwrite(screenshot_path, bgr)

    regions_path = str(tmp_path / 'ui_regions.json')
    preview_path = str(tmp_path / 'preview.png')
    monkeypatch.setattr(ui_calibrate, 'UI_REGIONS_PATH', regions_path)
    monkeypatch.setattr(ui_calibrate, 'CALIBRATION_PREVIEW_PATH', preview_path)

    fractions = calibrate(screenshot_path)

    assert fractions is not None and len(fractions) == 1
    assert os.path.exists(preview_path)
    with open(regions_path) as f:
        as_lists = [[list(x_range), list(y_range)] for x_range, y_range in fractions]
        assert json.load(f) == as_lists   # round-trips through JSON as nested lists, not tuples


def test_calibrate_writes_nothing_when_no_frame_is_found(tmp_path, monkeypatch):
    screenshot_path = str(tmp_path / 'shot.png')
    cv2.imwrite(screenshot_path, _canvas())   # plain background, no bars at all

    regions_path = str(tmp_path / 'ui_regions.json')
    monkeypatch.setattr(ui_calibrate, 'UI_REGIONS_PATH', regions_path)
    monkeypatch.setattr(ui_calibrate, 'CALIBRATION_PREVIEW_PATH', str(tmp_path / 'preview.png'))

    assert calibrate(screenshot_path) is None
    assert not os.path.exists(regions_path)


def test_calibrate_reports_an_unreadable_screenshot_instead_of_raising(tmp_path):
    assert calibrate(str(tmp_path / 'missing.png')) is None
