"""Auto-detect UI_EXCLUDE_REGIONS from a real screenshot instead of hand-measuring pixels.

UI_EXCLUDE_REGIONS in float_detector.py is a fixed pair of window-fraction rectangles measured once on
the author's own default-layout screenshot. Anyone who moved their unit frames, switched Edit Mode layouts,
or runs an addon that reskins them (ElvUI and friends) will have UI in a different place, and the color gate
will then find their gold-bordered, brightly-colored frame more saturated than any water and click it instead
of the float (see the README's "定位点跑到角色状态框上" troubleshooting entry).

Template-matching the frame's border art was considered and rejected: it depends on the exact skin, so it
breaks for precisely the addon case this tool needs to handle. Instead this looks for what almost every skin
still renders as a solid, saturated color: a health bar (green) directly paired with a resource bar above/below
it at a matching width. Health is present for every class and is the reliable anchor; the resource bar's own
color depends on class - mana (blue) for casters, rage (red) for warriors, energy (yellow) for rogues and cat
Druids - so RESOURCE_HSV_RANGES tries each in turn rather than assuming blue.

A resource bar at 0 (a warrior at rest, rage drains to 0 out of combat) renders no fill at all, so there is
nothing to pair with - confirmed on a real screenshot (2026-09-25) where the player's own warrior frame was
missed for exactly this reason. A lone health bar is still accepted then, but only when it is not near the
screen's horizontal center: the fishing cast's own progress bar is the other long green bar real screenshots
turn up, and it always sits centered at the bottom of the screen while casting - unit frames (moved or not)
essentially never do. See HEALTH_BAR_CENTER_EXCLUSION.

Run directly (`python ui_calibrate.py [screenshot] [--trace]`, screenshot defaults to var/fishing_session.png)
to detect the regions on a real screenshot, save an annotated preview to debug/ui_calibration.png for a
visual check, and write the regions to var/ui_regions.json - float_detector.py reads that file if present,
falling back to its own UI_EXCLUDE_REGIONS default otherwise. --trace prints every candidate bar found and
why each one was paired, accepted alone, or rejected - the same idea as float_detector.py's DEBUG_TRACE, for
the same reason: working out why a real screenshot didn't calibrate the way you expected without guessing.
"""
import json
import os
import sys

import cv2
import numpy as np

from float_detector import UI_REGIONS_PATH

CALIBRATION_PREVIEW_PATH = 'debug/ui_calibration.png'

# A bar's fill has real internal shading (WoW's bar textures are not flat color), so a solid-block extent
# floor like the one float_detector.py uses for base-color blobs is too strict here - measured on a real
# health bar it came out to 0.33-0.35, not the near-1.0 a flat rectangle would give. 0.25 keeps both real
# bars found on that screenshot with room to spare, while still rejecting non-bar-shaped clutter.
BAR_MIN_ASPECT = 4      # width / height: bars are wide and short
BAR_MIN_EXTENT = 0.25   # area / bounding-box area
BAR_MIN_AREA = 300      # px; drops thin slivers (anti-aliased edges, single-pixel-tall highlight lines)

# Measured on a real screenshot (var/fishing_session.png, 2026-09-25): health bar HSV (20,150,180)... i.e. the
# fill's own color, sampled from a clean strip away from any text overlay. Health can range from full green
# down through yellow to red as it drops, but the exclude-region calibration only needs the common "mostly
# healthy" case to find the bar at all - this is deliberately just the green end of that range.
#
# A dusk-harbor screenshot's green-sailed ship rigging also matched this range and got mistaken for a lone
# health bar (V 40-41 against a real bar's 150-184 - there's a clean gap here worth tightening the V floor
# for). Left alone on purpose: that false region landed nowhere near the search band (see the session notes
# around 2026-09-25 for why that makes it harmless in practice), so tightening this was not worth the risk
# of narrowing a range that already works for every real frame measured so far.
HEALTH_HSV_RANGE = ((35, 100, 40), (65, 255, 160))
# Mana is measured the same way as health above. Rage and energy are not yet measured against a real
# screenshot (no warrior/rogue capture on hand) - these are WoW's well-known standard bar colors, a starting
# point to refine against a real one later, the same spirit as FLOAT_BASE_COLOR_RANGES growing over time in
# float_detector.py. Rage needs two ranges since its red sits right where OpenCV's 0-179 hue wraps around.
RESOURCE_HSV_RANGES = (
	((95, 100, 30), (125, 255, 160)),    # mana - measured
	((0, 120, 30), (8, 255, 160)),       # rage, low-hue half - not yet measured
	((172, 120, 30), (179, 255, 160)),   # rage, high-hue half (wraps past 179) - not yet measured
	((22, 120, 60), (32, 255, 200)),     # energy - not yet measured
)

# A lone health bar (no resource bar found - see the module docstring) is trusted as a unit frame only when
# it is clearly not the fishing cast bar: measured on a real screenshot, that bar's own center sat at 49.5%
# of the window's width. Padded well past that single measurement, since only one has been measured so far.
HEALTH_BAR_CENTER_EXCLUSION = (0.30, 0.70)   # fraction of window width; a bar centered inside this is rejected

# ...and it must also be tall enough to be a real unit frame's own bar, not one of the small nameplate health
# bars WoW floats above every nearby player/NPC's head - measured on the same real screenshot, the player's
# own and a target's frame were 23-24px tall, three unrelated nameplates in a crowd nearby were 9-11px. A
# paired health bar (a resource bar was found to match it) does not need this - nameplates do not show one.
HEALTH_BAR_MIN_HEIGHT_FOR_LONE_BAR = 16

# A trace line per step (candidates found, how each health bar was paired or why a lone one was accepted or
# rejected) - same idea as float_detector.py's DEBUG_TRACE, its own opt-in switch since this tool has no
# debug-snapshot flag to piggyback on. Off by default so a normal `python ui_calibrate.py` run stays quiet.
UI_CALIBRATE_TRACE = False


def _debug(message):
    if UI_CALIBRATE_TRACE:
        print('[ui] ' + message)


def _bar_candidates(hsv: np.ndarray, lo, hi, label: str = ''):
    """[(x, y, w, h)] of bar-shaped blobs (wide, short, mostly filled) matching an HSV range."""
    mask = cv2.inRange(hsv, lo, hi)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    rejected = 0
    for i in range(1, n):
        x, y, w, h, area = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH]),
                             int(stats[i, cv2.CC_STAT_HEIGHT]), int(stats[i, cv2.CC_STAT_AREA]))
        if area < BAR_MIN_AREA or h == 0 or w / h < BAR_MIN_ASPECT or area / (w * h) < BAR_MIN_EXTENT:
            rejected += 1
            continue
        candidates.append((x, y, w, h))
    if label:
        _debug(label + ': ' + str(len(candidates)) + ' bar-shaped candidate(s) ' + str(candidates)
               + (', ' + str(rejected) + ' connected component(s) not bar-shaped enough' if rejected else ''))
    return candidates


def _health_bars_with_resource(hsv: np.ndarray, img_width: int):
    """(health box, resource box or None) for every health bar trusted as a unit frame: either it has a
    resource bar - mana, rage or energy, whichever the class shows, see RESOURCE_HSV_RANGES - of a similar
    width close below it (or above, if nothing qualifies below - some frame skins put it there instead), or
    (resource box is None) it has none but is clearly not the fishing cast bar either - see
    HEALTH_BAR_CENTER_EXCLUSION and the module docstring for why a lone bar needs that extra check and a
    paired one does not.

    Below is preferred over above, not just "whichever is closer": measured on a real screenshot
    (2026-09-25) a warrior's class-color name strip directly above the health bar was blue enough to pass
    the mana range and sat flush against it (gap 0px) - closer than the rogue target's real, correctly
    colored energy bar sitting flush below it (gap 1px) in the same frame. Comparing gaps alone would have
    picked the decoration over the real bar by that 1px; always trying "below" first does not."""
    health = _bar_candidates(hsv, *HEALTH_HSV_RANGE, label='health')
    resource = [box for i, (lo, hi) in enumerate(RESOURCE_HSV_RANGES)
                for box in _bar_candidates(hsv, lo, hi, label='resource range ' + str(i))]
    lo_frac, hi_frac = HEALTH_BAR_CENTER_EXCLUSION
    found = []
    for hx, hy, hw, hh in health:
        below, above = [], []
        for rx, ry, rw, rh in resource:
            if abs(rw - hw) > 0.5 * hw or abs(rx - hx) > 0.5 * hw:
                continue
            gap_below, gap_above = abs(ry - (hy + hh)), abs(hy - (ry + rh))
            if gap_below <= 3 * hh:
                below.append((gap_below, (rx, ry, rw, rh)))
            elif gap_above <= 3 * hh:
                above.append((gap_above, (rx, ry, rw, rh)))
        candidates = below or above
        if candidates:
            gap, resource_box = min(candidates, key=lambda c: c[0])
            found.append(((hx, hy, hw, hh), resource_box))
            _debug('health ' + str((hx, hy, hw, hh)) + ' paired with resource ' + str(resource_box)
                   + ' (' + ('below' if candidates is below else 'above') + ', gap ' + str(gap) + 'px, '
                   + str(len(below)) + ' below-candidate(s), ' + str(len(above)) + ' above-candidate(s) seen)')
            continue
        center_frac = (hx + hw / 2) / img_width
        centered = lo_frac <= center_frac <= hi_frac
        tall_enough = hh >= HEALTH_BAR_MIN_HEIGHT_FOR_LONE_BAR
        if tall_enough and not centered:
            found.append(((hx, hy, hw, hh), None))
            _debug('health ' + str((hx, hy, hw, hh)) + ' has no resource bar nearby, accepted as a lone bar '
                   '(center ' + str(round(center_frac, 2)) + ' of width, outside ' + str(HEALTH_BAR_CENTER_EXCLUSION) + ')')
        else:
            _debug('health ' + str((hx, hy, hw, hh)) + ' has no resource bar nearby, rejected ('
                   + ('too short: ' + str(hh) + 'px < ' + str(HEALTH_BAR_MIN_HEIGHT_FOR_LONE_BAR) + 'px'
                      if not tall_enough else 'centered at ' + str(round(center_frac, 2)) + ' of width, like the fishing cast bar') + ')')
    return found


def find_ui_regions(img_bgr: np.ndarray):
    """[(x0, y0, x1, y1)] in screenshot pixels, one per unit frame found (a player frame, a target frame,
    ...). Padding is symmetric left/right on purpose: the player frame hangs its portrait to the bar's left,
    but WoW's default target frame mirrors that with the portrait on the right, and one fixed asymmetric pad
    cannot fit both - see the module docstring's real-screenshot comparison.

    Padding scales off the bar's own width, not its height: measured on a real screenshot (2026-09-25) a
    compact addon skin's bars were only 9-18px tall but still fronted by a full-size portrait, and a
    height-scaled pad (5x height, ~45px) fell 120px short of that portrait's edge - a real miss this session
    watched happen. Width tracks a frame's overall footprint far more consistently across skins than a bar's
    height, which some skins make arbitrarily thin on its own."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    regions = []
    for (hx, hy, hw, hh), resource_box in _health_bars_with_resource(hsv, w):
        if resource_box is not None:
            rx, ry, rw, rh = resource_box
            top, bottom = min(hy, ry), max(hy + hh, ry + rh)
            left, right = min(hx, rx), max(hx + hw, rx + rw)
        else:
            # No resource bar to size against (see _health_bars_with_resource) - assume there is invisible
            # room for one the same height as the health bar, directly below, so the box still roughly
            # matches a paired one instead of hugging the health bar alone.
            top, bottom = hy, hy + 2 * hh
            left, right = hx, hx + hw
        pad_x = int(2.0 * hw)
        pad_top = int(0.6 * hw)
        pad_bottom = int(0.6 * hw)
        region = (max(0, left - pad_x), max(0, top - pad_top), min(w, right + pad_x), min(h, bottom + pad_bottom))
        regions.append(region)
        _debug('region ' + str(region) + ' padded from health ' + str((hx, hy, hw, hh))
               + (' + resource ' + str(resource_box) if resource_box is not None else ' alone (no resource bar)'))
    return regions


def _to_fractions(regions, w, h):
    return tuple(((x0 / w, x1 / w), (y0 / h, y1 / h)) for x0, y0, x1, y1 in regions)


def calibrate(screenshot_path: str):
    """Detect UI regions on `screenshot_path`, save an annotated preview, and write var/ui_regions.json.
    Returns the regions as window fractions, or None if the screenshot could not be read."""
    img = cv2.imread(screenshot_path)
    if img is None:
        print('Could not read ' + screenshot_path)
        return None
    h, w = img.shape[:2]
    regions = find_ui_regions(img)
    if not regions:
        print('No health+resource bar pair found - is a unit frame visible in this screenshot? '
              'Nothing written; float_detector.py keeps using its UI_EXCLUDE_REGIONS default. '
              'Re-run with --trace to see why each candidate bar was rejected.')
        return None

    annotated = img.copy()
    for x0, y0, x1, y1 in regions:
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 3)
    os.makedirs(os.path.dirname(CALIBRATION_PREVIEW_PATH), exist_ok=True)
    cv2.imwrite(CALIBRATION_PREVIEW_PATH, annotated)

    fractions = _to_fractions(regions, w, h)
    os.makedirs(os.path.dirname(UI_REGIONS_PATH), exist_ok=True)
    with open(UI_REGIONS_PATH, 'w') as f:
        json.dump(fractions, f, indent=1)

    print('Found ' + str(len(regions)) + ' region(s), saved to ' + UI_REGIONS_PATH)
    print('Look at ' + CALIBRATION_PREVIEW_PATH + ' - the red box(es) should fully cover each unit frame '
          '(portrait, bars, name) with some margin. If not, delete ' + UI_REGIONS_PATH + ' and try a '
          'screenshot where your own frame and a target\'s are both clearly visible, or re-run with --trace '
          'to see how each region was derived.')
    return fractions


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--trace' in args:
        args.remove('--trace')
        UI_CALIBRATE_TRACE = True
    path = args[0] if args else 'var/fishing_session.png'
    calibrate(path)
