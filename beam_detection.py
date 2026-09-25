# All the beam math
# Plan:
"""
 1. PROFILE: while the beams are far apart, fit each beam on its own
     to learn its size (sigma), brightness (amplitude) and position.

 2.  TRACK: after that, only the MOVING beam's x and y are unknown.
     Each frame we fit a model that already contains the reference beam
     (fixed) and the moving beam (fixed shape, free position).
     Because the shapes are known, this keeps working even when the
     beams overlap.
"""

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit


def clamp_box(box, shape):
    """Make sure the box stays inside the image."""
    x1, y1, x2, y2 = box
    height, width = shape
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))
    return x1, y1, x2, y2
 #whats shape?

 
def gaussian(x, y, amp, cx, cy, sx, sy):
    """One 2D gaussian beam (a smooth hill of light)."""
    return amp * np.exp(-((x - cx) ** 2 / (2 * sx ** 2) + (y - cy) ** 2 / (2 * sy ** 2)))

 
def distance(beam_a, beam_b):
    return np.hypot(beam_a["x"] - beam_b["x"], beam_a["y"] - beam_b["y"])


# ------------------------------------------------------------
# Step 1: PROFILE (beams far apart)
# ------------------------------------------------------------
 
def rough_fit(raw, box):
    """
    Quick estimate using moments (no curve fitting).
    Only used as the starting guess for profile_beam().
    """
    x1, y1, x2, y2 = clamp_box(box, raw.shape)
    area = raw[y1:y2, x1:x2].astype(np.float64)
 
    background = np.median(area)
    signal = area - background
    signal[signal < 0] = 0
 
    peak = signal.max()
    if peak <= 0:
        return None
    signal[signal < 0.2 * peak] = 0
 
    cy, cx = ndimage.center_of_mass(signal)
    rows, cols = np.indices(signal.shape)
    total = signal.sum()
    sx = np.sqrt(np.sum(signal * (cols - cx) ** 2) / total)
    sy = np.sqrt(np.sum(signal * (rows - cy) ** 2) / total)
 
    return {"x": x1 + cx, "y": y1 + cy,
            "sx": max(sx, 1.0), "sy": max(sy, 1.0),
            "amp": peak, "bg": background}



def one_gaussian_model(xy, amp, cx, cy, sx, sy, offset):
    x, y = xy
    return (gaussian(x, y, amp, cx, cy, sx, sy) + offset).ravel()
 
 
def profile_beam(raw, box):
    """
    Full gaussian fit of ONE beam inside its box.
    Everything is free here: amplitude, position, size, background.
    Returns a dict describing the beam, or None if it fails.
    """
    guess = rough_fit(raw, box)
    if guess is None:
        return None
 
    x1, y1, x2, y2 = clamp_box(box, raw.shape)
    area = raw[y1:y2, x1:x2].astype(np.float64)
    rows, cols = np.indices(area.shape)
    x = cols + x1
    y = rows + y1
 
    start = [guess["amp"], guess["x"], guess["y"], guess["sx"], guess["sy"], guess["bg"]]
    try:
        p, _ = curve_fit(one_gaussian_model, (x, y), area.ravel(), p0=start, maxfev=5000)
    except RuntimeError:
        return None
 
    return {"amp": p[0], "x": p[1], "y": p[2],
            "sx": abs(p[3]), "sy": abs(p[4]), "bg": p[5]}


def profiles_look_clean(ref, mov):
    """
    Profiling only works if each box sees just one beam.
    If the beams are within 3 sigma of each other, they are too close.
    """
    too_close = 3 * (max(ref["sx"], ref["sy"]) + max(mov["sx"], mov["sy"]))
    return distance(ref, mov) > too_close

#what do you mean to close?



# ------------------------------------------------------------
# Step 2: TRACK the moving beam
# ------------------------------------------------------------
 
def track_moving_beam(raw, ref, mov, guess_x, guess_y):
    """
    Find the moving beam's new x, y.
 
    ref      = profile of the reference (still) beam: everything fixed
    mov      = profile of the moving beam: amp, sx, sy fixed
    guess_x, guess_y = where the moving beam was last frame
 
    Only 3 numbers are fitted: moving x, moving y, background.
    Returns (x, y) or None if we lost the beam.
    """
    # Look in a window around where the beam was last frame.
    # 4 sigma covers the whole beam; +10 px allows it to move between frames.
    half = int(4 * max(mov["sx"], mov["sy"])) + 10
    box = clamp_box((guess_x - half, guess_y - half, guess_x + half, guess_y + half),
                    raw.shape)
    x1, y1, x2, y2 = box
    area = raw[y1:y2, x1:x2].astype(np.float64)
    rows, cols = np.indices(area.shape)
    x = cols + x1
    y = rows + y1
 
    # Light from the reference beam in this window (known, fixed).
    # If the beams are far apart this is basically zero, and that's fine.
    ref_light = gaussian(x, y, ref["amp"], ref["x"], ref["y"], ref["sx"], ref["sy"])
 
    def model(xy, cx, cy, offset):
        xx, yy = xy
        moving_light = gaussian(xx, yy, mov["amp"], cx, cy, mov["sx"], mov["sy"])
        return (ref_light + moving_light + offset).ravel()
 
    start = [guess_x, guess_y, mov["bg"]]
    try:
        p, _ = curve_fit(model, (x, y), area.ravel(), p0=start, maxfev=2000)
    except RuntimeError:
        return None
 
    new_x, new_y = p[0], p[1]
 
    # If the answer is outside the window, the fit ran off: treat as lost.
    if not (x1 <= new_x < x2 and y1 <= new_y < y2):
        return None
    return new_x, new_y
 
 
# ------------------------------------------------------------
# Overlap
# ------------------------------------------------------------
 
def overlap(beam1, beam2):
    """
    Number from 0 to 1. 1 = same place and same size, 0 = not touching.
    Multiply the two beams pixel by pixel, add up, and scale so a beam
    compared with itself gives exactly 1.
    """
    left = min(beam1["x"] - 4 * beam1["sx"], beam2["x"] - 4 * beam2["sx"])
    right = max(beam1["x"] + 4 * beam1["sx"], beam2["x"] + 4 * beam2["sx"])
    top = min(beam1["y"] - 4 * beam1["sy"], beam2["y"] - 4 * beam2["sy"])
    bottom = max(beam1["y"] + 4 * beam1["sy"], beam2["y"] + 4 * beam2["sy"])
    x, y = np.meshgrid(np.arange(left, right), np.arange(top, bottom))
 
    i1 = gaussian(x, y, 1.0, beam1["x"], beam1["y"], beam1["sx"], beam1["sy"])
    i2 = gaussian(x, y, 1.0, beam2["x"], beam2["y"], beam2["sx"], beam2["sy"])
    return np.sum(i1 * i2) / np.sqrt(np.sum(i1 ** 2) * np.sum(i2 ** 2))

 
