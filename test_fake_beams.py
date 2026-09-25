Test fake beams · PY
"""
test_fake_beams.py
Tests beam_detection.py WITHOUT the camera, using fake images where
we know the true answer. Run it after every change to the maths.
 
The moving beam starts far away, slides right across the reference
beam (full overlap), and keeps going out the other side.
"""
 
import numpy as np
import beam_detection as bd
 
rng = np.random.default_rng(0)
HEIGHT, WIDTH = 300, 400
rows, cols = np.indices((HEIGHT, WIDTH))
 
# True beams (what we want the code to find)
REF = {"amp": 1800, "x": 200.4, "y": 150.3, "sx": 9.0, "sy": 11.0}
MOV = {"amp": 2000, "sx": 8.0, "sy": 8.0}
BACKGROUND = 100
NOISE = 10
 
 
def fake_frame(mov_x, mov_y):
    img = BACKGROUND + rng.normal(0, NOISE, (HEIGHT, WIDTH))
    img += bd.gaussian(cols, rows, REF["amp"], REF["x"], REF["y"], REF["sx"], REF["sy"])
    img += bd.gaussian(cols, rows, MOV["amp"], mov_x, mov_y, MOV["sx"], MOV["sy"])
    return np.clip(img, 0, 65535).astype(np.uint16)
 
 
# ---- Profile with the beams far apart ----
start_x, start_y = 80.0, 147.0
frame = fake_frame(start_x, start_y)
ref = bd.profile_beam(frame, (160, 100, 240, 200))
mov = bd.profile_beam(frame, (40, 100, 120, 200))
print("Profile  ref: x=%.2f y=%.2f sx=%.2f sy=%.2f   (true %.2f %.2f %.1f %.1f)" %
      (ref["x"], ref["y"], ref["sx"], ref["sy"], REF["x"], REF["y"], REF["sx"], REF["sy"]))
print("Profile  mov: x=%.2f y=%.2f sx=%.2f sy=%.2f   (true %.2f %.2f %.1f %.1f)" %
      (mov["x"], mov["y"], mov["sx"], mov["sy"], start_x, start_y, MOV["sx"], MOV["sy"]))
print()
 
# ---- Slide the moving beam across the reference beam ----
print("%8s %10s %10s %9s %9s" % ("true x", "found x", "error x", "error y", "overlap"))
x_now, y_now = mov["x"], mov["y"]
worst = 0.0
for true_x in np.arange(80.0, 320.0, 2.37):
    true_y = 147.0 + 3.0 * np.sin(true_x / 30.0)   # small wobble in y too
    result = bd.track_moving_beam(fake_frame(true_x, true_y), ref, mov, x_now, y_now)
    if result is None:
        print("LOST at true x = %.2f" % true_x)
        break
    x_now, y_now = result
    err_x = x_now - true_x
    err_y = y_now - true_y
    worst = max(worst, abs(err_x), abs(err_y))
    moving_now = {"x": x_now, "y": y_now, "sx": mov["sx"], "sy": mov["sy"]}
    if abs(true_x - REF["x"]) < 40 or int(true_x) % 40 == 0:
        print("%8.2f %10.2f %+10.3f %+9.3f %9.3f" %
              (true_x, x_now, err_x, err_y, bd.overlap(ref, moving_now)))
 
print()
print("Worst error over the whole run: %.3f px" % worst)