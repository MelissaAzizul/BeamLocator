
Main · PY
"""
main.py
Runs the program: camera -> beam math -> screen.
 
How to use:
  1. Move the beams FAR apart.
  2. Drag a box around the REFERENCE beam (the one that stays still).
  3. Drag a box around the MOVING beam.
  4. The program profiles both beams, then tracks the moving one.
 
Keys:
  r = re-profile using the same boxes (beams must be far apart again)
  c = clear the boxes and start over
  q = quit
"""
 
import numpy as np
import cv2
 
import camera
import beam_detection as bd
 
# Camera settings. Adjust so the beam peaks are bright but NOT saturated.
EXPOSURE_US = 1000.0
GAIN_DB = 0.0
 
# Pixel size of the GS3-U3-32S4M sensor, for converting pixels to micrometres.
PIXEL_SIZE_UM = 3.45
 
 
# ------------------------------------------------------------
# Mouse drag (same as before)
# ------------------------------------------------------------
dragging = False
start_x = 0
start_y = 0
end_x = 0
end_y = 0
boxes = []  # box 0 = reference beam, box 1 = moving beam
 
 
def on_mouse(event, x, y, flags, param):
    global dragging, start_x, start_y, end_x, end_y
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_x, start_y, end_x, end_y = x, y, x, y
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        end_x, end_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        x1, x2 = min(start_x, x), max(start_x, x)
        y1, y2 = min(start_y, y), max(start_y, y)
        if (x2 - x1) > 2 and (y2 - y1) > 2 and len(boxes) < 2:
            boxes.append((x1, y1, x2, y2))
 
 
# ------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------
def draw_beam(display, beam, color, label):
    """Cross at the center, ellipse at 2 sigma (the 1/e^2 beam edge)."""
    cx = int(round(beam["x"]))
    cy = int(round(beam["y"]))
    cv2.drawMarker(display, (cx, cy), color, cv2.MARKER_CROSS, 15, 1)
    cv2.ellipse(display, (cx, cy), (int(2 * beam["sx"]), int(2 * beam["sy"])),
                0, 0, 360, color, 1)
    cv2.putText(display, label, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
 
 
def make_display(raw):
    """8-bit colour copy for the screen only. All maths uses 'raw'."""
    display = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
 
 
# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
def main():
    ref = None      # profile of the reference beam
    mov = None      # profile of the moving beam
    mov_x = None    # moving beam's latest position
    mov_y = None
    need_profile = False
    message = ""
 
    camera.open_camera(EXPOSURE_US, GAIN_DB)
    cv2.namedWindow("Beam Detection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Beam Detection", on_mouse)
 
    try:
        while True:
            raw = camera.grab_frame()
            if raw is None:
                continue
            display = make_display(raw)
 
            # ---- Boxes ----
            if dragging:
                cv2.rectangle(display, (start_x, start_y), (end_x, end_y), (0, 255, 255), 1)
            for box in boxes:
                cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
 
            lines = []
 
            # ---- Step 1: profile once both boxes exist ----
            if len(boxes) == 2 and ref is None:
                need_profile = True
 
            if need_profile:
                need_profile = False
                ref = bd.profile_beam(raw, boxes[0])
                mov = bd.profile_beam(raw, boxes[1])
                if ref is None or mov is None:
                    message = "Profiling failed. Press c and draw the boxes again."
                    ref = mov = None
                elif not bd.profiles_look_clean(ref, mov):
                    message = "Beams too close to profile. Move them apart, press r."
                    ref = mov = None
                else:
                    mov_x, mov_y = mov["x"], mov["y"]
                    message = "Profiled. Tracking."
                    print("Reference: x=%.2f y=%.2f sx=%.2f sy=%.2f amp=%.0f" %
                          (ref["x"], ref["y"], ref["sx"], ref["sy"], ref["amp"]))
                    print("Moving:    x=%.2f y=%.2f sx=%.2f sy=%.2f amp=%.0f" %
                          (mov["x"], mov["y"], mov["sx"], mov["sy"], mov["amp"]))
 
            # ---- Step 2: track ----
            if len(boxes) < 2:
                lines.append("Drag a box around the " +
                             ("REFERENCE (still) beam" if len(boxes) == 0 else "MOVING beam"))
            elif ref is not None:
                result = bd.track_moving_beam(raw, ref, mov, mov_x, mov_y)
                if result is None:
                    message = "Lost the moving beam. Move beams apart and press r."
                else:
                    mov_x, mov_y = result
 
                # The moving beam = its profile shape at the new position
                moving_now = {"x": mov_x, "y": mov_y, "sx": mov["sx"], "sy": mov["sy"]}
 
                draw_beam(display, ref, (255, 0, 0), "ref")
                draw_beam(display, moving_now, (0, 0, 255), "mov")
 
                sep_px = bd.distance(ref, moving_now)
                lines.append("Moving: x=%.2f y=%.2f px" % (mov_x, mov_y))
                lines.append("Separation: %.2f px (%.2f um on sensor)" %
                             (sep_px, sep_px * PIXEL_SIZE_UM))
                lines.append("Overlap: %.3f" % bd.overlap(ref, moving_now))
 
            if message:
                lines.append(message)
 
            for i, text in enumerate(lines):
                cv2.putText(display, text, (10, 30 + 28 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Beam Detection", display)
 
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                boxes.clear()
                ref = mov = None
                message = ""
            if key == ord('r') and len(boxes) == 2:
                need_profile = True
 
    finally:
        cv2.destroyAllWindows()
        camera.close_camera()
 
 
if __name__ == "__main__":
    main()
    