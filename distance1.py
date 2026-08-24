#!/usr/bin/env python3
"""
distance.py - Real-time stereo distance measurement (IMX219-83, Raspberry Pi 5)

IMPORTANT: Run check_calibration.py FIRST. If it reports a cx/image-size bug,
fix the calibration before trusting any numbers from this script - rectifying
with a wrong-sized camera matrix will always produce plausible-looking but
wrong disparities, no matter how good the matching code is.

Design decisions (why this is different from a "normal" OpenCV stereo demo):

  * Rectification maps are built ONCE at startup (cv2.CV_16SC2, fixed-point),
    not every frame. This is almost always the #1 cause of ~0.3 FPS.
  * Stereo matching runs on grayscale, and on a DOWNSCALED copy of the
    rectified images for speed. The disparity result is rescaled back to
    full-resolution-equivalent pixels using scale division only - the
    calibration matrices (P1/P2/Q) are NEVER resized or multiplied, which
    avoids the 3x4 vs 3x3 broadcasting bug from your earlier version.
  * Distance is computed from a median over a filtered ROI, not a single
    pixel and not the whole frame - background disparity is excluded by
    ROI + validity masking, not by hoping SGBM figures it out.
  * numDisparities is derived from your calibrated f and baseline for the
    working distance range you actually need (see MIN_DISTANCE_MM below),
    instead of being an arbitrary guess.
  * "NO VALID OBJECT" is shown whenever too few ROI pixels have valid,
    plausible disparity - it will not report a random background distance.

Keys while running:
    q - quit
    d - toggle debug view (rectified L/R side-by-side with epipolar lines,
        + disparity heatmap)
    r - reset the temporal smoothing buffer
"""

import os
import time
import collections

import cv2
import numpy as np
from picamera2 import Picamera2

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CALIB_FILE = "/home/pi/Desktop/MahirCode/calibration_images/stereo_calibration.npz"

# NOTE: HALF_W / HALF_H are no longer hardcoded. They are read from the
# calibration file's own 'image_width' / 'image_height' keys at startup,
# because your calibration images turned out to be full 1280x720 per-eye
# frames (see check_calibration.py output), not halves of a combined frame.
# Hardcoding 640 here was the earlier bug.

# Stereo MATCHING runs on a downscaled copy of the rectified images for
# speed. Rectification itself always happens at full HALF_W x HALF_H using
# the real calibration - only the expensive matching step is shrunk.
MATCH_SCALE = 0.5                   # 640x720 -> 320x360 for SGBM

# Practical near-limit for distance. A 62mm-baseline stereo pair needs very
# large disparities to see 20-30cm objects (e.g. ~450px at 20cm with your
# f=1444), which usually exceeds both the matchable range and the FOV
# overlap of the two lenses. Start conservative and shrink this only after
# you've confirmed matching works well at 50-100cm.
MIN_DISTANCE_MM = 350                # 35 cm near limit used to size numDisparities
MAX_DISTANCE_MM = 3000               # 3 m far limit (clips absurd disparity-noise distances)

# ROI size, in per-eye pixels. Set as a fraction of the full frame - computed
# in main() once HALF_W/HALF_H are known (read from the calibration file).
# 0.95 = nearly full-frame, leaving a small border so the box stays visible
# on screen. NOTE: a near-full-frame ROI means background WILL be mixed into
# the median if it's visible around your test object - fine for isolating
# the accuracy question right now, but shrink this back down (or switch to
# a detection-driven ROI, as final_project.py already does) once you're
# doing real flower-distance measurements, or background will bias readings.
ROI_FRACTION = 0.5

# Empirical scale correction. Leave at 1.0 until you've measured at least 2-3
# known distances. If the reported distance is consistently off by roughly
# the same PERCENTAGE at every distance (e.g. always ~18% too close), that's
# a calibration scale error (baseline/focal length), and can be corrected
# here directly: CALIBRATION_SCALE_CORRECTION = actual_distance / measured_distance
# from one of your test points. If the error is NOT a consistent percentage
# (e.g. fine at 50cm but bad at 30cm), it's not a scale issue - don't use
# this, come back and we'll look at the matching/ROI instead.
CALIBRATION_SCALE_CORRECTION = 1.0

MIN_VALID_PIXEL_PCT = 35.0           # min % of ROI with valid disparity to accept a reading
                                      # (raised from 8% - low percentages tend to be edge/
                                      # background bleed-through around a textureless object,
                                      # not a real reading on the object itself)
SMOOTH_WINDOW = 7                    # frames of median smoothing
MAX_JUMP_CM = 30                     # a single-frame jump larger than this vs. the running
                                      # median is treated as suspect (needs 2 confirmations)


# ---------------------------------------------------------------------------
# CALIBRATION LOADING + SANITY CHECK
# ---------------------------------------------------------------------------
def load_calibration(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")
    d = np.load(path)
    keys = set(d.keys())

    # Support both naming conventions: the "standard" K1/D1/K2/D2 and the
    # names your calibration script actually uses (mtxL/distL/mtxR/distR).
    name_map_options = [
        {"K1": "K1", "D1": "D1", "K2": "K2", "D2": "D2"},
        {"K1": "mtxL", "D1": "distL", "K2": "mtxR", "D2": "distR"},
    ]
    resolved = None
    for mapping in name_map_options:
        if all(v in keys for v in mapping.values()):
            resolved = mapping
            break
    if resolved is None:
        raise KeyError(
            f"Could not find camera matrix/distortion keys under either naming "
            f"convention. Available keys: {sorted(keys)}"
        )

    calib = {
        "K1": d[resolved["K1"]], "D1": d[resolved["D1"]],
        "K2": d[resolved["K2"]], "D2": d[resolved["D2"]],
    }
    for k in ("R1", "R2", "P1", "P2", "Q"):
        if k not in keys:
            raise KeyError(f"'{k}' missing from {path}. Available keys: {sorted(keys)}")
        calib[k] = d[k]

    # Optional but very useful: precomputed maps and stored image size, if present.
    for k in ("map1L", "map2L", "map1R", "map2R", "image_width", "image_height", "baseline_mm"):
        if k in keys:
            calib[k] = d[k]

    return calib


def sanity_check_calibration(calib, half_w, half_h):
    """
    Refuse to silently run on a calibration that is geometrically impossible
    for the runtime image size. This is exactly the bug that produced your
    cx=914 on a 640-wide image - catch it here instead of downstream.
    """
    errors = []
    for name in ("P1", "P2"):
        cx, cy = calib[name][0, 2], calib[name][1, 2]
        if not (0 <= cx <= half_w):
            errors.append(
                f"{name} cx={cx:.1f} is outside 0-{half_w}. This calibration was "
                f"almost certainly computed for a different image width than "
                f"{half_w}px. Run check_calibration.py and fix calibration before "
                f"trusting distances from this script."
            )
        if not (0 <= cy <= half_h):
            errors.append(f"{name} cy={cy:.1f} is outside 0-{half_h}.")
    return errors


def build_rectify_maps(calib, size_wh):
    """
    Get fixed-point rectification maps. Prefers the maps your calibration
    script already saved (map1L/map2L/map1R/map2R) if their shape matches
    the runtime capture size - only rebuilds if they're missing or stale.
    size_wh = (width, height).
    """
    want_shape = (size_wh[1], size_wh[0])  # (height, width) for a numpy array

    have_precomputed = all(k in calib for k in ("map1L", "map2L", "map1R", "map2R"))
    if have_precomputed:
        m1x = calib["map1L"]
        m2x = calib["map1R"]
        shape_ok = (m1x.shape[:2] == want_shape) and (m2x.shape[:2] == want_shape)
        if shape_ok:
            print(f"Using precomputed rectification maps from calibration file "
                  f"(shape {m1x.shape[:2]} matches capture size {want_shape}).")
            return (calib["map1L"], calib["map2L"]), (calib["map1R"], calib["map2R"])
        else:
            print(f"Precomputed maps have shape {m1x.shape[:2]}, but runtime capture "
                  f"size needs {want_shape}. Rebuilding maps instead.")

    map1 = cv2.initUndistortRectifyMap(
        calib["K1"], calib["D1"], calib["R1"], calib["P1"], size_wh, cv2.CV_16SC2
    )
    map2 = cv2.initUndistortRectifyMap(
        calib["K2"], calib["D2"], calib["R2"], calib["P2"], size_wh, cv2.CV_16SC2
    )
    return map1, map2


def compute_num_disparities(fx, baseline_mm, min_distance_mm, match_scale):
    """Size the disparity search range around the distances you actually need."""
    max_disp_full = fx * baseline_mm / min_distance_mm
    max_disp_scaled = max_disp_full * match_scale
    num_disp = int(np.ceil(max_disp_scaled / 16.0) * 16)
    return max(num_disp, 16)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Loading calibration...")
    calib = load_calibration(CALIB_FILE)

    # Per-eye image size comes from the calibration file itself now, not a
    # hardcoded assumption. Falls back to 640x720 only if the calibration
    # file genuinely doesn't record it (older calibration script version).
    HALF_W = int(calib["image_width"]) if "image_width" in calib else 640
    HALF_H = int(calib["image_height"]) if "image_height" in calib else 720
    print(f"Per-eye calibration image size: {HALF_W}x{HALF_H}")
    ROI_CX, ROI_CY = HALF_W // 2, HALF_H // 2
    ROI_W, ROI_H = int(HALF_W * ROI_FRACTION), int(HALF_H * ROI_FRACTION)

    errors = sanity_check_calibration(calib, HALF_W, HALF_H)
    if errors:
        print("\n" + "=" * 70)
        print("CALIBRATION SANITY CHECK FAILED")
        print("=" * 70)
        for e in errors:
            print(" -", e)
        print("=" * 70)
        print("Refusing to run with a geometrically-impossible calibration - the")
        print("distances you'd get would be meaningless. Run check_calibration.py,")
        print("fix the calibration, and re-run.\n")
        return
    else:
        print("Calibration sanity check passed (cx/cy consistent with "
              f"{HALF_W}x{HALF_H} per-eye images).")

    fx = calib["P1"][0, 0]
    baseline_mm = -calib["P2"][0, 3] / calib["P2"][0, 0]  # standard OpenCV convention
    print(f"fx = {fx:.2f} px   baseline = {baseline_mm:.3f} mm")

    print("Building rectification maps (once)...")
    (map1x, map1y), (map2x, map2y) = build_rectify_maps(calib, (HALF_W, HALF_H))

    num_disp = compute_num_disparities(fx, baseline_mm, MIN_DISTANCE_MM, MATCH_SCALE)
    print(f"numDisparities (at match scale {MATCH_SCALE}) = {num_disp} "
          f"(covers down to ~{MIN_DISTANCE_MM/10:.0f} cm)")

    # --- Detect camera topology: one combined-frame camera, or two independent ---
    cam_info = Picamera2.global_camera_info()
    print(f"Cameras detected by libcamera: {len(cam_info)}")
    dual_camera_mode = len(cam_info) >= 2
    if dual_camera_mode:
        print("Using DUAL-CAMERA mode: opening camera_num=0 (left) and "
              "camera_num=1 (right) independently, each captured at "
              f"{HALF_W}x{HALF_H} - no frame splitting.")
    else:
        print("Using SINGLE-CAMERA SPLIT mode: capturing one combined frame "
              f"and splitting into left/right halves of {HALF_W}x{HALF_H} each "
              f"(combined capture width = {HALF_W * 2}).")

    block_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 1 * block_size ** 2,
        P2=32 * 1 * block_size ** 2,
        disp12MaxDiff=2,
        uniquenessRatio=15,   # raised from 10 - rejects weaker/ambiguous matches more aggressively
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,  # fast mode, good fit for Pi 5
    )

    print("Starting camera(s)...")
    if dual_camera_mode:
        picam_left = Picamera2(camera_num=0)
        picam_right = Picamera2(camera_num=1)
        cfg_left = picam_left.create_video_configuration(
            main={"size": (HALF_W, HALF_H), "format": "RGB888"}
        )
        cfg_right = picam_right.create_video_configuration(
            main={"size": (HALF_W, HALF_H), "format": "RGB888"}
        )
        picam_left.configure(cfg_left)
        picam_right.configure(cfg_right)
        picam_left.start()
        picam_right.start()
    else:
        CAPTURE_SIZE = (HALF_W * 2, HALF_H)
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": CAPTURE_SIZE, "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
    time.sleep(1.0)  # let AE/AWB settle

    small_w = int(HALF_W * MATCH_SCALE)
    small_h = int(HALF_H * MATCH_SCALE)

    # ROI expressed in the small (matching-resolution) image, since that's
    # where we actually index the disparity array.
    roi_x0 = int((ROI_CX - ROI_W / 2) * MATCH_SCALE)
    roi_y0 = int((ROI_CY - ROI_H / 2) * MATCH_SCALE)
    roi_x1 = int((ROI_CX + ROI_W / 2) * MATCH_SCALE)
    roi_y1 = int((ROI_CY + ROI_H / 2) * MATCH_SCALE)
    roi_x0, roi_y0 = max(roi_x0, 0), max(roi_y0, 0)
    roi_x1, roi_y1 = min(roi_x1, small_w), min(roi_y1, small_h)

    history = collections.deque(maxlen=SMOOTH_WINDOW)
    pending_jump = None  # (value, confirmations) - for jump rejection

    debug_view = False
    prev_time = time.time()
    fps = 0.0

    print("\nRunning. Press 'q' to quit, 'd' to toggle debug view, 'r' to reset smoothing.\n")

    try:
        while True:
            if dual_camera_mode:
                left_raw = picam_left.capture_array()
                right_raw = picam_right.capture_array()
                if left_raw.shape[1] != HALF_W or left_raw.shape[0] != HALF_H:
                    print(f"[WARN] Left camera frame is {left_raw.shape[1]}x{left_raw.shape[0]}, "
                          f"expected {HALF_W}x{HALF_H}.")
            else:
                frame = picam2.capture_array()
                if frame.shape[1] != CAPTURE_SIZE[0] or frame.shape[0] != CAPTURE_SIZE[1]:
                    print(f"[WARN] Captured frame is {frame.shape[1]}x{frame.shape[0]}, "
                          f"expected {CAPTURE_SIZE[0]}x{CAPTURE_SIZE[1]}. "
                          f"Check Picamera2 configuration.")
                left_raw = frame[:, :HALF_W]
                right_raw = frame[:, HALF_W:]

            left_gray = cv2.cvtColor(left_raw, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_raw, cv2.COLOR_RGB2GRAY)

            # Rectify at FULL calibrated resolution - this is the geometry
            # that matches P1/P2/Q. Never skip or resize this step.
            rect_left = cv2.remap(left_gray, map1x, map1y, cv2.INTER_LINEAR)
            rect_right = cv2.remap(right_gray, map2x, map2y, cv2.INTER_LINEAR)

            # Downscale ONLY for the expensive matching step.
            small_left = cv2.resize(rect_left, (small_w, small_h), interpolation=cv2.INTER_AREA)
            small_right = cv2.resize(rect_right, (small_w, small_h), interpolation=cv2.INTER_AREA)

            raw_disp = stereo.compute(small_left, small_right).astype(np.float32) / 16.0

            # Rescale disparity values (not the array size) back to
            # full-resolution-equivalent pixels so we can use the
            # unscaled fx/baseline directly in Z = f*B/d.
            disp_full_equiv = raw_disp / MATCH_SCALE

            roi_disp = disp_full_equiv[roi_y0:roi_y1, roi_x0:roi_x1]

            # Valid disparity mask: positive, and within the distance range
            # we actually care about (rejects both SGBM's invalid marker
            # values and physically-impossible near/far noise).
            with np.errstate(divide="ignore", invalid="ignore"):
                z_mm_map = np.where(roi_disp > 0.5, (fx * baseline_mm) / roi_disp, 0)
            valid_mask = (roi_disp > 0.5) & (z_mm_map >= MIN_DISTANCE_MM) & (z_mm_map <= MAX_DISTANCE_MM)

            valid_count = int(valid_mask.sum())
            total_count = valid_mask.size
            valid_pct = 100.0 * valid_count / total_count if total_count else 0.0

            distance_cm = None
            median_disp = 0.0

            if valid_pct >= MIN_VALID_PIXEL_PCT:
                valid_z = z_mm_map[valid_mask]
                median_disp = float(np.median(roi_disp[valid_mask]))
                raw_distance_mm = float(np.median(valid_z)) * CALIBRATION_SCALE_CORRECTION
                raw_distance_cm = raw_distance_mm / 10.0

                # Jump rejection: if this reading is far from the current
                # smoothed median, require it to repeat before accepting it,
                # instead of either blindly trusting it or blindly ignoring it.
                if history:
                    current_median = float(np.median(history))
                    if abs(raw_distance_cm - current_median) > MAX_JUMP_CM:
                        if pending_jump is not None and abs(pending_jump[0] - raw_distance_cm) < 5:
                            pending_jump = (raw_distance_cm, pending_jump[1] + 1)
                        else:
                            pending_jump = (raw_distance_cm, 1)

                        if pending_jump[1] >= 2:
                            history.append(raw_distance_cm)
                            pending_jump = None
                        # else: hold - don't push this outlier into history yet
                    else:
                        pending_jump = None
                        history.append(raw_distance_cm)
                else:
                    history.append(raw_distance_cm)

                if history:
                    distance_cm = float(np.median(history))

            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = (0.9 * fps + 0.1 * (1.0 / dt)) if dt > 0 else fps

            # --------------------------- display ---------------------------
            display = cv2.cvtColor(rect_left, cv2.COLOR_GRAY2BGR)
            full_roi_pt1 = (ROI_CX - ROI_W // 2, ROI_CY - ROI_H // 2)
            full_roi_pt2 = (ROI_CX + ROI_W // 2, ROI_CY + ROI_H // 2)
            cv2.rectangle(display, full_roi_pt1, full_roi_pt2, (0, 255, 0), 2)

            if distance_cm is not None:
                label = f"Distance: {distance_cm:.1f} cm"
                color = (0, 255, 0)
            else:
                label = "NO VALID OBJECT"
                color = (0, 0, 255)

            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, f"Valid: {valid_pct:.1f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(display, f"Median disparity: {median_disp:.2f}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(display, f"Baseline: {baseline_mm:.2f} mm", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow("Distance", display)

            if debug_view:
                disp_vis = cv2.normalize(np.clip(raw_disp, 0, None), None, 0, 255, cv2.NORM_MINMAX)
                disp_vis = disp_vis.astype(np.uint8)
                disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
                cv2.imshow("Disparity", disp_color)

                # Epipolar sanity check: horizontal lines should cross the
                # SAME feature at the SAME height in both rectified images.
                stacked = np.hstack([rect_left, rect_right])
                stacked_bgr = cv2.cvtColor(stacked, cv2.COLOR_GRAY2BGR)
                for y in range(0, stacked_bgr.shape[0], 40):
                    cv2.line(stacked_bgr, (0, y), (stacked_bgr.shape[1], y), (0, 255, 0), 1)
                cv2.imshow("Rectified L | R (epipolar check)", stacked_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                debug_view = not debug_view
                if not debug_view:
                    cv2.destroyWindow("Disparity")
                    cv2.destroyWindow("Rectified L | R (epipolar check)")
            elif key == ord("r"):
                history.clear()
                pending_jump = None
                print("Smoothing buffer reset.")

    finally:
        if dual_camera_mode:
            picam_left.stop()
            picam_right.stop()
        else:
            picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
