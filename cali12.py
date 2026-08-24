#!/usr/bin/env python3
"""
check_calibration.py

Run this FIRST, before touching distance.py.

It answers three questions with hard evidence, not guesses:

  1. Are the saved calibration images (left_XX.jpg / right_XX.jpg) actually
     640 px wide (per-eye), or 1280 px wide (still-combined frame)?
  2. Is the camera matrix / P1 / P2 principal point physically possible for
     a 640x720 per-eye image?
  3. Does the Q matrix's baseline term match the reported baseline?

No camera required. Run on the Pi (or anywhere with the files present):

    python3 check_calibration.py
"""

import os
import sys
import glob
import numpy as np
import cv2

CALIB_DIR = "/home/pi/Desktop/MahirCode/calibration_images"
CALIB_FILE = os.path.join(CALIB_DIR, "stereo_calibration.npz")


def check_saved_image_sizes():
    print("=" * 70)
    print("1. CHECKING SAVED CALIBRATION IMAGE DIMENSIONS")
    print("=" * 70)

    for side in ("left", "right"):
        folder = os.path.join(CALIB_DIR, side)
        files = sorted(glob.glob(os.path.join(folder, f"{side}_*.jpg")))
        if not files:
            print(f"  [!] No files found in {folder}")
            continue

        sample = files[0]
        img = cv2.imread(sample)
        if img is None:
            print(f"  [!] Could not read {sample}")
            continue

        h, w = img.shape[:2]
        print(f"  {side}/{os.path.basename(sample)}: {w}x{h}")

        if w == 1280:
            print(f"  [BUG FOUND] '{side}' images are 1280 px wide.")
            print("              These are still COMBINED frames (or were saved")
            print("              without cropping to the individual camera half).")
            print("              This is almost certainly why cx/cy came out wrong.")
        elif w == 640:
            print(f"  [OK] '{side}' images are correctly 640 px wide (per-eye).")
        else:
            print(f"  [?] Unexpected width {w}. Investigate manually.")

    # Check left vs right for accidental duplication (common bug: both saved
    # from the same crop, or both are the full combined frame)
    left_files = sorted(glob.glob(os.path.join(CALIB_DIR, "left", "left_*.jpg")))
    right_files = sorted(glob.glob(os.path.join(CALIB_DIR, "right", "right_*.jpg")))
    if left_files and right_files:
        l0 = cv2.imread(left_files[0])
        r0 = cv2.imread(right_files[0])
        if l0 is not None and r0 is not None and l0.shape == r0.shape:
            diff = cv2.absdiff(l0, r0)
            mean_diff = diff.mean()
            print(f"\n  Mean pixel difference between left_01 and right_01: {mean_diff:.2f}")
            if mean_diff < 2.0:
                print("  [BUG FOUND] left_01.jpg and right_01.jpg are nearly IDENTICAL.")
                print("              The capture script is almost certainly saving the")
                print("              same crop (or the same full frame) into both folders.")
            else:
                print("  [OK] Left and right images are meaningfully different (expected).")
    print()


def check_matrix_sanity():
    print("=" * 70)
    print("2. CHECKING CAMERA MATRIX / P1 / P2 SANITY vs 640x720 PER-EYE IMAGE")
    print("=" * 70)

    if not os.path.exists(CALIB_FILE):
        print(f"  [!] Calibration file not found: {CALIB_FILE}")
        return None

    d = np.load(CALIB_FILE)
    print(f"  Keys in npz: {list(d.keys())}")

    HALF_W, HALF_H = 640, 720

    for key in ("K1", "K2", "P1", "P2"):
        if key not in d:
            print(f"  [!] '{key}' missing from npz — distance.py will need to derive it "
                  f"or you need to re-save it from your calibration script.")
            continue
        M = d[key]
        cx, cy = M[0, 2], M[1, 2]
        fx, fy = M[0, 0], M[1, 1]
        print(f"\n  {key}: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

        if not (0 <= cx <= HALF_W):
            print(f"    [BUG] cx={cx:.1f} is OUTSIDE the valid 0-{HALF_W} range for a "
                  f"{HALF_W}x{HALF_H} image.")
            print(f"          This matrix was almost certainly computed against a "
                  f"wider image (e.g. 1280 px), not the true {HALF_W}-px-wide per-eye image.")
        else:
            print(f"    [OK] cx is within the valid 0-{HALF_W} range.")

        if not (0 <= cy <= HALF_H):
            print(f"    [BUG] cy={cy:.1f} is OUTSIDE the valid 0-{HALF_H} range.")
        else:
            print(f"    [OK] cy is within the valid 0-{HALF_H} range.")

    return d


def check_q_consistency(d):
    print()
    print("=" * 70)
    print("3. CHECKING Q MATRIX / BASELINE CONSISTENCY")
    print("=" * 70)
    if d is None or "Q" not in d:
        print("  [!] Q not available, skipping.")
        return

    Q = d["Q"]
    q32 = Q[3, 2]
    implied_baseline_mm = 1.0 / q32 if q32 != 0 else float("inf")
    print(f"  Q[3,2] = {q32:.8f}  ->  implied baseline = {implied_baseline_mm:.3f} mm")

    if "T" in d:
        T = d["T"].flatten()
        baseline_from_T = np.linalg.norm(T)
        print(f"  |T| (from stereoCalibrate)      = {baseline_from_T:.3f} mm")
        if abs(baseline_from_T - implied_baseline_mm) > 1.0:
            print("  [WARNING] Q-implied baseline and |T| disagree by >1mm.")
        else:
            print("  [OK] Q and T agree on baseline.")
    else:
        print("  ('T' not stored in npz — add it when you re-save calibration, "
              "it's useful for debugging.)")


if __name__ == "__main__":
    check_saved_image_sizes()
    d = check_matrix_sanity()
    check_q_consistency(d)

    print()
    print("=" * 70)
    print("NEXT STEP")
    print("=" * 70)
    print("""
If the image-size bug is confirmed above (1280-wide calibration images,
or cx outside 0-640), you do NOT need to recapture your 57 photo pairs.
Options, in order of preference:

  A) If left_XX.jpg / right_XX.jpg are 1280 wide and are genuinely the
     FULL combined stereo frame (checkerboard visible in both halves),
     crop each to its correct half:
         left  = img[:, 0:640]
         right = img[:, 640:1280]
     then rerun your checkerboard-corner detection + calibration on the
     CROPPED images, with imageSize=(640, 720) passed to calibrateCamera
     and stereoCalibrate.

  B) If left_XX.jpg / right_XX.jpg are already individual per-eye 640-wide
     images but calibrateCamera/stereoCalibrate was simply called with
     imageSize=(1280, 720) instead of (640, 720), just fix that argument
     and rerun calibration on the SAME images. No recapture needed.

  C) If left_01.jpg and right_01.jpg turned out to be near-identical
     (see check #1), the capture script itself has a left/right split
     bug and needs to be fixed before ANY of the existing 57 pairs can
     be trusted — in that case recapture is the only option.

Re-run this script after regenerating stereo_calibration.npz to confirm
cx now falls inside 0-640.
""")
