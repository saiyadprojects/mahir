import cv2
import numpy as np
import os
import time

# ============================================================
# IMX219-83 STEREO DEPTH TEST
# Raspberry Pi 5
# ============================================================

CALIBRATION_FILE = "/home/pi/Desktop/MahirCode/calibration_images/stereo_calibration.npz"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# ============================================================
# LOAD CALIBRATION
# ============================================================

print("Loading calibration...")

data = np.load(CALIBRATION_FILE)

mtxL = data["mtxL"]
distL = data["distL"]

mtxR = data["mtxR"]
distR = data["distR"]

R1 = data["R1"]
R2 = data["R2"]

P1 = data["P1"]
P2 = data["P2"]

Q = data["Q"]

baseline = float(data["baseline_mm"])

print()
print("Calibration loaded")
print("Baseline:", baseline, "mm")

# ============================================================
# RECTIFICATION MAPS
# ============================================================

map1L, map2L = cv2.initUndistortRectifyMap(
    mtxL,
    distL,
    R1,
    P1,
    (IMAGE_WIDTH, IMAGE_HEIGHT),
    cv2.CV_32FC1
)

map1R, map2R = cv2.initUndistortRectifyMap(
    mtxR,
    distR,
    R2,
    P2,
    (IMAGE_WIDTH, IMAGE_HEIGHT),
    cv2.CV_32FC1
)

# ============================================================
# CAMERA
# ============================================================

from picamera2 import Picamera2

print()
print("Starting cameras...")

left_cam = Picamera2(0)
right_cam = Picamera2(1)

config_left = left_cam.create_preview_configuration(
    main={
        "size": (IMAGE_WIDTH, IMAGE_HEIGHT),
        "format": "RGB888"
    }
)

config_right = right_cam.create_preview_configuration(
    main={
        "size": (IMAGE_WIDTH, IMAGE_HEIGHT),
        "format": "RGB888"
    }
)

left_cam.configure(config_left)
right_cam.configure(config_right)

left_cam.start()
right_cam.start()

time.sleep(2)

print()
print("========================================")
print("STEREO TEST")
print("========================================")
print()
print("Q = Quit")
print()

# ============================================================
# STEREO MATCHER
# ============================================================

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=5,

    P1=8 * 3 * 5 * 5,
    P2=32 * 3 * 5 * 5,

    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2
)

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # CAPTURE
    # --------------------------------------------------------

    left = left_cam.capture_array()
    right = right_cam.capture_array()

    # --------------------------------------------------------
    # RGB → GRAY
    # --------------------------------------------------------

    grayL = cv2.cvtColor(
        left,
        cv2.COLOR_RGB2GRAY
    )

    grayR = cv2.cvtColor(
        right,
        cv2.COLOR_RGB2GRAY
    )

    # --------------------------------------------------------
    # RECTIFY
    # --------------------------------------------------------

    rectL = cv2.remap(
        grayL,
        map1L,
        map2L,
        cv2.INTER_LINEAR
    )

    rectR = cv2.remap(
        grayR,
        map1R,
        map2R,
        cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # DISPARITY
    # --------------------------------------------------------

    disparity = stereo.compute(
        rectL,
        rectR
    ).astype(np.float32) / 16.0

    # --------------------------------------------------------
    # DEPTH
    # --------------------------------------------------------

    # Q matrix converts disparity → 3D coordinates

    points_3d = cv2.reprojectImageTo3D(
        disparity,
        Q
    )

    # --------------------------------------------------------
    # DEPTH DISPLAY
    # --------------------------------------------------------

    valid = disparity > 0

    depth_display = np.zeros_like(
        disparity,
        dtype=np.uint8
    )

    if np.any(valid):

        disp_valid = disparity[valid]

        min_disp = np.percentile(
            disp_valid,
            5
        )

        max_disp = np.percentile(
            disp_valid,
            95
        )

        if max_disp > min_disp:

            normalized = (
                (disparity - min_disp) /
                (max_disp - min_disp)
            )

            normalized = np.clip(
                normalized,
                0,
                1
            )

            depth_display = (
                normalized * 255
            ).astype(np.uint8)

    depth_color = cv2.applyColorMap(
        depth_display,
        cv2.COLORMAP_JET
    )

    # --------------------------------------------------------
    # DRAW EPIPOLAR LINES
    # --------------------------------------------------------

    left_color = cv2.cvtColor(
        rectL,
        cv2.COLOR_GRAY2BGR
    )

    right_color = cv2.cvtColor(
        rectR,
        cv2.COLOR_GRAY2BGR
    )

    # horizontal lines every 60 pixels

    for y in range(
        0,
        IMAGE_HEIGHT,
        60
    ):

        cv2.line(
            left_color,
            (0, y),
            (IMAGE_WIDTH, y),
            (0, 255, 0),
            1
        )

        cv2.line(
            right_color,
            (0, y),
            (IMAGE_WIDTH, y),
            (0, 255, 0),
            1
        )

    # --------------------------------------------------------
    # RESIZE FOR DISPLAY
    # --------------------------------------------------------

    displayL = cv2.resize(
        left_color,
        (640, 360)
    )

    displayR = cv2.resize(
        right_color,
        (640, 360)
    )

    displayDepth = cv2.resize(
        depth_color,
        (640, 360)
    )

    top = cv2.hconcat([
        displayL,
        displayR
    ])

    bottom = cv2.hconcat([
        displayDepth,
        displayDepth
    ])

    output = cv2.vconcat([
        top,
        bottom
    ])

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    cv2.putText(
        output,
        "RECTIFIED LEFT",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        "RECTIFIED RIGHT",
        (660, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        "DISPARITY / DEPTH",
        (20, 395),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "IMX219-83 STEREO TEST",
        output
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

left_cam.stop()
right_cam.stop()

cv2.destroyAllWindows()

print()
print("Stereo test finished.")
