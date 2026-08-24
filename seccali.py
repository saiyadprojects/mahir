import cv2
import numpy as np
import glob
import os

# ============================================================
# IMX219-83 STEREO CAMERA CALIBRATION
# Raspberry Pi 5
# ============================================================

# ------------------------------------------------------------
# CHECKERBOARD
# ------------------------------------------------------------
# Physical board:
# 8 x 8 squares
# Therefore:
# 7 x 7 internal corners
#
# Each square = 40 mm
# ------------------------------------------------------------

CHECKERBOARD = (7, 7)
SQUARE_SIZE = 40.0       # mm

# ------------------------------------------------------------
# IMAGE FOLDERS
# ------------------------------------------------------------

LEFT_DIR = "left"
RIGHT_DIR = "right"

OUTPUT_FILE = "stereo_calibration.npz"

# ------------------------------------------------------------
# KNOWN BAD STEREO PAIRS
# ------------------------------------------------------------
# Pair 41 showed extremely large stereo/rectification error.
#
# We DO NOT delete the image.
# We simply don't use it for final calibration.
# ------------------------------------------------------------

BAD_PAIRS = [41]

# ============================================================
# FIND IMAGES
# ============================================================

left_images = sorted(
    glob.glob(os.path.join(LEFT_DIR, "*.jpg"))
)

right_images = sorted(
    glob.glob(os.path.join(RIGHT_DIR, "*.jpg"))
)

print()
print("==========================================")
print("IMX219-83 STEREO CAMERA CALIBRATION")
print("==========================================")

print()
print("Left images :", len(left_images))
print("Right images:", len(right_images))

if len(left_images) == 0:
    print("ERROR: No left images found.")
    exit()

if len(right_images) == 0:
    print("ERROR: No right images found.")
    exit()

if len(left_images) != len(right_images):
    print("ERROR: Left/right image count mismatch.")
    exit()

# ============================================================
# CREATE 3D CHECKERBOARD OBJECT POINTS
# ============================================================

objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

# ============================================================
# STORAGE
# ============================================================

objpoints = []

imgpoints_left = []
imgpoints_right = []

pair_numbers = []

# ============================================================
# SUBPIXEL CRITERIA
# ============================================================

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    50,
    0.0001
)

# ============================================================
# CHECKERBOARD DETECTION
# ============================================================

print()
print("==========================================")
print("DETECTING CHECKERBOARD CORNERS")
print("==========================================")
print()

for i, (left_file, right_file) in enumerate(
    zip(left_images, right_images)
):

    pair_number = i + 1

    left = cv2.imread(left_file)
    right = cv2.imread(right_file)

    if left is None or right is None:

        print(
            f"{pair_number:02d}: IMAGE ERROR"
        )

        continue

    gray_left = cv2.cvtColor(
        left,
        cv2.COLOR_BGR2GRAY
    )

    gray_right = cv2.cvtColor(
        right,
        cv2.COLOR_BGR2GRAY
    )

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    ret_left, corners_left = \
        cv2.findChessboardCorners(
            gray_left,
            CHECKERBOARD,
            flags
        )

    ret_right, corners_right = \
        cv2.findChessboardCorners(
            gray_right,
            CHECKERBOARD,
            flags
        )

    # --------------------------------------------------------
    # BOTH CAMERAS MUST DETECT THE BOARD
    # --------------------------------------------------------

    if ret_left and ret_right:

        corners_left = cv2.cornerSubPix(
            gray_left,
            corners_left,
            (11, 11),
            (-1, -1),
            criteria
        )

        corners_right = cv2.cornerSubPix(
            gray_right,
            corners_right,
            (11, 11),
            (-1, -1),
            criteria
        )

        objpoints.append(
            objp.copy()
        )

        imgpoints_left.append(
            corners_left
        )

        imgpoints_right.append(
            corners_right
        )

        pair_numbers.append(
            pair_number
        )

        print(
            f"{pair_number:02d}: OK"
        )

    else:

        print(
            f"{pair_number:02d}: FAILED"
        )

# ============================================================
# DETECTION SUMMARY
# ============================================================

print()
print("==========================================")
print("CORNER DETECTION SUMMARY")
print("==========================================")

print(
    "Detected pairs:",
    len(objpoints)
)

if len(objpoints) < 15:

    print()
    print(
        "ERROR: Too few successful pairs."
    )

    print(
        "Need at least 15 good pairs."
    )

    exit()

# ============================================================
# IMAGE SIZE
# ============================================================

image = cv2.imread(
    left_images[0]
)

image_size = (
    image.shape[1],
    image.shape[0]
)

print(
    "Image size:",
    image_size
)

# ============================================================
# INITIAL LEFT CAMERA CALIBRATION
# ============================================================

print()
print("==========================================")
print("INITIAL LEFT CALIBRATION")
print("==========================================")

rms_left, mtx_left, dist_left, \
rvecs_left, tvecs_left = \
    cv2.calibrateCamera(
        objpoints,
        imgpoints_left,
        image_size,
        None,
        None
    )

print(
    f"LEFT RMS: {rms_left:.4f} px"
)

# ============================================================
# INITIAL RIGHT CAMERA CALIBRATION
# ============================================================

print()
print("==========================================")
print("INITIAL RIGHT CALIBRATION")
print("==========================================")

rms_right, mtx_right, dist_right, \
rvecs_right, tvecs_right = \
    cv2.calibrateCamera(
        objpoints,
        imgpoints_right,
        image_size,
        None,
        None
    )

print(
    f"RIGHT RMS: {rms_right:.4f} px"
)

# ============================================================
# MONOCULAR REPROJECTION ERROR
# ============================================================

print()
print("==========================================")
print("PAIR REPROJECTION ERRORS")
print("==========================================")

errors = []

for i in range(len(objpoints)):

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    projected_left, _ = cv2.projectPoints(
        objpoints[i],
        rvecs_left[i],
        tvecs_left[i],
        mtx_left,
        dist_left
    )

    error_left = cv2.norm(
        imgpoints_left[i],
        projected_left,
        cv2.NORM_L2
    ) / len(projected_left)

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    projected_right, _ = cv2.projectPoints(
        objpoints[i],
        rvecs_right[i],
        tvecs_right[i],
        mtx_right,
        dist_right
    )

    error_right = cv2.norm(
        imgpoints_right[i],
        projected_right,
        cv2.NORM_L2
    ) / len(projected_right)

    combined_error = (
        error_left +
        error_right
    ) / 2.0

    errors.append(
        combined_error
    )

    print(
        f"Pair {pair_numbers[i]:02d}: "
        f"{combined_error:.3f} px"
    )

# ============================================================
# ERROR STATISTICS
# ============================================================

median_error = np.median(
    errors
)

print()
print("==========================================")
print("ERROR STATISTICS")
print("==========================================")

print(
    f"Median error: {median_error:.3f} px"
)

# ------------------------------------------------------------
# MONOCULAR ERROR THRESHOLD
# ------------------------------------------------------------

threshold = max(
    1.5,
    median_error * 2.5
)

print(
    f"Monocular threshold: {threshold:.3f} px"
)

# ============================================================
# SELECT GOOD MONOCULAR PAIRS
# ============================================================

good_indices = []

for i, error in enumerate(errors):

    if error <= threshold:

        good_indices.append(
            i
        )

print()
print(
    "Good pairs after monocular filtering:",
    len(good_indices)
)

# ============================================================
# REMOVE KNOWN STEREO OUTLIERS
# ============================================================

print()
print("==========================================")
print("STEREO OUTLIER FILTER")
print("==========================================")

print(
    "Known bad pairs:",
    BAD_PAIRS
)

filtered_indices = []

for i in good_indices:

    number = pair_numbers[i]

    if number in BAD_PAIRS:

        print(
            f"Removing Pair {number}"
        )

    else:

        filtered_indices.append(
            i
        )

good_indices = filtered_indices

print()
print(
    "Pairs remaining:",
    len(good_indices)
)

# ============================================================
# DISPLAY FINAL PAIRS
# ============================================================

print()
print("Pairs used for final calibration:")

for i in good_indices:

    print(
        f"{pair_numbers[i]:02d} "
        f"({errors[i]:.3f} px)"
    )

# ============================================================
# BUILD FINAL DATASET
# ============================================================

objpoints_final = [
    objpoints[i]
    for i in good_indices
]

imgpoints_left_final = [
    imgpoints_left[i]
    for i in good_indices
]

imgpoints_right_final = [
    imgpoints_right[i]
    for i in good_indices
]

# ============================================================
# FINAL LEFT CALIBRATION
# ============================================================

print()
print("==========================================")
print("FINAL LEFT CALIBRATION")
print("==========================================")

rms_left, mtx_left, dist_left, \
rvecs_left, tvecs_left = \
    cv2.calibrateCamera(
        objpoints_final,
        imgpoints_left_final,
        image_size,
        None,
        None
    )

print(
    f"LEFT RMS: {rms_left:.4f} px"
)

# ============================================================
# FINAL RIGHT CALIBRATION
# ============================================================

print()
print("==========================================")
print("FINAL RIGHT CALIBRATION")
print("==========================================")

rms_right, mtx_right, dist_right, \
rvecs_right, tvecs_right = \
    cv2.calibrateCamera(
        objpoints_final,
        imgpoints_right_final,
        image_size,
        None,
        None
    )

print(
    f"RIGHT RMS: {rms_right:.4f} px"
)

# ============================================================
# STEREO CALIBRATION
# ============================================================

print()
print("==========================================")
print("STEREO CALIBRATION")
print("==========================================")

stereo_criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    200,
    1e-7
)

# ------------------------------------------------------------
# Keep individual camera calibration fixed.
# Estimate only relative stereo position/orientation.
# ------------------------------------------------------------

stereo_flags = cv2.CALIB_FIX_INTRINSIC

stereo_rms, mtx_left, dist_left, \
mtx_right, dist_right, \
R, T, E, F = \
    cv2.stereoCalibrate(
        objpoints_final,
        imgpoints_left_final,
        imgpoints_right_final,
        mtx_left,
        dist_left,
        mtx_right,
        dist_right,
        image_size,
        criteria=stereo_criteria,
        flags=stereo_flags
    )

# ============================================================
# BASELINE
# ============================================================

baseline = np.linalg.norm(
    T
)

# ============================================================
# RECTIFICATION
# ============================================================

print()
print("Calculating rectification...")

R1, R2, P1, P2, Q, roi_left, roi_right = \
    cv2.stereoRectify(
        mtx_left,
        dist_left,
        mtx_right,
        dist_right,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

# ============================================================
# RECTIFICATION MAPS
# ============================================================

map1_left, map2_left = \
    cv2.initUndistortRectifyMap(
        mtx_left,
        dist_left,
        R1,
        P1,
        image_size,
        cv2.CV_32FC1
    )

map1_right, map2_right = \
    cv2.initUndistortRectifyMap(
        mtx_right,
        dist_right,
        R2,
        P2,
        image_size,
        cv2.CV_32FC1
    )

# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("==========================================")
print("FINAL CALIBRATION RESULTS")
print("==========================================")

print(
    "Pairs used:",
    len(objpoints_final)
)

print(
    f"Left RMS:   {rms_left:.4f} px"
)

print(
    f"Right RMS:  {rms_right:.4f} px"
)

print(
    f"Stereo RMS: {stereo_rms:.4f} px"
)

print()
print("Translation T:")
print(T)

print()
print("Rotation R:")
print(R)

print()
print(
    f"Baseline: {baseline:.3f} mm"
)

# ============================================================
# SAVE NPZ
# ============================================================

np.savez(
    OUTPUT_FILE,

    # Camera matrices
    mtxL=mtx_left,
    distL=dist_left,

    mtxR=mtx_right,
    distR=dist_right,

    # Stereo parameters
    R=R,
    T=T,
    E=E,
    F=F,

    # Rectification
    R1=R1,
    R2=R2,

    P1=P1,
    P2=P2,

    Q=Q,

    # Rectification maps
    map1L=map1_left,
    map2L=map2_left,

    map1R=map1_right,
    map2R=map2_right,

    # Image information
    image_width=image_size[0],
    image_height=image_size[1],

    # Physical information
    square_size_mm=SQUARE_SIZE,
    baseline_mm=baseline,

    # Checkerboard
    checkerboard_width=CHECKERBOARD[0],
    checkerboard_height=CHECKERBOARD[1]
)

# ============================================================
# COMPLETE
# ============================================================

print()
print("==========================================")
print("CALIBRATION COMPLETE")
print("==========================================")

print()
print("Saved file:")

print(
    os.path.abspath(OUTPUT_FILE)
)

print()
print("Baseline:")
print(
    f"{baseline:.3f} mm"
)

print()
print("Stereo RMS:")
print(
    f"{stereo_rms:.4f} px"
)

print()
print("==========================================")
