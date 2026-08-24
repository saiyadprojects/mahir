import cv2
import numpy as np
import glob
import os

# ============================================================
# SETTINGS
# ============================================================

CHECKERBOARD = (7, 7)       # INNER corners
SQUARE_SIZE = 40.0          # mm - change to your actual square size

IMAGE_SIZE = (1280, 720)

LEFT_DIR = "left"
RIGHT_DIR = "right"

OUTPUT_FILE = "stereo_calibration.npz"

# ============================================================
# FIND IMAGES
# ============================================================

left_images = sorted(
    glob.glob(os.path.join(LEFT_DIR, "*.jpg"))
)

right_images = sorted(
    glob.glob(os.path.join(RIGHT_DIR, "*.jpg"))
)

print("Left images :", len(left_images))
print("Right images:", len(right_images))

if len(left_images) == 0:
    print("ERROR: No left images found")
    exit()

if len(right_images) == 0:
    print("ERROR: No right images found")
    exit()

if len(left_images) != len(right_images):
    print("ERROR: Left and right image counts are different")
    exit()

# ============================================================
# PREPARE OBJECT POINTS
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

successful_pairs = 0

# ============================================================
# CORNER DETECTION
# ============================================================

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

print()
print("Detecting checkerboard corners...")
print()

for i, (left_file, right_file) in enumerate(
    zip(left_images, right_images)
):

    print(
        f"Processing pair {i + 1}: "
        f"{os.path.basename(left_file)} / "
        f"{os.path.basename(right_file)}"
    )

    img_left = cv2.imread(left_file)
    img_right = cv2.imread(right_file)

    if img_left is None or img_right is None:
        print("  ERROR reading image")
        continue

    gray_left = cv2.cvtColor(
        img_left,
        cv2.COLOR_BGR2GRAY
    )

    gray_right = cv2.cvtColor(
        img_right,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Find corners
    # --------------------------------------------------------

    ret_left, corners_left = cv2.findChessboardCorners(
        gray_left,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    ret_right, corners_right = cv2.findChessboardCorners(
        gray_right,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # --------------------------------------------------------
    # Only use pair if BOTH cameras detected the board
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

        objpoints.append(objp)

        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)

        successful_pairs += 1

        print("  OK")

    else:

        print(
            "  FAILED - checkerboard not detected "
            "in both images"
        )

print()
print("==========================================")
print("CORNER DETECTION COMPLETE")
print("==========================================")
print(f"Total pairs      : {len(left_images)}")
print(f"Successful pairs : {successful_pairs}")
print()

# ============================================================
# CHECK NUMBER OF GOOD IMAGES
# ============================================================

if successful_pairs < 10:

    print(
        "ERROR: Too few successful calibration pairs."
    )

    print(
        "Try to capture at least 15-20 good pairs."
    )

    exit()

# ============================================================
# CAMERA MATRIX INITIALIZATION
# ============================================================

gray = cv2.imread(
    left_images[0],
    cv2.IMREAD_GRAYSCALE
)

image_size = gray.shape[::-1]

print("Image size:", image_size)

# ============================================================
# LEFT CAMERA CALIBRATION
# ============================================================

print()
print("Calibrating LEFT camera...")

ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = \
    cv2.calibrateCamera(
        objpoints,
        imgpoints_left,
        image_size,
        None,
        None
    )

print("Left RMS error:", ret_left)

# ============================================================
# RIGHT CAMERA CALIBRATION
# ============================================================

print()
print("Calibrating RIGHT camera...")

ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = \
    cv2.calibrateCamera(
        objpoints,
        imgpoints_right,
        image_size,
        None,
        None
    )

print("Right RMS error:", ret_right)

# ============================================================
# STEREO CALIBRATION
# ============================================================

print()
print("Performing STEREO calibration...")

stereo_criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-5
)

stereo_flags = cv2.CALIB_FIX_INTRINSIC

retval, M1, D1, M2, D2, R, T, E, F = \
    cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx_left,
        dist_left,
        mtx_right,
        dist_right,
        image_size,
        criteria=stereo_criteria,
        flags=stereo_flags
    )

print("Stereo RMS error:", retval)

# ============================================================
# BASELINE
# ============================================================

baseline = np.linalg.norm(T)

print()
print("==========================================")
print("STEREO RESULTS")
print("==========================================")

print("Baseline:", baseline, "mm")

print()
print("Translation T:")
print(T)

print()
print("Rotation R:")
print(R)

# ============================================================
# RECTIFICATION
# ============================================================

print()
print("Calculating rectification...")

R1, R2, P1, P2, Q, roi_left, roi_right = \
    cv2.stereoRectify(
        M1,
        D1,
        M2,
        D2,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

# ============================================================
# RECTIFICATION MAPS
# ============================================================

map1_left, map2_left = cv2.initUndistortRectifyMap(
    M1,
    D1,
    R1,
    P1,
    image_size,
    cv2.CV_32FC1
)

map1_right, map2_right = cv2.initUndistortRectifyMap(
    M2,
    D2,
    R2,
    P2,
    image_size,
    cv2.CV_32FC1
)

# ============================================================
# SAVE EVERYTHING
# ============================================================

np.savez(
    OUTPUT_FILE,

    # Camera matrices
    mtxL=M1,
    distL=D1,

    mtxR=M2,
    distR=D2,

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

    # Image size
    image_width=image_size[0],
    image_height=image_size[1],

    # Baseline
    baseline_mm=baseline
)

print()
print("==========================================")
print("CALIBRATION COMPLETE")
print("==========================================")

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("Baseline:")
print(f"{baseline:.2f} mm")

print()
print("You can now use this file for stereo depth.")
