import cv2
import numpy as np
import time
import os

from picamera2 import Picamera2


# ============================================================
# IMX219-83 STEREO CAMERA
# REAL-TIME DISTANCE MEASUREMENT
# Raspberry Pi 5
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

WIDTH = 1280
HEIGHT = 720

CALIBRATION_FILE = (
    "/home/pi/Desktop/MahirCode/"
    "calibration_images/stereo_calibration.npz"
)

# ROI around the center of the image
ROI_SIZE = 80

# Minimum and maximum useful disparity
MIN_DISPARITY = 0
NUM_DISPARITIES = 128

# Stereo matching parameters
BLOCK_SIZE = 5

# Display size
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


# ============================================================
# CHECK CALIBRATION FILE
# ============================================================

if not os.path.exists(CALIBRATION_FILE):

    print()
    print("ERROR: Calibration file not found!")
    print()
    print(CALIBRATION_FILE)
    print()

    exit()


# ============================================================
# LOAD CALIBRATION
# ============================================================

print()
print("==============================================")
print("LOADING STEREO CALIBRATION")
print("==============================================")

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

baseline_mm = float(
    data["baseline_mm"]
)

print()
print("Calibration loaded successfully.")
print(
    f"Baseline: {baseline_mm:.2f} mm"
)

print(
    f"Image size: {WIDTH} x {HEIGHT}"
)


# ============================================================
# RECTIFICATION MAPS
# ============================================================

print()
print("Creating rectification maps...")

map1L, map2L = cv2.initUndistortRectifyMap(
    mtxL,
    distL,
    R1,
    P1,
    (WIDTH, HEIGHT),
    cv2.CV_32FC1
)

map1R, map2R = cv2.initUndistortRectifyMap(
    mtxR,
    distR,
    R2,
    P2,
    (WIDTH, HEIGHT),
    cv2.CV_32FC1
)

print("Rectification maps ready.")


# ============================================================
# CREATE STEREO MATCHER
# ============================================================

print()
print("Creating StereoSGBM...")

stereo = cv2.StereoSGBM_create(

    minDisparity=MIN_DISPARITY,

    numDisparities=NUM_DISPARITIES,

    blockSize=BLOCK_SIZE,

    P1=8 * 1 * BLOCK_SIZE * BLOCK_SIZE,

    P2=32 * 1 * BLOCK_SIZE * BLOCK_SIZE,

    disp12MaxDiff=1,

    uniquenessRatio=8,

    speckleWindowSize=100,

    speckleRange=2,

    preFilterCap=63,

    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

print("Stereo matcher ready.")


# ============================================================
# START CAMERAS
# ============================================================

print()
print("==============================================")
print("STARTING CAMERAS")
print("==============================================")

left_camera = Picamera2(0)
right_camera = Picamera2(1)


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

left_config = left_camera.create_preview_configuration(

    main={
        "size": (WIDTH, HEIGHT),
        "format": "RGB888"
    }

)

right_config = right_camera.create_preview_configuration(

    main={
        "size": (WIDTH, HEIGHT),
        "format": "RGB888"
    }

)


left_camera.configure(left_config)
right_camera.configure(right_config)


# ============================================================
# START
# ============================================================

left_camera.start()
right_camera.start()

time.sleep(2)


print()
print("==============================================")
print("STEREO DISTANCE MEASUREMENT")
print("==============================================")
print()
print("Place an object near the CENTER.")
print()
print("Controls:")
print("Q = Quit")
print("S = Save current frame")
print()
print("Starting...")
print()


# ============================================================
# CREATE WINDOW
# ============================================================

cv2.namedWindow(
    "STEREO DISTANCE",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "STEREO DISTANCE",
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT
)


# ============================================================
# FPS VARIABLES
# ============================================================

previous_time = time.time()

fps = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # CAPTURE
    # --------------------------------------------------------

    left_rgb = left_camera.capture_array()
    right_rgb = right_camera.capture_array()


    # --------------------------------------------------------
    # RGB → GRAYSCALE
    # --------------------------------------------------------

    left_gray = cv2.cvtColor(
        left_rgb,
        cv2.COLOR_RGB2GRAY
    )

    right_gray = cv2.cvtColor(
        right_rgb,
        cv2.COLOR_RGB2GRAY
    )


    # --------------------------------------------------------
    # RECTIFICATION
    # --------------------------------------------------------

    rect_left = cv2.remap(

        left_gray,

        map1L,
        map2L,

        cv2.INTER_LINEAR
    )

    rect_right = cv2.remap(

        right_gray,

        map1R,
        map2R,

        cv2.INTER_LINEAR
    )


    # --------------------------------------------------------
    # CALCULATE DISPARITY
    # --------------------------------------------------------

    disparity_raw = stereo.compute(
        rect_left,
        rect_right
    )


    disparity = (
        disparity_raw.astype(np.float32)
        / 16.0
    )


    # --------------------------------------------------------
    # 3D REPROJECTION
    # --------------------------------------------------------

    points_3d = cv2.reprojectImageTo3D(
        disparity,
        Q
    )


    # ========================================================
    # CENTER ROI
    # ========================================================

    center_x = WIDTH // 2
    center_y = HEIGHT // 2

    half_roi = ROI_SIZE // 2

    x1 = center_x - half_roi
    x2 = center_x + half_roi

    y1 = center_y - half_roi
    y2 = center_y + half_roi


    # --------------------------------------------------------
    # GET DISPARITY VALUES INSIDE ROI
    # --------------------------------------------------------

    roi_disparity = disparity[
        y1:y2,
        x1:x2
    ]


    # --------------------------------------------------------
    # VALID DISPARITY
    # --------------------------------------------------------

    valid = roi_disparity > 1.0


    distance_cm = None


    if np.count_nonzero(valid) > 50:

        valid_disparity = roi_disparity[
            valid
        ]


        # ----------------------------------------------------
        # REMOVE EXTREME VALUES
        # ----------------------------------------------------

        disparity_median = np.median(
            valid_disparity
        )


        # ----------------------------------------------------
        # FIND 3D POINTS
        # ----------------------------------------------------

        roi_points = points_3d[
            y1:y2,
            x1:x2
        ]

        valid_points = roi_points[
            valid
        ]


        if len(valid_points) > 50:

            # ------------------------------------------------
            # Z = DEPTH FROM CAMERA
            # ------------------------------------------------

            z_values = valid_points[:, 2]


            # Remove invalid values
            z_values = z_values[
                np.isfinite(z_values)
            ]


            if len(z_values) > 20:

                # Remove extreme depth values
                z_min = np.percentile(
                    z_values,
                    10
                )

                z_max = np.percentile(
                    z_values,
                    90
                )

                z_filtered = z_values[
                    (z_values >= z_min)
                    &
                    (z_values <= z_max)
                ]


                if len(z_filtered) > 10:

                    depth_mm = np.median(
                        z_filtered
                    )

                    distance_cm = (
                        abs(depth_mm) / 10.0
                    )


    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    display = cv2.cvtColor(
        rect_left,
        cv2.COLOR_GRAY2BGR
    )


    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    cv2.rectangle(

        display,

        (x1, y1),

        (x2, y2),

        (0, 255, 0),

        3
    )


    # --------------------------------------------------------
    # CENTER CROSSHAIR
    # --------------------------------------------------------

    cv2.line(

        display,

        (center_x - 15, center_y),

        (center_x + 15, center_y),

        (0, 0, 255),

        2
    )


    cv2.line(

        display,

        (center_x, center_y - 15),

        (center_x, center_y + 15),

        (0, 0, 255),

        2
    )


    # ========================================================
    # DISTANCE TEXT
    # ========================================================

    if distance_cm is not None:

        distance_text = (
            f"Distance: {distance_cm:.1f} cm"
        )

        cv2.putText(

            display,

            distance_text,

            (30, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.2,

            (0, 255, 0),

            3
        )


        # ----------------------------------------------------
        # DISPARITY
        # ----------------------------------------------------

        cv2.putText(

            display,

            f"Disparity: {disparity_median:.2f}",

            (30, 95),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 255),

            2
        )


    else:

        cv2.putText(

            display,

            "Distance: ---",

            (30, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.2,

            (0, 0, 255),

            3
        )


    # ========================================================
    # BASELINE
    # ========================================================

    cv2.putText(

        display,

        f"Baseline: {baseline_mm:.1f} mm",

        (30, 135),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    dt = current_time - previous_time

    if dt > 0:

        current_fps = 1.0 / dt

        fps = (
            0.9 * fps
            +
            0.1 * current_fps
        )

    previous_time = current_time


    cv2.putText(

        display,

        f"FPS: {fps:.1f}",

        (30, 175),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    # ========================================================
    # DISPLAY SIZE
    # ========================================================

    display_small = cv2.resize(

        display,

        (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(

        "STEREO DISTANCE",

        display_small
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    if key == ord("s"):

        filename = (
            f"distance_test_"
            f"{int(time.time())}.jpg"
        )

        cv2.imwrite(
            filename,
            display
        )

        print()
        print(
            f"Saved: {filename}"
        )

        if distance_cm is not None:

            print(
                f"Distance: "
                f"{distance_cm:.2f} cm"
            )

        else:

            print(
                "Distance could not be measured."
            )


# ============================================================
# CLEANUP
# ============================================================

print()
print("Stopping cameras...")

left_camera.stop()
right_camera.stop()

cv2.destroyAllWindows()

print()
print("==============================================")
print("DISTANCE TEST FINISHED")
print("==============================================")
