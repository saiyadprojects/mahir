import cv2
import numpy as np
import time
from picamera2 import Picamera2


# ============================================================
# IMX219-83 STEREO CAMERA
# FAST REAL-TIME DISTANCE
# Raspberry Pi 5
# ============================================================

CALIBRATION_FILE = "/home/pi/Desktop/MahirCode/calibration_images/stereo_calibration.npz"

# ------------------------------------------------------------
# IMPORTANT:
# Calibration was done at 1280x720.
# We process at half resolution for much better FPS.
# ------------------------------------------------------------

FULL_W = 1280
FULL_H = 720

W = 640
H = 360

SCALE_X = W / FULL_W
SCALE_Y = H / FULL_H


# ============================================================
# DISTANCE RANGE
# ============================================================

MIN_DISTANCE_CM = 15
MAX_DISTANCE_CM = 300


# ============================================================
# MEASUREMENT REGION
# ============================================================
#
# Smaller region = faster and less background interference.
#
# This is the area around the red cross.
#

ROI_W = 220
ROI_H = 160

cx = W // 2
cy = H // 2

rx1 = cx - ROI_W // 2
rx2 = cx + ROI_W // 2

ry1 = cy - ROI_H // 2
ry2 = cy + ROI_H // 2


# ============================================================
# LOAD CALIBRATION
# ============================================================

print()
print("==========================================")
print(" IMX219-83 FAST STEREO DISTANCE")
print("==========================================")

data = np.load(CALIBRATION_FILE)

mtxL = data["mtxL"].astype(np.float64)
distL = data["distL"].astype(np.float64)

mtxR = data["mtxR"].astype(np.float64)
distR = data["distR"].astype(np.float64)

R = data["R"].astype(np.float64)
T = data["T"].astype(np.float64)


# ============================================================
# SCALE CAMERA MATRICES
# ============================================================

mtxL_small = mtxL.copy()
mtxR_small = mtxR.copy()

mtxL_small[0, 0] *= SCALE_X
mtxL_small[1, 1] *= SCALE_Y
mtxL_small[0, 2] *= SCALE_X
mtxL_small[1, 2] *= SCALE_Y

mtxR_small[0, 0] *= SCALE_X
mtxR_small[1, 1] *= SCALE_Y
mtxR_small[0, 2] *= SCALE_X
mtxR_small[1, 2] *= SCALE_Y


baseline_mm = float(np.linalg.norm(T))

print(f"Calibration loaded")
print(f"Original: {FULL_W} x {FULL_H}")
print(f"Processing: {W} x {H}")
print(f"Baseline: {baseline_mm:.2f} mm")


# ============================================================
# RECTIFICATION
# ============================================================

print("Creating rectification maps...")

image_size = (W, H)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtxL_small,
    distL,
    mtxR_small,
    distR,
    image_size,
    R,
    T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)


mapLx, mapLy = cv2.initUndistortRectifyMap(
    mtxL_small,
    distL,
    R1,
    P1,
    image_size,
    cv2.CV_16SC2
)

mapRx, mapRy = cv2.initUndistortRectifyMap(
    mtxR_small,
    distR,
    R2,
    P2,
    image_size,
    cv2.CV_16SC2
)


# ============================================================
# CAMERA
# ============================================================

print("Starting cameras...")

left_cam = Picamera2(0)
right_cam = Picamera2(1)

config_left = left_cam.create_video_configuration(
    main={
        "size": (W, H),
        "format": "RGB888"
    },
    buffer_count=2
)

config_right = right_cam.create_video_configuration(
    main={
        "size": (W, H),
        "format": "RGB888"
    },
    buffer_count=2
)

left_cam.configure(config_left)
right_cam.configure(config_right)

left_cam.start()
right_cam.start()

time.sleep(1)

print("Cameras started")


# ============================================================
# STEREO SGBM
# ============================================================

NUM_DISPARITIES = 128
BLOCK_SIZE = 5

stereo = cv2.StereoSGBM_create(

    minDisparity=0,

    numDisparities=NUM_DISPARITIES,

    blockSize=BLOCK_SIZE,

    P1=8 * BLOCK_SIZE * BLOCK_SIZE,

    P2=32 * BLOCK_SIZE * BLOCK_SIZE,

    disp12MaxDiff=1,

    uniquenessRatio=15,

    speckleWindowSize=80,

    speckleRange=2,

    preFilterCap=31,

    mode=cv2.STEREO_SGBM_MODE_SGBM
)


# ============================================================
# DISTANCE FILTER
# ============================================================

distance_history = []

HISTORY_SIZE = 5


# ============================================================
# FPS
# ============================================================

last_time = time.time()

fps = 0


# ============================================================
# WINDOW
# ============================================================

cv2.namedWindow(
    "STEREO DISTANCE",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "STEREO DISTANCE",
    960,
    540
)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # CAPTURE
        # ----------------------------------------------------

        left = left_cam.capture_array()
        right = right_cam.capture_array()


        # ----------------------------------------------------
        # RECTIFY
        # ----------------------------------------------------

        rectL = cv2.remap(
            left,
            mapLx,
            mapLy,
            cv2.INTER_LINEAR
        )

        rectR = cv2.remap(
            right,
            mapRx,
            mapRy,
            cv2.INTER_LINEAR
        )


        # ----------------------------------------------------
        # GRAYSCALE
        # ----------------------------------------------------

        grayL = cv2.cvtColor(
            rectL,
            cv2.COLOR_RGB2GRAY
        )

        grayR = cv2.cvtColor(
            rectR,
            cv2.COLOR_RGB2GRAY
        )


        # ----------------------------------------------------
        # DISPARITY
        # ----------------------------------------------------

        disparity = stereo.compute(
            grayL,
            grayR
        ).astype(np.float32) / 16.0


        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        roi = disparity[
            ry1:ry2,
            rx1:rx2
        ]

        gray_roi = grayL[
            ry1:ry2,
            rx1:rx2
        ]


        # ----------------------------------------------------
        # TEXTURE / GRADIENT
        # ----------------------------------------------------

        gx = cv2.Sobel(
            gray_roi,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            gray_roi,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        gradient = cv2.magnitude(
            gx,
            gy
        )


        # ----------------------------------------------------
        # VALID DISPARITY
        # ----------------------------------------------------
        #
        # VERY IMPORTANT:
        #
        # Do NOT accept disparity near 127.
        #
        # 127 previously produced your false ~70 cm reading.
        #

        valid_mask = (

            (roi > 3.0)

            &

            (roi < NUM_DISPARITIES - 5)

            &

            (gradient > 12)

        )


        valid_values = roi[
            valid_mask
        ]


        # ----------------------------------------------------
        # REMOVE EXTREME VALUES
        # ----------------------------------------------------

        if len(valid_values) > 0:

            valid_values = valid_values[
                np.isfinite(valid_values)
            ]


        # ----------------------------------------------------
        # VALID PIXEL %
        # ----------------------------------------------------

        valid_percent = (

            len(valid_values)

            /

            roi.size

        ) * 100.0


        # ----------------------------------------------------
        # DISTANCE
        # ----------------------------------------------------

        current_distance = None
        disparity_value = None


        if (

            valid_percent >= 5.0

            and

            len(valid_values) >= 100

        ):

            # ------------------------------------------------
            # Remove extreme disparity values
            # ------------------------------------------------

            p10 = np.percentile(
                valid_values,
                10
            )

            p90 = np.percentile(
                valid_values,
                90
            )

            stable_values = valid_values[
                (valid_values >= p10)
                &
                (valid_values <= p90)
            ]


            if len(stable_values) >= 50:

                disparity_value = float(
                    np.median(stable_values)
                )


                # ------------------------------------------------
                # DEPTH
                # ------------------------------------------------

                focal_length = P1[0, 0]

                distance_mm = (

                    focal_length
                    *
                    baseline_mm
                    /
                    disparity_value

                )

                current_distance = (
                    distance_mm / 10.0
                )


                # ------------------------------------------------
                # RANGE CHECK
                # ------------------------------------------------

                if (

                    current_distance < MIN_DISTANCE_CM

                    or

                    current_distance > MAX_DISTANCE_CM

                ):

                    current_distance = None


        # =====================================================
        # TEMPORAL FILTER
        # =====================================================

        if current_distance is not None:

            # -----------------------------------------------
            # Reject sudden impossible jumps
            # -----------------------------------------------

            if len(distance_history) > 0:

                previous = distance_history[-1]

                if abs(
                    current_distance - previous
                ) > 80:

                    current_distance = None


        if current_distance is not None:

            distance_history.append(
                current_distance
            )

            if len(distance_history) > HISTORY_SIZE:

                distance_history.pop(0)


        # =====================================================
        # FINAL DISTANCE
        # =====================================================

        if len(distance_history) >= 2:

            final_distance = float(
                np.median(distance_history)
            )

        else:

            final_distance = None


        # =====================================================
        # FPS
        # =====================================================

        now = time.time()

        dt = now - last_time

        if dt > 0:

            instant_fps = 1.0 / dt

            fps = (
                0.8 * fps
                +
                0.2 * instant_fps
            )

        last_time = now


        # =====================================================
        # DISPLAY
        # =====================================================

        display = rectL.copy()


        # -----------------------------------------------------
        # ROI
        # -----------------------------------------------------

        cv2.rectangle(
            display,
            (rx1, ry1),
            (rx2, ry2),
            (0, 255, 0),
            2
        )


        # -----------------------------------------------------
        # CENTER CROSS
        # -----------------------------------------------------

        cv2.drawMarker(
            display,
            (cx, cy),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            25,
            2
        )


        # =====================================================
        # DISTANCE TEXT
        # =====================================================

        if final_distance is not None:

            cv2.putText(
                display,
                f"Distance: {final_distance:.1f} cm",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                display,
                "NO VALID OBJECT",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


        # -----------------------------------------------------
        # DISPARITY
        # -----------------------------------------------------

        if disparity_value is not None:

            disp_text = f"Disparity: {disparity_value:.2f}"

        else:

            disp_text = "Disparity: ---"


        cv2.putText(
            display,
            disp_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )


        # -----------------------------------------------------
        # VALID
        # -----------------------------------------------------

        cv2.putText(
            display,
            f"Valid: {valid_percent:.1f}%",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # -----------------------------------------------------
        # FPS
        # -----------------------------------------------------

        cv2.putText(
            display,
            f"FPS: {fps:.1f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # -----------------------------------------------------
        # BASELINE
        # -----------------------------------------------------

        cv2.putText(
            display,
            f"Baseline: {baseline_mm:.2f} mm",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # =====================================================
        # SHOW
        # =====================================================

        cv2.imshow(
            "STEREO DISTANCE",
            display
        )


        # =====================================================
        # KEY
        # =====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


finally:

    left_cam.stop()
    right_cam.stop()

    cv2.destroyAllWindows()

    print()
    print("==========================================")
    print(" CAMERA STOPPED")
    print("==========================================")
