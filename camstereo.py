from picamera2 import Picamera2
import cv2
import time

# ==========================================
# CAMERAS
# ==========================================

left_cam = Picamera2(0)
right_cam = Picamera2(1)

# ==========================================
# CONFIGURATION
# ==========================================

config_left = left_cam.create_preview_configuration(
    main={
        "size": (1280, 720),
        "format": "RGB888"
    }
)

config_right = right_cam.create_preview_configuration(
    main={
        "size": (1280, 720),
        "format": "RGB888"
    }
)

left_cam.configure(config_left)
right_cam.configure(config_right)

# ==========================================
# START
# ==========================================

left_cam.start()
right_cam.start()

time.sleep(2)

print("IMX219-83 Stereo Camera Started")
print("LEFT  = Camera 0")
print("RIGHT = Camera 1")
print("Q = Quit")
print("S = Save images")

# ==========================================
# WINDOW
# ==========================================

cv2.namedWindow(
    "IMX219-83 Stereo Camera",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "IMX219-83 Stereo Camera",
    1920,
    720
)

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # Capture images
    left = left_cam.capture_array()
    right = right_cam.capture_array()

    # IMPORTANT:
    # Do NOT use RGB2BGR conversion here.
    # Picamera2 RGB888 is already suitable for OpenCV.

    # ======================================
    # LABELS
    # ======================================

    cv2.putText(
        left,
        "LEFT CAMERA",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        right,
        "RIGHT CAMERA",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # ======================================
    # RESIZE FOR DISPLAY
    # ======================================

    left_display = cv2.resize(
        left,
        (960, 540)
    )

    right_display = cv2.resize(
        right,
        (960, 540)
    )

    # ======================================
    # SIDE BY SIDE
    # ======================================

    stereo = cv2.hconcat([
        left_display,
        right_display
    ])

    cv2.imshow(
        "IMX219-83 Stereo Camera",
        stereo
    )

    key = cv2.waitKey(1) & 0xFF

    # ======================================
    # SAVE
    # ======================================

    if key == ord('s'):

        cv2.imwrite("left.jpg", left)
        cv2.imwrite("right.jpg", right)

        print("Saved images")

    # ======================================
    # EXIT
    # ======================================

    elif key == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================

left_cam.stop()
right_cam.stop()

cv2.destroyAllWindows()

print("Camera stopped")