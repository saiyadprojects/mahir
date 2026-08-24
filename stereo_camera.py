from picamera2 import Picamera2
import cv2
import time

# ==========================================
# CREATE LEFT AND RIGHT CAMERAS
# ==========================================

left_cam = Picamera2(0)
right_cam = Picamera2(1)

# ==========================================
# CAMERA CONFIGURATION
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
# START BOTH CAMERAS
# ==========================================

left_cam.start()
right_cam.start()

time.sleep(2)

print("===================================")
print(" IMX219-83 STEREO CAMERA")
print("===================================")
print("Left Camera  : Camera 0")
print("Right Camera : Camera 1")
print()
print("S = Save stereo images")
print("Q = Quit")
print("===================================")

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # Capture images
    left = left_cam.capture_array()
    right = right_cam.capture_array()

    # Picamera2 gives RGB, OpenCV uses BGR
    left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)

    # --------------------------------------
    # Put camera names on images
    # --------------------------------------

    cv2.putText(
        left,
        "LEFT CAMERA",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        right,
        "RIGHT CAMERA",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # --------------------------------------
    # Combine side-by-side
    # --------------------------------------

    stereo_view = cv2.hconcat([left, right])

    # Show combined image
    cv2.imshow("IMX219-83 STEREO CAMERA", stereo_view)

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------
    # SAVE IMAGES
    # --------------------------------------

    if key == ord('s'):

        cv2.imwrite("left.jpg", left)
        cv2.imwrite("right.jpg", right)
        cv2.imwrite("stereo.jpg", stereo_view)

        print("Images saved:")
        print("  left.jpg")
        print("  right.jpg")
        print("  stereo.jpg")

    # --------------------------------------
    # EXIT
    # --------------------------------------

    elif key == ord('q'):
        break


# ==========================================
# CLEANUP
# ==========================================

left_cam.stop()
right_cam.stop()

cv2.destroyAllWindows()

print("Stereo camera stopped.")