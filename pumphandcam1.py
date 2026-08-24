import pygame
import cv2
import time

from picamera2 import Picamera2
from gpiozero import PWMOutputDevice, DigitalOutputDevice, OutputDevice
from adafruit_servokit import ServoKit
from time import sleep


# =========================================================
# CYTRON MOTOR DRIVER
# =========================================================

PWM1 = PWMOutputDevice(13)
DIR1 = DigitalOutputDevice(16)

PWM2 = PWMOutputDevice(12)
DIR2 = DigitalOutputDevice(20)

MAX_SPEED = 0.5
DEADZONE = 0.15


# =========================================================
# L293D PUMP
# =========================================================

PUMP_IN1 = OutputDevice(27)
PUMP_IN2 = OutputDevice(22)

pump_state = False


def pump_on():

    global pump_state

    PUMP_IN1.on()
    PUMP_IN2.off()

    pump_state = True

    print(">>> PUMP ON")


def pump_off():

    global pump_state

    PUMP_IN1.off()
    PUMP_IN2.off()

    pump_state = False

    print(">>> PUMP OFF")


def pump_toggle():

    if pump_state:
        pump_off()
    else:
        pump_on()


# =========================================================
# PCA9685 SERVO SETUP
# =========================================================

kit = ServoKit(channels=16)

servo_names = [
    "Base",
    "Shoulder",
    "Elbow",
    "Wrist",
    "Gripper",
    "Camera"
]

servo_angles = [
    90,
    90,
    90,
    90,
    90,
    90
]

selected_servo = 0

SERVO_SPEED = 2.5


# =========================================================
# INITIAL SERVO POSITIONS
# =========================================================

for i in range(6):

    kit.servo[i].angle = servo_angles[i]


# =========================================================
# STEREO CAMERA SETUP
# =========================================================

left_cam = Picamera2(0)
right_cam = Picamera2(1)


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


camera_active = False


# =========================================================
# CAMERA ON
# =========================================================

def camera_on():

    global camera_active

    if camera_active:
        return

    print()
    print("==========================================")
    print("       STEREO CAMERA: ON")
    print("       LEFT  = Camera 0")
    print("       RIGHT = Camera 1")
    print("==========================================")


    left_cam.start()
    right_cam.start()

    time.sleep(1)

    camera_active = True


# =========================================================
# CAMERA OFF
# =========================================================

def camera_off():

    global camera_active

    if not camera_active:
        return

    print()
    print("==========================================")
    print("       STEREO CAMERA: OFF")
    print("==========================================")


    try:
        left_cam.stop()
    except Exception:
        pass


    try:
        right_cam.stop()
    except Exception:
        pass


    camera_active = False

    try:
        cv2.destroyWindow(
            "IMX219-83 Stereo Camera"
        )
    except Exception:
        pass


# =========================================================
# CAMERA TOGGLE
# =========================================================

def camera_toggle():

    if camera_active:

        camera_off()

    else:

        camera_on()


# =========================================================
# PYGAME / PS5
# =========================================================

pygame.init()
pygame.joystick.init()


if pygame.joystick.get_count() == 0:

    print("==========================================")
    print("NO PS5 CONTROLLER CONNECTED")
    print("==========================================")

    pump_off()

    raise SystemExit


joystick = pygame.joystick.Joystick(0)
joystick.init()


print()
print("==========================================")
print("        PS5 ROBOT CONTROLLER")
print("==========================================")

print(
    "Controller:",
    joystick.get_name()
)

print(
    "Buttons:",
    joystick.get_numbuttons()
)

print(
    "Axes:",
    joystick.get_numaxes()
)

print(
    "Hats:",
    joystick.get_numhats()
)

print()
print("==========================================")
print("CONTROLS")
print("==========================================")

print("Left Stick       -> Robot movement")
print("Right Stick      -> Selected servo")
print()

print("D-PAD UP         -> Stereo Camera ON/OFF")
print()

print("Square           -> Base")
print("Triangle         -> Shoulder")
print("Circle           -> Elbow")
print("Cross            -> Wrist")
print("L1               -> Gripper")
print("R1               -> Camera servo")
print()

print("L2               -> Pump ON/OFF")
print("OPTIONS          -> EXIT")

print()
print("==========================================")
print()


# =========================================================
# BUTTON STATE
# =========================================================

last_buttons = [
    0
] * joystick.get_numbuttons()


# =========================================================
# D-PAD STATE
# =========================================================

if joystick.get_numhats() > 0:

    last_hat = joystick.get_hat(0)

else:

    last_hat = (0, 0)


# =========================================================
# SERVO BUTTON MAPPING
# =========================================================

mapping = {

    3: 0,       # Square   -> Base
    2: 1,       # Triangle -> Shoulder
    1: 2,       # Circle   -> Elbow
    0: 3,       # Cross    -> Wrist
    4: 4,       # L1       -> Gripper
    5: 5        # R1       -> Camera servo
}


# =========================================================
# SPECIAL BUTTONS
# =========================================================

PUMP_BUTTON = 6

EXIT_BUTTON = 9


# =========================================================
# MOTOR FUNCTION
# =========================================================

def set_motor(left_speed, right_speed):

    # -----------------------------------------------------
    # LEFT MOTOR
    # -----------------------------------------------------

    if left_speed >= 0:

        DIR1.off()

        PWM1.value = min(
            left_speed,
            MAX_SPEED
        )

    else:

        DIR1.on()

        PWM1.value = min(
            abs(left_speed),
            MAX_SPEED
        )


    # -----------------------------------------------------
    # RIGHT MOTOR
    # -----------------------------------------------------

    if right_speed >= 0:

        DIR2.off()

        PWM2.value = min(
            right_speed,
            MAX_SPEED
        )

    else:

        DIR2.on()

        PWM2.value = min(
            abs(right_speed),
            MAX_SPEED
        )


# =========================================================
# CAMERA WINDOW
# =========================================================

cv2.namedWindow(
    "IMX219-83 Stereo Camera",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "IMX219-83 Stereo Camera",
    1920,
    720
)


# =========================================================
# INITIAL STATE
# =========================================================

set_motor(
    0,
    0
)

pump_off()


# =========================================================
# CAMERA STARTS ON
# =========================================================

camera_on()


# =========================================================
# MAIN LOOP
# =========================================================

running = True


try:

    while running:

        pygame.event.pump()


        # =================================================
        # CAR CONTROL - LEFT STICK
        # =================================================

        y = -joystick.get_axis(1)

        x = joystick.get_axis(0)


        # -------------------------------------------------
        # DEADZONE
        # -------------------------------------------------

        if abs(y) < DEADZONE:

            y = 0


        if abs(x) < DEADZONE:

            x = 0


        # -------------------------------------------------
        # DIFFERENTIAL DRIVE
        # -------------------------------------------------

        left = max(
            -1,
            min(
                1,
                y + x
            )
        )


        right = max(
            -1,
            min(
                1,
                y - x
            )
        )


        set_motor(
            left,
            right
        )


        # =================================================
        # CAMERA DISPLAY
        # =================================================

        if camera_active:

            # ---------------------------------------------
            # CAPTURE LEFT
            # ---------------------------------------------

            left = left_cam.capture_array()


            # ---------------------------------------------
            # CAPTURE RIGHT
            # ---------------------------------------------

            right = right_cam.capture_array()


            # ---------------------------------------------
            # LABEL LEFT
            # ---------------------------------------------

            cv2.putText(
                left,
                "LEFT CAMERA",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            # ---------------------------------------------
            # LABEL RIGHT
            # ---------------------------------------------

            cv2.putText(
                right,
                "RIGHT CAMERA",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            # ---------------------------------------------
            # RESIZE
            # ---------------------------------------------

            left_display = cv2.resize(
                left,
                (960, 540)
            )


            right_display = cv2.resize(
                right,
                (960, 540)
            )


            # ---------------------------------------------
            # SIDE BY SIDE
            # ---------------------------------------------

            stereo = cv2.hconcat(
                [
                    left_display,
                    right_display
                ]
            )


            # ---------------------------------------------
            # SHOW
            # ---------------------------------------------

            cv2.imshow(
                "IMX219-83 Stereo Camera",
                stereo
            )


            # ---------------------------------------------
            # KEYBOARD
            # ---------------------------------------------

            key = cv2.waitKey(1) & 0xFF


            # ---------------------------------------------
            # S = SAVE IMAGES
            # ---------------------------------------------

            if key == ord('s'):

                cv2.imwrite(
                    "left.jpg",
                    left
                )

                cv2.imwrite(
                    "right.jpg",
                    right
                )

                print("Saved stereo images")


            # ---------------------------------------------
            # Q = CAMERA OFF
            # ---------------------------------------------

            elif key == ord('q'):

                camera_off()


        else:

            cv2.waitKey(1)


        # =================================================
        # READ PS5 BUTTONS
        # =================================================

        buttons = [
            joystick.get_button(i)
            for i in range(
                joystick.get_numbuttons()
            )
        ]


        # =================================================
        # SERVO SELECTION
        # =================================================

        for btn, servo in mapping.items():

            if btn < len(buttons):

                # NEW BUTTON PRESS

                if (
                    buttons[btn]
                    and not last_buttons[btn]
                ):

                    selected_servo = servo

                    print(
                        "Selected Servo:",
                        servo_names[
                            selected_servo
                        ]
                    )


        # =================================================
        # L2 -> PUMP ON/OFF
        # =================================================

        if PUMP_BUTTON < len(buttons):

            if (
                buttons[PUMP_BUTTON]
                and not last_buttons[PUMP_BUTTON]
            ):

                pump_toggle()


        # =================================================
        # D-PAD UP -> CAMERA ON/OFF
        # =================================================
        #
        # PS5 D-PAD is normally reported as a HAT.
        #
        # get_hat(0) returns:
        #
        # UP    = (0, 1)
        # DOWN  = (0,-1)
        # LEFT  = (-1,0)
        # RIGHT = (1,0)
        #

        if joystick.get_numhats() > 0:

            current_hat = joystick.get_hat(0)

            # Detect NEW D-PAD UP press

            if (
                current_hat[1] == 1
                and last_hat[1] != 1
            ):

                print()
                print(">>> D-PAD UP PRESSED")
                print(">>> TOGGLING STEREO CAMERA")

                camera_toggle()

            last_hat = current_hat


        # =================================================
        # OPTIONS -> EXIT
        # =================================================

        if EXIT_BUTTON < len(buttons):

            if (
                buttons[EXIT_BUTTON]
                and not last_buttons[EXIT_BUTTON]
            ):

                print()
                print("Exit requested")

                running = False


        # =================================================
        # SAVE BUTTON STATE
        # =================================================

        last_buttons = buttons


        # =================================================
        # RIGHT STICK Y -> SERVO
        # =================================================

        if joystick.get_numaxes() > 3:

            ry = -joystick.get_axis(3)


            if abs(ry) > DEADZONE:

                servo_angles[
                    selected_servo
                ] += (
                    ry * SERVO_SPEED
                )


                # -----------------------------------------
                # LIMIT SERVO ANGLE
                # -----------------------------------------

                servo_angles[
                    selected_servo
                ] = max(
                    0,
                    min(
                        180,
                        servo_angles[
                            selected_servo
                        ]
                    )
                )


                kit.servo[
                    selected_servo
                ].angle = servo_angles[
                    selected_servo
                ]


        sleep(0.02)


# =========================================================
# CLEANUP
# =========================================================

except KeyboardInterrupt:

    print()
    print("Keyboard interrupt")


finally:

    print()
    print("Stopping robot...")


    # =====================================================
    # STOP MOTORS
    # =====================================================

    PWM1.value = 0
    PWM2.value = 0


    # =====================================================
    # PUMP OFF
    # =====================================================

    pump_off()


    # =====================================================
    # CAMERA OFF
    # =====================================================

    if camera_active:

        camera_off()


    # =====================================================
    # CLOSE GPIO
    # =====================================================

    PWM1.close()
    PWM2.close()

    DIR1.close()
    DIR2.close()

    PUMP_IN1.close()
    PUMP_IN2.close()


    # =====================================================
    # CLOSE PYGAME
    # =====================================================

    pygame.quit()


    # =====================================================
    # CLOSE OPENCV
    # =====================================================

    cv2.destroyAllWindows()


    print()
    print("==========================================")
    print("Motors stopped")
    print("Pump OFF")
    print("Camera OFF")
    print("Program stopped")
    print("==========================================")
