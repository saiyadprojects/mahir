import pygame
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

servo_angles = [90, 90, 90, 90, 90, 90]

selected_servo = 0

SERVO_SPEED = 2.5


# =========================================================
# INITIAL SERVO POSITIONS
# =========================================================

for i in range(6):
    kit.servo[i].angle = servo_angles[i]


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


print("==========================================")
print("        PS5 ROBOT CONTROLLER")
print("==========================================")

print("Controller:", joystick.get_name())

print("Buttons:", joystick.get_numbuttons())
print("Axes:", joystick.get_numaxes())

print()
print("==========================================")
print("CONTROLS")
print("==========================================")

print("Left Stick  -> Robot movement")
print("Right Stick -> Selected servo")
print()

print("Square   -> Base")
print("Triangle -> Shoulder")
print("Circle   -> Elbow")
print("Cross    -> Wrist")
print("L1       -> Gripper")
print("R1       -> Camera")

print()

print("L2 -> PUMP ON/OFF TOGGLE")

print()

print("OPTIONS -> EXIT")

print("==========================================")
print()


# =========================================================
# BUTTON STATE
# =========================================================

last_buttons = [
    0
] * joystick.get_numbuttons()


# =========================================================
# SERVO SELECTION
# =========================================================

mapping = {
    3: 0,  # Square -> Base
    2: 1,  # Triangle -> Shoulder
    1: 2,  # Circle -> Elbow
    0: 3,  # Cross -> Wrist
    4: 4,  # L1 -> Gripper
    5: 5   # R1 -> Camera
}


# =========================================================
# MOTOR FUNCTION
# =========================================================

def set_motor(left_speed, right_speed):

    # -----------------------------
    # LEFT MOTOR
    # -----------------------------

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


    # -----------------------------
    # RIGHT MOTOR
    # -----------------------------

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
# INITIAL STATE
# =========================================================

set_motor(0, 0)

pump_off()


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


        # Deadzone

        if abs(y) < DEADZONE:
            y = 0

        if abs(x) < DEADZONE:
            x = 0


        # Differential drive

        left = max(
            -1,
            min(1, y + x)
        )

        right = max(
            -1,
            min(1, y - x)
        )


        set_motor(
            left,
            right
        )


        # =================================================
        # READ BUTTONS
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

                # Detect NEW button press

                if (
                    buttons[btn]
                    and not last_buttons[btn]
                ):

                    selected_servo = servo

                    print(
                        "Selected Servo:",
                        servo_names[selected_servo]
                    )


        # =================================================
        # PUMP TOGGLE
        # =================================================
        #
        # L2 is normally button 6 on many
        # PS5/DualSense mappings.
        #
        # If your controller reports L2
        # differently, change PUMP_BUTTON.
        #

        PUMP_BUTTON = 6

        if PUMP_BUTTON < len(buttons):

            if (
                buttons[PUMP_BUTTON]
                and not last_buttons[PUMP_BUTTON]
            ):

                pump_toggle()


        # =================================================
        # OPTIONS -> EXIT
        # =================================================

        EXIT_BUTTON = 9

        if EXIT_BUTTON < len(buttons):

            if (
                buttons[EXIT_BUTTON]
                and not last_buttons[EXIT_BUTTON]
            ):

                print("Exit requested")

                running = False


        # =================================================
        # SAVE BUTTON STATE
        # =================================================

        last_buttons = buttons


        # =================================================
        # SERVO MOTION - RIGHT STICK Y
        # =================================================

        if joystick.get_numaxes() > 3:

            ry = -joystick.get_axis(3)

            if abs(ry) > DEADZONE:

                servo_angles[selected_servo] += (
                    ry * SERVO_SPEED
                )


                # Limit angle

                servo_angles[selected_servo] = max(
                    0,
                    min(
                        180,
                        servo_angles[selected_servo]
                    )
                )


                kit.servo[
                    selected_servo
                ].angle = servo_angles[
                    selected_servo
                ]


        sleep(0.02)


# =========================================================
# STOP
# =========================================================

except KeyboardInterrupt:

    print()
    print("Keyboard interrupt")


finally:

    print()
    print("Stopping robot...")


    # Stop motors

    PWM1.value = 0
    PWM2.value = 0


    # Turn pump OFF

    pump_off()


    # Close GPIO

    PWM1.close()
    PWM2.close()

    DIR1.close()
    DIR2.close()

    PUMP_IN1.close()
    PUMP_IN2.close()


    pygame.quit()


    print("Motors stopped")
    print("Pump OFF")
    print("Program stopped")