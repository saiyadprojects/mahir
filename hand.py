import pygame
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from adafruit_servokit import ServoKit
from time import sleep

# -----------------------------
# Cytron Motor Driver Pins
# -----------------------------
PWM1 = PWMOutputDevice(13)
DIR1 = DigitalOutputDevice(16)

PWM2 = PWMOutputDevice(12)
DIR2 = DigitalOutputDevice(20)

MAX_SPEED = 0.5
DEADZONE = 0.15

# -----------------------------
# PCA9685 Servo Setup
# -----------------------------
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

for i in range(6):
    kit.servo[i].angle = servo_angles[i]

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No PS5 Controller Connected!")
    raise SystemExit

joystick = pygame.joystick.Joystick(0)
joystick.init()

print("Connected:", joystick.get_name())
print("Selected Servo:", servo_names[selected_servo])

last_buttons = [0] * joystick.get_numbuttons()

def set_motor(left_speed, right_speed):
    if left_speed >= 0:
        DIR1.off()
        PWM1.value = min(left_speed, MAX_SPEED)
    else:
        DIR1.on()
        PWM1.value = min(abs(left_speed), MAX_SPEED)

    if right_speed >= 0:
        DIR2.off()
        PWM2.value = min(right_speed, MAX_SPEED)
    else:
        DIR2.on()
        PWM2.value = min(abs(right_speed), MAX_SPEED)

try:
    while True:
        pygame.event.pump()

        # -------- Car Control (Left Stick) --------
        y = -joystick.get_axis(1)
        x = joystick.get_axis(0)

        if abs(y) < DEADZONE:
            y = 0
        if abs(x) < DEADZONE:
            x = 0

        left = max(-1, min(1, y + x))
        right = max(-1, min(1, y - x))
        set_motor(left, right)

        # -------- Servo Selection --------
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]

        mapping = {
            3: 0,  # Square -> Base
            2: 1,  # Triangle -> Shoulder
            1: 2,  # Circle -> Elbow
            0: 3,  # Cross -> Wrist
            4: 4,  # L1 -> Gripper
            5: 5   # R1 -> Camera
        }

        for btn, servo in mapping.items():
            if buttons[btn] and not last_buttons[btn]:
                selected_servo = servo
                print(f"Selected: {servo_names[selected_servo]}")

        last_buttons = buttons

        # -------- Servo Motion (Right Stick Y) --------
        ry = -joystick.get_axis(3)

        if abs(ry) > DEADZONE:
            servo_angles[selected_servo] += ry * SERVO_SPEED
            servo_angles[selected_servo] = max(0, min(180, servo_angles[selected_servo]))
            kit.servo[selected_servo].angle = servo_angles[selected_servo]

        sleep(0.02)

except KeyboardInterrupt:
    PWM1.value = 0
    PWM2.value = 0
    print("Stopped")
