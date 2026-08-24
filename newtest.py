import pygame
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import sleep

# -----------------------------
# Cytron Motor Driver Pins
# -----------------------------
PWM1 = PWMOutputDevice(13)
DIR1 = DigitalOutputDevice(16)

PWM2 = PWMOutputDevice(12)
DIR2 = DigitalOutputDevice(20)

MAX_SPEED = 0.5      # Change between 0.3 and 1.0
DEADZONE = 0.15

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No PS5 Controller Connected!")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print("Connected:", joystick.get_name())


def set_motor(left_speed, right_speed):

    # Left Motor
    if left_speed >= 0:
        DIR1.off()      # Forward
        PWM1.value = min(left_speed, MAX_SPEED)
    else:
        DIR1.on()       # Reverse
        PWM1.value = min(abs(left_speed), MAX_SPEED)

    # Right Motor
    if right_speed >= 0:
        DIR2.off()      # Forward
        PWM2.value = min(right_speed, MAX_SPEED)
    else:
        DIR2.on()       # Reverse
        PWM2.value = min(abs(right_speed), MAX_SPEED)


try:
    while True:

        pygame.event.pump()

        # Left joystick
        y = -joystick.get_axis(1)   # Forward/Backward
        x = joystick.get_axis(0)    # Left/Right

        if abs(y) < DEADZONE:
            y = 0

        if abs(x) < DEADZONE:
            x = 0

        left = y + x
        right = y - x

        left = max(-1, min(1, left))
        right = max(-1, min(1, right))

        set_motor(left, right)

        sleep(0.02)

except KeyboardInterrupt:
    PWM1.value = 0
    PWM2.value = 0
    print("Stopped")