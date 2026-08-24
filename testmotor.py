
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import sleep

# ------------------------
# Motor Pin Configuration
# ------------------------

# Left Motor
PWM1 = PWMOutputDevice(13)          # GPIO13
DIR1 = DigitalOutputDevice(16)      # GPIO16

# Right Motor
PWM2 = PWMOutputDevice(12)          # GPIO12
DIR2 = DigitalOutputDevice(20)      # GPIO20

# ------------------------
# Functions
# ------------------------

def forward(speed=0.7):
    DIR1.on()
    DIR2.on()

    PWM1.value = speed
    PWM2.value = speed


def reverse(speed=0.7):
    DIR1.off()
    DIR2.off()

    PWM1.value = speed
    PWM2.value = speed


def stop():
    PWM1.value = 0
    PWM2.value = 0


# ------------------------
# Main Program
# ------------------------

try:
    print("Forward")
    forward(0.6)
    sleep(3)

    print("Stop")
    stop()
    sleep(2)

    print("Reverse")
    reverse(0.6)
    sleep(3)

    print("Stop")
    stop()

finally:
    stop()