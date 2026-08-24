import pygame
from gpiozero import OutputDevice
from time import sleep

# ==========================================
# L293D PUMP CONNECTION
# ==========================================

IN1 = OutputDevice(27)
IN2 = OutputDevice(22)

# ==========================================
# PUMP FUNCTIONS
# ==========================================

def pump_on():
    IN1.on()
    IN2.off()
    print("PUMP ON")


def pump_off():
    IN1.off()
    IN2.off()
    print("PUMP OFF")


# ==========================================
# INITIALIZE
# ==========================================

pygame.init()
pygame.joystick.init()

print("==========================================")
print("       PS5 PUMP CONTROLLER")
print("==========================================")

if pygame.joystick.get_count() == 0:
    print("No PS5 controller detected!")
    print("Make sure the controller is connected.")
    pump_off()
    quit()

controller = pygame.joystick.Joystick(0)
controller.init()

print("Controller connected:")
print(controller.get_name())

print()
print("X  = PUMP ON")
print("O  = PUMP OFF")
print("OPTIONS = EXIT")
print()

# Pump OFF when program starts
pump_off()

running = True

try:

    while running:

        for event in pygame.event.get():

            # ==================================
            # BUTTON PRESSED
            # ==================================

            if event.type == pygame.JOYBUTTONDOWN:

                # PS5 X button
                if event.button == 0:
                    pump_on()

                # PS5 O button
                elif event.button == 1:
                    pump_off()

                # OPTIONS button
                elif event.button == 9:
                    print("Exiting...")
                    running = False

        sleep(0.01)

except KeyboardInterrupt:
    pass

finally:

    # ALWAYS TURN PUMP OFF
    pump_off()

    IN1.close()
    IN2.close()

    pygame.quit()

    print("Pump OFF")
    print("Program stopped.")