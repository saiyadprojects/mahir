from gpiozero import OutputDevice
from time import sleep

RELAY_PIN = 26

relay = OutputDevice(
    RELAY_PIN,
    active_high=True,
    initial_value=False
)

print("Relay test started")

try:
    while True:
        print("RELAY ON")
        relay.off()
        sleep(3)

        print("RELAY OFF")
        relay.on()
        sleep(3)

except KeyboardInterrupt:
    relay.off()
    print("Stopped")
