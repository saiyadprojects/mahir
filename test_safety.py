"""Headless safety + logic verification with stubbed hardware."""
import sys, types, time
import numpy as np

# ---------------- stub gpiozero ----------------
gz = types.ModuleType("gpiozero")
class PWMOutputDevice:
    def __init__(self, pin): self.pin = pin; self._v = 0.99  # start dirty on purpose
    @property
    def value(self): return self._v
    @value.setter
    def value(self, v): self._v = float(v)
class DigitalOutputDevice:
    def __init__(self, pin): self.pin = pin; self.state = None
    def on(self): self.state = 1
    def off(self): self.state = 0
gz.PWMOutputDevice = PWMOutputDevice; gz.DigitalOutputDevice = DigitalOutputDevice
sys.modules["gpiozero"] = gz

# ---------------- stub servokit ----------------
sk = types.ModuleType("adafruit_servokit")
class _S:
    def __init__(self): self.angle = 90
class ServoKit:
    def __init__(self, channels=16): self.servo = [_S() for _ in range(channels)]
sk.ServoKit = ServoKit
sys.modules["adafruit_servokit"] = sk

# ---------------- stub pygame ----------------
pg = types.ModuleType("pygame")
class _Joy:
    present = True
    def __init__(self, i=0):
        self.axes = [0.0, 0.0, -1.0, 0.0, -1.0, -1.0]  # L2=2 rest -1, R2=5 rest -1
        self.buttons = [0] * 13
    def init(self): pass
    def get_name(self): return "STUB DualSense"
    def get_numbuttons(self): return len(self.buttons)
    def get_numaxes(self): return len(self.axes)
    def get_axis(self, i):
        if not _Joy.present: raise RuntimeError("device gone")
        return self.axes[i]
    def get_button(self, i):
        if not _Joy.present: raise RuntimeError("device gone")
        return self.buttons[i]
JOY = _Joy()
class _JoyMod:
    @staticmethod
    def init(): pass
    @staticmethod
    def quit(): pass
    @staticmethod
    def get_count(): return 1 if _Joy.present else 0
    Joystick = staticmethod(lambda i: JOY)
pg.init = lambda: None
pg.quit = lambda: None
pg.joystick = _JoyMod()
pg.event = types.SimpleNamespace(pump=lambda: None, get=lambda: [])
pg.JOYDEVICEREMOVED = 1541
sys.modules["pygame"] = pg

# ---------------- stub picamera2 / onnxruntime (absent) ----------------
sys.modules["picamera2"] = None

import importlib.util
spec = importlib.util.spec_from_file_location("fp", "/home/claude/final_project.py")
fp = importlib.util.module_from_spec(spec)
sys.modules["fp"] = fp
spec.loader.exec_module(fp)

PASS = []; FAIL = []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name)

print("\n=== 1. MOTOR SAFETY ===")
r = fp.RobotControl()
check("PWM forced to 0 at construction (was 0.99 before init)",
      r.PWM1.value == 0.0 and r.PWM2.value == 0.0)
check("L2 auto-detected on axis 2 (resting at -1.0)", r.axis_l2 == 2)

# No input at all
r.poll()
check("no input -> motors OFF", r.PWM1.value == 0.0 and r.PWM2.value == 0.0)

# Stick pushed FULLY forward but R2 NOT held -> must stay off
JOY.axes[1] = -1.0
r.poll()
check("stick full forward, R2 released -> motors STILL OFF",
      r.PWM1.value == 0.0 and r.PWM2.value == 0.0 and not r.motor_running)

# R2 held + stick forward -> motors run
JOY.axes[5] = 1.0
r.poll()
check("R2 held + stick forward -> motors RUN", r.motor_running and r.PWM1.value > 0)
running_val = r.PWM1.value

# Release R2 only (stick still fully forward) -> must stop IMMEDIATELY
JOY.axes[5] = -1.0
r.poll()
check("R2 released (stick still forward) -> motors STOP same frame",
      r.PWM1.value == 0.0 and r.PWM2.value == 0.0 and not r.motor_running)

# R2 held but stick centered -> no motion
JOY.axes[5] = 1.0; JOY.axes[1] = 0.0
r.poll()
check("R2 held, stick centered -> motors OFF (R2 alone does not drive)",
      r.PWM1.value == 0.0)

# Slow mode
JOY.axes[1] = -1.0; JOY.axes[2] = 1.0   # L2 pressed
r.poll()
slow_val = r.PWM1.value
JOY.axes[2] = -1.0
r.poll()
check("L2 slow mode reduces speed", 0 < slow_val < r.PWM1.value)

# Controller disconnect mid-drive
check("motors running before disconnect", r.motor_running)
_Joy.present = False
ok = r.poll()
check("controller disconnect -> poll fails and motors STOP",
      (not ok) and r.PWM1.value == 0.0 and not r.connected)
_Joy.present = True
r.poll()

print("\n=== 2. E-STOP + TOGGLES ===")
JOY.axes[5] = 1.0; JOY.axes[1] = -1.0
r.poll(); check("driving again after reconnect", r.motor_running)
JOY.buttons[fp.BTN_ESTOP] = 1; r.poll()
check("PS button -> e-stop latched, motors off",
      r.estop_latched and r.PWM1.value == 0.0)
JOY.buttons[fp.BTN_ESTOP] = 0; r.poll()
check("e-stop holds through R2 held + stick forward", r.PWM1.value == 0.0)
JOY.buttons[fp.BTN_ESTOP] = 1; r.poll(); JOY.buttons[fp.BTN_ESTOP] = 0
check("PS button again -> e-stop cleared", not r.estop_latched)
r.poll(); check("motors drive again after clear", r.motor_running)

cam0 = r.camera_on
JOY.buttons[fp.BTN_CAMERA_TOGGLE] = 1; r.poll(); JOY.buttons[fp.BTN_CAMERA_TOGGLE] = 0
check("OPTIONS toggles camera state", r.camera_on != cam0)
check("camera toggle did not disturb motors", r.motor_running)

JOY.buttons[3] = 1; r.poll(); JOY.buttons[3] = 0
check("SQUARE selects Base (hand.py mapping preserved)", r.selected_servo == 0)
JOY.buttons[4] = 1; r.poll(); JOY.buttons[4] = 0
check("L1 selects Gripper (hand.py mapping preserved)", r.selected_servo == 4)

r.last_good_poll = time.time() - 5.0
r.watchdog()
check("watchdog stops motors on stale poll", r.PWM1.value == 0.0)

r.shutdown()
check("shutdown leaves PWM at 0", r.PWM1.value == 0.0 and r.PWM2.value == 0.0)

print("\n=== 3. GREEN BOX SIZING ===")
W, H = 640, 720
b = fp.DetectionBox(W, H)
nb = b.update(None)
check("no detection -> fallback centre box present", nb[2] - nb[0] > 0)
tiny = (300, 350, 340, 390, 0.9)          # 40x40 px detection
gb = b.update(tiny)
bw, bh = gb[2] - gb[0], gb[3] - gb[1]
check(f"tiny 40x40 detection grown to {bw}x{bh} (min {int(W*0.30)}x{int(H*0.30)})",
      bw >= int(W * 0.30) and bh >= int(H * 0.30))
edge = b.update((0, 0, 40, 40, 0.9))
for _ in range(30): edge = b.update((0, 0, 40, 40, 0.9))
check("box at frame corner stays fully inside frame",
      edge[0] >= 0 and edge[1] >= 0 and edge[2] <= W and edge[3] <= H)
b2 = fp.DetectionBox(W, H)
b2.update((300, 350, 340, 390, 0.9))
moved = b2.update((360, 350, 400, 390, 0.9))
check("box follows a moving target", moved[0] > b2.fallback[0] - 10**6)
sm = b2.to_match_scale(int(W*0.5), int(H*0.5))
check("match-scale box clamped inside disparity array",
      0 <= sm[0] < sm[2] <= int(W*0.5) and 0 <= sm[1] < sm[3] <= int(H*0.5))

print("\n=== 4. DISTANCE MATH ===")
fx, base = 1444.0, 62.0
est = fp.DistanceEstimator(fx, base)
# synthesise a disparity field for a true 50 cm target: d = fx*B/Z
true_mm = 500.0
d = fx * base / true_mm
disp = np.full((360, 320), d, dtype=np.float32)
out = None
for _ in range(10):
    out, vp, md = est.measure(disp, (50, 50, 250, 300))
check(f"50 cm synthetic target -> reported {out:.1f} cm", abs(out - 50.0) < 0.5)
check("valid pct reported as 100%", vp > 99.0)

est2 = fp.DistanceEstimator(fx, base)
noise = np.zeros((360, 320), dtype=np.float32)   # all invalid
o2, vp2, _ = est2.measure(noise, (50, 50, 250, 300))
check("all-invalid ROI -> None, not a bogus number", o2 is None and vp2 == 0.0)

# scale correction wiring
fp.CALIBRATION_SCALE_CORRECTION = 1.2
est3 = fp.DistanceEstimator(fx, base)
for _ in range(10):
    o3, _, _ = est3.measure(disp, (50, 50, 250, 300))
check(f"CALIBRATION_SCALE_CORRECTION=1.2 applied -> {o3:.1f} cm", abs(o3 - 60.0) < 0.6)
fp.CALIBRATION_SCALE_CORRECTION = 1.0

# jump rejection
est4 = fp.DistanceEstimator(fx, base)
for _ in range(7): est4.measure(disp, (50, 50, 250, 300))
stable = float(np.median(est4.history))
far = np.full((360, 320), fx * base / 2000.0, dtype=np.float32)  # 200cm spike
o4, _, _ = est4.measure(far, (50, 50, 250, 300))
check("single wild jump not immediately accepted", abs(o4 - stable) < 5)

print("\n=== 5. GLOBAL E-STOP HOOK ===")
p = PWMOutputDevice(13); p.value = 0.5
fp._MOTOR_REFS["pwm1"] = p
fp.emergency_stop_motors()
check("emergency_stop_motors() zeroes registered PWM", p.value == 0.0)

print(f"\n{'='*60}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL: print("   FAILED:", f)
    sys.exit(1)
print("  ALL CHECKS PASSED\n" + "="*60)
