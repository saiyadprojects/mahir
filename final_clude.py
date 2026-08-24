#!/usr/bin/env python3
"""
===============================================================================
 final_project.py - Smart Pollination Platform (integrated, safety-hardened)
===============================================================================

 Combines three previously-separate pieces:

   1. distance.py  (LATEST version) -> stereo rectification, SGBM disparity,
                                       robust median distance, temporal
                                       smoothing, jump rejection.
                                       *** Distance math is copied verbatim,
                                           including CALIBRATION_SCALE_CORRECTION
                                           and the 0.5 ROI fraction fallback. ***
   2. final_project.py (previous)   -> YOLO26n ONNX rose detection, detection-
                                       driven ROI, robot actuation.
   3. hand.py                       -> PS5 controller mappings for the Cytron
                                       motor driver + 6x PCA9685 servos.
                                       *** Button->servo mapping is copied
                                           verbatim from hand.py. ***

-------------------------------------------------------------------------------
 THE BUG THIS VERSION FIXES (why your motors ran continuously)
-------------------------------------------------------------------------------
 The previous final_project.py called set_motor() ONLY when the stick was
 outside the deadzone:

       if stick_active:
           self.set_motor(left, right)      # <-- and nothing when released

 A Cytron driver latches its last PWM value. So the moment you nudged the
 stick and let go, the last non-zero PWM stayed applied and the motors kept
 running forever. (hand.py did NOT have this bug, because it called
 set_motor() unconditionally every loop, which naturally wrote 0 on release.)

 This version goes further than restoring hand.py's behavior: motors now
 require a dead-man trigger to be physically held down. See MOTOR SAFETY.

-------------------------------------------------------------------------------
 PS5 CONTROLLER CONFIGURATION  (pygame indices)
-------------------------------------------------------------------------------
 BUTTONS - these four rows are taken verbatim from hand.py's `mapping` dict,
 not assumed:

   Button 3  = SQUARE    -> select servo 0 : Base        [from hand.py]
   Button 2  = TRIANGLE  -> select servo 1 : Shoulder    [from hand.py]
   Button 1  = CIRCLE    -> select servo 2 : Elbow       [from hand.py]
   Button 0  = CROSS (X) -> select servo 3 : Wrist       [from hand.py]
   Button 4  = L1        -> select servo 4 : Gripper     [from hand.py]
   Button 5  = R1        -> select servo 5 : Camera servo[from hand.py]

 BUTTONS added by this file (hand.py did not define these):

   Button 9  = OPTIONS   -> toggle CAMERA FEED ON / OFF
   Button 8  = CREATE/SHARE -> toggle ASSIST MODE arm/disarm
   Button 10 = PS logo   -> EMERGENCY STOP (motors off + assist disarmed)

 AXES:

   Axis 0    = LEFT STICK X   -> steering            [from hand.py]
   Axis 1    = LEFT STICK Y   -> forward / reverse   [from hand.py, inverted]
   Axis 3    = RIGHT STICK Y  -> move selected servo [from hand.py, inverted]
   Axis 5    = R2 TRIGGER     -> *** DEAD-MAN: motors only run while held ***
   Axis 2/4  = L2 TRIGGER     -> slow/precision mode (auto-detected, see below)

 !! IMPORTANT !! hand.py only ever proved out axes 0, 1, 3 and buttons 0-5.
 It never used the triggers, so there is nothing in hand.py to copy for R2/L2
 and I will not silently guess. Two DualSense layouts are common on Linux:

       SDL GameController layout : 0=LX 1=LY 2=RX 3=RY 4=L2 5=R2
       SDL raw DualSense layout  : 0=LX 1=LY 2=L2 3=RX 4=RY 5=R2

 Both agree that R2 = axis 5, which is why R2 is the dead-man trigger. They
 disagree on L2 (axis 2 vs axis 4), so L2 is AUTO-DETECTED at startup by
 finding which axis rests at ~-1.0 (triggers rest at -1.0, sticks rest at 0).

 CONFIRM ALL OF THIS ON YOUR OWN CONTROLLER BEFORE DRIVING:

       python3 final_project.py --map

 That prints every button and axis index live as you press them. If anything
 differs, change the BTN_* / AXIS_* constants in the CONFIG block - they are
 all in one place.

-------------------------------------------------------------------------------
 MOTOR SAFETY MODEL
-------------------------------------------------------------------------------
   * PWM is explicitly written to 0.0 at construction, before anything else.
   * set_motor()/stop_motors() are called on EVERY loop iteration without
     exception, so no PWM value is ever left latched.
   * Motors are hard-gated behind R2. R2 released -> PWM 0 that same frame.
   * Left stick centered while R2 held -> speed 0 (holding R2 alone does
     not move the robot).
   * Controller unplugged / JOYDEVICEREMOVED / any pygame exception -> PWM 0.
   * Watchdog: if a full controller poll has not succeeded within
     CONTROLLER_TIMEOUT_S, PWM 0.
   * Assist mode can never drive on its own - it also requires R2 held.
   * atexit + SIGINT/SIGTERM handlers + finally block all call an
     unconditional emergency stop, so a crash or Ctrl-C cannot leave the
     motors energised.

 Keyboard (video window focused): q quit | c camera | d debug | r reset | SPACE e-stop
===============================================================================
"""

import os
import sys
import time
import atexit
import signal
import collections

import cv2
import numpy as np

# --- optional: rose detector -------------------------------------------------
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[INFO] onnxruntime not available - running with the fixed centre ROI "
          "instead of detection-driven ROI.")

# --- optional: hardware ------------------------------------------------------
try:
    import pygame
    from gpiozero import PWMOutputDevice, DigitalOutputDevice
    from adafruit_servokit import ServoKit
    HARDWARE_AVAILABLE = True
except ImportError as e:
    print(f"[INFO] Hardware libraries unavailable ({e}). "
          f"VISION-ONLY mode: no motors, no servos.")
    HARDWARE_AVAILABLE = False

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


# =============================================================================
# CONFIG
# =============================================================================

# ---- files ----
CALIB_FILE = "/home/pi/Desktop/MahirCode/calibration_images/stereo_calibration.npz"
MODEL_FILE = "/home/pi/Desktop/MahirCode/calibration_images/rose_new (1).onnx"

# ---- stereo / distance (VERBATIM from the latest distance.py) ----
MATCH_SCALE = 0.5                    # SGBM runs on a half-size copy, for speed
MIN_DISTANCE_MM = 350                # 35 cm near limit -> sizes numDisparities
MAX_DISTANCE_MM = 3000               # 3 m far limit
ROI_FRACTION = 0.5                   # fallback centre ROI = 50% of the frame
CALIBRATION_SCALE_CORRECTION = 1.0   # leave 1.0 until you have 3 known-distance
                                     # readings; if the error is a consistent
                                     # percentage, set this to
                                     #   actual_cm / measured_cm
MIN_VALID_PIXEL_PCT = 35.0           # min % of ROI with usable disparity
SMOOTH_WINDOW = 7                    # frames of median smoothing
MAX_JUMP_CM = 30                     # single-frame jump needing confirmation

# ---- detection ----
CONF_THRESHOLD = 0.45
MODEL_INPUT_SIZE = 640

# ---- GREEN DETECTION BOX sizing ----
# A tight YOLO box around a small flower gives SGBM almost nothing to match and
# falls off the target the moment the object moves. These three numbers keep
# the green box big and forgiving.
BOX_PAD_FRAC = 0.35        # grow the detected box by 35% of its size per side
BOX_MIN_W_FRAC = 0.40      # never narrower than 40% of frame width
BOX_MIN_H_FRAC = 0.40      # never shorter than 40% of frame height
                           # (40% ~= the 50% centre ROI you have been testing
                           #  with, so the box never shrinks below something
                           #  you already know gives a usable reading)
BOX_TRACK_SMOOTH = 0.35    # 0 = box frozen, 1 = box snaps instantly.
                           # 0.35 lets the box follow a moving hand smoothly
                           # without jittering frame to frame.

# ---- assist mode (still requires R2 held; never self-starts) ----
APPROACH_TARGET_CM = 18.0
CENTER_DEADZONE_PX = 40
DRIVE_SPEED = 0.28
TURN_GAIN = 0.55
ARM_SETTLE_SECONDS = 1.5
COLLECT_POSE = {           # PRESET pose, not inverse kinematics. Tune by hand.
    "Base": 90,
    "Shoulder": 60,
    "Elbow": 120,
    "Wrist": 90,
    "Gripper": 40,
}

# ---- motor / servo (from hand.py) ----
MAX_SPEED = 0.5            # hand.py value, unchanged
DEADZONE = 0.15            # hand.py value, unchanged
SERVO_SPEED = 2.5          # hand.py value, unchanged
servo_names = ["Base", "Shoulder", "Elbow", "Wrist", "Gripper", "Camera"]

# GPIO pins - hand.py values, unchanged
PIN_PWM1, PIN_DIR1 = 13, 16
PIN_PWM2, PIN_DIR2 = 12, 20

# ---- PS5 mapping (see the header block for the full explanation) ----
BTN_SERVO_MAP = {          # VERBATIM from hand.py
    3: 0,   # Square   -> Base
    2: 1,   # Triangle -> Shoulder
    1: 2,   # Circle   -> Elbow
    0: 3,   # Cross    -> Wrist
    4: 4,   # L1       -> Gripper
    5: 5,   # R1       -> Camera servo
}
BTN_CAMERA_TOGGLE = 9      # Options
BTN_ASSIST_TOGGLE = 8      # Create / Share
BTN_ESTOP = 10             # PS logo

AXIS_LEFT_X = 0            # from hand.py
AXIS_LEFT_Y = 1            # from hand.py
AXIS_RIGHT_Y = 3           # from hand.py
AXIS_R2 = 5                # dead-man. Same index in both known layouts.
AXIS_L2 = None             # auto-detected at startup; set an int here to force.

TRIGGER_PRESSED = 0.20     # normalised 0..1 threshold for "held"
SLOW_MODE_SCALE = 0.40     # L2 held -> 40% speed, for fine approach
CONTROLLER_TIMEOUT_S = 0.5 # no successful poll in this long -> motors off


# =============================================================================
# GLOBAL EMERGENCY STOP
# Registered with atexit and the signal handlers so that NOTHING - not an
# exception, not Ctrl-C, not SIGTERM - can leave PWM energised.
# =============================================================================
_MOTOR_REFS = {"pwm1": None, "pwm2": None}


def emergency_stop_motors():
    for key in ("pwm1", "pwm2"):
        dev = _MOTOR_REFS.get(key)
        if dev is not None:
            try:
                dev.value = 0.0
            except Exception:
                pass


atexit.register(emergency_stop_motors)


def _signal_handler(signum, frame):
    emergency_stop_motors()
    print(f"\n[SAFETY] Signal {signum} received - motors stopped.")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# =============================================================================
# STEREO CALIBRATION + RECTIFICATION   (logic from the latest distance.py)
# =============================================================================
def load_calibration(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration file not found: {path}")
    d = np.load(path)
    keys = set(d.keys())

    name_map_options = [
        {"K1": "K1", "D1": "D1", "K2": "K2", "D2": "D2"},
        {"K1": "mtxL", "D1": "distL", "K2": "mtxR", "D2": "distR"},
    ]
    resolved = None
    for mapping in name_map_options:
        if all(v in keys for v in mapping.values()):
            resolved = mapping
            break
    if resolved is None:
        raise KeyError(f"Could not find camera matrix keys. Available: {sorted(keys)}")

    calib = {
        "K1": d[resolved["K1"]], "D1": d[resolved["D1"]],
        "K2": d[resolved["K2"]], "D2": d[resolved["D2"]],
    }
    for k in ("R1", "R2", "P1", "P2", "Q"):
        if k not in keys:
            raise KeyError(f"'{k}' missing from {path}. Available: {sorted(keys)}")
        calib[k] = d[k]

    for k in ("map1L", "map2L", "map1R", "map2R",
              "image_width", "image_height", "baseline_mm"):
        if k in keys:
            calib[k] = d[k]
    return calib


def sanity_check_calibration(calib, half_w, half_h):
    """Refuse to run on a calibration that is geometrically impossible for the
    runtime image size - this is what produced the old cx=914-on-a-640px-image
    bug, and it silently poisons every distance downstream."""
    errors = []
    for name in ("P1", "P2"):
        cx, cy = calib[name][0, 2], calib[name][1, 2]
        if not (0 <= cx <= half_w):
            errors.append(f"{name} cx={cx:.1f} is outside 0-{half_w}. This "
                          f"calibration was computed for a different image width.")
        if not (0 <= cy <= half_h):
            errors.append(f"{name} cy={cy:.1f} is outside 0-{half_h}.")
    return errors


def build_rectify_maps(calib, size_wh):
    want_shape = (size_wh[1], size_wh[0])
    if all(k in calib for k in ("map1L", "map2L", "map1R", "map2R")):
        if (calib["map1L"].shape[:2] == want_shape and
                calib["map1R"].shape[:2] == want_shape):
            print("Using precomputed rectification maps from the calibration file.")
            return ((calib["map1L"], calib["map2L"]),
                    (calib["map1R"], calib["map2R"]))
        print("Precomputed maps are the wrong size for this capture - rebuilding.")

    map1 = cv2.initUndistortRectifyMap(
        calib["K1"], calib["D1"], calib["R1"], calib["P1"], size_wh, cv2.CV_16SC2)
    map2 = cv2.initUndistortRectifyMap(
        calib["K2"], calib["D2"], calib["R2"], calib["P2"], size_wh, cv2.CV_16SC2)
    return map1, map2


def compute_num_disparities(fx, baseline_mm, min_distance_mm, match_scale):
    max_disp_full = fx * baseline_mm / min_distance_mm
    return max(int(np.ceil((max_disp_full * match_scale) / 16.0) * 16), 16)


# =============================================================================
# DISTANCE FROM A BOX   (the exact algorithm from the latest distance.py,
#                        factored into a function so the same code serves both
#                        the detection box and the fallback centre ROI)
# =============================================================================
class DistanceEstimator:
    """Holds the temporal smoothing state (history + pending-jump confirmation)
    so the per-frame call site stays readable."""

    def __init__(self, fx, baseline_mm):
        self.fx = fx
        self.baseline_mm = baseline_mm
        self.history = collections.deque(maxlen=SMOOTH_WINDOW)
        self.pending_jump = None

    def reset(self):
        self.history.clear()
        self.pending_jump = None

    def measure(self, disp_full_equiv, box_small):
        """box_small = (x0, y0, x1, y1) in MATCHING-resolution pixels.
        Returns (distance_cm or None, valid_pct, median_disparity)."""
        x0, y0, x1, y1 = box_small
        if x1 <= x0 or y1 <= y0:
            return None, 0.0, 0.0

        roi_disp = disp_full_equiv[y0:y1, x0:x1]
        if roi_disp.size == 0:
            return None, 0.0, 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            z_mm_map = np.where(roi_disp > 0.5,
                                (self.fx * self.baseline_mm) / roi_disp, 0)

        valid_mask = ((roi_disp > 0.5) &
                      (z_mm_map >= MIN_DISTANCE_MM) &
                      (z_mm_map <= MAX_DISTANCE_MM))

        valid_pct = 100.0 * valid_mask.sum() / valid_mask.size
        if valid_pct < MIN_VALID_PIXEL_PCT:
            return None, valid_pct, 0.0

        median_disp = float(np.median(roi_disp[valid_mask]))
        raw_cm = (float(np.median(z_mm_map[valid_mask]))
                  * CALIBRATION_SCALE_CORRECTION) / 10.0

        # Jump rejection: a big step has to repeat before it is believed,
        # rather than being blindly trusted or blindly discarded.
        if self.history:
            current_median = float(np.median(self.history))
            if abs(raw_cm - current_median) > MAX_JUMP_CM:
                if (self.pending_jump is not None and
                        abs(self.pending_jump[0] - raw_cm) < 5):
                    self.pending_jump = (raw_cm, self.pending_jump[1] + 1)
                else:
                    self.pending_jump = (raw_cm, 1)
                if self.pending_jump[1] >= 2:
                    self.history.append(raw_cm)
                    self.pending_jump = None
            else:
                self.pending_jump = None
                self.history.append(raw_cm)
        else:
            self.history.append(raw_cm)

        if not self.history:
            return None, valid_pct, median_disp
        return float(np.median(self.history)), valid_pct, median_disp


# =============================================================================
# ROSE DETECTION (ONNX)
# =============================================================================
def letterbox(img, size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = size - new_h, size - new_w
    top, left = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, scale, left, top


def detect_roses(session, input_name, bgr_img, conf_threshold):
    """Returns [(x1, y1, x2, y2, conf), ...] in bgr_img pixel coordinates."""
    padded, scale, pad_x, pad_y = letterbox(bgr_img, MODEL_INPUT_SIZE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])

    out = session.run(None, {input_name: tensor})[0][0]   # (300, 6)

    h, w = bgr_img.shape[:2]
    boxes = []
    for row in out:
        x1, y1, x2, y2, conf, _cls = row
        if conf < conf_threshold:
            continue
        # normalised [0,1] over the 640x640 letterboxed input
        x1p, y1p = x1 * MODEL_INPUT_SIZE, y1 * MODEL_INPUT_SIZE
        x2p, y2p = x2 * MODEL_INPUT_SIZE, y2 * MODEL_INPUT_SIZE
        ox1 = float(np.clip((x1p - pad_x) / scale, 0, w - 1))
        oy1 = float(np.clip((y1p - pad_y) / scale, 0, h - 1))
        ox2 = float(np.clip((x2p - pad_x) / scale, 0, w - 1))
        oy2 = float(np.clip((y2p - pad_y) / scale, 0, h - 1))
        boxes.append((ox1, oy1, ox2, oy2, float(conf)))

    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes


# =============================================================================
# GREEN DETECTION BOX
# =============================================================================
class DetectionBox:
    """Keeps the green box big, smooth and on-target.

    Three jobs:
      * pad a tight YOLO box outward so SGBM has real texture to work with,
      * enforce a minimum size so a moving hand cannot slip out of the box
        between frames,
      * low-pass the box position so it tracks smoothly instead of jittering.
    """

    def __init__(self, frame_w, frame_h):
        self.frame_w, self.frame_h = frame_w, frame_h
        self.min_w = int(frame_w * BOX_MIN_W_FRAC)
        self.min_h = int(frame_h * BOX_MIN_H_FRAC)
        # fallback: the centred ROI_FRACTION box from distance.py
        fw, fh = int(frame_w * ROI_FRACTION), int(frame_h * ROI_FRACTION)
        self.fallback = (frame_w // 2 - fw // 2, frame_h // 2 - fh // 2,
                         frame_w // 2 + fw // 2, frame_h // 2 + fh // 2)
        self.current = self.fallback
        self.tracking = False

    def _shape(self, x1, y1, x2, y2):
        """Pad outward, enforce the minimum size, clamp inside the frame."""
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        w = max(w * (1 + 2 * BOX_PAD_FRAC), self.min_w)
        h = max(h * (1 + 2 * BOX_PAD_FRAC), self.min_h)
        w = min(w, self.frame_w)
        h = min(h, self.frame_h)

        # clamp the centre so the full-size box still fits on screen
        cx = min(max(cx, w / 2.0), self.frame_w - w / 2.0)
        cy = min(max(cy, h / 2.0), self.frame_h - h / 2.0)

        return (int(cx - w / 2), int(cy - h / 2),
                int(cx + w / 2), int(cy + h / 2))

    def update(self, detection):
        """detection = (x1,y1,x2,y2,conf) or None. Returns the box to use."""
        if detection is None:
            self.tracking = False
            # ease back to the centre fallback rather than snapping
            self.current = tuple(
                int(c + BOX_TRACK_SMOOTH * (f - c))
                for c, f in zip(self.current, self.fallback))
            return self.current

        target = self._shape(*detection[:4])
        if not self.tracking:
            self.current = target          # first lock-on: snap straight to it
        else:
            self.current = tuple(
                int(c + BOX_TRACK_SMOOTH * (t - c))
                for c, t in zip(self.current, target))
        self.tracking = True
        return self.current

    def to_match_scale(self, small_w, small_h):
        x1, y1, x2, y2 = self.current
        return (max(int(x1 * MATCH_SCALE), 0),
                max(int(y1 * MATCH_SCALE), 0),
                min(int(x2 * MATCH_SCALE), small_w),
                min(int(y2 * MATCH_SCALE), small_h))


def draw_detection_box(img, box, distance_cm, valid_pct, locked, conf=None):
    """Big, obvious green box with corner brackets and the distance printed
    right on the box itself."""
    x1, y1, x2, y2 = box
    color = (0, 255, 0) if distance_cm is not None else (0, 165, 255)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    # corner brackets make the box readable even over busy backgrounds
    L = max(24, (x2 - x1) // 8)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, 6)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, 6)

    tag = "ROSE LOCKED" if locked else "SEARCH AREA"
    if conf is not None:
        tag += f" {conf:.2f}"
    cv2.putText(img, tag, (x1 + 8, max(y1 - 10, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # distance INSIDE the box, with a filled backing plate so it stays legible
    if distance_cm is not None:
        text = f"{distance_cm:.1f} cm"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        tx, ty = x1 + 10, y2 - 14
        cv2.rectangle(img, (tx - 6, ty - th - 10), (tx + tw + 6, ty + 8),
                      (0, 0, 0), -1)
        cv2.putText(img, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    else:
        cv2.putText(img, f"NO DEPTH ({valid_pct:.0f}%)", (x1 + 10, y2 - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)


# =============================================================================
# HARDWARE CONTROL
# =============================================================================
class RobotControl:
    """PS5 controller -> motors + servos, with hold-to-run motor safety."""

    def __init__(self):
        # --- motors: constructed and IMMEDIATELY forced to zero ---
        self.PWM1 = PWMOutputDevice(PIN_PWM1)
        self.DIR1 = DigitalOutputDevice(PIN_DIR1)
        self.PWM2 = PWMOutputDevice(PIN_PWM2)
        self.DIR2 = DigitalOutputDevice(PIN_DIR2)
        self.PWM1.value = 0.0
        self.PWM2.value = 0.0
        _MOTOR_REFS["pwm1"] = self.PWM1      # register for the global e-stop
        _MOTOR_REFS["pwm2"] = self.PWM2
        print("[SAFETY] Motors initialised to 0.0 PWM (OFF).")

        # --- servos ---
        self.kit = ServoKit(channels=16)
        self.servo_angles = [90, 90, 90, 90, 90, 90]
        self.selected_servo = 0
        for i in range(6):
            self.kit.servo[i].angle = self.servo_angles[i]

        # --- controller ---
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No PS5 controller connected.")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print("Connected:", self.joystick.get_name())

        self.num_buttons = self.joystick.get_numbuttons()
        self.num_axes = self.joystick.get_numaxes()
        self.last_buttons = [0] * self.num_buttons

        self.axis_l2 = self._autodetect_l2()

        # --- state ---
        self.connected = True
        self.assist_mode = False
        self.camera_on = True
        self.motor_running = False
        self.throttle_held = False
        self.manual_active = False
        self.slow_mode = False
        self.last_good_poll = time.time()
        self.estop_latched = False

    # ------------------------------------------------------------------ setup
    def _autodetect_l2(self):
        """Triggers rest at -1.0; sticks rest near 0.0. Whichever of the
        candidate axes is resting at the floor is the L2 trigger."""
        if AXIS_L2 is not None:
            print(f"[MAP] L2 forced to axis {AXIS_L2} by config.")
            return AXIS_L2
        pygame.event.pump()
        time.sleep(0.1)
        pygame.event.pump()
        for candidate in (2, 4):
            if candidate >= self.num_axes or candidate == AXIS_RIGHT_Y:
                continue
            try:
                if self.joystick.get_axis(candidate) < -0.85:
                    print(f"[MAP] L2 auto-detected on axis {candidate} "
                          f"(resting at -1.0).")
                    return candidate
            except Exception:
                continue
        print("[MAP] L2 not auto-detected - slow mode disabled. "
              "Run 'python3 final_project.py --map' to find it.")
        return None

    def _trigger(self, axis):
        """Raw trigger axis (-1 released .. +1 fully pressed) -> 0.0 .. 1.0."""
        if axis is None or axis >= self.num_axes:
            return 0.0
        return float(np.clip((self.joystick.get_axis(axis) + 1.0) / 2.0, 0.0, 1.0))

    # ----------------------------------------------------------------- motors
    def set_motor(self, left_speed, right_speed):
        """Unchanged sign/direction convention from hand.py."""
        if left_speed >= 0:
            self.DIR1.off(); self.PWM1.value = min(left_speed, MAX_SPEED)
        else:
            self.DIR1.on();  self.PWM1.value = min(abs(left_speed), MAX_SPEED)

        if right_speed >= 0:
            self.DIR2.off(); self.PWM2.value = min(right_speed, MAX_SPEED)
        else:
            self.DIR2.on();  self.PWM2.value = min(abs(right_speed), MAX_SPEED)

        self.motor_running = (self.PWM1.value > 0.001 or self.PWM2.value > 0.001)

    def stop_motors(self):
        self.PWM1.value = 0.0
        self.PWM2.value = 0.0
        self.motor_running = False

    # ----------------------------------------------------------------- servos
    def set_servo(self, name_or_idx, angle):
        idx = (servo_names.index(name_or_idx)
               if isinstance(name_or_idx, str) else int(name_or_idx))
        angle = max(0, min(180, angle))
        self.servo_angles[idx] = angle
        self.kit.servo[idx].angle = angle

    def run_collect_pose(self):
        print("Assist: moving to the PRESET collect pose (tune COLLECT_POSE).")
        for name, angle in COLLECT_POSE.items():
            self.set_servo(name, angle)
            time.sleep(0.15)      # stagger so the servos do not all slam at once
        time.sleep(ARM_SETTLE_SECONDS)

    # ------------------------------------------------------------------- poll
    def poll(self):
        """One controller frame. ALWAYS ends with the motors either explicitly
        commanded or explicitly stopped - never left latched.

        Returns True if the poll succeeded, False if the controller is gone
        (in which case the motors have already been stopped)."""
        try:
            for event in pygame.event.get():
                if event.type == getattr(pygame, "JOYDEVICEREMOVED", -1):
                    raise RuntimeError("controller removed")
            pygame.event.pump()

            if not pygame.joystick.get_count():
                raise RuntimeError("no joystick present")

            # ---------------- buttons (edge-triggered toggles) ----------------
            buttons = [self.joystick.get_button(i) for i in range(self.num_buttons)]

            def pressed(idx):
                return (idx < len(buttons) and buttons[idx]
                        and not self.last_buttons[idx])

            for btn, servo in BTN_SERVO_MAP.items():      # verbatim from hand.py
                if pressed(btn):
                    self.selected_servo = servo
                    print(f"Selected servo: {servo_names[servo]}")

            if pressed(BTN_CAMERA_TOGGLE):
                self.camera_on = not self.camera_on
                print(f"Camera: {'ON' if self.camera_on else 'OFF'}")

            if pressed(BTN_ASSIST_TOGGLE):
                self.assist_mode = not self.assist_mode
                print(f"Assist: {'ARMED' if self.assist_mode else 'OFF'} "
                      f"(still requires R2 held to move)")

            if pressed(BTN_ESTOP):
                # PS button toggles the latch: first press latches, second clears.
                self.estop_latched = not self.estop_latched
                if self.estop_latched:
                    self.assist_mode = False
                    self.stop_motors()
                    print("[SAFETY] EMERGENCY STOP latched. Press PS again to clear.")
                else:
                    print("[SAFETY] E-stop cleared.")

            self.last_buttons = buttons

            # ------------------------- triggers -------------------------
            r2 = self._trigger(AXIS_R2)
            l2 = self._trigger(self.axis_l2)
            self.throttle_held = (r2 >= TRIGGER_PRESSED) and not self.estop_latched
            self.slow_mode = l2 >= TRIGGER_PRESSED

            # ------------------------- servos ---------------------------
            # Right stick only moves a servo while it is actually deflected,
            # so a servo can never "run away" on its own either.
            ry = -self.joystick.get_axis(AXIS_RIGHT_Y)
            if abs(ry) > DEADZONE and not self.estop_latched:
                idx = self.selected_servo
                self.set_servo(idx, self.servo_angles[idx] + ry * SERVO_SPEED)

            # ------------------------- drive ----------------------------
            # THE SAFETY CORE. Every path through this block writes a motor
            # command; there is no branch that leaves the previous PWM latched.
            if not self.throttle_held:
                self.stop_motors()
                self.manual_active = False
            else:
                y = -self.joystick.get_axis(AXIS_LEFT_Y)
                x = self.joystick.get_axis(AXIS_LEFT_X)
                if abs(y) < DEADZONE:
                    y = 0.0
                if abs(x) < DEADZONE:
                    x = 0.0

                self.manual_active = (y != 0.0 or x != 0.0)
                if self.manual_active:
                    scale = r2 * (SLOW_MODE_SCALE if self.slow_mode else 1.0)
                    left = max(-1, min(1, y + x)) * scale
                    right = max(-1, min(1, y - x)) * scale
                    self.set_motor(left, right)
                else:
                    # R2 held but the stick is centred -> hand control to assist
                    # if it is armed, otherwise sit still.
                    self.stop_motors()

            self.last_good_poll = time.time()
            self.connected = True
            return True

        except Exception as e:
            self.stop_motors()
            self.connected = False
            self.throttle_held = False
            self.manual_active = False
            print(f"[SAFETY] Controller poll failed ({e}) - motors stopped.")
            return False

    def watchdog(self):
        """Belt and braces: even if poll() somehow stops being called, a stale
        controller means no motion."""
        if time.time() - self.last_good_poll > CONTROLLER_TIMEOUT_S:
            self.stop_motors()
            self.connected = False

    def try_reconnect(self):
        """Non-blocking hot-plug recovery. Motors stay off until it succeeds."""
        try:
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.num_buttons = self.joystick.get_numbuttons()
                self.num_axes = self.joystick.get_numaxes()
                self.last_buttons = [0] * self.num_buttons
                self.connected = True
                self.last_good_poll = time.time()
                print("[INFO] Controller reconnected.")
                return True
        except Exception:
            pass
        return False

    def shutdown(self):
        self.stop_motors()
        emergency_stop_motors()
        try:
            pygame.quit()
        except Exception:
            pass
        print("[SAFETY] Shutdown complete - motors off.")


# =============================================================================
# CONTROLLER MAPPING DIAGNOSTIC  (python3 final_project.py --map)
# =============================================================================
def controller_map_mode():
    if not HARDWARE_AVAILABLE:
        print("pygame not available on this machine.")
        return
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No controller connected.")
        return
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"\nController: {js.get_name()}")
    print(f"  buttons: {js.get_numbuttons()}   axes: {js.get_numaxes()}")
    print("\nPress buttons / move sticks / squeeze triggers.")
    print("Note the index printed for R2, L2 and Options, then set the")
    print("AXIS_R2 / AXIS_L2 / BTN_CAMERA_TOGGLE constants to match.")
    print("Ctrl-C to exit.\n")
    last = [0] * js.get_numbuttons()
    rest = None
    try:
        while True:
            pygame.event.pump()
            axes = [round(js.get_axis(i), 2) for i in range(js.get_numaxes())]
            if rest is None:
                rest = list(axes)
                print(f"Resting axis values: {rest}")
                print("  (axes resting at -1.0 are triggers)\n")
            for i in range(js.get_numbuttons()):
                v = js.get_button(i)
                if v and not last[i]:
                    print(f"  BUTTON {i:>2} pressed")
                last[i] = v
            for i, (a, r) in enumerate(zip(axes, rest)):
                if abs(a - r) > 0.3:
                    print(f"  AXIS   {i:>2} = {a:+.2f}   (rest {r:+.2f})")
            time.sleep(0.12)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        pygame.quit()


# =============================================================================
# MAIN
# =============================================================================
def main():
    # ---------------- detector (optional) ----------------
    session = input_name = None
    if ONNX_AVAILABLE and os.path.exists(MODEL_FILE):
        print("Loading rose detector...")
        session = ort.InferenceSession(MODEL_FILE, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
    else:
        print(f"[INFO] Detector not loaded (model missing or onnxruntime absent). "
              f"Using the fixed {int(ROI_FRACTION*100)}% centre box.")

    # ---------------- calibration ----------------
    print("Loading stereo calibration...")
    calib = load_calibration(CALIB_FILE)
    HALF_W = int(calib["image_width"]) if "image_width" in calib else 640
    HALF_H = int(calib["image_height"]) if "image_height" in calib else 720
    print(f"Per-eye calibration image size: {HALF_W}x{HALF_H}")

    errors = sanity_check_calibration(calib, HALF_W, HALF_H)
    if errors:
        print("\n" + "=" * 70)
        print("CALIBRATION SANITY CHECK FAILED")
        for e in errors:
            print(" -", e)
        print("Refusing to run - every distance would be meaningless.")
        print("=" * 70)
        return

    fx = float(calib["P1"][0, 0])
    baseline_mm = float(-calib["P2"][0, 3] / calib["P2"][0, 0])
    print(f"fx = {fx:.2f} px   baseline = {baseline_mm:.3f} mm")

    (map1x, map1y), (map2x, map2y) = build_rectify_maps(calib, (HALF_W, HALF_H))
    num_disp = compute_num_disparities(fx, baseline_mm, MIN_DISTANCE_MM, MATCH_SCALE)
    print(f"numDisparities = {num_disp} (covers down to ~{MIN_DISTANCE_MM/10:.0f} cm)")

    block_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block_size,
        P1=8 * block_size ** 2, P2=32 * block_size ** 2,
        disp12MaxDiff=2, uniquenessRatio=15,
        speckleWindowSize=80, speckleRange=2, preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    if not PICAMERA_AVAILABLE:
        print("Picamera2 not available - cannot capture on this machine.")
        return

    # ---------------- cameras ----------------
    cam_info = Picamera2.global_camera_info()
    dual_camera_mode = len(cam_info) >= 2
    print(f"Camera mode: {'DUAL' if dual_camera_mode else 'SINGLE-SPLIT'} "
          f"({len(cam_info)} detected)")

    picam_left = picam_right = picam2 = None
    if dual_camera_mode:
        picam_left = Picamera2(camera_num=0)
        picam_right = Picamera2(camera_num=1)
        picam_left.configure(picam_left.create_video_configuration(
            main={"size": (HALF_W, HALF_H), "format": "RGB888"}))
        picam_right.configure(picam_right.create_video_configuration(
            main={"size": (HALF_W, HALF_H), "format": "RGB888"}))
        picam_left.start(); picam_right.start()
    else:
        CAPTURE_SIZE = (HALF_W * 2, HALF_H)
        picam2 = Picamera2()
        picam2.configure(picam2.create_video_configuration(
            main={"size": CAPTURE_SIZE, "format": "RGB888"}))
        picam2.start()
    time.sleep(1.0)     # let AE/AWB settle

    # ---------------- robot ----------------
    robot = None
    if HARDWARE_AVAILABLE:
        try:
            robot = RobotControl()
        except Exception as e:
            print(f"[WARN] Robot control unavailable ({e}). Vision-only mode.")

    # ---------------- loop state ----------------
    small_w, small_h = int(HALF_W * MATCH_SCALE), int(HALF_H * MATCH_SCALE)
    estimator = DistanceEstimator(fx, baseline_mm)
    box = DetectionBox(HALF_W, HALF_H)
    frame_center_x = HALF_W // 2

    camera_on = True                # mirrors robot.camera_on when a pad exists
    debug_view = False
    prev_time = time.time()
    fps = 0.0
    blank = np.zeros((HALF_H, HALF_W, 3), dtype=np.uint8)

    print("\n" + "=" * 70)
    print("RUNNING")
    print("  R2 (hold)      = motors ON  -- release = immediate stop")
    print("  Left stick     = direction / speed (while R2 held)")
    print("  L2 (hold)      = slow / precision mode")
    print("  OPTIONS        = camera feed ON / OFF")
    print("  CREATE/SHARE   = arm / disarm assist")
    print("  PS button      = emergency stop")
    print("  Square/Triangle/Circle/Cross/L1/R1 = select servo (from hand.py)")
    print("  Right stick Y  = move the selected servo")
    print("  Window keys: q quit | c camera | d debug | r reset | SPACE e-stop")
    print("=" * 70 + "\n")

    try:
        while True:
            # ---------- 1. CONTROLLER FIRST, ALWAYS ----------
            # Polled before any vision work so that a released trigger stops the
            # motors even if a frame takes a long time to process.
            if robot is not None:
                if not robot.poll():
                    robot.try_reconnect()
                robot.watchdog()
                camera_on = robot.camera_on

            # ---------- 2. CAMERA (skipped entirely when OFF) ----------
            distance_cm = None
            valid_pct = 0.0
            median_disp = 0.0
            best = None
            raw_disp = None

            if camera_on:
                if dual_camera_mode:
                    left_raw = picam_left.capture_array()
                    right_raw = picam_right.capture_array()
                else:
                    frame = picam2.capture_array()
                    left_raw = frame[:, :HALF_W]
                    right_raw = frame[:, HALF_W:]

                left_gray = cv2.cvtColor(left_raw, cv2.COLOR_RGB2GRAY)
                right_gray = cv2.cvtColor(right_raw, cv2.COLOR_RGB2GRAY)

                # Rectify at FULL calibrated resolution - this is the geometry
                # P1/P2/Q describe. Never resize this step.
                rect_left_gray = cv2.remap(left_gray, map1x, map1y, cv2.INTER_LINEAR)
                rect_right_gray = cv2.remap(right_gray, map2x, map2y, cv2.INTER_LINEAR)
                rect_left_bgr = cv2.cvtColor(
                    cv2.remap(left_raw, map1x, map1y, cv2.INTER_LINEAR),
                    cv2.COLOR_RGB2BGR)

                # ---- detection ----
                if session is not None:
                    dets = detect_roses(session, input_name, rect_left_bgr,
                                        CONF_THRESHOLD)
                    best = dets[0] if dets else None

                # ---- stereo: downscale ONLY for the expensive match ----
                small_left = cv2.resize(rect_left_gray, (small_w, small_h),
                                        interpolation=cv2.INTER_AREA)
                small_right = cv2.resize(rect_right_gray, (small_w, small_h),
                                         interpolation=cv2.INTER_AREA)
                raw_disp = stereo.compute(small_left, small_right).astype(np.float32) / 16.0

                # Rescale disparity VALUES (not the array) back to full-res
                # equivalent pixels so unscaled fx/baseline stay valid.
                disp_full_equiv = raw_disp / MATCH_SCALE

                # ---- green box -> distance ----
                box.update(best)
                distance_cm, valid_pct, median_disp = estimator.measure(
                    disp_full_equiv, box.to_match_scale(small_w, small_h))

                display = rect_left_bgr.copy()
                draw_detection_box(display, box.current, distance_cm, valid_pct,
                                   box.tracking,
                                   conf=best[4] if best is not None else None)
            else:
                # Camera OFF: no capture, no rectification, no SGBM, no
                # inference. The window stays alive so the controller and the
                # keys keep working, and so the toggle is reversible.
                estimator.reset()
                display = blank.copy()
                cv2.putText(display, "CAMERA OFF", (HALF_W // 2 - 150, HALF_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(display, "Press OPTIONS on the pad (or 'c') to turn on",
                            (HALF_W // 2 - 240, HALF_H // 2 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # ---------- 3. ASSIST (only while R2 is held) ----------
            if (robot is not None and robot.connected and robot.assist_mode
                    and robot.throttle_held and not robot.manual_active
                    and camera_on):
                if best is not None and distance_cm is not None:
                    cx = (best[0] + best[2]) / 2.0
                    offset = cx - frame_center_x
                    if distance_cm > APPROACH_TARGET_CM:
                        speed = DRIVE_SPEED * (SLOW_MODE_SCALE if robot.slow_mode else 1.0)
                        if abs(offset) > CENTER_DEADZONE_PX:
                            turn = TURN_GAIN * float(np.clip(offset / frame_center_x, -1, 1))
                            robot.set_motor(speed + turn, speed - turn)
                        else:
                            robot.set_motor(speed, speed)
                    else:
                        robot.stop_motors()
                        robot.run_collect_pose()
                        robot.assist_mode = False   # one-shot; re-arm manually
                else:
                    robot.stop_motors()

            # ---------- 4. FPS ----------
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # ---------- 5. STATUS OVERLAY ----------
            y = 30
            dist_txt = (f"DISTANCE: {distance_cm:.1f} cm" if distance_cm is not None
                        else ("NO VALID READING" if camera_on else "CAMERA OFF"))
            dist_col = (0, 255, 0) if distance_cm is not None else (0, 0, 255)
            cv2.putText(display, dist_txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, dist_col, 2); y += 30

            cv2.putText(display, f"CAMERA: {'ON' if camera_on else 'OFF'}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if camera_on else (0, 0, 255), 2); y += 25

            if robot is not None:
                if robot.estop_latched:
                    mstat, mcol = "E-STOP LATCHED", (0, 0, 255)
                elif robot.motor_running:
                    mstat, mcol = "MOTOR: ON", (0, 255, 0)
                else:
                    mstat, mcol = "MOTOR: OFF", (0, 0, 255)
                cv2.putText(display, mstat, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, mcol, 2); y += 25

                cstat = "PAD: CONNECTED" if robot.connected else "PAD: DISCONNECTED"
                ccol = (0, 255, 0) if robot.connected else (0, 0, 255)
                cv2.putText(display, cstat, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ccol, 2); y += 25

                flags = []
                if robot.throttle_held:
                    flags.append("R2")
                if robot.slow_mode:
                    flags.append("SLOW")
                if robot.assist_mode:
                    flags.append("ASSIST")
                cv2.putText(display,
                            f"SERVO: {servo_names[robot.selected_servo]}"
                            + (f"   [{' '.join(flags)}]" if flags else ""),
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 200, 0), 1); y += 25
            else:
                cv2.putText(display, "MOTOR: OFF (no controller)", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2); y += 25

            if camera_on:
                cv2.putText(display, f"valid {valid_pct:.0f}%  disp {median_disp:.1f}"
                                     f"  base {baseline_mm:.1f}mm  {fps:.1f} FPS",
                            (10, HALF_H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 1)

            cv2.imshow("Smart Pollination - Distance + Control", display)

            # ---------- 6. DEBUG WINDOWS ----------
            if debug_view and camera_on and raw_disp is not None:
                dv = cv2.normalize(np.clip(raw_disp, 0, None), None, 0, 255,
                                   cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow("Disparity", cv2.applyColorMap(dv, cv2.COLORMAP_JET))
                stacked = cv2.cvtColor(np.hstack([rect_left_gray, rect_right_gray]),
                                       cv2.COLOR_GRAY2BGR)
                for yy in range(0, stacked.shape[0], 40):
                    cv2.line(stacked, (0, yy), (stacked.shape[1], yy), (0, 255, 0), 1)
                cv2.imshow("Rectified L | R (epipolar check)", stacked)

            # ---------- 7. KEYS ----------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                camera_on = not camera_on
                if robot is not None:
                    robot.camera_on = camera_on
                print(f"Camera: {'ON' if camera_on else 'OFF'}")
            elif key == ord("d"):
                debug_view = not debug_view
                if not debug_view:
                    cv2.destroyWindow("Disparity")
                    cv2.destroyWindow("Rectified L | R (epipolar check)")
            elif key == ord("r"):
                estimator.reset()
                print("Smoothing buffer reset.")
            elif key == ord(" "):
                if robot is not None:
                    robot.estop_latched = True
                    robot.assist_mode = False
                    robot.stop_motors()
                    print("[SAFETY] Keyboard emergency stop.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        # Any unexpected failure must not leave the motors energised.
        emergency_stop_motors()
        print(f"\n[SAFETY] Unhandled error - motors stopped. Error: {e}")
        raise
    finally:
        if robot is not None:
            robot.shutdown()
        else:
            emergency_stop_motors()
        try:
            if dual_camera_mode:
                if picam_left: picam_left.stop()
                if picam_right: picam_right.stop()
            elif picam2:
                picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Clean exit.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--map", "-m"):
        controller_map_mode()
    else:
        main()
