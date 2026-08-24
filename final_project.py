#!/usr/bin/env python3
"""
final_project.py - Smart Pollination Platform: integrated prototype

Pipeline (matches Project_Proposal_Mahir_Ranawat.pdf's system architecture):

    stereo camera --> rose detection (YOLO26n ONNX) --> bounding box
                   --> stereo disparity on that ROI --> distance in cm
                   --> robot actuation (car steering + arm)

Combines three pieces that were built/verified separately:
  1. Calibrated stereo distance pipeline (distance.py) - rectification,
     disparity, median/robust distance, temporal smoothing.
  2. rose_new__1_.onnx - Ultralytics YOLO26n, single class "rose",
     input 1x3x640x640, output 1x300x6 = [x1,y1,x2,y2,conf,class] in
     NORMALIZED [0,1] coordinates, NMS already applied (end2end export).
     (Verified directly against the model file before writing this code.)
  3. hand.py - PS5 controller car + 6-servo arm control (Cytron motor
     driver + PCA9685 via ServoKit). Manual control is preserved exactly;
     this file adds an optional assist mode on top of it.

HONEST LIMITATION: the "arm reach" step below moves the arm to a FIXED,
PRESET pose once the robot is close to a detected rose. It is NOT inverse
kinematics - that would require your arm's link lengths/geometry, which
this project doesn't have calibrated yet. Tune COLLECT_POSE by hand on
the real robot before relying on it. Treat autonomous mode as "drive up
to and center on the flower", with the final reach/collect step as a
rough starting point to refine during your Phase 4 testing.

Controls:
    Left stick        - manual drive (always overrides assist steering
                         the instant you move it)
    Square/Triangle/
    Circle/Cross/L1/R1 - select servo (Base/Shoulder/Elbow/Wrist/
                         Gripper/Camera) - unchanged from hand.py
    Right stick Y      - move selected servo - unchanged from hand.py
    OPTIONS (button 9) - toggle Assist Mode on/off (adjust index if your
                         controller maps differently - print button
                         indices with test_controller.py if unsure)
    Q on the video window - quit
"""

import os
import time
import collections

import cv2
import numpy as np
import onnxruntime as ort

# ---- hardware control imports (from hand.py). Wrapped so this script can
# also run on a dev machine with just a webcam for vision-only testing. ----
try:
    import pygame
    from gpiozero import PWMOutputDevice, DigitalOutputDevice
    from adafruit_servokit import ServoKit
    HARDWARE_AVAILABLE = True
except ImportError as e:
    print(f"[INFO] Hardware control libraries not available ({e}). "
          f"Running in VISION-ONLY mode (detection + distance, no motors/servos).")
    HARDWARE_AVAILABLE = False

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


# =============================================================================
# CONFIG
# =============================================================================
CALIB_FILE = "/home/pi/Desktop/MahirCode/calibration_images/stereo_calibration.npz"
MODEL_FILE = "/home/pi/Desktop/MahirCode/calibration_images/rose_new (1).onnx"

MATCH_SCALE = 0.5                    # downscale factor for stereo matching (speed)
MIN_DISTANCE_MM = 150                # near limit for numDisparities sizing (15cm)
MAX_DISTANCE_MM = 3000
MIN_VALID_PIXEL_PCT = 35.0
SMOOTH_WINDOW = 7
MAX_JUMP_CM = 30

CONF_THRESHOLD = 0.45                # rose detector confidence cutoff
MODEL_INPUT_SIZE = 640               # from the ONNX model's input shape

# Assist-mode behavior
APPROACH_TARGET_CM = 18.0            # stop driving forward once this close
CENTER_DEADZONE_PX = 40              # how far off-center (in rectified-left px)
                                      # before steering correction kicks in
DRIVE_SPEED = 0.28                   # capped low for a slow, controllable approach
TURN_GAIN = 0.55                     # steering aggressiveness while centering
ARM_SETTLE_SECONDS = 1.5             # pause after reaching target before running
                                      # the preset collection pose

# --- FIXED PRESET POSE for the final "reach" step. TUNE THIS ON YOUR ROBOT. ---
# Order matches servo_names below. This is a placeholder starting pose, not
# a calculated one - move the arm by hand/joystick to a pose that reaches a
# flower held at APPROACH_TARGET_CM in front of the robot, read the angles
# off `servo_angles` (printed live), and put those numbers here.
COLLECT_POSE = {
    "Base": 90,
    "Shoulder": 60,
    "Elbow": 120,
    "Wrist": 90,
    "Gripper": 40,   # closed-ish; adjust to your gripper's closed angle
}

# --- hand.py hardware config, unchanged ---
MAX_SPEED = 0.5
DEADZONE = 0.15
SERVO_SPEED = 2.5
servo_names = ["Base", "Shoulder", "Elbow", "Wrist", "Gripper", "Camera"]
ASSIST_TOGGLE_BUTTON = 9   # "Options" on most PS5-via-pygame mappings; check with
                           # a small test script (print button indices) if this
                           # doesn't toggle assist mode on your controller.


# =============================================================================
# STEREO CALIBRATION + RECTIFICATION  (same logic as distance.py)
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

    for k in ("map1L", "map2L", "map1R", "map2R", "image_width", "image_height", "baseline_mm"):
        if k in keys:
            calib[k] = d[k]
    return calib


def sanity_check_calibration(calib, half_w, half_h):
    errors = []
    for name in ("P1", "P2"):
        cx, cy = calib[name][0, 2], calib[name][1, 2]
        if not (0 <= cx <= half_w):
            errors.append(f"{name} cx={cx:.1f} outside 0-{half_w}.")
        if not (0 <= cy <= half_h):
            errors.append(f"{name} cy={cy:.1f} outside 0-{half_h}.")
    return errors


def build_rectify_maps(calib, size_wh):
    want_shape = (size_wh[1], size_wh[0])
    have_precomputed = all(k in calib for k in ("map1L", "map2L", "map1R", "map2R"))
    if have_precomputed:
        m1x = calib["map1L"]
        m2x = calib["map1R"]
        if m1x.shape[:2] == want_shape and m2x.shape[:2] == want_shape:
            return (calib["map1L"], calib["map2L"]), (calib["map1R"], calib["map2R"])

    map1 = cv2.initUndistortRectifyMap(
        calib["K1"], calib["D1"], calib["R1"], calib["P1"], size_wh, cv2.CV_16SC2
    )
    map2 = cv2.initUndistortRectifyMap(
        calib["K2"], calib["D2"], calib["R2"], calib["P2"], size_wh, cv2.CV_16SC2
    )
    return map1, map2


def compute_num_disparities(fx, baseline_mm, min_distance_mm, match_scale):
    max_disp_full = fx * baseline_mm / min_distance_mm
    max_disp_scaled = max_disp_full * match_scale
    return max(int(np.ceil(max_disp_scaled / 16.0) * 16), 16)


# =============================================================================
# ROSE DETECTION (ONNX)
# =============================================================================
def letterbox(img, size=640, color=(114, 114, 114)):
    """Resize keeping aspect ratio + pad to a square. Returns the padded image
    plus the scale factor and padding offsets needed to map boxes back."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = size - new_h, size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
    return padded, scale, left, top


def preprocess_for_yolo(bgr_img, size=640):
    padded, scale, pad_x, pad_y = letterbox(bgr_img, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))[None, ...]  # 1x3xHxW
    return np.ascontiguousarray(chw), scale, pad_x, pad_y


def detect_roses(session, input_name, bgr_img, conf_threshold):
    """Returns a list of (x1,y1,x2,y2,conf) in bgr_img's own pixel coordinates."""
    tensor, scale, pad_x, pad_y = preprocess_for_yolo(bgr_img, MODEL_INPUT_SIZE)
    out = session.run(None, {input_name: tensor})[0][0]  # (300, 6)

    boxes = []
    for row in out:
        x1, y1, x2, y2, conf, cls = row
        if conf < conf_threshold:
            continue
        # coords are normalized [0,1] over the 640x640 letterboxed input
        x1p, y1p, x2p, y2p = (x1 * MODEL_INPUT_SIZE, y1 * MODEL_INPUT_SIZE,
                               x2 * MODEL_INPUT_SIZE, y2 * MODEL_INPUT_SIZE)
        # undo letterbox padding + scale to get back to bgr_img coordinates
        ox1 = (x1p - pad_x) / scale
        oy1 = (y1p - pad_y) / scale
        ox2 = (x2p - pad_x) / scale
        oy2 = (y2p - pad_y) / scale
        h, w = bgr_img.shape[:2]
        ox1, ox2 = np.clip([ox1, ox2], 0, w - 1)
        oy1, oy2 = np.clip([oy1, oy2], 0, h - 1)
        boxes.append((float(ox1), float(oy1), float(ox2), float(oy2), float(conf)))

    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes


# =============================================================================
# HARDWARE CONTROL (from hand.py, preserved + extended)
# =============================================================================
class RobotControl:
    def __init__(self):
        self.PWM1 = PWMOutputDevice(13)
        self.DIR1 = DigitalOutputDevice(16)
        self.PWM2 = PWMOutputDevice(12)
        self.DIR2 = DigitalOutputDevice(20)

        self.kit = ServoKit(channels=16)
        self.servo_angles = [90, 90, 90, 90, 90, 90]
        self.selected_servo = 0
        for i in range(6):
            self.kit.servo[i].angle = self.servo_angles[i]

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No PS5 controller connected.")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print("Connected:", self.joystick.get_name())

        self.last_buttons = [0] * self.joystick.get_numbuttons()
        self.assist_mode = False

        self.mapping = {
            3: 0, 2: 1, 1: 2, 0: 3, 4: 4, 5: 5,
        }

    def set_motor(self, left_speed, right_speed):
        if left_speed >= 0:
            self.DIR1.off(); self.PWM1.value = min(left_speed, MAX_SPEED)
        else:
            self.DIR1.on(); self.PWM1.value = min(abs(left_speed), MAX_SPEED)
        if right_speed >= 0:
            self.DIR2.off(); self.PWM2.value = min(right_speed, MAX_SPEED)
        else:
            self.DIR2.on(); self.PWM2.value = min(abs(right_speed), MAX_SPEED)

    def stop_motors(self):
        self.PWM1.value = 0
        self.PWM2.value = 0

    def set_servo(self, name, angle):
        angle = max(0, min(180, angle))
        idx = servo_names.index(name)
        self.servo_angles[idx] = angle
        self.kit.servo[idx].angle = angle

    def poll_and_handle_manual(self):
        """Runs the exact hand.py manual-control logic for one frame.
        Returns (manual_x, manual_y) stick input so the caller can decide
        whether assist steering should be overridden this frame."""
        pygame.event.pump()

        y = -self.joystick.get_axis(1)
        x = self.joystick.get_axis(0)
        raw_x, raw_y = x, y
        if abs(y) < DEADZONE:
            y = 0
        if abs(x) < DEADZONE:
            x = 0

        stick_active = (abs(raw_x) >= DEADZONE) or (abs(raw_y) >= DEADZONE)

        if stick_active:
            left = max(-1, min(1, y + x))
            right = max(-1, min(1, y - x))
            self.set_motor(left, right)

        buttons = [self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())]

        for btn, servo in self.mapping.items():
            if btn < len(buttons) and buttons[btn] and not self.last_buttons[btn]:
                self.selected_servo = servo
                print(f"Selected: {servo_names[self.selected_servo]}")

        if (ASSIST_TOGGLE_BUTTON < len(buttons) and buttons[ASSIST_TOGGLE_BUTTON]
                and not self.last_buttons[ASSIST_TOGGLE_BUTTON]):
            self.assist_mode = not self.assist_mode
            print(f"Assist mode: {'ON' if self.assist_mode else 'OFF'}")
            if not self.assist_mode:
                self.stop_motors()

        self.last_buttons = buttons

        ry = -self.joystick.get_axis(3)
        if abs(ry) > DEADZONE:
            name = servo_names[self.selected_servo]
            new_angle = self.servo_angles[self.selected_servo] + ry * SERVO_SPEED
            self.set_servo(name, new_angle)

        return stick_active

    def run_collect_pose(self):
        print("Assist: reaching preset collection pose (TUNE THIS on your robot)...")
        for name, angle in COLLECT_POSE.items():
            self.set_servo(name, angle)
            time.sleep(0.15)  # small stagger so servos don't all slam at once
        time.sleep(ARM_SETTLE_SECONDS)
        print("Assist: collection pose reached. Toggle assist off/on to retry.")

    def shutdown(self):
        self.stop_motors()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("Loading rose detector...")
    session = ort.InferenceSession(MODEL_FILE, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print("Loading stereo calibration...")
    calib = load_calibration(CALIB_FILE)
    HALF_W = int(calib["image_width"]) if "image_width" in calib else 640
    HALF_H = int(calib["image_height"]) if "image_height" in calib else 720
    print(f"Per-eye calibration image size: {HALF_W}x{HALF_H}")

    errors = sanity_check_calibration(calib, HALF_W, HALF_H)
    if errors:
        print("CALIBRATION SANITY CHECK FAILED:")
        for e in errors:
            print(" -", e)
        return

    fx = calib["P1"][0, 0]
    baseline_mm = -calib["P2"][0, 3] / calib["P2"][0, 0]
    print(f"fx={fx:.2f}px  baseline={baseline_mm:.3f}mm")

    (map1x, map1y), (map2x, map2y) = build_rectify_maps(calib, (HALF_W, HALF_H))
    num_disp = compute_num_disparities(fx, baseline_mm, MIN_DISTANCE_MM, MATCH_SCALE)
    print(f"numDisparities={num_disp} (covers down to ~{MIN_DISTANCE_MM/10:.0f}cm)")

    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=5,
        P1=8 * 25, P2=32 * 25, disp12MaxDiff=2, uniquenessRatio=15,
        speckleWindowSize=80, speckleRange=2, preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    if not PICAMERA_AVAILABLE:
        print("Picamera2 not available - cannot run on this machine without a "
              "camera. (Vision logic above is ready for the Pi.)")
        return

    cam_info = Picamera2.global_camera_info()
    dual_camera_mode = len(cam_info) >= 2
    print(f"Camera mode: {'DUAL' if dual_camera_mode else 'SINGLE-SPLIT'}")

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
    time.sleep(1.0)

    robot = None
    if HARDWARE_AVAILABLE:
        try:
            robot = RobotControl()
        except Exception as e:
            print(f"[WARN] Could not initialize robot control ({e}). "
                  f"Continuing in vision-only mode.")

    small_w, small_h = int(HALF_W * MATCH_SCALE), int(HALF_H * MATCH_SCALE)
    history = collections.deque(maxlen=SMOOTH_WINDOW)
    pending_jump = None
    prev_time = time.time()
    fps = 0.0
    frame_center_x = HALF_W // 2

    print("\nRunning. Press 'q' in the video window to quit.\n")

    try:
        while True:
            # ---- capture ----
            if dual_camera_mode:
                left_raw = picam_left.capture_array()
                right_raw = picam_right.capture_array()
            else:
                frame = picam2.capture_array()
                left_raw = frame[:, :HALF_W]
                right_raw = frame[:, HALF_W:]

            left_gray = cv2.cvtColor(left_raw, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_raw, cv2.COLOR_RGB2GRAY)
            rect_left_gray = cv2.remap(left_gray, map1x, map1y, cv2.INTER_LINEAR)
            rect_right_gray = cv2.remap(right_gray, map2x, map2y, cv2.INTER_LINEAR)
            # color rectified left, for detection + display (same maps apply to any channel count)
            rect_left_color = cv2.remap(left_raw, map1x, map1y, cv2.INTER_LINEAR)
            rect_left_bgr = cv2.cvtColor(rect_left_color, cv2.COLOR_RGB2BGR)

            # ---- detection ----
            detections = detect_roses(session, input_name, rect_left_bgr, CONF_THRESHOLD)
            best = detections[0] if detections else None

            # ---- stereo matching (once per frame, full rectified pair) ----
            small_left = cv2.resize(rect_left_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
            small_right = cv2.resize(rect_right_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
            raw_disp = stereo.compute(small_left, small_right).astype(np.float32) / 16.0
            disp_full_equiv = raw_disp / MATCH_SCALE

            distance_cm = None
            valid_pct = 0.0

            if best is not None:
                x1, y1, x2, y2, conf = best
                # map detection bbox (full-res rectified-left coords) into the
                # small matching-resolution array for indexing
                sx0 = max(int(x1 * MATCH_SCALE), 0)
                sy0 = max(int(y1 * MATCH_SCALE), 0)
                sx1 = min(int(x2 * MATCH_SCALE), small_w)
                sy1 = min(int(y2 * MATCH_SCALE), small_h)

                if sx1 > sx0 and sy1 > sy0:
                    roi_disp = disp_full_equiv[sy0:sy1, sx0:sx1]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        z_mm_map = np.where(roi_disp > 0.5, (fx * baseline_mm) / roi_disp, 0)
                    valid_mask = (roi_disp > 0.5) & (z_mm_map >= MIN_DISTANCE_MM) & (z_mm_map <= MAX_DISTANCE_MM)
                    valid_pct = 100.0 * valid_mask.sum() / valid_mask.size if valid_mask.size else 0.0

                    if valid_pct >= MIN_VALID_PIXEL_PCT:
                        raw_distance_cm = float(np.median(z_mm_map[valid_mask])) / 10.0
                        if history:
                            current_median = float(np.median(history))
                            if abs(raw_distance_cm - current_median) > MAX_JUMP_CM:
                                if pending_jump is not None and abs(pending_jump[0] - raw_distance_cm) < 5:
                                    pending_jump = (raw_distance_cm, pending_jump[1] + 1)
                                else:
                                    pending_jump = (raw_distance_cm, 1)
                                if pending_jump[1] >= 2:
                                    history.append(raw_distance_cm)
                                    pending_jump = None
                            else:
                                pending_jump = None
                                history.append(raw_distance_cm)
                        else:
                            history.append(raw_distance_cm)
                        if history:
                            distance_cm = float(np.median(history))
                    else:
                        history.clear()
            else:
                history.clear()

            # ---- robot control ----
            if robot is not None:
                stick_active = robot.poll_and_handle_manual()

                if robot.assist_mode and not stick_active:
                    if best is not None and distance_cm is not None:
                        cx = (best[0] + best[2]) / 2.0
                        offset = cx - frame_center_x
                        if distance_cm > APPROACH_TARGET_CM:
                            if abs(offset) > CENTER_DEADZONE_PX:
                                turn = TURN_GAIN * np.clip(offset / frame_center_x, -1, 1)
                                robot.set_motor(DRIVE_SPEED + turn, DRIVE_SPEED - turn)
                            else:
                                robot.set_motor(DRIVE_SPEED, DRIVE_SPEED)
                        else:
                            robot.stop_motors()
                            robot.run_collect_pose()
                    else:
                        robot.stop_motors()

            # ---- FPS ----
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = (0.9 * fps + 0.1 * (1.0 / dt)) if dt > 0 else fps

            # ---- display ----
            display = rect_left_bgr.copy()
            if best is not None:
                x1, y1, x2, y2, conf = best
                cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(display, f"rose {conf:.2f}", (int(x1), max(int(y1) - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            label = f"Distance: {distance_cm:.1f} cm" if distance_cm is not None else (
                "NO VALID OBJECT" if best is not None else "NO ROSE DETECTED")
            color = (0, 255, 0) if distance_cm is not None else (0, 0, 255)
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, f"Valid: {valid_pct:.1f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            if robot is not None:
                mode_txt = f"Assist: {'ON' if robot.assist_mode else 'OFF'}  Servo: {servo_names[robot.selected_servo]}"
                cv2.putText(display, mode_txt, (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

            cv2.imshow("Smart Pollination - Rose Detection + Distance", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        if robot is not None:
            robot.shutdown()
        if dual_camera_mode:
            picam_left.stop(); picam_right.stop()
        else:
            picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
