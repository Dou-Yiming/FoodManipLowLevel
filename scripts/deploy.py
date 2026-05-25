#!/usr/bin/env python3
"""
Deploy a trained policy (ACT / Diffusion) on YAM bimanual arms.

Reads camera images + follower joint states, runs the policy, and sends
predicted actions to the follower arms. Leader arm top buttons act as
e-stop: press either one to freeze the robot immediately.

Usage:
    # From local checkpoint
    python scripts/deploy.py --policy outputs/act_blockincup/checkpoints/last/pretrained_model

    # From HuggingFace Hub
    python scripts/deploy.py --policy Jefferzn/act_blockincup

    # With live camera view
    python scripts/deploy.py --policy Jefferzn/act_blockincup --visualize
"""

import argparse
import contextlib
import glob
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as transforms_v2

# Create Qt font directory to suppress QFontDatabase warnings
os.makedirs(os.path.join(os.path.dirname(cv2.__file__), "qt", "fonts"), exist_ok=True)


@contextlib.contextmanager
def _quiet_stderr():
    """Suppress C-level stderr (Qt/OpenCV noise)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import GripperType
from i2rt.utils.utils import override_log_level

from resolve_leader_can import ArmInfo, ensure_can_up, resolve_arms

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

PORT_LEFT = 11333
PORT_RIGHT = 11334
NUM_ARM_JOINTS = 6
NUM_JOINTS = 7  # 6 arm + 1 gripper
# Optional: blend policy -> current pose over N steps (0 = off). Large ramps can look
# "frozen" when pred ≈ state because each step only moves (pred-state)/N.
DEFAULT_RAMP_STEPS = 0

_all_robots: List[MotorChainRobot] = []


# ── Camera reader ─────────────────────────────────────────────────────────

class CameraReader:
    """Continuously reads frames from a camera in a background thread."""

    def __init__(self, device: str, width: int = 640, height: int = 360) -> None:
        self.device = device
        self.index = int(device.replace("/dev/video", ""))
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name=f"cam_{self.index}"
        )

    def start(self) -> None:
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()


def detect_cameras() -> List[CameraReader]:
    """Auto-detect cameras using stable USB path symlinks.

    Uses /dev/v4l/by-path/ so that camera order is determined by
    physical USB port, not by boot/plug order. Falls back to
    /dev/videoN if by-path is not available.
    """
    by_path = sorted(glob.glob("/dev/v4l/by-path/*-video-index0"))
    if by_path:
        # Resolve symlinks to /dev/videoN, keep sorted by USB path
        paths = [os.path.realpath(p) for p in by_path]
        logging.info(f"Using stable USB paths: {dict(zip(by_path, paths))}")
    else:
        paths = sorted(
            (p for p in glob.glob("/dev/video*") if p[len("/dev/video"):].isdigit()),
            key=lambda p: int(p[len("/dev/video"):]),
        )

    readers = []
    with _quiet_stderr():
        for path in paths:
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                ret, _ = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if ret and w > 0 and h > 0:
                    reader = CameraReader(path)
                    reader.start()
                    readers.append(reader)
                    logging.info(f"Camera {path}: {reader.width}x{reader.height}")
            else:
                cap.release()
    if not readers:
        logging.warning("No cameras detected")
    return readers


# ── Robot helpers ─────────────────────────────────────────────────────────

class ServerRobot:
    def __init__(self, robot: MotorChainRobot, port: int) -> None:
        import portal
        self._robot = robot
        self._server = portal.Server(port)
        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)
        self._server.bind(
            "is_motor_chain_running", lambda: self._robot.motor_chain.running
        )

    def serve(self) -> None:
        self._server.start()


class ClientRobot:
    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        import portal
        self._client = portal.Client(f"{host}:{port}")

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)

    def is_motor_chain_running(self) -> bool:
        return bool(self._client.is_motor_chain_running().result())


def load_home_poses(
    path: Optional[str],
    follower_keys: List[str],
) -> Dict[str, np.ndarray]:
    """Load per-follower home joint poses from YAML/JSON."""
    default_path = Path(__file__).resolve().parent.parent / "config" / "deploy_home.yaml"
    pose_path = Path(path) if path else default_path
    if not pose_path.exists():
        raise FileNotFoundError(f"Home pose file not found: {pose_path}")

    raw: Union[dict, list]
    if pose_path.suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(pose_path.read_text())
    else:
        raw = json.loads(pose_path.read_text())

    poses: Dict[str, np.ndarray] = {}
    if isinstance(raw, dict) and all(isinstance(v, list) for v in raw.values()):
        for key in follower_keys:
            if key not in raw:
                raise KeyError(f"Home pose file missing key '{key}' (have {list(raw.keys())})")
            arr = np.asarray(raw[key], dtype=np.float32)
            if arr.shape != (NUM_JOINTS,):
                raise ValueError(f"{key}: expected {NUM_JOINTS} joints, got shape {arr.shape}")
            poses[key] = arr
    elif isinstance(raw, list):
        if len(raw) != len(follower_keys) * NUM_JOINTS:
            raise ValueError(
                f"Flat home pose must have {len(follower_keys) * NUM_JOINTS} values, got {len(raw)}"
            )
        flat = np.asarray(raw, dtype=np.float32)
        for i, key in enumerate(follower_keys):
            poses[key] = flat[i * NUM_JOINTS:(i + 1) * NUM_JOINTS]
    else:
        raise ValueError(f"Unsupported home pose format in {pose_path}")
    return poses


def move_followers_to_pose(
    clients: Dict[str, ClientRobot],
    follower_pairs: List[Tuple[str, int]],
    target_by_key: Dict[str, np.ndarray],
    duration: float,
    hz: float = 50.0,
) -> None:
    """Smoothly command followers from current pose to target."""
    starts = {f_key: clients[f_key].get_joint_pos() for f_key, _ in follower_pairs}
    steps = max(int(duration * hz), 1)
    for i in range(1, steps + 1):
        alpha = i / steps
        for f_key, _ in follower_pairs:
            cmd = (1.0 - alpha) * starts[f_key] + alpha * target_by_key[f_key]
            clients[f_key].command_joint_pos(cmd.astype(np.float32))
        time.sleep(1.0 / hz)


class YAMLeaderRobot:
    """Leader arm wrapper — only used for e-stop button reading."""

    def __init__(self, robot: MotorChainRobot) -> None:
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_buttons(self) -> list:
        """Read teaching handle button states. Returns [top_button, bottom_button]."""
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        return encoder_obs[0].io_inputs


def start_follower_server(arm_info: ArmInfo, port: int, label: str) -> threading.Thread:
    gripper_type = GripperType.from_string_name(arm_info.gripper_type)
    logging.info(f"[{label}] Creating follower on {arm_info.channel}")
    robot = get_yam_robot(
        channel=arm_info.channel,
        gripper_type=gripper_type,
        zero_gravity_mode=False,
    )
    _all_robots.append(robot)
    server = ServerRobot(robot, port)
    thread = threading.Thread(target=server.serve, name=f"follower_{label}", daemon=True)
    thread.start()
    return thread


# ── Button monitor thread ─────────────────────────────────────────────────

def button_monitor(
    leaders: List[YAMLeaderRobot],
    stop_event: threading.Event,
    running_event: threading.Event,
) -> None:
    """Poll leader arm buttons. Top button toggles between running and paused.

    Starts in paused state. First press = start running. Second press = e-stop (pause).
    Third press = resume. And so on.
    """
    while not stop_event.is_set():
        for leader in leaders:
            try:
                buttons = leader.get_buttons()
                if buttons[0] > 0.5:
                    if not running_event.is_set():
                        running_event.set()
                        logging.info("START — policy running")
                    else:
                        running_event.clear()
                        logging.warning("E-STOP — robot paused (press top button to resume)")
                    # Wait for button release
                    while not stop_event.is_set():
                        time.sleep(0.03)
                        buttons = leader.get_buttons()
                        if buttons[0] < 0.5:
                            break
            except Exception:
                pass
        time.sleep(0.05)


def blend_actions(
    current: np.ndarray, target: np.ndarray, step: int, ramp_steps: int
) -> np.ndarray:
    """Linearly ramp from current joint positions to the policy target."""
    if ramp_steps <= 0 or step >= ramp_steps:
        return target
    # (step + 1) so step 0 still moves; step ramp_steps-1 reaches full target.
    alpha = min(1.0, (step + 1) / ramp_steps)
    return current + alpha * (target - current)


def _fmt_joints(arr: np.ndarray, precision: int = 3) -> str:
    return np.array2string(
        arr, precision=precision, suppress_small=True, separator=", "
    )


def configure_act_action_steps(policy: ACTPolicy, n_action_steps: int) -> None:
    """Override ACT open-loop chunk length for closed-loop deploy."""
    chunk_size = policy.config.chunk_size
    if n_action_steps > chunk_size:
        logging.warning(
            f"n_action_steps={n_action_steps} > chunk_size={chunk_size}; clamping to {chunk_size}"
        )
        n_action_steps = chunk_size
    policy.config.n_action_steps = n_action_steps
    policy.reset()


def log_prediction_step(
    step: int,
    infer_ms: float,
    state: np.ndarray,
    pred: np.ndarray,
    cmd: np.ndarray,
    follower_labels: List[str],
    ramp_steps: int,
    cameras_ok: List[bool],
    chunk_replan: bool = False,
    action_queue_left: int = 0,
) -> None:
    """Print state, network prediction, and command sent to the robot."""
    parts = []
    for i, label in enumerate(follower_labels):
        sl = slice(i * NUM_JOINTS, (i + 1) * NUM_JOINTS)
        delta = cmd[sl] - state[sl]
        parts.append(
            f"{label}: state={_fmt_joints(state[sl])} "
            f"pred={_fmt_joints(pred[sl])} "
            f"cmd={_fmt_joints(cmd[sl])} "
            f"|delta|_max={np.max(np.abs(delta)):.3f}"
        )
    ramp_note = (
        f" (ramp {step}/{ramp_steps})" if ramp_steps > 0 and step < ramp_steps else ""
    )
    cam_note = f" cameras_ok={cameras_ok}"
    chunk_note = (
        f" chunk_replan={'yes' if chunk_replan else 'no'}"
        f" queue_left={action_queue_left}"
    )
    logging.info(
        f"step {step} infer={infer_ms:.0f}ms{ramp_note}{cam_note}{chunk_note}\n  "
        + "\n  ".join(parts)
    )


# ── Visualisation helper ──────────────────────────────────────────────────

def tile_frames(
    frames: List[Optional[np.ndarray]],
    labels: List[str],
    target_w: int,
    target_h: int,
) -> np.ndarray:
    n = len(frames)
    if n == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    cell_w = target_w
    cell_h = target_h // n
    canvas = np.zeros((n * cell_h, cell_w, 3), dtype=np.uint8)
    for i, (frame, label) in enumerate(zip(frames, labels)):
        y0 = i * cell_h
        cell = cv2.resize(frame, (cell_w, cell_h)) if frame is not None else \
               np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        canvas[y0:y0 + cell_h] = cell
    return canvas


# ── Policy image sizing ───────────────────────────────────────────────────

def get_policy_image_sizes(policy) -> Dict[str, Tuple[int, int]]:
    """Return {feature_key: (height, width)} from the policy config."""
    sizes: Dict[str, Tuple[int, int]] = {}
    for key, feat in policy.config.input_features.items():
        if "images" not in key:
            continue
        shape = feat.shape
        # PolicyFeature shape is (C, H, W)
        if len(shape) == 3 and shape[0] == 3:
            sizes[key] = (int(shape[1]), int(shape[2]))
        elif len(shape) == 3:
            sizes[key] = (int(shape[0]), int(shape[1]))
    return sizes


def detect_training_resize(policy_path: str) -> Optional[Tuple[int, int]]:
    """Read train_config.json to find if an explicit resize transform was used."""
    candidates = [
        Path(policy_path) / "train_config.json",
        Path(policy_path).parent / "train_config.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                cfg = json.loads(p.read_text())
                resize = cfg.get("dataset", {}).get("image_transforms", {}).get("resize")
                if resize and len(resize) == 2:
                    return tuple(resize)
            except (json.JSONDecodeError, KeyError):
                pass
    return None


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy trained policy on YAM arms.")
    parser.add_argument("--policy", required=True,
                        help="Path to pretrained model dir or HuggingFace repo id.")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Control frequency in Hz (default: 10, should match training).")
    parser.add_argument("--visualize", action="store_true",
                        help="Show live camera feeds while running.")
    parser.add_argument("--cameras", type=int, nargs="+", default=None,
                        help="Camera indices. Auto-detects if not specified.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (default: cuda).")
    parser.add_argument(
        "--ramp-steps", type=int, default=DEFAULT_RAMP_STEPS,
        help="Blend policy toward current pose over N steps (default: 0=off). "
        "Use ~10 only if the first policy jump is violent; large values feel frozen "
        "when pred ≈ state.",
    )
    parser.add_argument(
        "--log-interval", type=int, default=1,
        help="Log state/prediction/command every N policy steps (0=off, default: 1).",
    )
    parser.add_argument(
        "--n-action-steps", type=int, default=10,
        help="ACT only: replan every N control steps (default: 10). "
        "Training often uses 100, which replays one chunk open-loop for ~10s at 10 Hz "
        "and makes pred look frozen. Use 1 for max reactivity, 100 to match training.",
    )
    parser.add_argument(
        "--home-pose",
        type=str,
        default=None,
        help="YAML/JSON with Fleft/Fright 7-DOF home poses (default: config/deploy_home.yaml).",
    )
    parser.add_argument(
        "--home-duration", type=float, default=3.0,
        help="Seconds to blend followers to home pose on exit (default: 3).",
    )
    parser.add_argument(
        "--no-home-on-exit", action="store_true",
        help="Skip moving to home pose before shutdown.",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

    args = parse_args()
    override_log_level(level=logging.INFO)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logging.warning("CUDA not available, falling back to CPU")
    if device == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        logging.info(
            f"GPU memory: {free_bytes / 1e9:.2f} GB free / {total_bytes / 1e9:.2f} GB total"
        )
        if free_bytes < 1.5e9:
            logging.warning(
                "Low free GPU memory — stop lerobot_train.py first, or deploy with --device cpu"
            )

    # ── Load policy ──────────────────────────────────────────────────────
    logging.info(f"Loading policy from {args.policy}...")
    policy_cfg = json.loads((Path(args.policy) / "config.json").read_text())
    policy_type = policy_cfg.get("type", "act")
    if policy_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(args.policy)
    else:
        policy = ACTPolicy.from_pretrained(args.policy)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    logging.info(f"Policy loaded: {policy.config.type} on {device}")
    logging.info(f"  Input:  {list(policy.config.input_features.keys())}")
    logging.info(f"  Output: {list(policy.config.output_features.keys())}")
    if policy_type == "act":
        trained_n = policy.config.n_action_steps
        configure_act_action_steps(policy, args.n_action_steps)
        logging.info(
            f"ACT deploy: n_action_steps {trained_n} (checkpoint) -> {policy.config.n_action_steps} "
            f"(replan every {policy.config.n_action_steps / args.hz:.1f}s at {args.hz} Hz)"
        )

    # Build pre/post processors (handles normalization/unnormalization)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy,
    )

    # ── Image resize to match policy input (required for correct + memory-safe inference)
    policy_image_sizes = get_policy_image_sizes(policy)
    training_resize = detect_training_resize(args.policy)
    if training_resize:
        logging.info(f"Training resize in train_config.json: {training_resize[0]}×{training_resize[1]}")
    for key, (h, w) in policy_image_sizes.items():
        logging.info(f"Policy expects {key}: {h}×{w}")

    # Open cameras near policy resolution to avoid decoding 3K frames every tick.
    cap_w = max((w for _, w in policy_image_sizes.values()), default=640)
    cap_h = max((h for h, _ in policy_image_sizes.values()), default=480)

    # ── Cameras ──────────────────────────────────────────────────────────
    if args.cameras:
        devices = [f"/dev/video{i}" for i in args.cameras]
        logging.info(f"Opening cameras: {devices} (capture ~{cap_w}×{cap_h})")
        cameras = []
        for dev in devices:
            r = CameraReader(dev, width=cap_w, height=cap_h)
            r.start()
            cameras.append(r)
            logging.info(f"  {dev}: {r.width}x{r.height}")
    else:
        logging.info("Auto-detecting cameras...")
        cameras = detect_cameras()
    time.sleep(0.3)
    logging.info(f"{len(cameras)} camera(s) active")
    n_policy_cams = len(policy_image_sizes)
    if len(cameras) != n_policy_cams:
        logging.error(
            f"Policy expects {n_policy_cams} camera(s) but {len(cameras)} are open — "
            f"use --cameras with the same indices/order as recording"
        )
        sys.exit(1)
    for i, cam in enumerate(cameras):
        key = f"observation.images.cam_{i}"
        if key in policy_image_sizes:
            eh, ew = policy_image_sizes[key]
            if cam.height != eh or cam.width != ew:
                logging.info(
                    f"  {cam.device} native {cam.height}×{cam.width} → resize to {eh}×{ew} for {key}"
                )

    # ── Resolve arms ─────────────────────────────────────────────────────
    logging.info("Resolving arm CAN interfaces...")
    leaders_info = resolve_arms("leader_arms")
    followers_info = resolve_arms("follower_arms")
    ensure_can_up({**leaders_info, **followers_info})

    # Always use both arms — bimanual policy controls both
    follower_pairs = [("Fleft", PORT_LEFT), ("Fright", PORT_RIGHT)]
    leader_keys = ["Lleft", "Lright"]

    # ── Follower servers ─────────────────────────────────────────────────
    clients: Dict[str, ClientRobot] = {}
    for f_key, port in follower_pairs:
        start_follower_server(followers_info[f_key], port, label=f_key)
    time.sleep(1.0)
    for f_key, port in follower_pairs:
        clients[f_key] = ClientRobot(port)

    # ── Leader arms (for e-stop buttons only) ────────────────────────────
    leader_robots: List[YAMLeaderRobot] = []
    for l_key in leader_keys:
        l_info = leaders_info[l_key]
        gripper_type = GripperType.from_string_name(l_info.gripper_type)
        robot = get_yam_robot(
            channel=l_info.channel,
            gripper_type=gripper_type,
            zero_gravity_mode=True,
            gravity_comp_factor=0.0,
        )
        _all_robots.append(robot)
        leader_robots.append(YAMLeaderRobot(robot))
    logging.info(f"Leader arms connected for e-stop buttons")

    # ── Shutdown + run state events ──────────────────────────────────────
    stop_event = threading.Event()
    running_event = threading.Event()  # clear = paused, set = running
    _force_count = 0

    def _shutdown(sig, frame):
        nonlocal _force_count
        _force_count += 1
        if _force_count >= 2:
            logging.warning("Force exit")
            os._exit(1)
        logging.info("Shutting down (Ctrl-C again to force)...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start button monitor thread
    btn_thread = threading.Thread(
        target=button_monitor,
        args=(leader_robots, stop_event, running_event),
        name="button_monitor", daemon=True,
    )
    btn_thread.start()

    # Warm up GPU inference before motors move (avoids a multi-second first-step stall).
    if cameras:
        state_dim = int(policy.config.input_features["observation.state"].shape[0])
        warmup_obs: dict = {"observation.state": torch.zeros(state_dim)}
        for i in range(len(cameras)):
            key = f"observation.images.cam_{i}"
            if key in policy_image_sizes:
                eh, ew = policy_image_sizes[key]
                warmup_obs[key] = torch.zeros(3, eh, ew)
        try:
            with torch.inference_mode():
                _ = policy.select_action(preprocessor(warmup_obs))
            logging.info("Policy inference warmup done")
        except Exception as e:
            logging.warning(f"Policy warmup failed (first step may be slow): {e}")

    # ── Control loop ─────────────────────────────────────────────────────
    period = 1.0 / args.hz
    step_count = 0
    ramp_steps = max(0, args.ramp_steps)

    logging.info(
        f"\nReady — {len(follower_pairs)} follower(s), {len(cameras)} camera(s) @ {args.hz} Hz\n"
        f"  Action ramp: {ramp_steps} steps\n"
        f"  Prediction log: every {args.log_interval} step(s) "
        f"({'off' if args.log_interval == 0 else 'on'})\n"
        f"  Press top button  → start policy / e-stop toggle\n"
        f"  Ctrl-C            → stop and exit\n"
        f"  Do not run teleop.py or record.py at the same time (CAN bus conflict).\n"
        f"  Waiting for top button press to start...\n"
    )

    # Set up visualisation window if requested
    win = None
    cam_labels = []
    if args.visualize and cameras:
        cam_labels = [f"cam {c.index}" for c in cameras]
        win = "deploy.py — Q/ESC to quit"
        with _quiet_stderr():
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    was_running = False
    hold_steps = 0
    try:
        while not stop_event.is_set():
            t_start = time.monotonic()

            # Wait for button press to start / e-stop pause
            if not running_event.is_set():
                was_running = False
                if win is not None:
                    frames = [c.get_frame() for c in cameras]
                    rect = cv2.getWindowImageRect(win)
                    disp_w, disp_h = max(rect[2], 640), max(rect[3], 480)
                    canvas = tile_frames(frames, cam_labels, disp_w, disp_h)
                    label = "PAUSED — press top button to start" if step_count == 0 \
                            else "E-STOP — press top button to resume"
                    cv2.putText(canvas, label, (20, disp_h - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
                    cv2.imshow(win, canvas)
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        stop_event.set()
                else:
                    time.sleep(0.05)
                continue

            if not was_running:
                if hasattr(policy, "reset"):
                    policy.reset()
                    logging.info("Policy action queue reset — next step plans a new chunk")
                was_running = True

            # 1. Read follower state (observation)
            state_parts = []
            for f_key, _ in follower_pairs:
                qpos = clients[f_key].get_joint_pos()
                state_parts.append(qpos.astype(np.float32))
            state = np.concatenate(state_parts)

            # 2. Read camera images
            observation: dict = {
                "observation.state": torch.from_numpy(state),
            }
            cameras_ok: List[bool] = []
            for i, cam in enumerate(cameras):
                bgr = cam.get_frame()
                cameras_ok.append(bgr is not None)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                    key = f"observation.images.cam_{i}"
                    if key in policy_image_sizes:
                        eh, ew = policy_image_sizes[key]
                        if img_tensor.shape[1] != eh or img_tensor.shape[2] != ew:
                            img_tensor = transforms_v2.functional.resize(
                                img_tensor, [eh, ew], antialias=True,
                            )
                    observation[key] = img_tensor
                elif args.log_interval > 0 and step_count % max(args.log_interval, 1) == 0:
                    logging.warning(f"step {step_count}: cam_{i} ({cam.device}) — no frame")

            # 3. Run policy
            t_infer = time.monotonic()
            queue_before = (
                len(policy._action_queue)
                if policy_type == "act" and hasattr(policy, "_action_queue")
                else 0
            )
            processed_obs = preprocessor(observation)
            with torch.inference_mode():
                action = policy.select_action(processed_obs)
            action = postprocessor(action)
            infer_ms = (time.monotonic() - t_infer) * 1000.0
            chunk_replan = policy_type == "act" and queue_before == 0
            queue_left = (
                len(policy._action_queue)
                if policy_type == "act" and hasattr(policy, "_action_queue")
                else 0
            )
            pred_np = action.squeeze(0).cpu().numpy()
            cmd_np = pred_np
            if ramp_steps > 0 and step_count < ramp_steps:
                cmd_np = blend_actions(state, pred_np, step_count, ramp_steps)

            if args.log_interval > 0 and step_count % args.log_interval == 0:
                log_prediction_step(
                    step=step_count,
                    infer_ms=infer_ms,
                    state=state,
                    pred=pred_np,
                    cmd=cmd_np,
                    follower_labels=[f_key for f_key, _ in follower_pairs],
                    ramp_steps=ramp_steps,
                    cameras_ok=cameras_ok,
                    chunk_replan=chunk_replan,
                    action_queue_left=queue_left,
                )

            # 4. Send actions to followers (skip if e-stopped during inference)
            if running_event.is_set():
                offset = 0
                for f_key, _ in follower_pairs:
                    try:
                        if not clients[f_key].is_motor_chain_running():
                            running_event.clear()
                            logging.error(
                                f"[{f_key}] motor control loop stopped — policy paused. "
                                "Power-cycle arm or restart deploy; check for joint limit errors."
                            )
                            break
                        clients[f_key].command_joint_pos(
                            cmd_np[offset:offset + NUM_JOINTS]
                        )
                    except Exception as e:
                        running_event.clear()
                        logging.error(
                            f"[{f_key}] command failed — pausing policy ({e}). "
                            "One arm may have hit a motor fault (timeout/disabled). "
                            "Press top button to retry after checking the arm."
                        )
                        break
                    offset += NUM_JOINTS

                max_cmd_delta = float(np.max(np.abs(cmd_np - state)))
                if max_cmd_delta < 0.02:
                    hold_steps += 1
                    if hold_steps == 30:
                        logging.warning(
                            "Policy command ≈ current state for 30 steps — arm may look stopped "
                            "even though inference runs. Try --n-action-steps 1, a better checkpoint, "
                            "or start from a pose closer to the demos."
                        )
                else:
                    hold_steps = 0

            step_count += 1
            if step_count % 100 == 0:
                logging.info(f"Step {step_count}")

            # 5. Visualisation
            if win is not None:
                frames = [c.get_frame() for c in cameras]
                rect = cv2.getWindowImageRect(win)
                disp_w, disp_h = max(rect[2], 640), max(rect[3], 480)
                canvas = tile_frames(frames, cam_labels, disp_w, disp_h)
                label = f"RUNNING  step {step_count}"
                cv2.putText(canvas, label, (disp_w - 350, disp_h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)
                cv2.imshow(win, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    stop_event.set()

            # Rate limiting
            elapsed = time.monotonic() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        logging.error(f"Error in control loop: {e}", exc_info=True)
    finally:
        logging.info(f"Stopping after {step_count} steps")
        if win is not None:
            cv2.destroyAllWindows()
        for cam in cameras:
            try:
                cam.stop()
            except Exception as e:
                logging.warning(f"Error stopping camera: {e}")
        if (
            not args.no_home_on_exit
            and clients
            and _force_count < 2
        ):
            try:
                home = load_home_poses(
                    args.home_pose, [f_key for f_key, _ in follower_pairs]
                )
                logging.info(
                    f"Moving followers to home pose over {args.home_duration:.1f}s..."
                )
                move_followers_to_pose(
                    clients,
                    follower_pairs,
                    home,
                    duration=args.home_duration,
                )
                logging.info("Home pose reached.")
            except Exception as e:
                logging.warning(f"Could not move to home pose: {e}")
        for robot in _all_robots:
            try:
                robot.close()
            except Exception as e:
                logging.warning(f"Error closing robot: {e}")
        logging.info("Deploy stopped.")


if __name__ == "__main__":
    main()
