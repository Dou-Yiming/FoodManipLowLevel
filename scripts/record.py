#!/usr/bin/env python3
"""
Bimanual recording script for YAM arms — LeRobot dataset format.

Records follower joint states (observation), leader joint commands (action),
and RGB camera streams in LeRobot v3 format, ready for training ACT,
Diffusion Policy, or any other LeRobot-compatible policy.

Controls (teaching handle buttons):
  Top button    — toggle sync + gravity compensation (same as teleop.py)
  Bottom button — press once to START an episode, press again to SAVE it (repeat for more episodes)
  Ctrl-C        — quit the recorder (all episodes saved so far are kept)

Data is saved to:  data/<name>/
Format:            LeRobot v3 (parquet + mp4)

Usage:
    python scripts/record.py --name pick_place --task "pick up the block"
    python scripts/record.py --name pick_place --task "..." --visualize
    python scripts/record.py --name pick_place --task "..." --hz 15
    python scripts/record.py --name pick_place --task "..." --left
"""

import argparse
import contextlib
import glob
import json
import logging
import math
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
from PIL import Image

if not hasattr(cv2, "VideoCapture"):
    sys.exit(
        "OpenCV install is incomplete (cv2 has no VideoCapture). "
        "Reinstall in your venv:\n"
        "  uv pip uninstall opencv-python opencv-python-headless -y\n"
        "  uv pip install opencv-python"
    )

# Best-effort Qt font dir to suppress QFontDatabase warnings (may be unavailable or read-only)
_cv2_file = getattr(cv2, "__file__", None)
if _cv2_file:
    try:
        os.makedirs(os.path.join(os.path.dirname(_cv2_file), "qt", "fonts"), exist_ok=True)
    except OSError:
        pass


@contextlib.contextmanager
def _quiet_stderr():
    """Suppress C-level stderr (Qt/OpenCV noise that bypasses Python logging)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def opencv_gui_available() -> bool:
    """True if OpenCV was built with a windowing backend (not headless)."""
    import re

    info = cv2.getBuildInformation()
    if re.search(r"GUI:\s+NONE", info):
        return False
    return any(k in info for k in ("GTK", "Qt", "Cocoa"))

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import GripperType
from i2rt.utils.utils import override_log_level

from resolve_leader_can import ArmInfo, ensure_can_up, load_teleop_config, resolve_arms

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

PORT_LEFT = 11333
PORT_RIGHT = 11334
NUM_ARM_JOINTS = 6
NUM_JOINTS = 7  # 6 arm + 1 gripper
# 480×640 is enough for manipulation policies and keeps AV1/h264 encoding fast.
DEFAULT_RECORD_RESOLUTION = (480, 640)
SLOW_SAVE_PIXELS = 1280 * 720  # warn above ~720p per frame

_all_robots: List[MotorChainRobot] = []


# ── Shared state ──────────────────────────────────────────────────────────

class ArmState:
    """Thread-safe snapshot of one arm pair's current state."""

    def __init__(self) -> None:
        self.leader_qpos: Optional[np.ndarray] = None
        self.follower_qpos: Optional[np.ndarray] = None
        self.synchronized: bool = False
        self._lock = threading.Lock()

    def update(self, leader_qpos: np.ndarray, follower_qpos: np.ndarray, synced: bool) -> None:
        with self._lock:
            self.leader_qpos = leader_qpos.copy()
            self.follower_qpos = follower_qpos.copy()
            self.synchronized = synced

    def snapshot(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        with self._lock:
            lq = self.leader_qpos.copy() if self.leader_qpos is not None else None
            fq = self.follower_qpos.copy() if self.follower_qpos is not None else None
            return lq, fq, self.synchronized


class RecordingState:
    """Controls episode recording lifecycle, safe to call from multiple threads."""

    PHASE_IDLE = "idle"
    PHASE_RECORDING = "recording"
    PHASE_SAVING = "saving"
    PHASE_DONE = "done"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.is_recording = False
        self.episode_idx = 0
        self._save_requested = False
        self._last_toggle_time = 0.0
        self._phase = self.PHASE_IDLE
        self._frames_captured = 0
        self._recording_t0 = 0.0
        self._save_ep_num = 0
        self._save_t0 = 0.0
        self._save_step = ""
        self._save_current = 0
        self._save_total = 0
        self._save_camera_idx = 0
        self._save_camera_count = 1
        self._save_camera_label = ""
        self._done_message = ""
        self._done_detail = ""
        self._done_until = 0.0

    def toggle(self, label: str = "") -> Optional[str]:
        """Toggle recording. Returns 'started', 'stopped', or None (cooldown)."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_toggle_time < 1.0:
                return None
            self._last_toggle_time = now
            if not self.is_recording:
                self.is_recording = True
                self._phase = self.PHASE_RECORDING
                self._frames_captured = 0
                self._recording_t0 = now
                logging.info(
                    f"[{label}] ● Recording STARTED — episode {self.episode_idx + 1}"
                )
                return "started"
            else:
                self.is_recording = False
                self._save_requested = True
                self._save_ep_num = self.episode_idx + 1
                self._phase = self.PHASE_SAVING
                self._save_t0 = now
                logging.info(
                    f"[{label}] ■ Recording STOPPED — saving episode {self._save_ep_num}"
                )
                return "stopped"

    def take_save_request(self) -> bool:
        with self._lock:
            if self._save_requested:
                self._save_requested = False
                return True
            return False

    def set_frames_captured(self, frames: int) -> None:
        with self._lock:
            self._frames_captured = frames

    def begin_save(self, ep_num: int, frames: int) -> None:
        with self._lock:
            self._phase = self.PHASE_SAVING
            self._save_ep_num = ep_num
            self._frames_captured = frames
            self._save_t0 = time.monotonic()
            self._save_step = "Starting"
            self._save_current = 0
            self._save_total = frames
            self._save_camera_idx = 0
            self._save_camera_count = 1
            self._save_camera_label = ""

    def set_save_progress(
        self,
        step: str,
        current: int,
        total: int,
        *,
        cam_idx: int = 0,
        cam_count: int = 1,
        cam_label: str = "",
    ) -> None:
        with self._lock:
            self._save_step = step
            self._save_current = current
            self._save_total = total
            self._save_camera_idx = cam_idx
            self._save_camera_count = max(1, cam_count)
            self._save_camera_label = cam_label

    def save_progress_fraction(self) -> Optional[float]:
        """Overall save progress in [0, 1] during multi-camera video encode, else None."""
        with self._lock:
            if self._phase != self.PHASE_SAVING or self._save_total <= 0:
                return None
            if self._save_step != "Encoding video":
                return None
            done = self._save_camera_idx * self._save_total + self._save_current
            total = self._save_camera_count * self._save_total
            return min(1.0, done / total)

    def complete_save(self, ep_num: int, frames: int, hz: float, total_episodes: int) -> None:
        duration_s = frames / hz if hz > 0 else 0.0
        with self._lock:
            self._phase = self.PHASE_DONE
            self._done_message = f"Episode {ep_num} saved"
            self._done_detail = (
                f"{frames} frames · {duration_s:.1f}s · {total_episodes} in dataset"
            )
            self._done_until = time.monotonic() + 4.0

    def discard_empty(self) -> None:
        with self._lock:
            self._phase = self.PHASE_DONE
            self._done_message = "Empty episode discarded"
            self._done_detail = "No frames captured — check cameras / arms"
            self._done_until = time.monotonic() + 4.0

    def fail_save(self, error: str) -> None:
        with self._lock:
            self._phase = self.PHASE_DONE
            self._done_message = "Save failed"
            self._done_detail = error[:80]
            self._done_until = time.monotonic() + 5.0

    def get_overlay(self) -> Tuple[str, str, str, str]:
        """Return (line1, line2, line3, style) for the visualization overlay."""
        with self._lock:
            now = time.monotonic()
            if self._phase == self.PHASE_DONE and now > self._done_until:
                self._phase = self.PHASE_IDLE

            if self._phase == self.PHASE_RECORDING:
                elapsed = now - self._recording_t0
                return (
                    f"REC  episode {self.episode_idx + 1}",
                    f"capturing · {self._frames_captured} frames · {elapsed:.1f}s",
                    "bottom button · stop & save",
                    "rec",
                )
            if self._phase == self.PHASE_SAVING:
                elapsed = now - self._save_t0 if self._save_t0 else 0.0
                step = self._save_step or "Saving"
                if self._save_total > 0 and step == "Encoding video":
                    line2 = f"{step}: frame {self._save_current}/{self._save_total}"
                elif self._save_total > 0:
                    line2 = f"{step} ({self._save_total} frames)"
                else:
                    line2 = step
                if self._save_camera_count > 1 and self._save_camera_label:
                    line3 = (
                        f"{self._save_camera_label} "
                        f"({self._save_camera_idx + 1}/{self._save_camera_count}) · {elapsed:.1f}s"
                    )
                else:
                    line3 = f"{elapsed:.1f}s"
                return (
                    f"Saving episode {self._save_ep_num}...",
                    line2,
                    line3,
                    "save",
                )
            if self._phase == self.PHASE_DONE:
                style = "ok" if "saved" in self._done_message else "warn"
                return (self._done_message, self._done_detail, "", style)
            return (
                f"{self.episode_idx} episode(s) saved",
                "bottom button · start recording",
                "",
                "idle",
            )

    def mark_episode_saved(self) -> int:
        """Call after a successful save_episode(); returns 1-based episode number."""
        with self._lock:
            self.episode_idx += 1
            return self.episode_idx

    @property
    def next_episode_number(self) -> int:
        """1-based episode number for the episode currently being recorded."""
        with self._lock:
            return self.episode_idx + 1


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
        if self.width <= 0 or self.height <= 0:
            self._cap.release()
            self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
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


def _usb_port_id(dev_path: str) -> str:
    """Stable physical port id for a /dev/videoN node (groups index0/index1)."""
    try:
        sysfs = os.path.realpath(
            f"/sys/class/video4linux/{os.path.basename(dev_path)}/device"
        )
        while sysfs and not os.path.exists(os.path.join(sysfs, "busnum")):
            parent = os.path.dirname(sysfs)
            if parent == sysfs:
                break
            sysfs = parent
        if os.path.exists(os.path.join(sysfs, "busnum")):
            bus = open(os.path.join(sysfs, "busnum")).read().strip()
            dev = open(os.path.join(sysfs, "devnum")).read().strip()
            return f"usb-{bus}-{dev}"
    except OSError:
        pass
    return dev_path


def detect_cameras() -> List[CameraReader]:
    """Auto-detect one CameraReader per physical USB camera.

    Each camera exposes two /dev/videoN nodes (capture + metadata). We open
    candidates, wait for a real frame, and keep at most one reader per port.
    """
    by_path_all = sorted(glob.glob("/dev/v4l/by-path/*-video-index*"))
    candidates: List[str] = []
    port_of: dict[str, str] = {}

    if by_path_all:
        ports: dict[str, list[str]] = {}
        for p in by_path_all:
            port_key = p.rsplit("-video-index", 1)[0]
            ports.setdefault(port_key, []).append(p)
        logging.info(f"Found {len(ports)} USB camera port(s) via by-path")
        for port_key in sorted(ports):
            for lnk in sorted(ports[port_key]):
                dev = os.path.realpath(lnk)
                candidates.append(dev)
                port_of[dev] = port_key
    else:
        logging.info("No by-path entries; scanning /dev/video*")
        candidates = sorted(
            (p for p in glob.glob("/dev/video*") if p[len("/dev/video"):].isdigit()),
            key=lambda p: int(p[len("/dev/video"):]),
        )
        port_of = {dev: _usb_port_id(dev) for dev in candidates}

    readers: List[CameraReader] = []
    seen_ports: set[str] = set()
    with _quiet_stderr():
        for dev in candidates:
            port = port_of[dev]
            if port in seen_ports:
                continue
            try:
                reader = CameraReader(dev)
            except Exception:
                logging.info(f"  ✗ {dev} (cannot open)")
                continue
            reader.start()
            deadline = time.monotonic() + 2.0
            frame = None
            while time.monotonic() < deadline:
                frame = reader.get_frame()
                if frame is not None:
                    break
                time.sleep(0.05)
            if frame is not None:
                readers.append(reader)
                seen_ports.add(port)
                logging.info(f"  ✓ {dev}: {reader.width}×{reader.height}")
            else:
                reader.stop()
                logging.info(f"  ✗ {dev} (no frames)")

    if not readers:
        logging.warning("No cameras detected — recording without images")
    return readers


def _wait_for_camera_frames(cameras: List[CameraReader], timeout_s: float = 3.0) -> None:
    if not cameras:
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(c.get_frame() is not None for c in cameras):
            return
        time.sleep(0.05)
    missing = [c.device for c in cameras if c.get_frame() is None]
    logging.warning(f"Cameras still missing frames after warm-up: {missing}")


def open_cameras(
    indices: Optional[List[int]],
    warmup_s: float = 3.0,
    capture_resolution: Optional[Tuple[int, int]] = None,
) -> List[CameraReader]:
    """Open cameras by /dev/video index, keeping only devices that stream frames."""
    if indices is None:
        logging.info("Auto-detecting cameras...")
        return detect_cameras()

    devices = [f"/dev/video{i}" for i in indices]
    if capture_resolution is not None:
        ch, cw = capture_resolution
        logging.info(f"Opening cameras: {devices} (capture {ch}×{cw})")
    else:
        logging.info(f"Opening cameras: {devices} (native capture)")
    readers: List[CameraReader] = []
    for dev in devices:
        try:
            if capture_resolution is not None:
                ch, cw = capture_resolution
                reader = CameraReader(dev, width=cw, height=ch)
            else:
                reader = CameraReader(dev)
        except Exception as e:
            logging.warning(f"  ✗ {dev} (cannot open: {e})")
            continue
        reader.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if reader.get_frame() is not None:
                readers.append(reader)
                logging.info(f"  ✓ {dev}: {reader.width}×{reader.height}")
                break
            time.sleep(0.05)
        else:
            reader.stop()
            logging.warning(
                f"  ✗ {dev} (no frames after 2s — skipped; try the other index "
                f"for this USB camera, e.g. index0 vs index1)"
            )
    if len(readers) != len(indices):
        logging.warning(
            f"Only {len(readers)}/{len(indices)} requested cameras are streaming. "
            "Episodes stay empty while recording if the dataset expects more cameras."
        )
    _wait_for_camera_frames(readers, timeout_s=warmup_s)
    return readers


def recording_image_sizes(
    dataset: "LeRobotDataset",
    cameras: List[CameraReader],
    resolution: Optional[Tuple[int, int]],
) -> List[Optional[Tuple[int, int]]]:
    """Per-camera (height, width) for resize; uses --resolution, else dataset features, else native."""
    if resolution is not None:
        h, w = resolution
        return [(h, w)] * len(cameras)
    sizes: List[Optional[Tuple[int, int]]] = []
    for i in range(len(cameras)):
        key = f"observation.images.cam_{i}"
        feat = dataset.meta.features.get(key)
        if feat is not None:
            shape = feat["shape"]
            sizes.append((int(shape[0]), int(shape[1])))
        else:
            sizes.append(None)
    return sizes


def load_reference_info(reference_root: Path) -> dict:
    path = reference_root / "meta" / "info.json"
    if not path.exists():
        raise FileNotFoundError(f"No meta/info.json in {reference_root}")
    return json.loads(path.read_text())


def validate_against_reference(
    features: dict,
    fps: float,
    n_cameras: int,
    arm_labels: List[str],
    reference_root: Path,
) -> None:
    """Log warnings if recording format diverges from a reference dataset (e.g. wire_insert)."""
    ref = load_reference_info(reference_root)
    ref_feats = ref["features"]
    issues: List[str] = []

    if int(ref.get("fps", fps)) != int(fps):
        issues.append(f"fps: recording {fps} vs reference {ref.get('fps')}")

    for key in ("observation.state", "action"):
        ref_shape = tuple(ref_feats[key]["shape"])
        our_shape = tuple(features[key]["shape"])
        if ref_shape != our_shape:
            issues.append(f"{key} shape {our_shape} vs reference {ref_shape}")

    ref_cams = sorted(k for k in ref_feats if k.startswith("observation.images."))
    our_cams = sorted(k for k in features if k.startswith("observation.images."))
    if len(our_cams) != len(ref_cams):
        issues.append(f"camera count {len(our_cams)} vs reference {len(ref_cams)}")
    for key in our_cams:
        if key not in ref_feats:
            issues.append(f"missing in reference: {key}")
            continue
        ref_shape = tuple(ref_feats[key]["shape"])
        our_shape = tuple(features[key]["shape"])
        if ref_shape != our_shape:
            issues.append(f"{key} shape {our_shape} vs reference {ref_shape}")

    expected_names = []
    for arm in arm_labels:
        expected_names += [f"{arm}_j{i}" for i in range(NUM_ARM_JOINTS)]
        expected_names.append(f"{arm}_gripper")
    ref_names = list(ref_feats["observation.state"]["names"])
    if ref_names != expected_names:
        issues.append(
            f"joint names order differs from reference (reference starts with {ref_names[:3]}...)"
        )

    logging.info(f"Reference dataset: {reference_root.resolve()}")
    logging.info(
        f"  Reference format: {len(ref_cams)} cameras, "
        f"{ref_feats[ref_cams[0]]['shape'][0]}×{ref_feats[ref_cams[0]]['shape'][1]} "
        f"@ {ref.get('fps')} Hz, state dim {ref_feats['observation.state']['shape'][0]}"
    )
    if issues:
        for msg in issues:
            logging.warning(f"Format mismatch vs reference: {msg}")
    else:
        logging.info("Format check vs reference: OK (matches wire_insert-style layout)")


# ── Visualisation helper ──────────────────────────────────────────────────

def _grid_shape(n: int, target_w: int, target_h: int) -> Tuple[int, int]:
    """Return (ncols, nrows) for tiling n camera feeds in the window."""
    if n <= 1:
        return 1, 1
    if n == 2 and target_w >= target_h:
        return 2, 1
    if n == 2:
        return 1, 2
    ncols = max(1, min(n, math.ceil(math.sqrt(n * target_w / max(target_h, 1)))))
    nrows = math.ceil(n / ncols)
    return ncols, nrows


def _fit_frame_in_cell(
    frame: Optional[np.ndarray], cell_w: int, cell_h: int,
) -> np.ndarray:
    """Letterbox frame into cell_w×cell_h, preserving aspect ratio."""
    cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    if frame is None or frame.size == 0:
        return cell
    fh, fw = frame.shape[:2]
    if fw <= 0 or fh <= 0:
        return cell
    scale = min(cell_w / fw, cell_h / fh)
    new_w = max(1, int(fw * scale))
    new_h = max(1, int(fh * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (cell_w - new_w) // 2
    y0 = (cell_h - new_h) // 2
    cell[y0:y0 + new_h, x0:x0 + new_w] = resized
    return cell


def tile_frames(
    frames: List[Optional[np.ndarray]],
    labels: List[str],
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Tile camera frames in a grid; each cell keeps the source aspect ratio."""
    n = len(frames)
    if n == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    ncols, nrows = _grid_shape(n, target_w, target_h)
    cell_w = target_w // ncols
    cell_h = target_h // nrows
    canvas = np.zeros((nrows * cell_h, ncols * cell_w, 3), dtype=np.uint8)
    for i, (frame, label) in enumerate(zip(frames, labels)):
        row, col = i // ncols, i % ncols
        y0, x0 = row * cell_h, col * cell_w
        cell = _fit_frame_in_cell(frame, cell_w, cell_h)
        cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(cell, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = cell
    return canvas


def draw_recording_overlay(canvas: np.ndarray, recording_state: RecordingState) -> None:
    """Draw recording / saving progress on the visualization window."""
    line1, line2, line3, style = recording_state.get_overlay()
    h, w = canvas.shape[:2]
    palette = {
        "rec": ((80, 80, 255), (0, 0, 200)),
        "save": ((0, 220, 255), (0, 130, 200)),
        "ok": ((120, 255, 120), (0, 180, 0)),
        "warn": ((100, 180, 255), (0, 60, 255)),
        "idle": ((180, 180, 180), (60, 60, 60)),
    }
    fg, shadow = palette.get(style, palette["idle"])
    lines = [ln for ln in (line1, line2, line3) if ln]
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = max(12, w - 520)
    y = h - 16

    progress = recording_state.save_progress_fraction()
    if progress is not None:
        bar_w = min(480, w - 40)
        bar_h = 14
        bar_x = max(12, w - bar_w - 20)
        bar_y = h - 16 - len(lines) * 36 - bar_h - 8
        fill_w = int(bar_w * progress)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        if fill_w > 0:
            cv2.rectangle(
                canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 255), -1,
            )
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        pct = f"{int(progress * 100)}%"
        cv2.putText(
            canvas, pct, (bar_x + bar_w + 8, bar_y + bar_h - 2),
            font, 0.55, fg, 1, cv2.LINE_AA,
        )

    for line in reversed(lines):
        scale = 1.0 if style in ("rec", "save") else 0.85
        thickness_fg = 2 if style in ("rec", "save") else 1
        cv2.putText(canvas, line, (x, y), font, scale, shadow, 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (x, y), font, scale, fg, thickness_fg, cv2.LINE_AA)
        y -= int(34 * scale)


# ── Robot classes ─────────────────────────────────────────────────────────

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


class YAMLeaderRobot:
    def __init__(self, robot: MotorChainRobot) -> None:
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> Tuple[np.ndarray, list]:
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        gripper_cmd = 1 - encoder_obs[0].position
        return np.concatenate([qpos, [gripper_cmd]]), encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert joint_pos.shape[0] == NUM_ARM_JOINTS
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._robot.update_kp_kd(kp, kd)


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


# ── Teleop loop ───────────────────────────────────────────────────────────

def run_leader_follower_loop(
    leader: YAMLeaderRobot,
    client: ClientRobot,
    bilateral_kp: float,
    gravity_comp_factor: float,
    arm_state: ArmState,
    recording_state: RecordingState,
    label: str,
    stop_event: threading.Event,
    handle_recording: bool = True,
) -> None:
    leader_kp = leader._robot._kp.copy()
    current_joint_pos, _ = leader.get_info()
    current_follower_joint_pos = client.get_joint_pos()

    def slow_move(target: np.ndarray, start: np.ndarray, duration: float = 1.0) -> None:
        steps = 100
        for i in range(steps):
            if stop_event.is_set():
                return
            blend = i / steps
            client.command_joint_pos((1 - blend) * start + blend * target)
            time.sleep(duration / steps)

    synchronized = False
    while not stop_event.is_set():
        current_joint_pos, current_button = leader.get_info()

        # Top button: toggle sync + gravity comp
        if current_button[0] > 0.5:
            if not synchronized:
                leader._robot.gravity_comp_factor = gravity_comp_factor
                leader.update_kp_kd(kp=leader_kp * bilateral_kp, kd=np.zeros(NUM_ARM_JOINTS))
                leader.command_joint_pos(current_joint_pos[:NUM_ARM_JOINTS])
                slow_move(current_joint_pos, current_follower_joint_pos)
                logging.info(f"[{label}] Synchronized")
            else:
                leader._robot.gravity_comp_factor = 0.0
                leader.update_kp_kd(kp=np.zeros(NUM_ARM_JOINTS), kd=np.zeros(NUM_ARM_JOINTS))
                leader.command_joint_pos(current_follower_joint_pos[:NUM_ARM_JOINTS])
                logging.info(f"[{label}] Un-synchronized")
            synchronized = not synchronized
            while current_button[0] > 0.5 and not stop_event.is_set():
                time.sleep(0.03)
                current_joint_pos, current_button = leader.get_info()

        # Bottom button: start/stop episode recording (one leader only)
        if handle_recording and current_button[1] > 0.5:
            recording_state.toggle(label=label)
            while current_button[1] > 0.5 and not stop_event.is_set():
                time.sleep(0.03)
                current_joint_pos, current_button = leader.get_info()

        current_follower_joint_pos = client.get_joint_pos()
        if synchronized:
            client.command_joint_pos(current_joint_pos)
            leader.command_joint_pos(current_follower_joint_pos[:NUM_ARM_JOINTS])

        arm_state.update(current_joint_pos, current_follower_joint_pos, synchronized)
        time.sleep(0.01)


# ── Episode save with UI progress ─────────────────────────────────────────

def save_episode_with_progress(
    dataset: "LeRobotDataset",
    recording_state: RecordingState,
) -> None:
    """Like LeRobotDataset.save_episode(), reporting progress for the visualization overlay."""
    from lerobot.datasets.lerobot_dataset import DEFAULT_IMAGE_PATH
    from lerobot.datasets.utils import validate_episode_buffer
    from lerobot.datasets.compute_stats import compute_episode_stats
    from lerobot.datasets.video_utils import encode_video_frames

    episode_buffer = dataset.episode_buffer
    validate_episode_buffer(episode_buffer, dataset.meta.total_episodes, dataset.features)

    episode_length = episode_buffer.pop("size")
    tasks = episode_buffer.pop("task")
    episode_tasks = list(set(tasks))
    episode_index = episode_buffer["episode_index"]

    episode_buffer["index"] = np.arange(
        dataset.meta.total_frames, dataset.meta.total_frames + episode_length
    )
    episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

    dataset.meta.save_episode_tasks(episode_tasks)
    episode_buffer["task_index"] = np.array(
        [dataset.meta.get_task_index(task) for task in tasks]
    )

    for key, ft in dataset.features.items():
        if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
            continue
        episode_buffer[key] = np.stack(episode_buffer[key])

    recording_state.set_save_progress("Flushing images", 0, episode_length)
    dataset._wait_image_writer()
    recording_state.set_save_progress("Flushing images", episode_length, episode_length)

    recording_state.set_save_progress("Computing stats", 0, 0)
    ep_stats = compute_episode_stats(episode_buffer, dataset.features)

    recording_state.set_save_progress("Writing joint data", 0, 0)
    ep_metadata = dataset._save_episode_data(episode_buffer)

    video_keys = dataset.meta.video_keys
    use_batched = dataset.batch_encoding_size > 1
    if video_keys and not use_batched:
        cam_count = len(video_keys)
        for cam_idx, video_key in enumerate(video_keys):
            cam_label = video_key.rsplit(".", 1)[-1]

            def _on_encode_frame(current: int, total: int, _vk=video_key, _ci=cam_idx, _cl=cam_label) -> None:
                recording_state.set_save_progress(
                    "Encoding video",
                    current,
                    total,
                    cam_idx=_ci,
                    cam_count=cam_count,
                    cam_label=_cl,
                )

            recording_state.set_save_progress(
                "Encoding video", 0, episode_length,
                cam_idx=cam_idx, cam_count=cam_count, cam_label=cam_label,
            )
            temp_path = Path(tempfile.mkdtemp(dir=dataset.root)) / f"{video_key}_{episode_index:03d}.mp4"
            fpath = DEFAULT_IMAGE_PATH.format(
                image_key=video_key, episode_index=episode_index, frame_index=0,
            )
            img_dir = (dataset.root / fpath).parent
            encode_video_frames(
                img_dir,
                temp_path,
                dataset.fps,
                vcodec=dataset.vcodec,
                overwrite=True,
                progress_callback=_on_encode_frame,
            )
            shutil.rmtree(img_dir)
            ep_metadata.update(
                dataset._save_episode_video(video_key, episode_index, temp_path=temp_path)
            )

    recording_state.set_save_progress("Finalizing metadata", episode_length, episode_length)
    dataset.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

    if video_keys and use_batched:
        dataset.episodes_since_last_encoding += 1
        if dataset.episodes_since_last_encoding == dataset.batch_encoding_size:
            start_ep = dataset.num_episodes - dataset.batch_encoding_size
            end_ep = dataset.num_episodes
            dataset._batch_save_episode_video(start_ep, end_ep)
            dataset.episodes_since_last_encoding = 0

    dataset.clear_episode_buffer(delete_images=len(dataset.meta.image_keys) > 0)


# ── Recording thread ──────────────────────────────────────────────────────

def recording_loop(
    state_left: Optional[ArmState],
    state_right: Optional[ArmState],
    cameras: List[CameraReader],
    dataset: "LeRobotDataset",
    task: str,
    hz: float,
    recording_state: RecordingState,
    stop_event: threading.Event,
    image_sizes: Optional[List[Optional[Tuple[int, int]]]] = None,
) -> None:
    period = 1.0 / hz
    frames_in_episode = 0
    skips = {"arms": 0, "cameras": 0, "add_frame": 0}
    last_skip_log = 0.0

    while not stop_event.is_set():
        t_start = time.monotonic()

        if recording_state.is_recording:
            left_leader, left_follower = None, None
            right_leader, right_follower = None, None
            if state_left is not None:
                left_leader, left_follower, _ = state_left.snapshot()
            if state_right is not None:
                right_leader, right_follower, _ = state_right.snapshot()

            left_ready = state_left is None or left_leader is not None
            right_ready = state_right is None or right_leader is not None

            if not (left_ready and right_ready):
                skips["arms"] += 1
            else:
                action_parts, state_parts = [], []
                if state_left is not None:
                    action_parts.append(left_leader.astype(np.float32))
                    state_parts.append(left_follower.astype(np.float32))
                if state_right is not None:
                    action_parts.append(right_leader.astype(np.float32))
                    state_parts.append(right_follower.astype(np.float32))

                frame: dict = {
                    "task": task,
                    "action": np.concatenate(action_parts),
                    "observation.state": np.concatenate(state_parts),
                }
                images_ok = True
                for i, cam in enumerate(cameras):
                    bgr = cam.get_frame()
                    if bgr is None:
                        images_ok = False
                        break
                    target = image_sizes[i] if image_sizes and i < len(image_sizes) else None
                    if target is not None:
                        th, tw = target
                        if bgr.shape[0] != th or bgr.shape[1] != tw:
                            bgr = cv2.resize(bgr, (tw, th))
                    frame[f"observation.images.cam_{i}"] = Image.fromarray(
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    )

                if images_ok or not cameras:
                    try:
                        dataset.add_frame(frame)
                        frames_in_episode += 1
                        recording_state.set_frames_captured(frames_in_episode)
                    except Exception as e:
                        skips["add_frame"] += 1
                        if skips["add_frame"] <= 3:
                            logging.error(f"add_frame failed (frame not counted): {e}")
                else:
                    skips["cameras"] += 1

            if frames_in_episode == 0 and sum(skips.values()) > 0:
                now = time.monotonic()
                if now - last_skip_log >= 5.0:
                    logging.warning(
                        "Recording but no frames saved yet — "
                        f"skipped ticks: arms={skips['arms']}, "
                        f"camera(s) returned None={skips['cameras']}, "
                        f"add_frame errors={skips['add_frame']}"
                    )
                    last_skip_log = now

        if recording_state.take_save_request():
            ep_num = recording_state.next_episode_number
            recording_state.begin_save(ep_num, frames_in_episode)
            if frames_in_episode > 0:
                try:
                    t_save = time.monotonic()
                    save_episode_with_progress(dataset, recording_state)
                    save_s = time.monotonic() - t_save
                    ep_num = recording_state.mark_episode_saved()
                    recording_state.complete_save(
                        ep_num, frames_in_episode, hz, dataset.meta.total_episodes,
                    )
                    logging.info(
                        f"Episode {ep_num} saved "
                        f"({frames_in_episode} frames, {frames_in_episode / hz:.1f}s) "
                        f"— total in dataset: {dataset.meta.total_episodes} "
                        f"[save_episode took {save_s:.1f}s]"
                    )
                except Exception as e:
                    recording_state.fail_save(str(e))
                    logging.error(f"Failed to save episode: {e}")
            else:
                recording_state.discard_empty()
                cam_list = ", ".join(c.device for c in cameras) or "(none)"
                logging.warning(
                    "Empty episode discarded — no frames were written while REC was on. "
                    f"Skipped ticks: arms not ready={skips['arms']}, "
                    f"missing camera frame={skips['cameras']}, add_frame errors={skips['add_frame']}. "
                    f"Active cameras: {cam_list}. "
                    "If resuming a dataset, camera count and devices must match episode 1; "
                    "try `v4l2-ctl --list-devices` and omit --cameras to auto-detect."
                )
            frames_in_episode = 0
            skips = {k: 0 for k in skips}
            last_skip_log = 0.0

        elapsed = time.monotonic() - t_start
        sleep_time = period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ── Dataset features ──────────────────────────────────────────────────────

def build_features(
    arm_labels: List[str],
    cameras: List[CameraReader],
    resolution: Optional[Tuple[int, int]] = None,
) -> dict:
    joint_names = []
    for arm in arm_labels:
        joint_names += [f"{arm}_j{i}" for i in range(NUM_ARM_JOINTS)]
        joint_names.append(f"{arm}_gripper")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
    }
    for i, cam in enumerate(cameras):
        h = resolution[0] if resolution else cam.height
        w = resolution[1] if resolution else cam.width
        features[f"observation.images.cam_{i}"] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return features


# ── Argument parsing ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    teleop_cfg = load_teleop_config()
    parser = argparse.ArgumentParser(
        description="Record YAM bimanual demonstrations in LeRobot format."
    )
    parser.add_argument("--name", required=True,
                        help="Dataset name. Data saved to data/<name>/.")
    parser.add_argument("--task", required=True,
                        help='Task description, e.g. "pick up the block".')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--left", action="store_true", help="Left arm only.")
    group.add_argument("--right", action="store_true", help="Right arm only.")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Recording frequency in Hz (default: 10).")
    parser.add_argument("--visualize", action="store_true",
                        help="Show live camera feeds (grid, native aspect ratio) while recording.")
    parser.add_argument(
        "--cameras", type=int, nargs="+", default=None,
        help="Camera device indices (/dev/videoN), e.g. --cameras 0 2. "
             "Auto-detects all working cameras if not specified.",
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=None, metavar=("H", "W"),
        help="Record at HxW resolution (e.g. --resolution 480 640). "
             f"Default: {DEFAULT_RECORD_RESOLUTION[0]}×{DEFAULT_RECORD_RESOLUTION[1]}. "
             "Use --native-resolution for full camera size.",
    )
    parser.add_argument(
        "--native-resolution", action="store_true",
        help="Record at each camera's native resolution (slow to save if >720p).",
    )
    parser.add_argument(
        "--vcodec", choices=["h264", "hevc", "libsvtav1"], default="h264",
        help="Video codec used when saving episodes (default: h264, fastest to encode). "
             "libsvtav1 yields smaller files but is much slower.",
    )
    parser.add_argument(
        "--reference-dataset", type=str, default="data/wire_insert",
        help="Path to an existing dataset meta to compare format against "
             "(default: data/wire_insert). Pass '' to skip.",
    )
    parser.add_argument(
        "--bilateral_kp", type=float,
        default=teleop_cfg.get("bilateral_kp", 0.2),
        help="Bilateral PD gain factor.",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

    if not LEROBOT_AVAILABLE:
        sys.exit("LeRobot not found. Install: pip install -e lerobot/")

    args = parse_args()
    override_log_level(level=logging.INFO)
    _all_robots.clear()

    if args.native_resolution and args.resolution is not None:
        sys.exit("Use either --resolution or --native-resolution, not both.")
    if args.native_resolution:
        record_resolution = None
    elif args.resolution is not None:
        record_resolution = tuple(args.resolution)
    else:
        record_resolution = DEFAULT_RECORD_RESOLUTION

    if record_resolution is not None:
        logging.info(f"Recording resolution: {record_resolution[0]}×{record_resolution[1]}")
    else:
        logging.info("Recording resolution: native (per camera) — not recommended for ACT training")

    # ── Cameras ──────────────────────────────────────────────────────────
    cameras = open_cameras(args.cameras, capture_resolution=record_resolution)
    logging.info(
        f"{len(cameras)} camera(s) active → "
        + ", ".join(f"cam_{i}={c.device}" for i, c in enumerate(cameras))
    )

    # ── Resolve arms ─────────────────────────────────────────────────────
    logging.info("Resolving arm CAN interfaces...")
    leaders = resolve_arms("leader_arms")
    followers = resolve_arms("follower_arms")
    ensure_can_up({**leaders, **followers})

    if args.left:
        pairs = [("Lleft", "Fleft", PORT_LEFT)]
    elif args.right:
        pairs = [("Lright", "Fright", PORT_RIGHT)]
    else:
        pairs = [("Lleft", "Fleft", PORT_LEFT), ("Lright", "Fright", PORT_RIGHT)]

    arm_labels = [l_key for l_key, _, _ in pairs]

    # ── Dataset ───────────────────────────────────────────────────────────
    root = Path("data") / args.name
    features = build_features(arm_labels, cameras, resolution=record_resolution)
    repo_id = f"yam/{args.name}"
    n_threads = max(4 * len(cameras), 4)

    import json
    import shutil
    _info_path = root / "meta" / "info.json"
    _can_resume = False
    if _info_path.exists():
        try:
            _info = json.loads(_info_path.read_text())
            _can_resume = _info.get("total_episodes", 0) > 0
        except (json.JSONDecodeError, KeyError):
            pass
    if not _can_resume and root.exists():
        shutil.rmtree(root)
        logging.info(f"Removed incomplete dataset at {root}")

    if _can_resume:
        logging.info(f"Resuming existing dataset at {root}")
        dataset = LeRobotDataset(repo_id=repo_id, root=root, vcodec=args.vcodec)
        dataset.start_image_writer(num_threads=n_threads)
        dataset.episode_buffer = dataset.create_episode_buffer()
        n_cams_in_dataset = sum(
            1 for k in dataset.meta.features if k.startswith("observation.images.")
        )
        if n_cams_in_dataset != len(cameras):
            logging.warning(
                f"Dataset was created with {n_cams_in_dataset} camera(s) but "
                f"{len(cameras)} are active now. LeRobot cannot change camera count "
                f"on resume — use a new --name or delete {root} and re-record."
            )
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=int(args.hz),
            features=features,
            robot_type="yam_bimanual",
            root=root,
            use_videos=True,
            image_writer_threads=n_threads,
            vcodec=args.vcodec,
        )
    logging.info(f"Dataset → {root.resolve()}  "
                 f"(episodes so far: {dataset.meta.total_episodes})")
    logging.info(f"State/action dim: {len(features['action']['names'])}, "
                 f"cameras: {len(cameras)}, hz: {args.hz}")

    image_sizes = recording_image_sizes(dataset, cameras, record_resolution)
    logging.info(f"Video codec: {args.vcodec}")
    if args.reference_dataset:
        validate_against_reference(
            features,
            args.hz,
            len(cameras),
            arm_labels,
            Path(args.reference_dataset),
        )
    n_cams_in_dataset = sum(
        1 for k in dataset.meta.features if k.startswith("observation.images.")
    )
    if n_cams_in_dataset != len(cameras):
        logging.warning(
            f"Dataset expects {n_cams_in_dataset} camera(s) but {len(cameras)} are streaming — "
            "episodes will stay empty until this matches. "
            f"Delete {root} and re-record, or fix --cameras."
        )
    for i, sz in enumerate(image_sizes):
        if sz is not None:
            logging.info(f"  cam_{i} record size: {sz[0]}×{sz[1]}")
            if sz[0] * sz[1] > SLOW_SAVE_PIXELS:
                mp = sz[0] * sz[1] / 1e6
                logging.warning(
                    f"  cam_{i} is {mp:.1f} MP/frame — episode saves will be slow "
                    f"(PNG flush + {args.vcodec} video encode). "
                    "Use --resolution 480 640 with a fresh --name for faster saves."
                )

    # ── Shared state + stop event ─────────────────────────────────────────
    arm_states: Dict[str, ArmState] = {l_key: ArmState() for l_key, _, _ in pairs}
    recording_state = RecordingState()
    recording_state.episode_idx = dataset.meta.total_episodes
    stop_event = threading.Event()

    # ── Shutdown handler ──────────────────────────────────────────────────
    _shutting_down = False
    _force_count = 0

    def _shutdown(sig: int, frame: object) -> None:
        nonlocal _shutting_down, _force_count
        if _shutting_down:
            _force_count += 1
            if _force_count >= 2:
                logging.warning("Force exit (data may be incomplete)")
                import os; os._exit(1)
            logging.warning("Press Ctrl-C twice more to force quit without cleanup")
        _shutting_down = True
        logging.info("Shutting down (finalizing dataset)...")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logging.info(
        f"\nRecorder ready — {len(pairs)} arm pair(s), {len(cameras)} camera(s) @ {args.hz} Hz\n"
        f"  Top button    → sync on/off + gravity comp\n"
        f"  Bottom button → start episode / save (repeat for more episodes)\n"
        f"  Ctrl-C        → finalize and quit\n"
    )

    # ── Main loop: optional live visualisation ────────────────────────────
    show_gui = args.visualize and cameras and opencv_gui_available()
    if args.visualize and cameras and not show_gui:
        logging.warning(
            "OpenCV has no GUI support (opencv-python-headless). "
            "Recording continues without a live window.\n"
            "  pip uninstall opencv-python-headless -y && pip install opencv-python\n"
            "  sudo apt install libgtk2.0-dev pkg-config   # Ubuntu/Debian"
        )

    try:
        # ── Follower servers ──────────────────────────────────────────────
        for _, f_key, port in pairs:
            start_follower_server(followers[f_key], port, label=f_key)
        time.sleep(1.0)

        # ── Recording thread ──────────────────────────────────────────────
        rec_thread = threading.Thread(
            target=recording_loop,
            args=(arm_states.get("Lleft"), arm_states.get("Lright"),
                  cameras, dataset, args.task, args.hz, recording_state, stop_event,
                  image_sizes),
            name="recording", daemon=True,
        )
        rec_thread.start()

        # ── Teleop threads ────────────────────────────────────────────────
        for pair_i, (l_key, f_key, port) in enumerate(pairs):
            l_info = leaders[l_key]
            gripper_type = GripperType.from_string_name(l_info.gripper_type)
            robot = get_yam_robot(
                channel=l_info.channel,
                gripper_type=gripper_type,
                zero_gravity_mode=True,
                gravity_comp_factor=0.0,
            )
            _all_robots.append(robot)
            leader = YAMLeaderRobot(robot)
            client = ClientRobot(port)
            t = threading.Thread(
                target=run_leader_follower_loop,
                args=(leader, client, args.bilateral_kp, l_info.gravity_comp_factor,
                      arm_states[l_key], recording_state, f"{l_key}→{f_key}", stop_event),
                kwargs={"handle_recording": pair_i == 0},
                name=f"teleop_{l_key}", daemon=True,
            )
            t.start()
        if len(pairs) > 1:
            logging.info(
                f"Episode start/stop uses bottom button on {pairs[0][0]} only"
            )

        if show_gui:
            cam_labels = [f"cam {c.index}" for c in cameras]
            win = "record.py — Q/ESC to quit"
            with _quiet_stderr():
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            while not stop_event.is_set():
                frames = [c.get_frame() for c in cameras]
                rect = cv2.getWindowImageRect(win)
                disp_w, disp_h = max(rect[2], 640), max(rect[3], 480)
                canvas = tile_frames(frames, cam_labels, disp_w, disp_h)
                draw_recording_overlay(canvas, recording_state)

                cv2.imshow(win, canvas)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    stop_event.set()

            cv2.destroyAllWindows()
        else:
            while not stop_event.is_set():
                time.sleep(0.5)
    except Exception as e:
        logging.error(f"Recorder error: {e}")
        stop_event.set()
    finally:
        # ── Cleanup (always runs, even on crash) ─────────────────────────
        try:
            if recording_state.is_recording:
                logging.info("Saving in-progress episode before exit...")
                dataset.save_episode()
                recording_state.episode_idx += 1
        except Exception as e:
            logging.warning(f"Could not save in-progress episode: {e}")

        try:
            dataset.finalize()
            logging.info(f"Done — {recording_state.episode_idx} episode(s) → {root.resolve()}")
        except Exception as e:
            logging.warning(f"Error finalizing dataset: {e}")

        for cam in cameras:
            try:
                cam.stop()
            except Exception:
                pass
        for robot in reversed(_all_robots):
            try:
                robot.close()
            except Exception as e:
                logging.warning(f"Error closing robot: {e}")
        _all_robots.clear()
        time.sleep(0.5)  # let CAN adapters release before a quick re-run


if __name__ == "__main__":
    main()
