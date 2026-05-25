# YAM Bimanual Food Manipulation

Low-level control and data collection stack for bimanual food manipulation using [YAM](https://i2rt.com/) robot arms and [LeRobot](https://github.com/huggingface/lerobot) for policy training.

## Hardware


| Component                    | Qty | Role                               |
| ---------------------------- | --- | ---------------------------------- |
| YAM arm + teaching handle    | 2   | Leader — you hold and move these   |
| YAM arm + crank_4310 gripper | 2   | Follower — copies your motion      |
| CANable USB-CAN adapter      | 4   | One per arm                        |
| RGB USB cameras              | 1–4 | Scene observation for policy input |


CAN channel names (can0, can1, …) are resolved automatically at runtime from USB serial numbers — no manual renaming needed. See `[config/arms.yaml](./config/arms.yaml)`.

---

## Install

### 1. Clone

```bash
git clone --recurse-submodules git@github.com:jefferzn2001/FoodManipLowLevel.git
cd FoodManipLowLevel
```

Already cloned without submodules?

```bash
git submodule update --init --depth 1
```

### 2. Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.11
source .venv/bin/activate
sudo apt update && sudo apt install build-essential python3-dev linux-headers-$(uname -r)
uv pip install -e .
```

### 3. LeRobot (for recording + training)

```bash
pip install -e lerobot/
```

### 4. CAN and udev (one-time)

**Option A — Install device rules (add user to plugdev/video, optional udev rules):**

```bash
cd /path/to/FoodManipLowLevel
sudo sh devices/install_devices.sh
```

If you don’t have a `devices/rules/*.rules` directory, the script skips udev rule install and only adds your user to `plugdev` and `video`.

**Option B — Install CAN rename rule (optional):**

```bash
sudo bash scripts/install_udev_rules.sh
```

**Bringing CAN up:** `teleop.py` and `record.py` bring up configured CAN interfaces when they start. After replugging adapters, run:

```bash
sudo bash scripts/setup_devices.sh
```

**Auto-setup on plug-in (recommended):** install a udev hook so CAN is brought up whenever a new CAN adapter appears:

```bash
sudo bash scripts/install_hotplug_setup.sh
```

This installs `/usr/local/bin/foodmanip-setup-devices` and reloads udev. You can still run `sudo foodmanip-setup-devices` manually anytime.

---

## Usage

### Check cameras

```bash
python scripts/cam.py
```

Shows all detected cameras tiled in a fullscreen window. Press Q or ESC to quit.

### Test arms — zero-gravity mode

```bash
python scripts/zero_grav.py            # both leaders
python scripts/zero_grav.py --arm Lleft
python scripts/zero_grav.py --arm Lright
```

### Teleop only (no recording)

```bash
python scripts/teleop.py              # both pairs: Lleft→Fleft + Lright→Fright
python scripts/teleop.py --left       # left pair only
python scripts/teleop.py --right      # right pair only
```

**Teaching handle controls:**

- **Top button** — press once to start sync (gravity comp on, bilateral feedback on); press again to stop
- Sync must be active before the follower mirrors your motion

**`fail to communicate with motor N` during startup**

Usually not loose wiring if motors 1–3 on that bus succeed. Common causes:

1. **`deploy.py` / another script still running** — only one process may open each CAN interface. Stop deploy, then retry teleop.
2. **Stale CAN socket after a crash** — `sudo ip link set can0 down && sudo ip link set can0 up` (repeat for `can1`, …), or unplug/replug the USB-CAN adapter.
3. **Bus timing** — startup now retries more and flushes the RX buffer; if motor 4 still fails, power-cycle that arm and retry once.

The motor firmware `timeout` register (see `i2rt/motor_config_tool/set_timeout.py`) is separate from host receive timeouts.

### Record demonstrations

```bash
python scripts/record.py --name pick_place --task "pick up the block"
python scripts/record.py --name pick_place --task "..." --visualize   # live camera view (needs opencv-python, not headless)
python scripts/record.py --name pick_place --task "..." --cameras 4 2  # same USB order as wire_insert / deploy
python scripts/record.py --name pick_place --task "..." --hz 15       # higher frequency
python scripts/record.py --name pick_place --task "..." --left        # one arm only
```

**Match `data/wire_insert` format (for ACT training):** defaults are already aligned — **480×640**, **10 Hz**, **2 cameras**, **h264**, **14-DOF** state/action (`Lleft_*` then `Lright_*`). On startup, `record.py` compares against `data/wire_insert/meta/info.json` and logs `Format check vs reference: OK` or warnings. Explicit:

```bash
python scripts/record.py --name pick_place --task "pick up the block" \
  --cameras 4 2 --resolution 480 640 --hz 10 --vcodec h264
```

Do **not** use `--native-resolution` unless you intend to retrain at that size. `--resolution` and `--hz` can be omitted (they are the defaults).

**Teaching handle controls during recording:**

- **Top button** — sync on/off (same as teleop)
- **Bottom button** — press once to **start** episode *N*, press again to **save** it; repeat for episode *N+1*, *N+2*, … without restarting the script
- **Ctrl-C** — exit the recorder (keeps all saved episodes)

**Multiple episodes in one session:** leave `record.py` running. Sync with the top button, then use the bottom button as start/stop for each demo. The overlay shows `REC ep N` while recording.

**More episodes later:** run the same command again with the same `--name` — the dataset resumes and new episodes append (`data/pick_place/meta/info.json` tracks the count).

**If re-starting `record.py` fails on CAN** (`fail to communicate with motor …`): wait ~2 s after the previous run exits, or run `sudo bash scripts/setup_devices.sh`, then retry. Recent versions shut down the CAN bus cleanly on exit to avoid this.

Data is saved to `data/<name>/` in LeRobot v3 format (parquet + mp4), ready for training.

**`--visualize` shows no window:** LeRobot installs `opencv-python-headless`, which cannot open GUI windows. Recording still works; to enable the live view, replace headless OpenCV and install GTK dev packages:

```bash
uv pip uninstall opencv-python opencv-python-headless -y
uv pip install opencv-python
sudo apt install libgtk2.0-dev pkg-config   # Ubuntu/Debian
```

If you see `cv2 has no VideoCapture`, the OpenCV wheel did not install correctly — run the `uv pip` commands above (do not mix a broken partial install with `pip` and `uv` in the same venv).

---

## Training

No modifications to LeRobot needed — the recorded dataset is directly compatible.

```bash
# ACT (recommended starting point — most sample-efficient)
python lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=yam/pick_place \
  --dataset.root=data/pick_place \
  --policy.type=act \
  --output_dir=outputs/act_pick_place

# Diffusion Policy
python lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=yam/pick_place \
  --dataset.root=data/pick_place \
  --policy.type=diffusion \
  --output_dir=outputs/dp_pick_place
```

**How many episodes?**

- ACT: 50–200 (start here — designed for bimanual, very sample-efficient)
- Diffusion Policy: 100–500 (better generalisation, needs more data)

Start with 50 clean ACT demonstrations. Keep episodes short (5–15 s) and consistent.

### Deploy LeRobot policy (ACT / Diffusion)

```bash
python scripts/deploy.py --policy outputs/act_blockincup/checkpoints/last/pretrained_model
python scripts/deploy.py --policy outputs/act_blockincup/checkpoints/last/pretrained_model --visualize
python scripts/deploy.py --policy outputs/act_wire_insert/checkpoints/last/pretrained_model --cameras 4 2 --n-action-steps 1
```

Stop `lerobot_train.py` before deploy if GPU memory is tight. Use the same `--cameras` indices/order as recording.

On **Ctrl-C**, followers blend to the pose in `config/deploy_home.yaml` over 3 s (override with `--home-pose`, `--home-duration`, or `--no-home-on-exit`).

**Deploy disconnects / `motor timeout` / `error: disabled`**

Usually the right or left **follower** arm (e.g. `can1`) loses CAN sync when the policy starts:

1. Do not run `teleop.py` or `record.py` while deploy is active — only one process may own each CAN adapter.
2. Start with the arms in a pose similar to your demos (large first-step jumps can trip motor protection).
3. Action ramp is **off** by default (`--ramp-steps 0`). A long ramp can look frozen when the policy already predicts a hold pose.
4. Cameras are opened near policy resolution (480×640) so a 3K sensor does not stall the control loop.

Each policy step logs **state**, **pred** (network output), and **cmd** (sent to motors, after ramp). Use `--log-interval 5` for less noise, or `--log-interval 0` to disable.

**ACT `pred` looks frozen:** training uses `n_action_steps=100`, so the policy replays one 100-step action chunk open-loop (~10 s at 10 Hz) without re-reading cameras. Deploy defaults to `--n-action-steps 10` (replan every 1 s). Logs show `chunk_replan=yes` only when a new chunk is planned; `queue_left` counts remaining steps. Use `--n-action-steps 1` for fully reactive control.

**Arm stops moving but logs keep going:** often `pred ≈ state` (policy says “hold here”) or a follower motor loop crashed — deploy now pauses on `motor control loop stopped` and warns after 30 steps of tiny commands. Chunk size 2 only replans every 0.2 s at 10 Hz; it does not fix a hold command.

### Deploy VLA policy (Dexbotic OFT)

The VLA (Vision-Language-Action) model is goal-conditioned: it sees the current camera frame + a goal image and predicts 16 bimanual delta-action steps. This requires two terminals — one for the GPU inference server, one for the robot control loop.

#### VLA environment setup (one-time)

```bash
conda create -n dexbotic python=3.10 -y
conda activate dexbotic

# Install PyTorch with Blackwell GPU support (sm_120)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install Dexbotic
cd food_manipulation
pip install -e .
pip install ipdb

# Verify GPU
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### Step 1 — Start the VLA inference server (Terminal 1)

```bash
conda activate dexbotic
cd ~/Desktop/FoodManipLowLevel/food_manipulation
CUDA_VISIBLE_DEVICES=0 python playground/custom/blockincup_oft.py --task inference
```

Wait for `Model loaded successfully` and `Running on http://127.0.0.1:7891`.

#### Step 2 — Capture a goal image (Terminal 2)

Set up the scene as the **desired end state** (e.g. block inside cup), then:

```bash
cd ~/Desktop/FoodManipLowLevel
source .venv/bin/activate
python -c "import cv2; cap=cv2.VideoCapture('/dev/video0',cv2.CAP_V4L2); _, f=cap.read(); cv2.imwrite('goal.png',f); cap.release(); print('Saved goal.png')"
eog goal.png   # verify the image
```

Then reset the scene to the starting state (e.g. block outside cup).

#### Step 3 — Run the VLA deployment (Terminal 2)

```bash
cd ~/Desktop/FoodManipLowLevel
source .venv/bin/activate
python scripts/deploy_vla.py --goal_image goal.png              # no camera view
python scripts/deploy_vla.py --goal_image goal.png --visualize  # with live camera view
```

**Controls:**

- Robot starts **paused** — press **top button** on leader arm to start
- Press **top button** again → **e-stop** (pause)
- Press **top button** again → resume
- **Ctrl-C** → stop and exit

The model checkpoint is at `food_manipulation/user_checkpoints/dexbotic/blockincup_oft/checkpoint-725/` (downloaded from `ymdou/blockincup_oft`).

---

## Configuration — `config/arms.yaml`

```yaml
leader_arms:
  Lleft:
    usb_serial: "canable.io_canable2_gs_usb_00330052594E501820313332"
    gripper_type: "yam_teaching_handle"
    gravity_comp_factor: 1.2   # increase if arm feels too heavy to move
  Lright:
    usb_serial: "canable.io_canable2_gs_usb_005F0056594E501820313332"
    gripper_type: "yam_teaching_handle"
    gravity_comp_factor: 1.2

follower_arms:
  Fleft:
    usb_serial: "canable.io_canable2_gs_usb_002A0064594E501820313332"
    gripper_type: "crank_4310"
  Fright:
    usb_serial: "canable.io_canable2_gs_usb_004C0053594E501820313332"
    gripper_type: "crank_4310"

teleop:
  bilateral_kp: 0.2   # how much the leader resists follower (0.1–0.3)
```

To find USB serials for new adapters:

```bash
udevadm info -q property -p /sys/class/net/can0 | grep ID_SERIAL
```

---

## Project Structure

```
FoodManipLowLevel/
├── config/
│   └── arms.yaml               # CAN serial → arm mapping, gains, bilateral_kp
├── scripts/
│   ├── teleop.py               # Bimanual leader-follower teleop
│   ├── record.py               # Record demonstrations → LeRobot dataset
│   ├── cam.py                  # Live multi-camera fullscreen viewer
│   ├── zero_grav.py            # Zero-gravity mode for arm testing
│   └── resolve_leader_can.py   # USB serial → CAN channel resolver
├── i2rt/                       # YAM robot SDK
│   ├── robots/                 # MotorChainRobot, get_robot
│   ├── motor_drivers/          # CAN / DM motor drivers
│   ├── robot_models/           # MuJoCo XMLs for gravity compensation
│   └── utils/
├── data/                       # Recorded datasets (gitignored)
│   └── <name>/                 # LeRobot v3 format
├── outputs/                    # Training checkpoints (gitignored)
├── lerobot/                    # HuggingFace LeRobot (submodule)
└── devices/
    └── install_devices.sh      # udev rule installer for CAN auto-start
```

---

## Gripper Types


| Name                  | Description                                         |
| --------------------- | --------------------------------------------------- |
| `yam_teaching_handle` | Teaching handle (trigger + 2 buttons) — leader arms |
| `crank_4310`          | Zero-linkage crank gripper — follower arms          |
| `linear_3507`         | Linear gripper with DM3507 (requires calibration)   |
| `linear_4310`         | Linear gripper with DM4310                          |
| `no_gripper`          | No gripper                                          |


---

## Advanced

- Set persistent CAN IDs: `python i2rt/motor_config_tool/set_zero.py --channel can0 --motor_id 1`
- Motor timeout: `python i2rt/motor_config_tool/set_timeout.py --channel can0`
- Full YAM docs: [I2RT upstream](https://github.com/i2rt-robotics/i2rt)

## Acknowledgments

- [I2RT](https://i2rt.com/) — YAM robot arms and SDK
- [LeRobot](https://github.com/huggingface/lerobot) — Policy training framework
- [GELLO](https://github.com/wuphilipp/gello_software) — Teleop inspiration

