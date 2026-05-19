# Mass Task Teleop

Local keyboard teleop for the custom can-of-icetea mass-sensing task.

## Run

From the repo root:

```bash
conda activate libero_pro
python scripts/weight_dev/mass_tasks/keyboard_teleop_demo.py \
  --camera agentview \
  --out outputs/opened_empty_can_keyboard_demo.hdf5
```

This opens the robosuite on-screen viewer, lets you control the Panda arm
with the keyboard, and saves the demo after the task succeeds.

On newer robosuite builds, `mjviewer` is not a valid renderer name. This script
defaults to `--renderer mujoco`, and also treats `--renderer mjviewer` as a
legacy alias for `mujoco`.

## WSL GUI Note

Keyboard teleop needs a working Linux GUI. On WSL, use WSLg or an X server. If
OpenCV crashes with a Qt `xcb` plugin error, install the missing XCB runtime
libraries:

```bash
sudo apt update
sudo apt install -y \
  libxcb-xinerama0 \
  libxcb-cursor0 \
  libxkbcommon-x11-0 \
  libxcb-icccm4 \
  libxcb-image0 \
  libxcb-keysyms1 \
  libxcb-randr0 \
  libxcb-render-util0
```

Then reopen the terminal and rerun the teleop command. Check that a display is
visible to WSL with:

```bash
echo $DISPLAY
```

## Mass Reveal

Mass starts hidden and is revealed after you grip and lift an object for the
configured hold time:

```text
[mass] opened_empty_can_1=unknown, opened_light_can_1=unknown, unopened_can_1=unknown, unopened_can_2=unknown
[mass] opened_empty_can_1=0.0350 kg, opened_light_can_1=unknown, unopened_can_1=unknown, unopened_can_2=unknown
```

To also try labels above objects in the viewer:

```bash
python scripts/weight_dev/mass_tasks/keyboard_teleop_demo.py --viewer-mass-labels
```

If viewer labels are unsupported in your robosuite build, terminal mass output
still works.

## Useful Args

```bash
--bddl-file        task BDDL path
--camera           viewer camera, default agentview
--renderer         on-screen renderer, default mujoco
--max-steps        collection step limit
--grip-duration    seconds of gripper contact before reveal
--lift-duration    seconds lifted before reveal
--lift-threshold   lift distance in meters before reveal
--tmp-dir          raw DataCollectionWrapper output
--out              final HDF5 demo path
```

Task files live in:

```text
scripts/weight_dev/mass_tasks/bddl/
scripts/weight_dev/mass_tasks/custom_assets/
```
