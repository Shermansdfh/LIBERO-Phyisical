# Mass Task Teleop

This folder contains the custom empty-mug mass-sensing task and a keyboard
teleop collector:

```bash
python scripts/weight_dev/mass_tasks/keyboard_teleop_demo.py \
  --camera agentview \
  --out outputs/empty_mug_keyboard_demo.hdf5
```

The script opens the robosuite MuJoCo viewer, records the rollout through
`DataCollectionWrapper`, and writes an HDF5 demo after the task succeeds for
`--success-hold` control steps.

## SSH Display Options

Interactive teleop needs a real OpenGL viewer. Over SSH, use one of these
display routes.

### X11 Forwarding

From your local machine:

```bash
ssh -Y <user>@<workstation>
cd /home/shermanchang/LIBERO-PRO
conda activate libero_pro
python scripts/weight_dev/mass_tasks/keyboard_teleop_demo.py --camera agentview
```

No manual port forwarding is usually needed. SSH creates the display tunnel for
you; on the workstation `echo $DISPLAY` should look like `localhost:10.0`.
Internally, display `:10` maps to TCP port `6010`, display `:11` maps to
`6011`, and so on.

### VNC Or TurboVNC

Use this when X11 forwarding is too slow or OpenGL fails.

On the workstation:

```bash
vncserver :1 -localhost yes -geometry 1600x1000
export DISPLAY=:1
cd /home/shermanchang/LIBERO-PRO
conda activate libero_pro
python scripts/weight_dev/mass_tasks/keyboard_teleop_demo.py --camera agentview
```

From your local machine, tunnel VNC display `:1`:

```bash
ssh -L 5901:localhost:5901 <user>@<workstation>
```

Then connect your VNC viewer to `localhost:5901`. The port is `5900 + display`;
for example, `:2` uses port `5902`.

If you use noVNC, run the web bridge on the workstation:

```bash
websockify --web=/usr/share/novnc 6080 localhost:5901
```

Then tunnel the browser port:

```bash
ssh -L 6080:localhost:6080 <user>@<workstation>
```

Open `http://localhost:6080/vnc.html` locally.

## Mass Reveal

Mass is hidden until an object is gripped and lifted for the configured hold
times. Terminal output updates only when the reveal state changes:

```text
[mass] empty_mug_1=unknown, plain_mug_1=unknown, plain_mug_2=unknown, yellow_mug_1=unknown
[mass] empty_mug_1=0.0004 kg, plain_mug_1=unknown, plain_mug_2=unknown, yellow_mug_1=unknown
```

To also try object-anchored labels in the live viewer:

```bash
python scripts/weight_dev/mass_tasks/keyboard_teleop_demo.py \
  --camera agentview \
  --viewer-mass-labels
```

The labels use MuJoCo viewer markers. If the installed robosuite viewer does
not expose marker labels, the script prints one warning and keeps the terminal
mass reveal working.

## Notes

- The task is defined in `bddl/put_empty_mug_in_basket.bddl`.
- Custom mug assets are registered by importing `libero_mass_sensing.py`.
- Pressing the keyboard device reset/quit control stops collection and does not
  save the demo. Let the task reach success if you want the HDF5 file.
- If `ModuleNotFoundError` appears for `robosuite`, `h5py`, or MuJoCo packages,
  activate the LIBERO conda environment before running the teleop script.
