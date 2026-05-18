# Mass Task Teleop

Local keyboard teleop for the custom empty-mug mass-sensing task.

## Run

From the repo root:

```bash
conda activate libero_pro
python scripts/weight_dev/mass_tasks/keyboard_teleop_demo.py \
  --camera agentview \
  --out outputs/empty_mug_keyboard_demo.hdf5
```

This opens the robosuite MuJoCo viewer, lets you control the Panda arm with the
keyboard, and saves the demo after the task succeeds.

## Mass Reveal

Mass starts hidden and is revealed after you grip and lift an object for the
configured hold time:

```text
[mass] empty_mug_1=unknown, plain_mug_1=unknown, plain_mug_2=unknown, yellow_mug_1=unknown
[mass] empty_mug_1=0.0004 kg, plain_mug_1=unknown, plain_mug_2=unknown, yellow_mug_1=unknown
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
