from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_BDDL_FILE = (
    Path(__file__).resolve().parent / "bddl" / "put_opened_empty_can_in_basket.bddl"
)


def _load_runtime_dependencies() -> dict[str, Any]:
    try:
        from .libero_mass_sensing import Opened_Empty_Can_Mass_Sensing
        from .mass_sensing_utils import MassSensingTracker, object_names_from_problem
        from .object import ensure_libero_import_path
    except ImportError:
        current_dir = Path(__file__).resolve().parent
        if os.fspath(current_dir) not in sys.path:
            sys.path.insert(0, os.fspath(current_dir))

        from libero_mass_sensing import (  # type: ignore[no-redef]
            Opened_Empty_Can_Mass_Sensing,
        )
        from mass_sensing_utils import (  # type: ignore[no-redef]
            MassSensingTracker,
            object_names_from_problem,
        )
        from object import ensure_libero_import_path  # type: ignore[no-redef]

    del Opened_Empty_Can_Mass_Sensing
    ensure_libero_import_path()

    import libero.libero.envs.bddl_utils as BDDLUtils
    from libero.libero.envs import TASK_MAPPING
    from robosuite import load_controller_config
    from robosuite.devices import Keyboard
    from robosuite.utils.input_utils import input2action
    from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

    return {
        "BDDLUtils": BDDLUtils,
        "DataCollectionWrapper": DataCollectionWrapper,
        "Keyboard": Keyboard,
        "MassSensingTracker": MassSensingTracker,
        "TASK_MAPPING": TASK_MAPPING,
        "VisualizationWrapper": VisualizationWrapper,
        "input2action": input2action,
        "load_controller_config": load_controller_config,
        "object_names_from_problem": object_names_from_problem,
    }


class MassPrinter:
    def __init__(self, tracker: Any) -> None:
        self.tracker = tracker
        self._last_printed: dict[str, float | None] | None = None
        self.print(force=True)

    def update(self) -> None:
        self.tracker.update()
        self.print()

    def print(self, *, force: bool = False) -> None:
        masses = self.tracker.masses()
        if force or masses != self._last_printed:
            print(f"[mass] {_format_masses(masses)}", flush=True)
            self._last_printed = masses


class MassViewerLabels:
    def __init__(
        self,
        env: Any,
        tracker: Any,
        *,
        enabled: bool,
        z_offset_m: float,
    ) -> None:
        self.env = env
        self.tracker = tracker
        self.enabled = enabled
        self.z_offset_m = float(z_offset_m)
        self._warned = False

    def draw(self) -> None:
        if not self.enabled:
            return

        add_marker = self._add_marker()
        if add_marker is None:
            self._warn_once(
                "viewer does not expose add_marker; using terminal masses only"
            )
            return

        sim = self.env.sim
        for state in self.tracker.states.values():
            mass = float(state.mass) if state.revealed else None
            label = _format_mass_value(mass)
            pos = np.array(sim.data.xpos[state.root_body_id], dtype=float)
            pos[2] += self.z_offset_m

            try:
                add_marker(
                    pos=pos,
                    size=np.array([0.012, 0.012, 0.012]),
                    rgba=np.array([0.1, 0.75, 0.95, 0.75]),
                    label=label,
                    type=2,
                )
            except TypeError:
                try:
                    add_marker(pos=pos, label=label)
                except Exception as exc:
                    self._warn_once(f"viewer mass labels failed: {exc}")
                    self.enabled = False
                    return
            except Exception as exc:
                self._warn_once(f"viewer mass labels failed: {exc}")
                self.enabled = False
                return

    def _add_marker(self) -> Any | None:
        viewer = getattr(self.env, "viewer", None)
        for candidate in (viewer, getattr(viewer, "viewer", None)):
            add_marker = getattr(candidate, "add_marker", None)
            if callable(add_marker):
                return add_marker
        return None

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        print(f"[mass-labels] {message}", flush=True)
        self._warned = True


def _format_mass_value(mass: float | None) -> str:
    return "unknown" if mass is None else f"{mass:.4f} kg"


def _format_masses(masses: dict[str, float | None]) -> str:
    return ", ".join(
        f"{name}={_format_mass_value(mass)}" for name, mass in sorted(masses.items())
    )


def _add_viewer_callback(
    viewer: Any,
    method_name: str,
    callback: Any,
    *,
    required: bool = False,
) -> bool:
    method = getattr(viewer, method_name, None)
    if method is None:
        if required:
            raise AttributeError(
                f"{type(viewer).__name__} does not expose {method_name}"
            )
        return False

    try:
        method("any", callback)
    except TypeError:
        method(callback)
    return True


def _register_keyboard_callbacks(viewer: Any, device: Any) -> None:
    _add_viewer_callback(
        viewer,
        "add_keypress_callback",
        device.on_press,
        required=True,
    )
    _add_viewer_callback(viewer, "add_keyup_callback", device.on_release)
    _add_viewer_callback(viewer, "add_keyrepeat_callback", device.on_press)


def _looks_like_wsl() -> bool:
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def _check_display_environment() -> None:
    if sys.platform != "linux":
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return

    hint = (
        "No Linux display was detected. Keyboard teleop needs a GUI display. "
        "In WSL, run from WSLg or start an X server and export DISPLAY before "
        "launching this script."
    )
    if _looks_like_wsl():
        hint += (
            " If DISPLAY is set but OpenCV later reports a Qt xcb plugin error, "
            "install the xcb runtime packages listed in README.md."
        )
    raise RuntimeError(hint)


def _save_hdf5(
    directory: str | Path,
    out_path: str | Path,
    env_info: dict[str, Any],
    bddl_path: str | Path,
) -> None:
    import h5py

    directory = Path(directory)
    bddl_path = Path(bddl_path)

    with h5py.File(out_path, "w") as h5:
        group = h5.create_group("data")
        now = dt.datetime.now()
        group.attrs["date"] = now.strftime("%Y-%m-%d")
        group.attrs["time"] = now.strftime("%H:%M:%S")
        group.attrs["env_info"] = json.dumps(env_info)
        group.attrs["bddl_file_name"] = os.fspath(bddl_path)
        group.attrs["bddl_file_content"] = bddl_path.read_text()

        demo_id = 0
        for ep_directory in sorted(directory.iterdir()):
            if not ep_directory.is_dir():
                continue

            state_paths = sorted(glob(os.fspath(ep_directory / "state_*.npz")))
            states: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            for state_file in state_paths:
                data = np.load(state_file, allow_pickle=True)
                states.extend(data["states"])
                actions.extend(info["actions"] for info in data["action_infos"])

            if len(states) < 2 or not actions:
                continue

            states = states[:-1]
            actions = actions[: len(states)]
            demo_id += 1

            demo = group.create_group(f"demo_{demo_id}")
            model_path = ep_directory / "model.xml"
            if model_path.exists():
                demo.attrs["model_file"] = model_path.read_text()
            demo.create_dataset("states", data=np.asarray(states))
            demo.create_dataset("actions", data=np.asarray(actions))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a keyboard teleop demo for the can-of-icetea mass task."
    )
    parser.add_argument("--bddl-file", default=os.fspath(DEFAULT_BDDL_FILE))
    parser.add_argument("--out", default="outputs/opened_empty_can_keyboard_demo.hdf5")
    parser.add_argument("--tmp-dir", default="outputs/opened_empty_can_keyboard_demo_raw")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument(
        "--renderer",
        default="mjviewer",
        help="On-screen renderer. Use mjviewer for GLFW; mujoco uses OpenCV.",
    )
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--pos-sensitivity", type=float, default=1.5)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--success-hold", type=int, default=10)
    parser.add_argument("--grip-duration", type=float, default=0.5)
    parser.add_argument("--lift-duration", type=float, default=0.5)
    parser.add_argument("--lift-threshold", type=float, default=0.03)
    parser.add_argument(
        "--viewer-mass-labels",
        action="store_true",
        help="Try to draw object-anchored mass labels in the MuJoCo viewer.",
    )
    parser.add_argument("--label-z-offset", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _check_display_environment()
    deps = _load_runtime_dependencies()
    BDDLUtils = deps["BDDLUtils"]
    DataCollectionWrapper = deps["DataCollectionWrapper"]
    Keyboard = deps["Keyboard"]
    MassSensingTracker = deps["MassSensingTracker"]
    TASK_MAPPING = deps["TASK_MAPPING"]
    VisualizationWrapper = deps["VisualizationWrapper"]
    input2action = deps["input2action"]
    load_controller_config = deps["load_controller_config"]
    object_names_from_problem = deps["object_names_from_problem"]

    bddl_path = Path(args.bddl_file).expanduser().resolve()
    problem_info = BDDLUtils.get_problem_info(os.fspath(bddl_path))
    parsed_problem = BDDLUtils.robosuite_parse_problem(os.fspath(bddl_path))
    problem_name = problem_info["problem_name"]
    controller_config = load_controller_config(default_controller=args.controller)
    config = {"robots": ["Panda"], "controller_configs": controller_config}

    print(problem_info["language_instruction"], flush=True)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=os.fspath(bddl_path),
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        renderer=args.renderer,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env = VisualizationWrapper(env)
    env = DataCollectionWrapper(env, args.tmp_dir)

    device = Keyboard(
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
    )
    _register_keyboard_callbacks(env.viewer, device)

    env.reset()

    tracker = MassSensingTracker(
        env,
        object_names_from_problem(parsed_problem),
        grip_duration_s=args.grip_duration,
        lift_duration_s=args.lift_duration,
        lift_threshold_m=args.lift_threshold,
    )
    mass_printer = MassPrinter(tracker)
    mass_labels = MassViewerLabels(
        env,
        tracker,
        enabled=args.viewer_mass_labels,
        z_offset_m=args.label_z_offset,
    )
    mass_labels.draw()
    env.render()
    device.start_control()

    success_hold = -1
    saved = True
    for _ in range(args.max_steps):
        action, _ = input2action(
            device=device,
            robot=env.robots[0],
            active_arm="right",
            env_configuration="single-arm-opposed",
        )
        if action is None:
            saved = False
            print("Reset requested; demo will not be saved.", flush=True)
            break

        env.step(action)
        tracker.update()
        mass_printer.print()
        mass_labels.draw()
        env.render()

        if success_hold == 0:
            break
        if env._check_success():
            success_hold = success_hold - 1 if success_hold > 0 else args.success_hold
        else:
            success_hold = -1

    env.close()
    if saved:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        _save_hdf5(args.tmp_dir, args.out, config, bddl_path)
        print(f"Saved demo to {args.out}", flush=True)


if __name__ == "__main__":
    main()
