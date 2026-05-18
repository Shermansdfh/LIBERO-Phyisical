import argparse
import os
import sys
import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path

from scripts.weight_dev.weight_wrapper import WeightAwareWrapper


ROLLOUT_PATH = os.path.join("scripts", "weight_dev", "rollout.gif")


def make_base_env(task_suite_name="libero_10", task_id=0, init_state_id=0):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=128,
        camera_widths=128,
    )

    env.seed(0)
    env.reset()

    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[init_state_id])

    return env


def get_task_spec(task_suite_name="libero_10", task_id=0):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    return task_suite, task, task_bddl_file


def make_teleop_env(
    task_suite_name="libero_10",
    task_id=0,
    init_state_id=0,
    camera="agentview",
    controller="OSC_POSE",
):
    import robosuite as suite
    from robosuite import load_controller_config
    from robosuite.wrappers import VisualizationWrapper

    import libero.libero.envs.bddl_utils as BDDLUtils
    from libero.libero.envs import TASK_MAPPING

    task_suite, task, task_bddl_file = get_task_spec(task_suite_name, task_id)
    problem_info = BDDLUtils.get_problem_info(task_bddl_file)
    problem_name = problem_info["problem_name"]

    config = {
        "robots": ["Panda"],
        "controller_configs": load_controller_config(default_controller=controller),
    }

    if "TwoArm" in problem_name:
        config["env_configuration"] = "single-arm-opposed"

    print("Task:", task.name)
    print("Language:", task.language)
    print("BDDL:", task_bddl_file)
    print("robosuite:", suite.__version__)

    env = TASK_MAPPING[problem_name](
        bddl_file_name=task_bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    env = VisualizationWrapper(env)
    env.seed(0)

    init_states = task_suite.get_task_init_states(task_id)
    return env, init_states[init_state_id]


def get_task_env(env):
    return getattr(env, "env", env)


def teleport_object_up(env, obj_name, dz=0.12):
    task_env = get_task_env(env)
    body = task_env.get_object(obj_name)
    joint_name = body.joints[-1]

    qpos = task_env.sim.data.get_joint_qpos(joint_name).copy()
    qpos[2] += dz
    task_env.sim.data.set_joint_qpos(joint_name, qpos)
    task_env.sim.forward()


def dummy_contact_with_eef(env, obj_name, obs):
    """
    Teleport the object to the observed EEF position.

    The wrapper's event is observation-based:
        lifted and ||robot0_eef_pos - <obj>_pos|| < threshold
    so this creates a synthetic near-EFF contact for testing that path.
    """
    task_env = get_task_env(env)
    body = task_env.get_object(obj_name)
    joint_name = body.joints[-1]

    target_pos = obs["robot0_eef_pos"].copy()

    qpos = task_env.sim.data.get_joint_qpos(joint_name).copy()
    qpos[:3] = target_pos
    task_env.sim.data.set_joint_qpos(joint_name, qpos)
    task_env.sim.forward()


def add_frame(frames, obs, camera_key="agentview_image"):
    frame = np.flipud(obs[camera_key])
    frames.append(Image.fromarray(frame.astype(np.uint8)))


def save_rollout(frames, path=ROLLOUT_PATH, fps=10):
    if not frames:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print("saved rollout:", path)


def make_device(env, args):
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
        return device

    if args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        return SpaceMouse(
            args.vendor_id,
            args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )

    raise ValueError("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")


def run_teleop(args):
    from robosuite.utils.input_utils import input2action

    base_env, init_state = make_teleop_env(
        task_suite_name=args.task_suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        camera=args.camera,
        controller=args.controller,
    )

    env = WeightAwareWrapper(
        base_env,
        lift_threshold=args.lift_threshold,
        hold_steps=args.hold_steps,
    )

    try:
        obs = env.reset()
        obs = env.set_init_state(init_state)
        base_env.render()

        device = make_device(base_env, args)
        device.start_control()

        print("Initial weight_state:", obs["weight_state"])
        print("Teleop running. Reset/quit from the input device to stop.")

        last_known = dict(env.weight_known)
        for t in range(args.max_steps):
            active_robot = base_env.robots[0]
            action, _ = input2action(
                device=device,
                robot=active_robot,
                active_arm=args.arm,
                env_configuration=args.config,
            )

            if action is None:
                print("Teleop stopped by device reset.")
                break

            obs, reward, done, info = env.step(action)
            base_env.render()

            newly_known = [
                name
                for name, known in info["weight_known"].items()
                if known and not last_known.get(name, False)
            ]
            if newly_known or t % args.print_every == 0:
                print("step", t)
                print("weight_state:", obs["weight_state"])
                print("weight_known:", info["weight_known"])
                print("weight_estimate:", info["weight_estimate"])
                print("weight_hold_counter:", info["weight_hold_counter"])

            last_known = dict(info["weight_known"])
    finally:
        env.close()


def run_synthetic(args):
    base_env = make_base_env()

    env = WeightAwareWrapper(
        base_env,
        lift_threshold=args.lift_threshold,
        hold_steps=args.hold_steps,
    )
    frames = []

    try:
        obs = env.reset()
        add_frame(frames, obs)
        print("weight_state at reset:", obs["weight_state"])

        target_obj = env.object_names[0]
        print("Testing fake lift on:", target_obj)

        for t in range(20):
            if 3 <= t < 3 + env.hold_steps + 2:
                dummy_contact_with_eef(base_env, target_obj, obs)
                env.initial_z[target_obj] = (
                    float(obs["robot0_eef_pos"][2]) - env.lift_threshold - 0.02
                )

            obs, reward, done, info = env.step(np.zeros(7))
            add_frame(frames, obs)

            print(
                t,
                "known=", info["weight_known"][target_obj],
                "est=", info["weight_estimate"][target_obj],
                "counter=", info["weight_hold_counter"][target_obj],
            )

        dummy_action = np.zeros(7)

        for t in range(20):
            obs, reward, done, info = env.step(dummy_action)
            add_frame(frames, obs)

            if t % 5 == 0:
                print("step", t)
                print("weight_state:", obs["weight_state"])
                print("weight_known:", info["weight_known"])
                print("weight_estimate:", info["weight_estimate"])
        save_rollout(frames)
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("synthetic", "teleop"),
        default="synthetic",
        help="synthetic tests the wrapper with teleportation; teleop uses keyboard/spacemouse.",
    )
    parser.add_argument("--task-suite", type=str, default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--lift-threshold", type=float, default=0.08)
    parser.add_argument("--hold-steps", type=int, default=10)

    parser.add_argument("--device", choices=("keyboard", "spacemouse"), default="keyboard")
    parser.add_argument("--camera", type=str, default="agentview")
    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--config", type=str, default="single-arm-opposed")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument("--pos-sensitivity", type=float, default=1.5)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "teleop":
        run_teleop(args)
    else:
        run_synthetic(args)


if __name__ == "__main__":
    main()
