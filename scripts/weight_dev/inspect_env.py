import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path


def make_env(task_suite_name="libero_10", task_id=0, init_state_id=0):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    print("Task:", task.name)
    print("Language:", task.language)
    print("BDDL:", task_bddl_file)

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    obs = env.reset()

    init_states = task_suite.get_task_init_states(task_id)
    obs = env.set_init_state(init_states[init_state_id])

    return env, obs


def get_task_env(env):
    """Return the underlying robosuite task env behind LIBERO's wrapper."""
    return getattr(env, "env", env)


def main():
    env, obs = make_env()

    try:
        task_env = get_task_env(env)

        print("\n=== Observation keys ===")
        for k in obs.keys():
            print(k, np.shape(obs[k]))

        print("\n=== Objects ===")
        print("objects_dict keys:", list(task_env.objects_dict.keys()))
        print("fixtures_dict keys:", list(task_env.fixtures_dict.keys()))
        print("obj_body_id:", task_env.obj_body_id)

        print("\n=== Object positions and masses ===")
        for obj_name in task_env.objects_dict.keys():
            body_id = task_env.obj_body_id[obj_name]
            pos = env.sim.data.body_xpos[body_id]
            mass = env.sim.model.body_mass[body_id]
            print(obj_name, "body_id=", body_id, "pos=", pos, "mass=", mass)
    finally:
        env.close()


if __name__ == "__main__":
    main()
