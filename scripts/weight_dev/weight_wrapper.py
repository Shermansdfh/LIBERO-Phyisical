import numpy as np


class WeightAwareWrapper:
    """
    Wrapper around a LIBERO env.

    Goal:
    - remember initial object heights
    - detect when object is lifted above threshold
    - expose a weight_state vector in obs
    - expose debug information in info
    """

    def __init__(
        self,
        env,
        object_mass_table=None,
        lift_threshold=0.08,
        hold_steps=10,
        noise_std=0.03,
    ):
        self.env = env
        self.object_mass_table = object_mass_table or {}
        self.lift_threshold = lift_threshold
        self.hold_steps = hold_steps
        self.noise_std = noise_std

        self.initial_z = {}
        self.hold_counter = {}
        self.weight_known = {}
        self.weight_estimate = {}

    @property
    def task_env(self):
        """Underlying LIBERO task env; OffScreenRenderEnv keeps it at .env."""
        return getattr(self.env, "env", self.env)

    @property
    def object_names(self):
        return list(self.task_env.objects_dict.keys())

    def reset(self):
        obs = self.env.reset()
        self.latest_obs = obs
        self._reset_memory()
        return self._augment_obs(obs)

    def set_init_state(self, init_state):
        if hasattr(self.env, "set_init_state"):
            result = self.env.set_init_state(init_state)
        else:
            self.env.sim.set_state_from_flattened(init_state)
            self.env.sim.forward()
            if hasattr(self.env, "_post_process"):
                self.env._post_process()
            if hasattr(self.env, "_update_observables"):
                self.env._update_observables(force=True)
            result = self.env._get_observations()

        self.latest_obs = result
        self._reset_memory()
        return self._augment_obs(result)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.latest_obs = obs

        for obj_name in self.object_names:
            lifted = self._is_lifted(obj_name)
            near_eef = self._is_near_eef(obj_name)
            event = lifted and near_eef

            if event:
                self.hold_counter[obj_name] += 1
            else:
                self.hold_counter[obj_name] = 0

            if self.hold_counter[obj_name] >= self.hold_steps:
                self.weight_known[obj_name] = True
                true_mass = self._get_true_mass(obj_name)
                estimate = true_mass + np.random.normal(0.0, self.noise_std)
                self.weight_estimate[obj_name] = float(max(0.0, estimate))

        info["weight_known"] = dict(self.weight_known)
        info["weight_estimate"] = dict(self.weight_estimate)
        info["weight_hold_counter"] = dict(self.hold_counter)

        obs = self._augment_obs(obs)
        return obs, reward, done, info

    def close(self):
        return self.env.close()

    def _reset_memory(self):
        self.initial_z = {}
        self.hold_counter = {}
        self.weight_known = {}
        self.weight_estimate = {}

        task_env = self.task_env
        for obj_name in self.object_names:
            body_id = task_env.obj_body_id[obj_name]
            z = float(self.env.sim.data.body_xpos[body_id][2])

            self.initial_z[obj_name] = z
            self.hold_counter[obj_name] = 0
            self.weight_known[obj_name] = False
            self.weight_estimate[obj_name] = 0.0

    def _is_lifted(self, obj_name):
        body_id = self.task_env.obj_body_id[obj_name]
        current_z = float(self.env.sim.data.body_xpos[body_id][2])
        return current_z > self.initial_z[obj_name] + self.lift_threshold

    def _is_near_eef(self, obj_name, max_dist=0.08):
        if not hasattr(self, "latest_obs"):
            return False

        eef_key = "robot0_eef_pos"
        obj_key = f"{obj_name}_pos"

        if eef_key not in self.latest_obs or obj_key not in self.latest_obs:
            return False

        eef_pos = self.latest_obs[eef_key]
        obj_pos = self.latest_obs[obj_key]

        return np.linalg.norm(eef_pos - obj_pos) < max_dist

    def _get_true_mass(self, obj_name):
        if obj_name in self.object_mass_table:
            return float(self.object_mass_table[obj_name])

        body_id = self.task_env.obj_body_id[obj_name]
        return float(self.env.sim.model.body_mass[body_id])

    def _augment_obs(self, obs):
        """
        Adds a fixed-length vector:
        [obj1_known, obj1_mass_est, obj2_known, obj2_mass_est, ...]
        """
        weight_vec = []

        for obj_name in self.object_names:
            weight_vec.append(float(self.weight_known[obj_name]))
            weight_vec.append(float(self.weight_estimate[obj_name]))

        obs["weight_state"] = np.array(weight_vec, dtype=np.float32)
        return obs
