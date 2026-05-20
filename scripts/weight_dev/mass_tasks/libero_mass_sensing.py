from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .mass_sensing_utils import (
        MassSensingTracker,
        add_mass_observation,
        object_names_from_observation,
    )
    from .object import ensure_libero_import_path
except ImportError:
    from mass_sensing_utils import (  # type: ignore[no-redef]
        MassSensingTracker,
        add_mass_observation,
        object_names_from_observation,
    )
    from object import ensure_libero_import_path  # type: ignore[no-redef]


ensure_libero_import_path()

try:
    from . import object as _custom_objects  # noqa: F401
except ImportError:
    import object as _custom_objects  # type: ignore[no-redef]  # noqa: F401

from libero.libero.envs.bddl_base_domain import TASK_MAPPING, register_problem  # noqa: E402
from libero.libero.envs.env_wrapper import OffScreenRenderEnv  # noqa: E402
from libero.libero.envs.regions import REGION_SAMPLERS  # noqa: E402
import libero.libero.envs.problems.libero_tabletop_manipulation  # noqa: E402,F401


DEFAULT_BDDL_FILE = (
    Path(__file__).resolve().parent / "bddl" / "put_opened_empty_can_in_basket.bddl"
)


Libero_Tabletop_Manipulation = TASK_MAPPING["libero_tabletop_manipulation"]


class Opened_Empty_Can_Mass_Sensing(Libero_Tabletop_Manipulation):
    """Custom tabletop task registered from ``opened_empty_can_mass_sensing`` BDDL."""


register_problem(Opened_Empty_Can_Mass_Sensing)
REGION_SAMPLERS["opened_empty_can_mass_sensing"] = REGION_SAMPLERS[
    "libero_tabletop_manipulation"
]


class Empty_Cup_Mass_Sensing(Libero_Tabletop_Manipulation):
    """Custom tabletop task registered from ``empty_cup_mass_sensing`` BDDL."""


register_problem(Empty_Cup_Mass_Sensing)
REGION_SAMPLERS["empty_cup_mass_sensing"] = REGION_SAMPLERS[
    "libero_tabletop_manipulation"
]


class MassSensingEnv:
    """LIBERO env wrapper that reveals object masses after grip-and-lift probing."""

    def __init__(
        self,
        *,
        bddl_file_name: str | Path = DEFAULT_BDDL_FILE,
        grip_duration_s: float = 0.5,
        lift_duration_s: float = 0.5,
        lift_threshold_m: float = 0.03,
        env: Any | None = None,
        **env_kwargs: Any,
    ) -> None:
        self.env = env or OffScreenRenderEnv(
            bddl_file_name=str(Path(bddl_file_name).expanduser().resolve()),
            **env_kwargs,
        )
        self.tracker = MassSensingTracker(
            self.env,
            (),
            grip_duration_s=grip_duration_s,
            lift_duration_s=lift_duration_s,
            lift_threshold_m=lift_threshold_m,
        )

    @property
    def sim(self) -> Any:
        return self.env.sim

    @property
    def robots(self) -> Any:
        return self.env.robots

    @property
    def control_freq(self) -> int:
        return int(getattr(self.env, "control_freq", 20))

    def reset(self) -> dict[str, Any]:
        obs = self.env.reset()
        self.tracker.refresh(object_names_from_observation(obs))
        return add_mass_observation(obs, self.tracker)

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        self.tracker.update()
        masses = self.tracker.masses()
        info["object_masses"] = masses
        obs["object_masses"] = masses
        return obs, reward, done, info

    def set_init_state(self, init_state: Any) -> dict[str, Any]:
        obs = self.env.set_init_state(init_state)
        self.tracker.refresh(object_names_from_observation(obs))
        return add_mass_observation(obs, self.tracker)

    def check_success(self) -> bool:
        return bool(self.env.check_success())

    def close(self) -> None:
        self.env.close()


class FrankaLiberoMassSensingEnv(MassSensingEnv):
    """Compatibility wrapper for the old Franka-specific constructor."""

    def __init__(
        self,
        suite_name: str = "self_defined",
        task_id: int = 0,
        privileged: bool = True,
        max_steps: int = 4000,
        seed: int | None = None,
        enable_render: bool = False,
        control_freq: int = 20,
        viser_debug: bool = False,
        bddl_file_name: str | Path | None = None,
        mass_grip_duration_s: float = 0.5,
        mass_lift_duration_s: float = 0.5,
        mass_lift_threshold_m: float = 0.03,
        **env_kwargs: Any,
    ) -> None:
        del suite_name, task_id, privileged, enable_render
        if viser_debug:
            raise NotImplementedError("Viser debugging is not available in this repo.")

        super().__init__(
            bddl_file_name=bddl_file_name or DEFAULT_BDDL_FILE,
            grip_duration_s=mass_grip_duration_s,
            lift_duration_s=mass_lift_duration_s,
            lift_threshold_m=mass_lift_threshold_m,
            robots=["Panda"],
            controller="JOINT_POSITION",
            horizon=max_steps,
            ignore_done=True,
            control_freq=control_freq,
            camera_depths=True,
            **env_kwargs,
        )
        if seed is not None:
            self.env.seed(seed)


__all__ = [
    "DEFAULT_BDDL_FILE",
    "Empty_Cup_Mass_Sensing",
    "FrankaLiberoMassSensingEnv",
    "MassSensingEnv",
    "Opened_Empty_Can_Mass_Sensing",
]
