from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from typing import Any

def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_libero_import_path() -> None:
    root = repo_root()
    root_str = os.fspath(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    importlib.invalidate_caches()


ensure_libero_import_path()

from robosuite.models.objects import MujocoXMLObject  # noqa: E402

from libero.libero.envs.base_object import OBJECTS_DICT  # noqa: E402


LOCAL_ASSET_DIR = Path(__file__).resolve().parent / "custom_assets"


class _CustomCanObject(MujocoXMLObject):
    def __init__(
        self,
        xml_path: Path,
        name: str,
        joints: list[dict[str, Any]] | None = None,
    ) -> None:
        if joints is None:
            joints = [dict(type="free", damping="0.0005")]

        super().__init__(
            os.fspath(xml_path),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


class CanOfIcetea(_CustomCanObject):
    """Unopened can-of-icetea asset."""

    def __init__(
        self,
        name: str = "can_of_icetea",
        joints: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            LOCAL_ASSET_DIR / "can_of_icetea" / "unopened" / "can_of_icetea.xml",
            name=name,
            joints=joints,
        )
        self.category_name = "can_of_icetea"


class OpenedLightCanOfIcetea(_CustomCanObject):
    """Opened can-of-icetea asset with 90% of the unopened can mass."""

    def __init__(
        self,
        name: str = "opened_light_can_of_icetea",
        joints: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            LOCAL_ASSET_DIR
            / "can_of_icetea"
            / "opened_light"
            / "opened_light_can_of_icetea.xml",
            name=name,
            joints=joints,
        )
        self.category_name = "opened_light_can_of_icetea"


class OpenedEmptyCanOfIcetea(_CustomCanObject):
    """Opened can-of-icetea asset with 10% of the unopened can mass."""

    def __init__(
        self,
        name: str = "opened_empty_can_of_icetea",
        joints: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            LOCAL_ASSET_DIR
            / "can_of_icetea"
            / "opened_empty"
            / "opened_empty_can_of_icetea.xml",
            name=name,
            joints=joints,
        )
        self.category_name = "opened_empty_can_of_icetea"


OBJECTS_DICT.update(
    {
        "can_of_icetea": CanOfIcetea,
        "opened_empty_can_of_icetea": OpenedEmptyCanOfIcetea,
        "opened_light_can_of_icetea": OpenedLightCanOfIcetea,
    }
)


__all__ = [
    "CanOfIcetea",
    "OpenedEmptyCanOfIcetea",
    "OpenedLightCanOfIcetea",
    "ensure_libero_import_path",
    "repo_root",
]
