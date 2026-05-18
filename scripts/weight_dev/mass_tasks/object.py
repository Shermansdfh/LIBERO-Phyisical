from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


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
NOTEBOOK_ASSET_DIR = repo_root() / "notebooks" / "custom_assets"


class _CustomMugObject(MujocoXMLObject):
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
        self.rotation = {
            "x": (-np.pi / 2, -np.pi / 2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None
        self.object_properties = {"vis_site_names": {}}


class LiberoMug(_CustomMugObject):
    """Plain LIBERO mug asset."""

    def __init__(
        self,
        name: str = "libero_mug",
        joints: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            NOTEBOOK_ASSET_DIR / "libero_mug" / "libero_mug.xml",
            name=name,
            joints=joints,
        )
        self.category_name = "libero_mug"


class LiberoMugYellow(_CustomMugObject):
    """Yellow mug asset used as the heavier filled-mug proxy."""

    def __init__(
        self,
        name: str = "libero_mug_yellow",
        joints: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            NOTEBOOK_ASSET_DIR / "libero_mug_yellow" / "libero_mug_yellow.xml",
            name=name,
            joints=joints,
        )
        self.category_name = "libero_mug_yellow"


class EmptyLiberoMugYellow(_CustomMugObject):
    """Visually yellow mug with lower collision-geom density."""

    def __init__(
        self,
        name: str = "empty_libero_mug_yellow",
        joints: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            LOCAL_ASSET_DIR / "empty_libero_mug_yellow" / "empty_libero_mug_yellow.xml",
            name=name,
            joints=joints,
        )
        self.category_name = "empty_libero_mug_yellow"


OBJECTS_DICT.update(
    {
        "empty_libero_mug_yellow": EmptyLiberoMugYellow,
        "libero_mug": LiberoMug,
        "libero_mug_yellow": LiberoMugYellow,
    }
)


__all__ = [
    "EmptyLiberoMugYellow",
    "LiberoMug",
    "LiberoMugYellow",
    "ensure_libero_import_path",
    "repo_root",
]
