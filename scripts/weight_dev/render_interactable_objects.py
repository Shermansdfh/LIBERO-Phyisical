#!/usr/bin/env python3
"""Render LIBERO interactable object assets.

By default this renders movable object classes registered in
``libero/libero/envs/objects`` and skips articulated fixtures such as the
microwave, cabinets, faucets, and stoves.
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "weight_dev" / "object_renders"
DEFAULT_EXTERNAL_ROOT = Path("/tmp2/shermanchang/new_objects")
VIEW_NAMES = ("front", "side", "top")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render registered LIBERO movable/interactable object assets."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--camera-distance", type=float, default=1.05)
    parser.add_argument("--camera-height", type=float, default=0.55)
    parser.add_argument("--sheet-cols", type=int, default=6)
    parser.add_argument(
        "--source",
        choices=("libero", "external", "both"),
        default="libero",
        help="Object source to render.",
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=DEFAULT_EXTERNAL_ROOT,
        help="Root containing category/model/usd/MJCF/*.xml external objects.",
    )
    parser.add_argument(
        "--external-limit",
        type=int,
        default=None,
        help="Optional maximum number of external MJCF objects to render.",
    )
    parser.add_argument(
        "--variants-per-object",
        type=int,
        default=None,
        help=(
            "For external objects, render at most N model variants per category. "
            "Use 1 to render one variant of every object category."
        ),
    )
    parser.add_argument(
        "--one-variant-per-object",
        dest="variants_per_object",
        action="store_const",
        const=1,
        help="Shortcut for --variants-per-object 1.",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["front"],
        choices=(*VIEW_NAMES, "all"),
        help="Camera views to render. Use '--views all' for front, side, and top.",
    )
    parser.add_argument(
        "--include-articulated",
        action="store_true",
        help="Also render fixtures such as microwave, cabinets, faucets, and stoves.",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help=(
            "Optional names to render. For external objects this matches category, "
            "model id, or category_model_id."
        ),
    )
    parser.add_argument(
        "--mujoco-gl",
        default=None,
        choices=("egl", "osmesa", "glfw"),
        help="Set MUJOCO_GL before importing robosuite. Useful on headless machines.",
    )
    parser.add_argument(
        "--no-contact-sheet",
        action="store_true",
        help="Only save individual object PNGs.",
    )
    return parser.parse_args()


def set_mujoco_backend(args):
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")


def import_render_deps():
    try:
        from robosuite.models import MujocoWorldBase
        from robosuite.utils.binding_utils import (
            MjRenderContext,
            MjRenderContextOffscreen,
            MjSim,
        )
        from robosuite.utils.mjcf_utils import array_to_string
    except ImportError as exc:
        raise SystemExit(
            "Could not import robosuite. Activate the LIBERO environment first, "
            "then rerun this script."
        ) from exc

    silence_egl_destructor_noise(MjRenderContext)
    return MujocoWorldBase, MjSim, MjRenderContextOffscreen, array_to_string


def silence_egl_destructor_noise(MjRenderContext):
    """Robosuite EGL destructors can print harmless shutdown tracebacks."""

    def quiet_mj_render_context_del(self):
        try:
            original_mj_render_context_del(self)
        except Exception:
            pass

    original_mj_render_context_del = MjRenderContext.__del__
    MjRenderContext.__del__ = quiet_mj_render_context_del

    try:
        from robosuite.renderers.context.egl_context import EGLGLContext
    except ImportError:
        return

    original_egl_del = EGLGLContext.__del__
    original_egl_free = EGLGLContext.free

    def quiet_egl_del(self):
        try:
            original_egl_del(self)
        except Exception:
            pass

    def quiet_egl_free(self):
        try:
            original_egl_free(self)
        except Exception:
            self._context = None

    EGLGLContext.__del__ = quiet_egl_del
    EGLGLContext.free = quiet_egl_free


def import_object_deps():
    from libero.libero.envs.objects import OBJECTS_DICT
    from libero.libero.envs.objects.articulated_objects import ArticulatedObject
    from libero.libero.envs.objects.google_scanned_objects import GoogleScannedObject
    from libero.libero.envs.objects.hope_objects import HopeBaseObject
    from libero.libero.envs.objects.self_designed_object import CustomObjects
    from libero.libero.envs.objects.turbosquid_objects import TurbosquidObjects

    movable_bases = (
        HopeBaseObject,
        GoogleScannedObject,
        TurbosquidObjects,
        CustomObjects,
    )
    return OBJECTS_DICT, ArticulatedObject, movable_bases


def camera_quat(camera_pos, target=(0.0, 0.0, 0.08), up=(0.0, 0.0, 1.0)):
    """Return MuJoCo camera quaternion looking from camera_pos to target."""
    camera_pos = np.asarray(camera_pos, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    z_axis = camera_pos - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])

    trace = np.trace(rot)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s

    quat = np.array([qw, qx, qy, qz], dtype=float)
    return quat / np.linalg.norm(quat)


def normalize_views(views):
    if "all" in views:
        return list(VIEW_NAMES)
    return views


def camera_config(view_name, args):
    if view_name == "front":
        return np.array([0.0, -args.camera_distance, args.camera_height]), np.array(
            [0.0, 0.0, 1.0]
        )
    if view_name == "side":
        return np.array([args.camera_distance, 0.0, args.camera_height]), np.array(
            [0.0, 0.0, 1.0]
        )
    if view_name == "top":
        return np.array([0.0, 0.0, args.camera_distance]), np.array([0.0, 1.0, 0.0])

    raise ValueError(f"Unknown view: {view_name}")


def registered_interactable_objects(object_dict, articulated_base, movable_bases, args):
    requested = set(args.objects) if args.objects else None
    selected = []

    for key, cls in sorted(object_dict.items()):
        if requested is not None and key not in requested:
            continue
        if issubclass(cls, articulated_base) and not args.include_articulated:
            continue
        if not args.include_articulated and not issubclass(cls, movable_bases):
            continue
        selected.append((key, cls))

    if requested is not None:
        found = {key for key, _ in selected}
        missing = sorted(requested - found)
        if missing:
            print(f"Skipped unknown or filtered object keys: {', '.join(missing)}")

    return selected


def external_interactable_objects(args):
    if not args.external_root.exists():
        return []

    requested = set(args.objects) if args.objects else None
    selected = []
    variants_by_category = {}
    xml_paths = sorted(args.external_root.glob("*/*/usd/MJCF/*.xml"))

    for xml_path in xml_paths:
        rel = xml_path.relative_to(args.external_root)
        if len(rel.parts) < 5:
            continue
        category, model_id = rel.parts[0], rel.parts[1]
        label = safe_name(f"{category}_{model_id}")
        candidates = {category, model_id, label}
        if requested is not None and candidates.isdisjoint(requested):
            continue
        if args.variants_per_object is not None:
            variant_count = variants_by_category.get(category, 0)
            if variant_count >= args.variants_per_object:
                continue

        selected.append((label, xml_path))
        variants_by_category[category] = variants_by_category.get(category, 0) + 1
        if args.external_limit is not None and len(selected) >= args.external_limit:
            break

    return selected


def safe_name(name):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def render_object(key, cls, args, deps, view_name):
    MujocoWorldBase, MjSim, MjRenderContextOffscreen, array_to_string = deps

    obj = cls(name=key)
    world = MujocoWorldBase()
    world.merge_assets(obj)

    body = obj.get_obj()
    body.set("pos", "0 0 0.04")
    world.worldbody.append(body)

    cam_pos, cam_up = camera_config(view_name, args)
    ET.SubElement(
        world.worldbody,
        "camera",
        name="objectview",
        mode="fixed",
        pos=array_to_string(cam_pos),
        quat=array_to_string(camera_quat(cam_pos, up=cam_up)),
        fovy="35",
    )
    ET.SubElement(
        world.worldbody,
        "light",
        name="object_light",
        pos="0 -1.0 1.5",
        dir="0 1 -1",
        diffuse="0.8 0.8 0.8",
        specular="0.2 0.2 0.2",
    )

    sim = make_sim(world.get_xml(), MjSim)
    if sim._render_context_offscreen is None:
        MjRenderContextOffscreen(
            sim,
            device_id=-1,
            max_width=args.width,
            max_height=args.height,
        )
    sim._render_context_offscreen.vopt.geomgroup[0] = 0
    sim._render_context_offscreen.vopt.geomgroup[1] = 1
    sim.forward()
    frame = sim.render(
        camera_name="objectview",
        width=args.width,
        height=args.height,
        depth=False,
    )
    return np.flipud(frame)


def render_external_object(xml_path, args, deps, view_name):
    _, MjSim, MjRenderContextOffscreen, array_to_string = deps

    tree = ET.parse(xml_path)
    root = tree.getroot()
    resolve_asset_paths(root, xml_path.parent)

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    cam_pos, cam_up = camera_config(view_name, args)
    ET.SubElement(
        worldbody,
        "camera",
        name="objectview",
        mode="fixed",
        pos=array_to_string(cam_pos),
        quat=array_to_string(camera_quat(cam_pos, up=cam_up)),
        fovy="35",
    )
    ET.SubElement(
        worldbody,
        "light",
        name="object_light",
        pos="0 -1.0 1.5",
        dir="0 1 -1",
        diffuse="0.8 0.8 0.8",
        specular="0.2 0.2 0.2",
    )

    sim = make_sim(ET.tostring(root, encoding="unicode"), MjSim)
    if sim._render_context_offscreen is None:
        MjRenderContextOffscreen(
            sim,
            device_id=-1,
            max_width=args.width,
            max_height=args.height,
        )
    sim._render_context_offscreen.vopt.geomgroup[0] = 0
    sim._render_context_offscreen.vopt.geomgroup[1] = 1
    sim.forward()
    frame = sim.render(
        camera_name="objectview",
        width=args.width,
        height=args.height,
        depth=False,
    )
    return np.flipud(frame)


def resolve_asset_paths(root, xml_dir):
    for node in root.findall(".//*[@file]"):
        file_path = Path(node.get("file"))
        if not file_path.is_absolute():
            node.set("file", str((xml_dir / file_path).resolve()))


def make_sim(xml, MjSim):
    if hasattr(MjSim, "from_xml_string"):
        return MjSim.from_xml_string(xml)

    try:
        import mujoco_py

        return MjSim(mujoco_py.load_model_from_xml(xml))
    except ImportError:
        import mujoco

        return MjSim(mujoco.MjModel.from_xml_string(xml))


def make_contact_sheet(rendered, output_path, cols):
    if not rendered:
        return

    label_h = 34
    tile_w, tile_h = rendered[0][1].size
    rows = math.ceil(len(rendered) / cols)
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, (name, image) in enumerate(rendered):
        row, col = divmod(idx, cols)
        x = col * tile_w
        y = row * (tile_h + label_h)
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + tile_h + 8), name, fill=(20, 20, 20), font=font)

    sheet.save(output_path)


def main():
    args = parse_args()
    if args.variants_per_object is not None and args.variants_per_object < 1:
        raise SystemExit("--variants-per-object must be >= 1")
    args.views = normalize_views(args.views)
    set_mujoco_backend(args)

    render_deps = import_render_deps()
    selected = []

    if args.source in ("libero", "both"):
        object_dict, articulated_base, movable_bases = import_object_deps()
        selected.extend(
            ("libero", key, cls)
            for key, cls in registered_interactable_objects(
                object_dict, articulated_base, movable_bases, args
            )
        )

    if args.source in ("external", "both"):
        selected.extend(
            ("external", key, xml_path)
            for key, xml_path in external_interactable_objects(args)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    failures = []

    print(
        f"Rendering {len(selected)} object(s) x {len(args.views)} view(s) "
        f"to {args.output_dir}"
    )
    for source, key, spec in selected:
        for view_name in args.views:
            try:
                if source == "libero":
                    frame = render_object(key, spec, args, render_deps, view_name)
                else:
                    frame = render_external_object(spec, args, render_deps, view_name)
                image = Image.fromarray(frame)
                suffix = "" if args.views == ["front"] else f"_{view_name}"
                image.save(args.output_dir / f"{key}{suffix}.png")
                label = key if args.views == ["front"] else f"{key}:{view_name}"
                rendered.append((label, image))
                print(f"[ok] {label}")
            except Exception as exc:
                label = key if args.views == ["front"] else f"{key}:{view_name}"
                failures.append((label, repr(exc)))
                print(f"[failed] {label}: {exc}")

    if not args.no_contact_sheet:
        make_contact_sheet(
            rendered, args.output_dir / "all_interactable_objects.png", args.sheet_cols
        )

    print(f"Done. Rendered {len(rendered)} object(s); {len(failures)} failure(s).")
    if failures:
        print("Failures:")
        for key, exc in failures:
            print(f"  {key}: {exc}")


if __name__ == "__main__":
    main()
