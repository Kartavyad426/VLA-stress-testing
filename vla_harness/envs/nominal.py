"""What "nominal" means for a LIBERO-Plus vision variant -- pure helpers.

R-039's paired source re-renders the CURRENT sim state under the trained
condition. For that it needs, per variant, which knobs are perturbed and what
their nominal values are. Everything here is read off the fork's own files,
because the fork bakes the perturbation into the variant NAME (camera, noise)
or into a swapped scene XML (lights); the sim carries no record of what the
nominal was.

Sources, all in third_party/LIBERO-plus/libero/libero:
  * camera: `pos_av`/`quat_av` in each domain's `_setup_camera`
    (envs/problems/libero_*_manipulation.py). `view_H_V_S_R_E` in the variant
    name is applied on top of those, so zeros and scale 100 ARE the nominal.
  * noise: `_noise_N` in the name; the wrapper blurs the agentview image in
    its own step()/reset() (envs/env_wrapper.py:294-327), so the inner env's
    observation is already the clean frame.
  * lights: a `_light_N` variant subclasses the base domain with
    `scene_xml = scenes/lights/<domain>_light_sync_modified_N.xml`, and the
    base domain's scene XML holds the nominal <light> elements by name.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

_VIEW = re.compile(r"_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)")
_NOISE = re.compile(r"_noise_(\d+)")
_LIGHT = re.compile(r"_light_(\d+)$")

# pos_av / quat_av per domain, from `_setup_camera` in the fork. Checked
# against the source by tests/test_paired_source.py, so drift fails a test
# rather than silently re-rendering the wrong nominal.
_CAMERA = {
    "tabletop": ([0.6586131746834771, 0.0, 1.6103500240372423],
                 [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]),
    "kitchen_tabletop": ([0.6586131746834771, 0.0, 1.6103500240372423],
                         [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]),
    "floor": ([0.8965773716836134, 5.216182733499864e-07, 0.65],
              [0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979]),
    "living_room_tabletop": ([0.6065773716836134, 0.0, 0.96],
                             [0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979]),
    "study_tabletop": ([0.4586131746834771, 0.0, 1.6103500240372423],
                       [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]),
    "coffee_table": ([1.5, 0.0, 0.9],
                     [0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979]),
}

# scene-XML prefix used by the light variants -> the domain's base scene
_BASE_SCENE = {
    "tabletop": "scenes/libero_tabletop_base_style.xml",
    "kitchen": "scenes/libero_kitchen_tabletop_base_style.xml",
    "floor": "scenes/libero_floor_base_style.xml",
    "living_room": "scenes/libero_living_room_tabletop_base_style.xml",
    "study": "scenes/libero_study_base_style.xml",
    "coffee_table": "scenes/libero_coffee_table_base_style.xml",
}


def variant_perturbations(name: str) -> dict:
    """Which vision knobs a LIBERO-Plus variant NAME perturbs.

    view  -> (horizon_deg, vertical_deg, scale, end_rot_deg, end_vert_deg) or
             None when every value is the trained one
    light -> the light id, or None
    noise -> the wrapper's severity, 0 when absent
    """
    view = None
    m = _VIEW.search(name)
    if m:
        h, v, s, r, e = (int(m.group(i)) for i in range(1, 6))
        scale = s / 100.0
        if (h, v, r, e) != (0, 0, 0, 0) or scale != 1.0:
            view = (h, v, scale, r, e)
    m = _NOISE.search(name)
    noise = int(m.group(1)) if m else 0
    m = _LIGHT.search(name)
    light = int(m.group(1)) if m else None
    return {"view": view, "light": light, "noise": noise}


def nominal_camera(domain: str) -> tuple[list[float], list[float]]:
    """(pos, quat) of the trained `agentview` for a domain such as 'tabletop'."""
    pos, quat = _CAMERA[domain]
    return list(pos), list(quat)


def base_scene_for(scene_xml: str) -> str:
    """The base scene that a `scenes/lights/<domain>_light_sync_modified_N.xml`
    variant was derived from; a base scene maps to itself."""
    base = os.path.basename(scene_xml)
    if "light_sync_modified" not in base:
        return scene_xml
    prefix = base.split("_light_sync_modified")[0]
    return _BASE_SCENE[prefix]


def nominal_lights(base_xml_path: str) -> dict[str, dict[str, list[float]]]:
    """<light> elements of a base scene, by name, as float lists."""
    root = ET.parse(base_xml_path).getroot()
    out: dict[str, dict[str, list[float]]] = {}
    for el in root.iter("light"):
        name = el.get("name")
        if name is None:
            continue
        fields = {}
        for key in ("diffuse", "dir", "specular", "pos"):
            if el.get(key) is not None:
                fields[key] = [float(x) for x in el.get(key).split()]
        out[name] = fields
    return out


def joint_direction(seed: int, n: int = 7):
    """Isotropic random unit direction in joint space, fixed by seed -- the
    R-035 / LIBERO-Plus definition of a robot-initial-state perturbation of
    radius r: init_qpos + r * direction (fork: robots/mounted_panda.py, each
    MountedPandaN is such a point)."""
    import numpy as np
    v = np.random.default_rng(int(seed)).standard_normal(n)
    return v / np.linalg.norm(v)
