"""Commercially usable asset packs for real-to-sim retrieval (AssetPack, 10-19; docs/ASSET_PACK_ADOPTION.html).

Packs live under data/real2sim/asset_packs/ (gitignored; download steps in the doc):
  kitchen -> NVIDIA/Lightwheel Kitchen-MJCF objects (CC BY 4.0): objects/<category>/<Model>/model.xml
  gso     -> mujoco_scanned_objects (Google Scanned Objects; meshes CC BY 4.0, XML MIT): src/models/<Name>/model.xml
  ycb     -> YCB object set, google_16k meshes (CC BY 4.0, AWS Open Data registry): src/<NNN_name>/google_16k/
             textured.obj + texture_map.png; no MJCF ships, so ycb_mjcf() writes one per model under ycb/mjcf/.
--assets takes one pack or a comma list (e.g. gso,ycb); no combined 'cc' option yet (user, 10-20: ideation phase).
Kitchen and GSO ship convex collision parts that keep container cavities open, so callers should keep them
(collision='shipped'), unlike the LIBERO-Plus meshes build_scene collapses to one hull. YCB has none: its
collision is the visual mesh's convex hull (collision='hull'), which fills a cavity, so YCB containers are not
offered for the bowl (see CAVITY).

Also: object property defaults + per-scene user overrides (mass / friction / scale), see resolve_props().
"""
import glob, json, os, re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKS = os.path.join(REPO, "data/real2sim/asset_packs")
LICENCE = {"kitchen": "CC BY 4.0 (NVIDIA Kitchen-MJCF)", "gso": "CC BY 4.0 meshes + MIT XML (Google Scanned Objects)",
           "ycb": "CC BY 4.0 (YCB object set, ycbbenchmarks.com)"}
PACK_NAMES = ("kitchen", "gso", "ycb")


def sources(arg):
    """'gso,ycb' -> ('gso', 'ycb'); None if arg names no pack (libero / none)."""
    names = tuple(p for p in arg.split(",") if p)
    return names if names and all(p in PACK_NAMES for p in names) else None

# Defaults (user, 10-19: ours unless the user overrides them per scene). density_kg_m3 is applied to the visual
# mesh's convex-hull volume, the same density build_scene already uses for retrieved meshes (500), so a pack object
# weighs what today's single-hull object would; friction is the bowl's existing [0.95, 0.3, 0.1].
DEFAULTS = dict(density_kg_m3=500.0, friction=[0.95, 0.3, 0.1], scale_mult=1.0)
CATEGORY_DEFAULTS = {}  # "<category>": {mass: kg, friction: [...]}; only sourced values go here


def _tokens(s):
    """'Now_Designs_Bowl_Akita_Black' / 'glass_cup' -> {'now', 'designs', 'bowl', 'akita', 'black'} (singular)."""
    out = set()
    for t in re.split(r"[_\W]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", s).lower()):
        if t:
            out.add(t[:-1] if t.endswith("s") and not t.endswith("ss") and len(t) > 3 else t)
    return out


def ycb_mjcf(obj_dir):
    """MJCF for one YCB google_16k folder, written once under ycb/mjcf/<name>/model.xml: textured visual mesh plus
    the same mesh as one colliding geom (MuJoCo collides a mesh as its convex hull)."""
    name = obj_dir.rstrip("/").split("/")[-2]
    out = f"{PACKS}/ycb/mjcf/{name}/model.xml"
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(f"""<mujoco model="{name}">
  <asset>
    <texture type="2d" name="texture" file="{obj_dir}/texture_map.png"/>
    <material name="material" texture="texture" specular="0.3" shininess="0.3"/>
    <mesh name="model" file="{obj_dir}/textured.obj"/>
  </asset>
  <worldbody>
    <body name="object">
      <geom type="mesh" mesh="model" material="material" contype="0" conaffinity="0" group="1"/>
      <geom type="mesh" mesh="model" group="3" rgba="1 1 1 0"/>
    </body>
  </worldbody>
</mujoco>
""")
    return out


def _ycb_tokens(name):
    """'065-a_cups' -> {'cup'}: the set number and letter are not words."""
    return _tokens(re.sub(r"^\d+(-[a-z])?_", "", name))


# YCB models whose convex-hull collision would fill an opening the task needs (bowl, mug, cups, pitcher)
CAVITY = ("024_bowl", "025_mug", "019_pitcher_base", "065-")


def pack_catalogue(source):
    """name -> entry dict(xml, pack, category, tokens, licence, collision). Sorted, deterministic.
    source: a pack name or a comma list of them (sources())."""
    cat = {}
    for pack in sources(source):
        if pack == "kitchen":
            for x in sorted(glob.glob(f"{PACKS}/kitchen_mjcf/objects/*/*/model.xml")):
                c, name = x.split("/")[-3:-1]
                cat[f"kitchen/{name}"] = dict(xml=x, pack=pack, category=c, tokens=_tokens(c), licence=LICENCE[pack],
                                              collision="shipped")
        elif pack == "gso":
            for x in sorted(glob.glob(f"{PACKS}/gso_mjcf/src/models/*/model.xml")):
                name = x.split("/")[-2]
                cat[f"gso/{name}"] = dict(xml=x, pack=pack, category=None, tokens=_tokens(name), licence=LICENCE[pack],
                                          collision="shipped")
        elif pack == "ycb":
            for d in sorted(glob.glob(f"{PACKS}/ycb/src/*/google_16k/")):
                name = d.split("/")[-3]
                cat[f"ycb/{name}"] = dict(xml=ycb_mjcf(d), pack=pack, category=None, tokens=_ycb_tokens(name),
                                          licence=LICENCE[pack], collision="hull", cavity=name.startswith(CAVITY))
    return cat


def match(label, cat, colours=()):
    """Detected label ('black bowl', 'red radish#1') -> catalogue key, or None.
    Head noun (last non-colour word) must be a category token (Kitchen) or a name token (GSO); candidates sharing
    more of the label's other words (e.g. the colour 'black') win, then Kitchen categories, then name order.
    Returns all candidates in that order so the caller can skip ones failing its size check (SCALE_RANGE).
    Name matching only and weak (GSO names are product names: 'apple' hits a card game); the image index replaces
    this ranking in stage 4."""
    words = [w for w in _tokens_ordered(label.split("#")[0])]
    nouns = [w for w in words if w not in colours]
    for w in reversed(nouns):
        cands = [k for k, e in cat.items() if w in e["tokens"]]
        if cands:
            extra = set(words) - {w}
            cands.sort(key=lambda k: (-len(extra & cat[k]["tokens"]), cat[k]["category"] is None, k))
            return cands
    return []


# Both packs are metric, so a retrieved model that needs a big rescale to fit the detection is the wrong object
# (e.g. a 20 cm game box for a 6 cm apple); it is skipped and the next candidate tried.
SCALE_RANGE = (0.5, 2.0)


def _tokens_ordered(s):
    return [t[:-1] if t.endswith("s") and not t.endswith("ss") and len(t) > 3 else t
            for t in re.split(r"[_\W]+", s.lower()) if t]


def _scaled_spec(xml, k):
    import mujoco
    s = mujoco.MjSpec.from_file(xml)
    for lt in list(s.lights):
        s.delete(lt)
    for ms in s.meshes:
        ms.scale = [k * v for v in ms.scale]
    return s


def _is_collider(g):
    return bool(g.contype or g.conaffinity)


def prepare_child(xml, k, props):
    """Pack model scaled by k, ready for spec.attach: shipped collision parts kept (cavities stay open), every
    non-colliding geom (visual, Kitchen's 'region' boxes) massless, collision parts given one density so the body's
    mass is props['mass'] or density_kg_m3 x the visual mesh's convex-hull volume, and props['friction'].
    Returns (spec, info) with the numbers to record in scene_fit.json."""
    import mujoco
    # volumes at density 1: (a) visual meshes' convex hulls, (b) the collision parts
    s = _scaled_spec(xml, k)
    for g in s.geoms:
        g.mass = float("nan")  # let density decide
        g.density = 0.0 if _is_collider(g) or g.type != mujoco.mjtGeom.mjGEOM_MESH else 1.0
    for ms in s.meshes:
        ms.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
    hull_volume = float(s.compile().body_mass.sum())
    s = _scaled_spec(xml, k)
    for g in s.geoms:
        g.mass = float("nan")
        g.density = 1.0 if _is_collider(g) else 0.0
    col_volume = float(s.compile().body_mass.sum())
    mass = props.get("mass", props.get("density_kg_m3", DEFAULTS["density_kg_m3"]) * hull_volume)
    s = _scaled_spec(xml, k)
    n_col = 0
    for g in s.geoms:
        g.mass = float("nan")
        if _is_collider(g):
            g.density = mass / col_volume
            g.friction = list(props.get("friction", DEFAULTS["friction"]))
            n_col += 1
        else:
            g.density = 0.0
    check = float(s.compile().body_mass.sum())
    return s, dict(mass_kg=check, hull_volume_m3=hull_volume, collision_volume_m3=col_volume, n_collision_parts=n_col)


def load_props(path):
    """User overrides: {"<label>": {...}, "category:<cat>": {...}} with keys mass (kg), friction ([3]), scale_mult."""
    if not path:
        return {}
    props = json.load(open(path))
    bad = {k: set(v) - {"mass", "friction", "scale_mult"} for k, v in props.items()}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise SystemExit(f"--object_props: unknown keys {bad} (allowed: mass, friction, scale_mult)")
    return props


def resolve_props(label, category, props):
    """Per-field value + where it came from: label override > category override > category default > default."""
    layers = [("label", props.get(label.split("#")[0], {}) | props.get(label, {})),
              ("category_override", props.get(f"category:{category}", {}) if category else {}),
              ("category_default", CATEGORY_DEFAULTS.get(category, {}) if category else {}),
              ("default", DEFAULTS)]
    out, src = {}, {}
    for field in ("mass", "density_kg_m3", "friction", "scale_mult"):
        for name, layer in layers:
            if field in layer:
                out[field], src[field] = layer[field], name
                break
    if "mass" in out and src["mass"] != "default":  # an explicit mass beats the density rule
        out.pop("density_kg_m3", None)
        src.pop("density_kg_m3", None)
    else:
        out.pop("mass", None)
        src.pop("mass", None)
    return out, src
