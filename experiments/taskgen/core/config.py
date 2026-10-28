"""Rubric 1, the scene config (docs/TASKGEN_DESIGN.html §3a): the source of truth for a modified scene. Every edit
goes here first; the scene is rebuilt from it. One format with res31's planned B8 overrides (D1): base_objects carry
optional overrides (not applied yet). Simulator-neutral. Exploratory, UNVERIFIED.
"""
import hashlib, json, os
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class Relation(BaseModel):
    type: str                         # on_table | next_to | on | inside
    target: Optional[str] = None


class AddedObject(BaseModel):
    name: str
    source: str                       # primitive:box | primitive:drawer | pack:<key>  (later objgen:<id>)
    kind: str                         # block | cup | bowl | fruit | item | drawer
    colour: Optional[str] = None
    rgba: Optional[Tuple[float, float, float, float]] = None
    size_m: Tuple[float, float, float]           # full extents at the current scale
    scale: float = 1.0                # pack assets: scale vs real size
    licence: Optional[str] = None
    relation: Optional[Relation] = None          # None = added, not placed yet
    pose: Optional[Tuple[float, float, float, float]] = None   # x, y, z_bottom, yaw (written by place)
    resized_from_m: Optional[Tuple[float, float, float]] = None
    resize_reason: Optional[str] = None


class BaseObject(BaseModel):
    kind: str
    locked: bool = True
    source: str = "reconstructed"
    overrides: Dict = Field(default_factory=dict)  # B8 placeholder (size, pose, mass, friction, colour, mesh, removed)


class SceneConfig(BaseModel):
    base_scene: str
    base_version: str
    episode: str
    simulator: str
    base_objects: Dict[str, BaseObject] = Field(default_factory=dict)
    added_objects: Dict[str, AddedObject] = Field(default_factory=dict)
    articulation: Dict[str, float] = Field(default_factory=dict)
    checks: Dict = Field(default_factory=dict)
    log: List[Dict] = Field(default_factory=list)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        open(path, "w").write(self.model_dump_json(indent=1))

    @classmethod
    def load(cls, path):
        return cls.model_validate_json(open(path).read())


def base_version(scene_dir):
    """Hash of the files the base scene is defined by (scene_fit.json, calib.json, scene.xml)."""
    h = hashlib.sha256()
    for f in ("scene_fit.json", "calib.json", "scene.xml"):
        p = os.path.join(scene_dir, f)
        if os.path.exists(p):
            h.update(f.encode() + open(p, "rb").read())
    return h.hexdigest()[:16]


def dump(obj):
    return json.loads(obj.model_dump_json())
