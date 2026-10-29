"""The agent's answer, v3: two optional parts (docs/TASKGEN_DESIGN.html §1a, §1b; D20 = one general agent).
  scene  job A, scene modification: objects to ADD (base objects are locked for the setup)
  task   job B, task creation: what the robot must do; may move ANY object
  infeasible  a list, one entry per refused part, each with the reason and the closest alternative used (if any)
A request can fill scene only, task only, or both. Simulator-neutral. New file next to core/templates.py (v1/v2 stay
as they are for the runs still using them). Exploratory, UNVERIFIED.
"""
import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from core.templates import PassCondition, RelationT

SourceT = Literal["box", "drawer", "cabinet", "pack"]


class AddSpec(BaseModel):
    """One object to add. box: needs colour; pack: needs noun; cabinet: levels (1-3) and open_level."""
    name: str = Field(description="unique name, e.g. red_block_1, cabinet_1")
    source: SourceT
    noun: Optional[str] = Field(None, description="source=pack only: what it is, e.g. 'cup'")
    colour: Optional[str] = Field(None, description="source=box only: red, blue, green, yellow, orange, purple, white, black")
    size_cm: Optional[float] = Field(None, description="optional; blocks 2-6 cm")
    levels: Optional[int] = Field(None, description="source=cabinet only: number of drawers stacked (1-3)")
    open_level: Optional[str] = Field(None, description="source=cabinet only: which drawer starts open: bottom, middle, top")
    relation: RelationT
    target: Optional[str] = Field(None, description="an existing or earlier-added object; a cabinet drawer as 'cabinet_1/middle'; none for on_table")


class ScenePart(BaseModel):
    add: List[AddSpec] = Field(default_factory=list)

    @field_validator("add")
    @classmethod
    def unique(cls, v):
        names = [o.name for o in v]
        if len(names) != len(set(names)):
            raise ValueError(f"object names must be unique: {names}")
        return v


class TaskPart(BaseModel):
    instruction: str
    targets: List[str]
    pass_all: List[PassCondition] = Field(description="every condition must hold at the end")


class Refusal(BaseModel):
    part: Literal["scene", "task"]
    reason: str
    alternative: Optional[str] = Field(None, description="what was done instead, if anything")


class Request(BaseModel):
    scene: Optional[ScenePart] = None
    task: Optional[TaskPart] = None
    infeasible: List[Refusal] = Field(default_factory=list)


def schema_json():
    return json.dumps(Request.model_json_schema())
