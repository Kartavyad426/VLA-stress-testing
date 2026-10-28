"""Templates with placeholders (docs/TASKGEN_DESIGN.html §4a mode A, §6). Simulator-neutral.

A SceneRequest is what a filled template looks like: objects to add (each with where it goes) and the task. The agent
(mode A) returns one as JSON; presets and the user write them by hand. A Template is a SceneRequest with "{slot}"
placeholders plus slot domains; expand() fills them by permutations / combinations, reports how many variants exist,
and samples up to a cap (D9). Exploratory, UNVERIFIED.
"""
import itertools, json, math, re
from typing import Dict, List, Literal, Optional, Union
import numpy as np
from pydantic import BaseModel, Field, field_validator

RelationT = Literal["on_table", "next_to", "on", "inside"]
RuleT = Literal["inside", "on", "stacked", "ordered_stack", "out_of", "next_to"]


class ObjectSpec(BaseModel):
    """One object to add. source: 'box' (primitive block), 'drawer', or 'pack' with a noun ('cup')."""
    name: str = Field(description="unique name to refer to it, e.g. red_block_1")
    source: Literal["box", "drawer", "pack"]
    noun: Optional[str] = Field(None, description="for source=pack: what it is, e.g. 'cup'")
    colour: Optional[str] = Field(None, description="for source=box: red, blue, green, yellow, orange, purple, white, black")
    size_cm: Optional[float] = Field(None, description="optional; blocks 2-6 cm")
    relation: RelationT
    target: Optional[str] = Field(None, description="the object it relates to (base or added); none for on_table")


class PassCondition(BaseModel):
    type: RuleT
    object: Optional[str] = None
    target: Optional[str] = None
    order: Optional[List[str]] = Field(None, description="ordered_stack: bottom to top")


class TaskSpec(BaseModel):
    instruction: str
    targets: List[str]
    pass_condition: PassCondition


class SceneRequest(BaseModel):
    add: List[ObjectSpec] = Field(default_factory=list)
    task: Optional[TaskSpec] = None
    infeasible: Optional[str] = Field(None, description="set instead of add/task when the request can't be done, with why")

    @field_validator("add")
    @classmethod
    def unique_names(cls, v):
        names = [o.name for o in v]
        if len(names) != len(set(names)):
            raise ValueError(f"object names must be unique: {names}")
        return v


# ---------------- slots and expansion ----------------
class Slot(BaseModel):
    """kind: 'choice' (one of values), 'choose' (k distinct values, unordered), 'permute' (k distinct, ordered),
    'permutation_of' (an order of another slot's chosen values)."""
    kind: Literal["choice", "choose", "permute", "permutation_of"]
    values: Optional[List[Union[str, int, float]]] = None
    k: Optional[int] = None
    of: Optional[str] = None
    scope: Literal["task", "scene"] = "task"


class Template(BaseModel):
    name: str
    slots: Dict[str, Slot]
    request: Dict          # a SceneRequest as a dict, with "{slot}" / "{slot[i]}" placeholders in strings

    def count(self):
        """How many variants exist (before any filtering)."""
        n = 1
        for s in self.slots.values():
            if s.kind == "choice":
                n *= len(s.values)
            elif s.kind == "choose":
                n *= math.comb(len(s.values), s.k)
            elif s.kind == "permute":
                n *= math.perm(len(s.values), s.k)
            else:
                n *= math.factorial(self._len_of(s.of))
        return n

    def _len_of(self, name):
        s = self.slots[name]
        return s.k if s.kind in ("choose", "permute") else 1

    def _domains(self):
        out = {}
        for n, s in self.slots.items():
            if s.kind == "choice":
                out[n] = list(s.values)
            elif s.kind == "choose":
                out[n] = [list(c) for c in itertools.combinations(s.values, s.k)]
            elif s.kind == "permute":
                out[n] = [list(c) for c in itertools.permutations(s.values, s.k)]
        return out

    def expand(self, cap=None, seed=0):
        """All variants (slot values -> SceneRequest), or a seeded sample of `cap` of them. Returns (total, variants)."""
        doms = self._domains()
        base_names = [n for n in self.slots if self.slots[n].kind != "permutation_of"]
        dep = [n for n in self.slots if self.slots[n].kind == "permutation_of"]
        combos = []
        for vals in itertools.product(*[doms[n] for n in base_names]):
            fill = dict(zip(base_names, vals))
            dep_doms = [[list(p) for p in itertools.permutations(fill[self.slots[d].of])] for d in dep]
            for dv in itertools.product(*dep_doms):
                combos.append({**fill, **dict(zip(dep, dv))})
        total = len(combos)
        assert total == self.count(), (total, self.count())
        if cap is not None and total > cap:
            idx = np.random.default_rng(seed).choice(total, cap, replace=False)
            combos = [combos[i] for i in sorted(idx)]
        return total, [(c, self.fill(c)) for c in combos]

    def fill(self, values):
        def sub(x):
            if isinstance(x, str):
                full = re.fullmatch(r"\{(\w+)(\[(\d+)\])?\}", x)
                if full:  # a whole-string placeholder keeps the value's type (list, number)
                    v = values[full.group(1)]
                    return v[int(full.group(3))] if full.group(3) else v
                return re.sub(r"\{(\w+)(\[(\d+)\])?\}", lambda mm: str(values[mm.group(1)][int(mm.group(3))] if mm.group(3) else values[mm.group(1)]), x)
            if isinstance(x, list):
                return [sub(i) for i in x]
            if isinstance(x, dict):
                return {k: sub(v) for k, v in x.items()}
            return x
        return SceneRequest.model_validate(sub(self.request))


def schema_json():
    """The JSON schema a model must answer in (mode A)."""
    return json.dumps(SceneRequest.model_json_schema())
