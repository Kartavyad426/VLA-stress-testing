"""Rubric 2, the task config (docs/TASKGEN_DESIGN.html §5, §1a job B): what the robot must do on a scene config.
Simulator-neutral. The do-nothing check (checklist E2) holds the arm at its start pose for the time limit and
evaluates the pass conditions: a valid task must NOT be solved by doing nothing. Exploratory, UNVERIFIED.
"""
import json
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from core.templates import PassCondition
from core import rules


class Check(BaseModel):
    passed: Optional[bool] = None
    detail: Dict = Field(default_factory=dict)


class TaskConfig(BaseModel):
    id: str
    source: Literal["preset", "agent", "user"]
    template: Optional[str] = None
    slot_values: Dict = Field(default_factory=dict)
    scene_config: str                    # path of the scene config (rubric 1)
    base_version: str
    instruction: str
    targets: List[str]
    pass_all: List[PassCondition]
    start: Dict = Field(default_factory=dict)     # arm joints (episode frame 0), articulation, placement seed
    time_limit_s: float = 30.0
    others_must_stay: List[str] = Field(default_factory=list)  # optional: objects that must not move (V2, V3, V10)
    checks: Dict[str, Check] = Field(default_factory=dict)     # rule_known (E4), do_nothing (E2), expert (E1)
    status: Literal["draft", "checked", "failed", "approved"] = "draft"

    def save(self, path):
        open(path, "w").write(self.model_dump_json(indent=1))


KNOWN = {"inside", "out_of", "on", "stacked", "ordered_stack", "next_to"}


def rule_known(task):
    bad = [c.type for c in task.pass_all if c.type not in KNOWN]
    return Check(passed=not bad and bool(task.pass_all), detail=dict(unknown=bad))


def do_nothing(task, session, step_s=None):
    """Hold the arm at the start pose for the time limit; the task must not count as solved."""
    s = session.build()
    s.settle(step_s or task.time_limit_s)
    ok, rows = rules.check_all(task.pass_all, session, s)
    return Check(passed=not ok, detail=dict(solved_by_doing_nothing=ok, seconds=step_s or task.time_limit_s, rules=rows,
                                            warnings=s.warnings()))


def summarise(task):
    c = task.checks
    need = ("rule_known", "do_nothing", "expert")
    if any(k in c and c[k].passed is False for k in need):
        task.status = "failed"
    elif all(k in c and c[k].passed for k in need):
        task.status = "checked"
    return task.status


def dump(task):
    return json.loads(task.model_dump_json())
