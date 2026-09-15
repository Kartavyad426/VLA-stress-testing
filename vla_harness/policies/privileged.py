"""B6/DG-6 -- the audited escape hatch for §7c discriminator 4.

Discriminator 4 ("give it ground-truth object pose instead of pixels")
distinguishes a PERCEPTION failure from a CONTROL failure: if the policy
succeeds with perfect state, the problem is upstream of control.

It could not be run. `Observation.policy_view()` strips every `_gt_` key and
ARCHITECTURE.md §3 makes that ENFORCED rather than conventional -- which is the
architecture's best feature and must not be weakened. A
`policy_view(allow_privileged=True)` flag would make the strict path optional at
every call site, which is precisely the convention-vs-enforcement weakness that
produced the original `_gt_` leak.

So the exception is explicit, opt-in and audited instead: this wrapper receives
the full observation, stamps `privileged: True` on every rollout it produces,
and is excluded from every headline number by the same filter that excludes
tier-3 diagnoses (I13).
"""
from __future__ import annotations

from ..schema import Observation, Action, derive_id


class PrivilegedProbePolicy:
    """Wraps a policy and feeds it privileged simulator state.

    `substitute` maps the full observation onto the keys the wrapped policy
    reads -- e.g. replacing the noisy camera estimate with `_gt_obj_xy`. That
    is the ablation: perfect perception, unchanged control.
    """

    privileged = True          # runner passes the FULL observation

    def __init__(self, inner, substitute=None, probe="gt_object_pose"):
        self.inner = inner
        self.substitute = substitute or (lambda obs: obs)
        self.probe = probe
        self.action_dims = inner.action_dims
        self.policy_id = derive_id(self.identity())

    def identity(self) -> dict:
        inner = self.inner.identity()
        return {"name": f"privileged[{self.probe}]<{inner.get('name')}>",
                "probe": self.probe, "inner": inner, "privileged": True}

    def reset(self) -> None:
        self.inner.reset()

    def __call__(self, obs: Observation) -> Action:
        return self.inner(self.substitute(obs))


def gt_object_pose(obs: Observation) -> Observation:
    """Perfect perception: the policy sees the object exactly where it is.

    Returns a POLICY VIEW (privileged keys stripped) with the camera estimate
    overwritten by ground truth -- so the wrapped policy still cannot read
    `_gt_*` directly, and the only thing that changed is perception quality.
    """
    view = obs.policy_view()
    gt = obs.get("_gt_obj_xy")
    if gt is not None:
        view.state["obj_cam_xy"] = tuple(gt)
        view.state["distractors_cam_xy"] = []
        view.state["_env_camera_yaw_deg"] = 0.0
    return view
