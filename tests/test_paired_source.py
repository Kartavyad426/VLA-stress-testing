"""R-039 paired source: the pieces that turn "render the nominal condition at
the current sim state" into the SOURCE features the splice consumes.

Three seams, each tested on its own:
  * CaptureSink.paused()  -- the nominal pass runs the same vlln hooks the
    capture taps; it must not stage into the forward being assembled.
  * vla_harness.envs.nominal -- pure helpers: which knobs a LIBERO-Plus variant
    name perturbs, the per-domain nominal camera, and the base scene's lights.
  * LeRobotPolicy.features_for(obs) -- backbone + encode on a nominal
    Observation WITHOUT acting, queueing, or counting a forward.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pytest
import torch

from conftest import FakePolicy, STATE_DIM, VL_DIM, VL_TOKENS

from vla_harness.capture import CaptureSink
from vla_harness.capture.taps import Role
from vla_harness.envs import nominal
from vla_harness.policies.lerobot_policy import LeRobotPolicy
from vla_harness.schema import Observation

FORK = os.path.join(os.path.dirname(__file__), "..", "third_party", "LIBERO-plus", "libero", "libero")


# --- CaptureSink.paused -------------------------------------------------------

def test_paused_sink_drops_stages_and_notes_and_resumes_after():
    sink = CaptureSink()
    with sink.paused():
        sink.stage(Role.VL_ENCODER, torch.ones(1))
        sink.note("k", 1)
    assert sink._pending == {}
    sink.stage(Role.VL_ENCODER, torch.ones(1))
    assert Role.VL_ENCODER.value in sink._pending


# --- variant name parsing -------------------------------------------------------

def test_view_variant_name_yields_its_five_camera_params():
    p = nominal.variant_perturbations(
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_120_0_0_initstate_0")
    assert p["view"] == (0, 0, 1.2, 0, 0)
    assert p["light"] is None and p["noise"] == 0


def test_noise_variant_name_yields_noise_level_and_a_canonical_view():
    p = nominal.variant_perturbations(
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_9")
    assert p["noise"] == 9
    assert p["view"] is None, "scale 1.00 and all-zero angles is the trained view"


def test_light_variant_name_yields_light_id():
    p = nominal.variant_perturbations(
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_light_12")
    assert p["light"] == 12 and p["view"] is None and p["noise"] == 0


def test_base_task_name_perturbs_nothing():
    p = nominal.variant_perturbations(
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate")
    assert p == {"view": None, "light": None, "noise": 0}


# --- nominal camera, checked against the fork's own source ----------------------

@pytest.mark.parametrize("domain", ["tabletop", "kitchen_tabletop", "floor",
                                    "living_room_tabletop", "study_tabletop",
                                    "coffee_table"])
def test_nominal_camera_matches_the_forks_setup_camera_constants(domain):
    pos, quat = nominal.nominal_camera(domain)
    src = open(os.path.join(FORK, "envs", "problems", f"libero_{domain}_manipulation.py")).read()
    m = re.search(r"pos_av = \[([^\]]+)\]", src)
    assert m, "fork source moved"
    want = [float(x) for x in m.group(1).split(",")]
    assert np.allclose(pos, want, atol=1e-9)
    assert len(quat) == 4 and abs(np.linalg.norm(quat) - 1) < 1e-4


def test_nominal_camera_refuses_an_unknown_domain():
    with pytest.raises(KeyError):
        nominal.nominal_camera("moon_base")


# --- base scene lookup and its lights -------------------------------------------

def test_light_variant_scene_maps_to_its_domains_base_scene():
    assert nominal.base_scene_for("scenes/lights/tabletop_light_sync_modified_1005.xml") \
        == "scenes/libero_tabletop_base_style.xml"
    assert nominal.base_scene_for("scenes/lights/kitchen_light_sync_modified_410.xml") \
        == "scenes/libero_kitchen_tabletop_base_style.xml"


def test_base_scene_maps_to_itself():
    assert nominal.base_scene_for("scenes/libero_floor_base_style.xml") \
        == "scenes/libero_floor_base_style.xml"


def test_nominal_lights_are_read_by_name_from_the_base_scene():
    lights = nominal.nominal_lights(os.path.join(FORK, "assets", "scenes", "libero_tabletop_base_style.xml"))
    assert set(lights) == {"light1", "light2"}
    assert lights["light1"]["diffuse"] == pytest.approx([0.8, 0.8, 0.8])
    assert lights["light1"]["dir"] == pytest.approx([0.0, -0.15, -1.0])
    assert lights["light1"]["specular"] == pytest.approx([0.3, 0.3, 0.3])
    assert lights["light2"]["pos"] == pytest.approx([-3.0, -3.0, 4.0])


# --- LeRobotPolicy.features_for -------------------------------------------------

class _FakeGroot:
    """`_groot_model` as the splice source needs it: prepare_input, backbone,
    action_head. Mirrors GR00TN17.get_action (groot_n1_7.py:874-877) minus
    the head's denoise."""

    def __init__(self, head):
        self.action_head = head
        self.calls = 0

    def prepare_input(self, inputs):
        return inputs, {"state": inputs["observation.state"].view(1, 1, -1)}

    def backbone(self, vl_input):
        self.calls += 1
        torch.manual_seed(int(vl_input["observation.images.image"].sum().item() * 1e3) % 10_000)
        mask = torch.zeros(1, VL_TOKENS, dtype=torch.bool)
        mask[0, :VL_TOKENS - 3] = True
        return {"backbone_features": torch.randn(1, VL_TOKENS, VL_DIM), "image_mask": mask}


def _wired_policy(with_sink: bool):
    p = LeRobotPolicy(checkpoint="x", device="cpu")
    fake = FakePolicy()
    fake._groot_model = _FakeGroot(fake.action_head)
    fake._action_queue = []
    p._policy = fake
    p._pre = lambda b: b
    p._post = None
    p._env_pre = p._env_post = None
    p.state_fn = lambda obs: np.zeros(STATE_DIM, dtype=np.float32)
    if with_sink:
        p._sink = CaptureSink()
    return p


def _obs(seed: int) -> Observation:
    rng = np.random.default_rng(seed)
    frames = {"image": rng.integers(0, 255, (8, 8, 3), dtype=np.uint8),
              "image2": rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)}
    return Observation(instruction="pick up the bowl", state={}, t=0, frames=frames)


def test_features_for_returns_the_three_source_tensors():
    p = _wired_policy(with_sink=False)
    f = p.features_for(_obs(1))
    assert set(f) >= {"backbone_features", "state_features", "image_mask"}
    assert f["backbone_features"].shape == (1, VL_TOKENS, VL_DIM)
    assert f["image_mask"].shape == (1, VL_TOKENS)


def test_features_for_neither_acts_nor_counts_a_forward_nor_touches_the_queue():
    p = _wired_policy(with_sink=False)
    p._policy._action_queue.append("cached")
    before = p.model_forwards
    p.features_for(_obs(1))
    assert p.model_forwards == before
    assert p._policy._action_queue == ["cached"]


def test_features_for_leaves_nothing_pending_in_the_capture_sink():
    from vla_harness.capture.groot_features import attach_groot_capture
    p = _wired_policy(with_sink=True)
    detach = attach_groot_capture(p._policy, p._sink)      # the real vlln hooks
    try:
        # control: those hooks DO stage when the head encodes unpaused
        p._policy.action_head._encode_features(
            p._policy._groot_model.backbone(p._build_batch(_obs(1))), {"state": torch.zeros(1, 1, STATE_DIM)})
        assert p._sink._pending != {}
        p._sink._pending = {}
        p.features_for(_obs(1))
        assert p._sink._pending == {}
        assert p._sink.current() == []
    finally:
        detach()


def test_features_for_is_a_function_of_the_frames():
    p = _wired_policy(with_sink=False)
    a = p.features_for(_obs(1))["backbone_features"]
    b = p.features_for(_obs(1))["backbone_features"]
    c = p.features_for(_obs(2))["backbone_features"]
    assert torch.equal(a, b) and not torch.equal(a, c)
