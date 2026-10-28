"""The simulator interface the framework talks to (docs/TASKGEN_DESIGN.html §2a). core/ never imports a simulator;
each simulator layer (sim_mujoco/, later others) implements this class. Exploratory, UNVERIFIED.

Lifecycle: load_base(scene) -> add(obj) ... -> build() -> set_robot(q) -> settle / overlaps / render / ...
Units: metres, seconds, robot base frame.
"""
from abc import ABC, abstractmethod


class Sim(ABC):
    name = "abstract"

    @abstractmethod
    def load_base(self, scene_dir):
        """Load a reconstructed base scene. Returns dict(table_z, base_objects={name: kind})."""

    @abstractmethod
    def add(self, obj):
        """Queue a neutral object (core.objects) to add; takes effect at build()."""

    @abstractmethod
    def build(self):
        """Compile the base + queued objects into a runnable model (never by patching a compiled file)."""

    @abstractmethod
    def set_robot(self, q_arm):
        """Put the arm at joint angles q_arm and hold it there (position targets); gripper open."""

    @abstractmethod
    def free_objects(self):
        """Names of the movable objects (base + added)."""

    @abstractmethod
    def position(self, name):
        """World position of a body (3,)."""

    @abstractmethod
    def settle(self, seconds):
        """Step with the arm held; returns {object: distance moved (m)}."""

    @abstractmethod
    def overlaps(self, tol=-0.001):
        """Interpenetrations deeper than -tol between two different objects: [dict(pair=[a, b], depth_cm=..)]."""

    @abstractmethod
    def warnings(self):
        """The simulator's own runtime warnings that fired: {name: count}."""

    @abstractmethod
    def joint(self, name):
        """Position of a 1-dof joint."""

    @abstractmethod
    def push_joint(self, name, force, seconds):
        """Apply a constant generalised force on a 1-dof joint while stepping."""

    @abstractmethod
    def set_pose(self, name, x, y, z_bottom, yaw):
        """Move a free object so its collision footprint is centred at (x, y), its lowest point at z_bottom, yawed."""

    @abstractmethod
    def snapshot(self):
        """Opaque copy of the full dynamic state."""

    @abstractmethod
    def restore(self, snap):
        """Return to a snapshot."""

    @abstractmethod
    def collision_points(self, name):
        """World-frame points (N, 3) on the collision shapes of a top-level object (box corners, mesh vertices)."""

    @abstractmethod
    def reachable(self, point):
        """Residual (m) of a top-down gripper pose at `point` from the current arm pose (IK)."""

    @abstractmethod
    def visibility(self, name, view, height=180, width=320):
        """(visible pixels, pixels it would cover with nothing in front) of an object from a camera, at the policy's
        input size by default."""

    @abstractmethod
    def arm_above(self, point, height):
        """Arm joint angles putting the gripper `height` above `point`, pointing down (IK), or None if unreachable."""

    @abstractmethod
    def render(self, view, height=360, width=640):
        """RGB image (H, W, 3) from 'ext', 'wrist' or 'free'."""

    @abstractmethod
    def save(self, path):
        """Write the compiled model (for a viewer)."""
