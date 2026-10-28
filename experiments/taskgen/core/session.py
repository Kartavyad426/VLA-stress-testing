"""Stage 1 tools: a session that edits a scene config and answers every placement question with the simulator
(docs/TASKGEN_DESIGN.html §3e). Simulator-neutral: the simulator and the asset lookup are injected.

Tools (each returns a short JSON-able dict; errors are {"error": ...} the caller or a model can act on):
  list_scene()                                  base + added objects with sim-measured sizes, table, gripper
  list_assets(query)                            pack models matching a noun, with real size, kind, licence
  add_object(source, colour, size_cm, name)     register an object (not placed yet)
  fits(name, relation, target)                  geometry-only answer + the limiting dimension
  place(name, relation, target, allow_resize)   searches poses, settles physics, resizes within bounds if allowed
  remove_object(name)
Exploratory, UNVERIFIED.
"""
import numpy as np

from core.config import SceneConfig, AddedObject, BaseObject, Relation, base_version, dump
from core.objects import Box, PackAsset, Drawer
from core.geometry import aabb, cavity, horizontal_gap, oriented_size, footprint_box
from core import kinds, place as P

RELATIONS = ("on_table", "next_to", "on", "inside")
SETTLE_S = 1.0
MOVE_LIMIT = 0.02          # the placed object may move this much while settling (checklist B1)
OTHERS_LIMIT = 0.005       # everything else must stay put
CLEARANCE_BASE = 0.015     # horizontal clearance to base objects that are not the target (D8 proposal)
CLEARANCE_ADDED = 0.005
REACH_TOL = 0.01
BASE_KIND = {"box": "block", "bowl": "bowl", "ellipsoid": "item", "asset": "item"}


def cm(v):
    return [round(100 * float(x), 1) for x in v]


class Session:
    def __init__(self, sim_factory, assets, scene_dir, episode, q_start):
        self.factory, self.assets, self.q0 = sim_factory, assets, np.asarray(q_start)
        s = sim_factory()
        info = s.load_base(scene_dir)
        self.fit, self.table_z = s.fit, info["table_z"]
        self.table = dict(z=self.table_z, xy_lo=self.fit["table"]["xy_lo"], xy_hi=self.fit["table"]["xy_hi"])
        self.base_bodies = {o.get("body", lab.replace(" ", "_")): BASE_KIND.get(o["kind"], "item") for lab, o in self.fit["objects"].items()}
        self.cfg = SceneConfig(base_scene=scene_dir, base_version=base_version(scene_dir), episode=episode, simulator=s.name,
                               base_objects={b: BaseObject(kind=k) for b, k in self.base_bodies.items()})

    # ---------- building ----------
    def _neutral(self, a: AddedObject, pose=None):
        x, y, zb, yaw = pose or a.pose
        if a.kind == "drawer":
            return Drawer(a.name, (x, y))
        if a.source.startswith("pack:"):
            return PackAsset(a.name, a.source[5:], (x, y), scale=a.scale, z_bottom=zb, yaw=yaw)
        return Box(a.name, tuple(a.size_m), tuple(a.rgba), (x, y), z_bottom=zb, yaw=yaw)

    def build(self, extra=None, exclude=()):
        """A simulator with the base, every placed added object (except `exclude`) and `extra` (neutral objects)."""
        s = self.factory()
        s.load_base(self.cfg.base_scene)
        for a in self.cfg.added_objects.values():
            if a.pose is not None and a.name not in exclude:
                s.add(self._neutral(a))
        for o in extra or []:
            s.add(o)
        s.build()
        s.set_robot(self.q0)
        return s

    def _body(self, name):
        a = self.cfg.added_objects.get(name)
        return f"{name}_cabinet" if a is not None and a.kind == "drawer" else name

    def _target(self, s, name):
        """Geometry the placement logic needs about a target, measured in the simulator."""
        box = aabb(s.collision_points(self._body(name)))
        t = dict(aabb=box)
        a = self.cfg.added_objects.get(name)
        if a is not None and a.kind == "drawer":
            d = Drawer(a.name, a.pose[:2])
            c, h, floor = d.inner(self.table_z)
            shift = s.joint(d.body_names()["joint"])
            t["inner"] = ((c[0] + shift, c[1], c[2]), h, floor)
        else:
            cv = cavity(s.collision_points(name))
            if cv:
                t["cavity"] = cv
        return t

    def _exists(self, name):
        return name in self.base_bodies or (name in self.cfg.added_objects and self.cfg.added_objects[name].pose is not None)

    def _log(self, call, args, result):
        self.cfg.log.append(dict(by="tool", call=call, args=args, result=result))
        return result

    # ---------- tools ----------
    def list_scene(self):
        s = self.build()
        out = dict(base={}, added={}, table=dict(z_cm=round(100 * self.table_z, 1), x_cm=cm([self.table["xy_lo"][0], self.table["xy_hi"][0]]),
                                                 y_cm=cm([self.table["xy_lo"][1], self.table["xy_hi"][1]])),
                   gripper_opening_cm=100 * kinds.GRIPPER_OPENING_M)
        for b, k in self.base_bodies.items():
            box = aabb(s.collision_points(b))
            row = dict(kind=k, locked=True, size_cm=cm(oriented_size(s.collision_points(b))), centre_cm=cm(box["centre"]))
            cv = cavity(s.collision_points(b)) if k == "bowl" else None
            if cv:
                row["opening_cm"], row["inner_depth_cm"] = round(200 * cv["inner_radius"], 1), round(100 * cv["depth"], 1)
            out["base"][b] = row
        for n, a in self.cfg.added_objects.items():
            if a.pose is None:
                out["added"][n] = dict(kind=a.kind, size_cm=cm(a.size_m), placed=False)
                continue
            box = aabb(s.collision_points(self._body(n)))
            row = dict(kind=a.kind, size_cm=cm(oriented_size(s.collision_points(self._body(n)))), centre_cm=cm(box["centre"]),
                       relation=dump(a.relation))
            if a.kind == "drawer":
                c, h, _ = self._target(s, n)["inner"]
                row["inside_cm"] = cm(2 * np.array(h))
            out["added"][n] = row
        return out

    def list_assets(self, query, limit=8):
        keys = self.assets.search(query, limit)
        if not keys:
            return dict(error=f"no pack model matches '{query}'", hint="use a primitive (primitive:box) or another noun")
        return dict(assets=[dict(source=f"pack:{k}", kind=(i := self.assets.info(k))["kind"], size_cm=cm(i["size_m"]),
                                 licence=i["licence"]) for k in keys])

    def add_object(self, source, colour=None, size_cm=None, name=None):
        args = dict(source=source, colour=colour, size_cm=size_cm, name=name)
        if source == "primitive:box":
            colour = colour or "red"
            if colour not in kinds.PALETTE:
                return self._log("add_object", args, dict(error=f"unknown colour '{colour}'", colours=list(kinds.PALETTE)))
            e = float(size_cm or 3.0) / 100
            kind, size, rgba, scale, lic = "block", (e, e, e), kinds.PALETTE[colour], 1.0, None
        elif source == "primitive:drawer":
            d = Drawer("tmp", (0, 0))
            kind, size, rgba, scale, lic, colour = "drawer", d.outer, None, 1.0, None, None
        elif source.startswith("pack:"):
            key = source[5:]
            if not self.assets.exists(key):  # a noun: the first match (catalogue order) whose real size is within bounds
                hits = self.assets.search(key, 50)
                if not hits:
                    return self._log("add_object", args, dict(error=f"no pack model '{key}'", hint="call list_assets first"))
                ok_hits = [h for h in hits if kinds.within_bounds(self.assets.info(h)["kind"], self.assets.info(h)["size_m"])[0]]
                key = (ok_hits or hits)[0]
            i = self.assets.info(key)
            kind, size, rgba, scale, lic, colour = i["kind"], i["size_m"], None, 1.0, i["licence"], None
            source = f"pack:{key}"
            if size_cm:
                k = float(size_cm) / 100 / kinds.bound_dim(kind, size) if kind in kinds.BOUNDS else float(size_cm) / 100 / max(size)
                size, scale = tuple(float(k * v) for v in size), k
        else:
            return self._log("add_object", args, dict(error=f"unknown source '{source}'",
                                                      sources=["primitive:box", "primitive:drawer", "pack:<key or noun>"]))
        ok, b = kinds.within_bounds(kind, size)
        if not ok:
            return self._log("add_object", args, dict(error=f"{kind} size {100 * b[2]:.1f} cm outside bounds {100 * b[0]:.0f}-{100 * b[1]:.0f} cm",
                                                      hint="pass size_cm within the bounds"))
        n = name or f"{colour + '_' if colour else ''}{kind}_{1 + sum(a.kind == kind for a in self.cfg.added_objects.values())}"
        if n in self.base_bodies or n in self.cfg.added_objects:
            return self._log("add_object", args, dict(error=f"name '{n}' already used"))
        self.cfg.added_objects[n] = AddedObject(name=n, source=source, kind=kind, colour=colour, rgba=rgba, size_m=size, scale=scale, licence=lic)
        return self._log("add_object", args, dict(added=n, kind=kind, size_cm=cm(size), graspable=kinds.graspable(size), placed=False))

    def fits(self, name, relation, target=None):
        a = self.cfg.added_objects.get(name)
        if a is None:
            return dict(error=f"'{name}' is not an added object")
        if target and not self._exists(target):
            return dict(error=f"target '{target}' is not a placed object in the scene")
        s = self.build(exclude=(name,))
        t = self._target(s, target) if target else None
        ok, why, k = P.fits(relation, a.size_m, t, self.table)
        return dict(fits=ok, reason=why, max_scale=None if k is None else round(k, 3))

    def remove_object(self, name):
        if name in self.base_bodies:
            return self._log("remove_object", dict(name=name), dict(error=f"'{name}' is a locked base object"))
        if self.cfg.added_objects.pop(name, None) is None:
            return dict(error=f"no added object '{name}'")
        return self._log("remove_object", dict(name=name), dict(removed=name))

    def _resize(self, a, k, why):
        """Scale an added object by k within its kind's bounds; False if that leaves the bounds."""
        new = tuple(float(k * v) for v in a.size_m)
        ok, _ = kinds.within_bounds(a.kind, new)
        if not ok or a.kind not in kinds.BOUNDS:
            return False
        a.resized_from_m = a.resized_from_m or a.size_m
        a.size_m, a.scale = new, a.scale * k
        a.resize_reason = why
        return True

    def place(self, name, relation, target=None, allow_resize=True, seed=0, n=24, rounds=3):
        args = dict(name=name, relation=relation, target=target, allow_resize=allow_resize)
        if name in self.base_bodies:
            return self._log("place", args, dict(infeasible=f"'{name}' is a locked base object; only added objects can be placed or resized"))
        a = self.cfg.added_objects.get(name)
        if a is None:
            return self._log("place", args, dict(error=f"no added object '{name}'", hint="add_object first"))
        if relation not in RELATIONS:
            return self._log("place", args, dict(error=f"unknown relation '{relation}'", relations=list(RELATIONS)))
        if relation != "on_table" and (target is None or not self._exists(target) or target == name):
            return self._log("place", args, dict(error=f"target '{target}' is not a placed object in the scene"))
        if a.kind == "drawer" and relation != "on_table":
            return self._log("place", args, dict(infeasible="a drawer stands on the table only"))
        a.pose, a.relation = None, None
        s = self.build()
        t = self._target(s, target) if target else None
        ok, why, k = P.fits(relation, a.size_m, t, self.table)
        if not ok:
            if allow_resize and k and self._resize(a, 0.97 * k, f"did not fit {relation} {target}: {why}"):
                pass
            else:
                lo = kinds.BOUNDS.get(a.kind)
                extra = f"; smallest allowed {a.kind} is {100 * lo[1]:.0f} cm" if lo else ""
                return self._log("place", args, dict(infeasible=f"{name} {relation} {target}: {why}{extra}"))
        rng = np.random.default_rng(seed)
        reasons = {}
        for r in range(rounds):
            fp = Drawer(a.name, (0, 0)).footprint() if a.kind == "drawer" else None
            cands = P.candidates(relation, a.size_m, self.table, t, rng, 8 * n, footprint=fp)
            res = self._try(a, relation, target, t, cands, reasons)
            if res:
                a.pose, a.relation = res, Relation(type=relation, target=target)
                out = dict(placed=name, relation=relation, target=target, at_cm=cm(res[:3]), size_cm=cm(a.size_m))
                if a.resized_from_m:
                    out["resized"] = f"{cm(a.resized_from_m)} -> {cm(a.size_m)} cm ({a.resize_reason})"
                return self._log("place", args, out)
            if not (allow_resize and self._resize(a, 0.85, f"no free pose {relation} {target or 'table'} at the original size")):
                break
        top = sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
        return self._log("place", args, dict(infeasible=f"no pose found for {name} {relation} {target or 'table'}",
                                             tried=sum(reasons.values()), reasons=dict(top)))

    def _footprint(self, a, c):
        if a.kind == "drawer":
            lo, hi = Drawer(a.name, c[:2]).footprint()
            return footprint_box((c[0] + (lo[0] + hi[0]) / 2, c[1] + (lo[1] + hi[1]) / 2), ((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2))
        r = float(np.linalg.norm(a.size_m[:2])) / 2  # any yaw
        return footprint_box(c[:2], (r, r))

    def _prefilter(self, a, relation, target, cands, boxes, reasons, keep):
        """Cheap geometry before physics: usable table and clearance to every non-target object."""
        out = []
        for c in cands:
            fp = self._footprint(a, c)
            if relation != "inside" and not P.table_ok(fp, self.table):
                reasons["off the usable table (pre-filter)"] = reasons.get("off the usable table (pre-filter)", 0) + 1
                continue
            near = next((o for o, b in boxes.items() if o != target and
                         horizontal_gap(fp, b) < (CLEARANCE_BASE if o in self.base_bodies else CLEARANCE_ADDED)), None)
            if near and relation in ("on_table", "next_to"):
                reasons[f"too close to {near} (pre-filter)"] = reasons.get(f"too close to {near} (pre-filter)", 0) + 1
                continue
            out.append(c)
            if len(out) == keep:
                break
        return out

    def _try(self, a, relation, target, t, cands, reasons):
        """Try candidates; return the settled pose (x, y, z_bottom, yaw) of the first that passes, else None."""
        def fail(r):
            reasons[r] = reasons.get(r, 0) + 1
        s0 = self.build()
        boxes = {o: aabb(s0.collision_points(self._body(o))) for o in list(self.base_bodies) +
                 [n for n, x in self.cfg.added_objects.items() if x.pose is not None and n != a.name]}
        cands = self._prefilter(a, relation, target, cands, boxes, reasons, keep=12 if a.kind == "drawer" else 24)
        if not cands:
            return None
        if a.kind == "drawer":  # fixed body: one build per candidate
            for c in cands:
                s = self.build(extra=[self._neutral(a, c)])
                ok = self._check(s, a, relation, t, target, c, moving=False)
                if ok is True:
                    return c
                fail(ok)
            return None
        s = self.build(extra=[self._neutral(a, cands[0])])
        start = s.snapshot()
        for c in cands:
            s.restore(start)
            s.set_pose(a.name, *c)
            ok = self._check(s, a, relation, t, target, c, moving=True)
            if ok is True:  # settled footprint centre and lowest point; yaw as placed (settling turns it little)
                box = aabb(s.collision_points(a.name))
                return (float(box["centre"][0]), float(box["centre"][1]), float(box["lo"][2]), float(c[3]))
            fail(ok)
        return None

    def _check(self, s, a, relation, t, target, pose, moving):
        body = self._body(a.name)
        box = aabb(s.collision_points(body))
        if relation != "inside" and not P.table_ok(box, self.table):
            return "off the usable table"
        for o in list(self.base_bodies) + [n for n, x in self.cfg.added_objects.items() if x.pose is not None and n != a.name]:
            if o == target:
                continue
            g = horizontal_gap(box, aabb(s.collision_points(self._body(o))))
            if g < (CLEARANCE_BASE if o in self.base_bodies else CLEARANCE_ADDED):
                return f"too close to {o}"
        if any(body in ov["pair"] for ov in s.overlaps()):
            return "overlaps at start"
        others = [n for n in s.free_objects() if n != body]
        before = {n: s.position(n) for n in others}
        moved = s.settle(SETTLE_S)
        if moving and moved.get(body, 0) > MOVE_LIMIT:
            return "moved while settling"
        if any(np.linalg.norm(s.position(n) - p) > OTHERS_LIMIT for n, p in before.items()):
            return "disturbed another object"
        if s.warnings():
            return "simulator warning"
        box = aabb(s.collision_points(body))
        t2 = self._target(s, target) if target else None
        ok, why = P.holds(relation, box, t2, self.table, None)
        if not ok:
            return why
        reach_pt = (box["centre"][0], box["centre"][1], box["hi"][2] - 0.01)
        if a.kind == "drawer":  # the arm must reach into the open drawer
            c, _, floor = Drawer(a.name, pose[:2]).inner(self.table_z)
            reach_pt = (c[0], c[1], floor + 0.02)
        if s.reachable(reach_pt) > REACH_TOL:
            return "out of the arm's reach"
        return True
