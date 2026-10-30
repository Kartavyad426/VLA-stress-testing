"""TaskGen chat: a local browser chat with the agent (user 2026-10-29: "chat kind of thing in browser"). Interface only:
the agent (agent/mode_a3.py), the tools (core/session.py) and the checks (core/taskconfig.py, core/experts.py) are
the existing ones; every step the agent takes is streamed to the page as it happens (model call, validation, each
tool call with its result, the second call, the task config), then the scene is rendered.

Local only (127.0.0.1). Start from the main checkout:
    runs/real2sim/v1_dev/py_triton_first.sh ../VLA-exp-taskgen/experiments/taskgen/chat/server.py --port 8765
then open http://127.0.0.1:8765. Models: NIM (key from robotics_agentic/.env) or "fake" (a fixed answer, no network).
Exploratory, UNVERIFIED.
"""
import argparse, asyncio, base64, json, os, shutil, subprocess, sys, threading, time
import cv2
import h5py
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from sim_mujoco.base import MAIN  # noqa: E402
from sim_mujoco.sim import MujocoSim  # noqa: E402
from sim_mujoco.expert import Expert  # noqa: E402
from sim_mujoco import assets  # noqa: E402
from core.session import Session  # noqa: E402
from core.taskconfig import TaskConfig, Check, do_nothing, summarise  # noqa: E402
from core import experts  # noqa: E402
from agent.models import Nim, FakeModel  # noqa: E402
from agent import mode_a3  # noqa: E402

WT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(WT, "runs/taskgen/chat")
SCENES = {"orange": ("runs/real2sim/v1/orange", "data/real2sim/droid/ep_orange_block"),
          "c11": ("runs/real2sim/v1/c11", "data/real2sim/droid/ep_c11"),
          "c1b": ("runs/real2sim/v1/c1b", "data/real2sim/droid/ep_c1b"),
          "c9": ("runs/real2sim/v1/c9", "data/real2sim/droid/ep_c9"),
          "orange_stereo": ("runs/real2sim/ep_orange_block_stereo", "data/real2sim/droid/ep_orange_block")}  # pre-v1 build: has the composite assets
MODELS = ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-lightning-30b-a3b", "poolside/laguna-xs-2.1", "openai/gpt-oss-20b", "fake"]  # 10-29 12:22: ultra and laguna added (NIM ping OK); gpt-oss-20b not responding then
FAKE_ANSWER = json.dumps({"scene": {"add": [
    {"name": "cabinet_1", "source": "cabinet", "levels": 3, "open_level": "middle", "relation": "on_table"},
    {"name": "red_block_1", "source": "box", "colour": "red", "size_cm": 3, "relation": "inside", "target": "cabinet_1/middle"}]},
    "task": {"instruction": "take the red block out of the middle drawer and put it in the black bowl", "targets": ["red_block_1"],
             "pass_all": [{"type": "inside", "object": "red_block_1", "target": "bowl_1"}]}, "infeasible": []})


POLICIES = ["expert", "groot"]  # expert: TaskGen's scripted expert in the policy's sim + cameras (CPU; proof the task can be solved);
# groot: GPU via run_groot.sh, only if the server was started with --allow_groot. (hold stays in run_policy.py, CLI only)
TASKGEN = HERE  # experiments/taskgen
# res31's one-command builder for a base scene's composite assets (plate, light, wrist plate, wrist table texture, gripper
# library, wrist light), 10-30: CPU only, ~10.5 min on v1/orange, exit 0 only if every step + check passes, last stdout
# line = <scene>/composite_assets.json, idempotent (existing outputs skipped). argv with {scene} / {episode} placeholders.
BUILD_ASSETS_CMD = [os.path.join(MAIN, "runs/real2sim/v1_dev/py_triton_first.sh"), os.path.join(MAIN, "experiments/real2sim/build_composite_assets.py"),
                    "{scene}", "{episode}"]


def png_data_url(img):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


class Chat:
    def __init__(self, scene="orange", model="openai/gpt-oss-20b"):
        self.set_scene(scene)
        self.model_name = model
        self.busy = False

    def set_scene(self, scene):
        sd, ep = SCENES[scene]
        q0 = h5py.File(os.path.join(MAIN, ep, "trajectory.h5"))["observation/robot_state/joint_positions"][0]
        self.S = Session(MujocoSim, assets, os.path.join(MAIN, sd), os.path.join(MAIN, ep), q0)
        self.scene = scene  # only after the scene loaded: a failed load keeps the old scene and its name
        self.task, self.history = None, []
        if not hasattr(self, "turns"):
            self.turns, self.turn = [], 0
            self.session_id = time.strftime("%Y%m%d_%H%M%S")
            self.trace_dir = os.path.join(OUT, "traces", self.session_id)
            os.makedirs(self.trace_dir, exist_ok=True)

    def record(self, kind, text, events, reply, images=None, video=None):
        """Observability O1: everything about one turn, on disk: the prompt, every event (incl. the exact messages sent
        to the model), the reply, scene + task config after the turn, renders; plus a row for the sidebar."""
        self.turn += 1
        n = self.turn
        base = os.path.join(self.trace_dir, f"turn_{n:03d}")
        scfg = base + "_scene_config.json"
        self.S.cfg.save(scfg)
        tcfg = None
        if self.task is not None:
            tcfg = base + "_task_config.json"
            self.task.save(tcfg)
        imgs = {}
        for k, v in (images or {}).items():
            p = f"{base}_{k}.png"
            open(p, "wb").write(base64.b64decode(v.split(",", 1)[1]))
            imgs[k] = p
        added = [n_ for n_, a in self.S.cfg.added_objects.items() if a.pose is not None]
        row = dict(turn=n, time=time.strftime("%H:%M:%S"), kind=kind, text=text, reply=reply, scene=self.scene, model=self.model_name,
                   added=added, task=None if self.task is None else self.task.instruction,
                   status=None if self.task is None else self.task.status, video=video,
                   scene_config=scfg, task_config=tcfg, renders=imgs)
        json.dump(dict(row, events=events), open(base + ".json", "w"), indent=1, default=str)
        self.turns.append(row)
        self.cur_turn = n
        return row

    def open_turn(self, n):
        row = next(r for r in self.turns if r["turn"] == n)
        self.cur_turn = n
        from core.config import SceneConfig
        self.snapshot()
        self.S.cfg = SceneConfig.load(row["scene_config"])
        self.task = TaskConfig(**json.load(open(row["task_config"]))) if row["task_config"] else None
        return row

    def resume(self, session_dir):
        """Reload a session's turns from its traces (after a server restart): sidebar rows, replies, and the last scene."""
        import glob
        rows = []
        for f in sorted(glob.glob(os.path.join(session_dir, "turn_[0-9][0-9][0-9].json"))):
            t = json.load(open(f))
            t.pop("events", None)
            rows.append(t)
        if not rows:
            return 0
        self.turns, self.turn = rows, rows[-1]["turn"]
        self.trace_dir, self.session_id = session_dir, os.path.basename(session_dir)
        if rows[-1]["scene"] != self.scene:
            self.set_scene(rows[-1]["scene"])
        self.open_turn(rows[-1]["turn"])
        self.history = []
        return len(rows)

    def model(self):
        return FakeModel([FAKE_ANSWER]) if self.model_name == "fake" else Nim(self.model_name)

    def snapshot(self):
        self.history.append((self.S.cfg.model_copy(deep=True), None if self.task is None else self.task.model_copy(deep=True)))

    def undo(self):
        if not self.history:
            return False
        self.S.cfg, self.task = self.history.pop()
        return True

    def labels(self):
        """Top-level body -> label: every object's scene name (base ones with their colour), so prompts can use them."""
        out = {}
        for b in self.S.base_bodies:
            c = self.S.colour_of(b)
            out[b] = f"{b} ({c})" if c and c not in b else b
        for n, a in self.S.cfg.added_objects.items():
            if a.pose is not None:
                out[self.S._body(n)] = n
        return out

    def renders(self):
        s = self.S.build()
        lab = self.labels()
        return {v: png_data_url(s.render_labelled(v, lab, 270, 480)) for v in ("ext", "wrist", "free")}

    def summary_text(self, r):
        """The final reply, written from the result (the model answers in JSON only)."""
        if not r.get("ok"):
            return f"I couldn't produce a valid answer: {r.get('reason')}."
        req, sc = r["request"], r["scene"]
        parts = []
        added = [st["result"]["placed"] for st in sc["steps"] if st["call"] == "place" and "placed" in st["result"]]
        if added:
            parts.append("Added and placed: " + ", ".join(added) + ".")
        if not sc["ok"]:
            parts.append(f"The scene step stopped: {sc.get('reason')}.")
        for c in sc["changed"]:
            parts.append(c[0].upper() + c[1:] + ".")
        for ref in req.get("infeasible") or []:
            alt = f" Instead: {ref['alternative']}." if ref.get("alternative") else ""
            parts.append(f"Refused ({ref['part']}): {ref['reason']}.{alt}")
        if r.get("task"):
            t = r["task"]
            conds = "; ".join(f"{c['type']}({', '.join(x for x in (c.get('object'), c.get('target')) if x) or ' → '.join(c.get('order') or [])})" for c in t["pass_all"])
            parts.append(f"Task: \"{t['instruction']}\". Passes when: {conds}.")
            if r.get("task_missing_names"):
                parts.append(f"Warning: the task names objects not in the scene: {r['task_missing_names']}.")
        elif req.get("task") is None and not req.get("infeasible"):
            parts.append("No task was asked for.")
        if not parts:
            parts.append("Nothing to do.")
        return " ".join(parts)

    def check(self, emit):
        if self.task is None:
            return dict(error="no task yet: give a prompt with a task first")
        if not self.task.pass_all:
            return dict(error="this task has no pass conditions, so nothing can be checked (no rule exists for what it asks)")
        t = TaskConfig(**{**self.task.model_dump(), "checks": {}})
        emit(dict(type="tool_start", tool="do_nothing", args=dict(seconds=t.time_limit_s)))
        t.checks["rule_known"] = self.task.checks.get("rule_known") or Check(passed=True)
        t.checks["do_nothing"] = do_nothing(t, self.S)
        emit(dict(type="tool_end", tool="do_nothing", ok=bool(t.checks["do_nothing"].passed),
                  result=dict(passed=t.checks["do_nothing"].passed, solved_by_doing_nothing=t.checks["do_nothing"].detail["solved_by_doing_nothing"])))
        emit(dict(type="tool_start", tool="expert", args=dict(steps=[list(x) for x in experts.steps_for(t.pass_all)])))
        ok, detail, ex = experts.run(t, self.S, Expert, record=True)
        t.checks["expert"] = Check(passed=ok, detail=detail)
        summarise(t)
        os.makedirs(os.path.join(OUT, "media"), exist_ok=True)
        name = f"expert_{int(time.time())}.mp4"
        ex.save_video(os.path.join(OUT, "media", name))
        video = f"/media/{name}" if os.path.exists(os.path.join(OUT, "media", name)) else None  # no frames -> no file, no link
        emit(dict(type="tool_end", tool="expert", ok=ok, result=dict(passed=ok, steps=detail["steps"], rules=detail["rules"]), video=video))
        self.task = t
        return dict(status=t.status, video=video)

    def composite(self):
        """What the policy would see: the exterior composite (cleaned real plate + sim) and the wrist view, at the policy's
        320x180, with the runner's own compositors (wrist v2, as GR00T gets it), for scenes whose base has the composite assets (res31 10-29: ep_orange_block_stereo only)."""
        from sim_mujoco.policy_runner import need_assets
        sdir = self.S.cfg.base_scene
        miss = need_assets(sdir, "v2")
        if miss:
            return None, dict(scene=self.scene, base=os.path.relpath(sdir, MAIN), missing=miss)
        from sim_mujoco.policy_runner import compositors
        s = self.S.build()
        comp, wcomp = compositors(self.S, s.m, "v2")  # the runner's own setup: what GR00T gets (wrist v2)
        ext = comp(s.d).clip(0, 255).astype(np.uint8)
        wr = wcomp(s.d).clip(0, 255).astype(np.uint8)
        imgs = {"exterior composite (policy view)": png_data_url(ext), "wrist composite (policy view, v2)": png_data_url(wr)}
        js = os.path.join(sdir, "composite_assets.json")  # res31's builder summary: warn if its checks failed
        if os.path.exists(js):
            summ = json.load(open(js))
            if not summ.get("ok"):
                bad = [f"{k} {v.get('value'):.2f}" for k, v in (summ.get("checks") or {}).items() if isinstance(v, dict) and not v.get("pass_") and v.get("value") is not None]
                imgs["_warning"] = f"this scene's composite assets failed checks ({', '.join(bad)}; thresholds provisional): the policy's view may be off"
        return imgs, None

    def start_policy(self, policy, emit, allow_groot=False):
        """Run a policy on the current turn's scene + task, as a separate process (the chat stays usable meanwhile):
        hold on CPU via run_policy.py; groot on the GPU via run_groot.sh (flock, GPU-idle check, RAM abort), only when
        the server was started with --allow_groot (prim's GPU gate). The result is added to the turn's row and trace."""
        if POLICY.get("proc") is not None:
            return f"a policy run is already going: {POLICY['policy']} on turn {POLICY['turn']} (started {POLICY['started']})"
        from sim_mujoco.policy_runner import need_assets
        miss = need_assets(self.S.cfg.base_scene, "v2")
        if miss:
            emit(dict(type="assets_missing", scene=self.scene, base=os.path.relpath(self.S.cfg.base_scene, MAIN), missing=miss,
                      builder=BUILD_ASSETS_CMD is not None))
            return "this scene has no composite assets, so no policy can see it yet: see the missing-assets message"
        if self.task is None or not self.task.pass_all:
            return "no task with pass conditions on this turn, so a policy run can't be scored"
        if not getattr(self, "cur_turn", None):
            return "no turn yet: give a prompt first"
        if policy not in POLICIES:
            return f"unknown policy {policy}; choose one of {', '.join(POLICIES)}"
        if policy == "groot" and not allow_groot:
            return ("GR00T runs on the GPU, which needs prim's go for each use; this server was started without --allow_groot. "
                    "Ask, then restart the server with --allow_groot (or run run_groot.sh by hand).")
        row = next(r for r in self.turns if r["turn"] == self.cur_turn)
        out = os.path.join(OUT, "policy", f"{self.session_id}_turn{row['turn']:03d}_{policy}_{time.strftime('%H%M%S')}")
        os.makedirs(out, exist_ok=True)
        sc, tc = os.path.join(out, "scene_config.json"), os.path.join(out, "task_config.json")
        self.S.cfg.save(sc)
        self.task.save(tc)
        if policy == "groot":
            cmd = [os.path.join(TASKGEN, "run_groot.sh"), sc, tc, out, "30"]
        else:
            cmd = [os.path.join(MAIN, "runs/real2sim/v1_dev/py_triton_first.sh"), os.path.join(TASKGEN, "run_policy.py"),
                   "--scene_config", sc, "--task_config", tc, "--policy", policy, "--seconds", "30", "--out", out]
        log = open(os.path.join(out, "stdout.log"), "w")
        proc = subprocess.Popen(cmd, cwd=MAIN, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)  # own group: Stop kills it all
        POLICY.update(proc=proc, policy=policy, turn=row["turn"], started=time.strftime("%H:%M:%S"), out=out)
        emit(dict(type="policy_start", policy=policy, turn=row["turn"], out=out, task=self.task.instruction, started=POLICY["started"]))
        threading.Thread(target=self._wait_policy, args=(proc, policy, row, out, log, emit), daemon=True).start()
        return None

    def _wait_policy(self, proc, policy, row, out, log, emit):
        t0 = time.time()
        rc = proc.wait()
        log.close()
        res, video = {}, None
        rj = os.path.join(out, "rollout.json")
        if os.path.exists(rj):
            r = json.load(open(rj))
            res = {k: r.get(k) for k in ("success", "stage_target", "steps", "policy_calls", "wall_s", "stages", "gpu_used_max_gib", "rules")}
        mp4 = os.path.join(out, "rollout.mp4")
        if os.path.exists(mp4):
            os.makedirs(os.path.join(OUT, "media"), exist_ok=True)
            name = os.path.basename(out) + ".mp4"
            shutil.copy(mp4, os.path.join(OUT, "media", name))
            video = f"/media/{name}"
        tail = open(os.path.join(out, "stdout.log")).read().splitlines()[-15:]
        run = dict(policy=policy, started=POLICY.get("started"), rc=rc, wall_s=round(time.time() - t0), success=res.get("success"),
                   stages=res.get("stages"), video=video, out=out)
        row.setdefault("runs", []).append(run)
        tj = os.path.join(self.trace_dir, f"turn_{row['turn']:03d}.json")
        if os.path.exists(tj):
            t = json.load(open(tj))
            t["runs"] = row["runs"]
            json.dump(t, open(tj, "w"), indent=1, default=str)
        POLICY.clear()
        emit(dict(type="policy_end", turn=row["turn"], rc=rc, result=res, run=run, log_tail=None if res else tail))
        emit(dict(type="turns", turns=self.turns, current=getattr(self, "cur_turn", None)))

    def build_assets(self, emit):
        """Build the base scene's composite assets with res31's builder, in the background; then show the composite."""
        sd, ep, name = self.S.cfg.base_scene, self.S.cfg.episode, self.scene  # fixed at the start: the user may switch scenes meanwhile
        if BUILD_ASSETS_CMD is None:
            return ("Can't build yet: res31's one-command builder for composite assets doesn't exist yet (asked 10-30; "
                    "it needs res31 to write it). Until then only orange_stereo has a composite.")
        if POLICY.get("proc") is not None:
            return "a background run is going; build after it ends"
        out = os.path.join(OUT, "assets", f"{self.scene}_{time.strftime('%H%M%S')}")
        os.makedirs(out, exist_ok=True)
        cmd = [c.format(scene=sd, episode=ep) for c in BUILD_ASSETS_CMD]
        log = open(os.path.join(out, "stdout.log"), "w")
        proc = subprocess.Popen(cmd, cwd=MAIN, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        POLICY.update(proc=proc, policy="build_assets", scene=name, turn=None, started=time.strftime("%H:%M:%S"), out=out)

        def wait():
            rc = proc.wait()
            log.close()
            POLICY.clear()
            lines = open(os.path.join(out, "stdout.log")).read().splitlines()
            summ = {}
            js = os.path.join(sd, "composite_assets.json")
            if os.path.exists(js):
                summ = json.load(open(js))
            checks = {k: (v.get("value"), v.get("pass_", v.get("pass"))) if isinstance(v, dict) else v for k, v in (summ.get("checks") or {}).items()}
            emit(dict(type="assets_built", scene=name, rc=rc, ok=bool(summ.get("ok")) and rc == 0, seconds=summ.get("seconds"),
                      peak_ram_gib=summ.get("peak_ram_gib"), checks=checks,
                      steps=[{k: st.get(k) for k in ("step", "status", "seconds", "device")} for st in summ.get("steps") or []],
                      log_tail=lines[-10:], summary=js))
            imgs = {}
            for pth in summ.get("check_images") or []:
                img = cv2.imread(pth)
                if img is not None:
                    hh, ww = img.shape[:2]
                    img = cv2.resize(img, (640, int(hh * 640 / ww))) if ww > 640 else img
                    imgs[os.path.basename(pth)] = png_data_url(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if imgs:
                emit(dict(type="render", images=imgs))
        threading.Thread(target=wait, daemon=True).start()
        return f"Building the composite for {name}: about 8–10 min. You can keep chatting; the result appears here when it's done."

    def new_session(self):
        """New chat: a fresh session (new trace folder, empty sidebar), the scene as reconstructed; old sessions stay on disk."""
        del self.turns
        self.set_scene(self.scene)

    def save(self):
        d = os.path.join(OUT, "saved")
        os.makedirs(d, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        paths = dict(scene_config=os.path.join(d, f"{stamp}_{self.scene}_scene_config.json"))
        self.S.cfg.save(paths["scene_config"])
        if self.task is not None:
            paths["task_config"] = os.path.join(d, f"{stamp}_{self.scene}_task_config.json")
            self.task.scene_config = paths["scene_config"]
            self.task.save(paths["task_config"])
        return paths


app = FastAPI()
chat = None
SUBS = set()          # every open page gets every event (a reloaded page keeps seeing a running request)
RUNNING = {}          # what is running now: kind, text, started
POLICY = {}           # the policy run going now (one at a time): proc, policy, turn, started, out
ALLOW_GROOT = False
STOP = threading.Event()


class Stopped(Exception):
    pass


@app.get("/")
def index():
    return HTMLResponse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")).read())


@app.get("/trace/{turn}")
def trace(turn: int):
    p = os.path.join(chat.trace_dir, f"turn_{turn:03d}.json")
    return FileResponse(p) if os.path.exists(p) else HTMLResponse("not found", 404)


@app.get("/media/{name}")
def media(name):
    p = os.path.join(OUT, "media", os.path.basename(name))
    return FileResponse(p) if os.path.exists(p) else HTMLResponse("not found", 404)


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    SUBS.add((loop, q))

    def emit(ev):  # to every open page
        msg = json.loads(json.dumps(ev, default=str))
        for lp, qq in list(SUBS):
            lp.call_soon_threadsafe(qq.put_nowait, msg)

    async def pump():
        while True:
            await sock.send_json(await q.get())

    pumper = asyncio.create_task(pump())
    # this page only: the scene, the session's past turns (with replies, so a reload keeps the chat), what is running
    q.put_nowait(json.loads(json.dumps(dict(type="hello", scenes=list(SCENES), models=MODELS, scene=chat.scene, model=chat.model_name,
                                            base=chat.S.list_scene()["base"], turns=chat.turns, trace_dir=chat.trace_dir,
                                            running=RUNNING or None, policies=POLICIES,
                                            policy_running={k: v for k, v in POLICY.items() if k != "proc"} or None), default=str)))

    def worker(kind, payload):
        chat.busy = True
        STOP.clear()
        RUNNING.update(kind=kind, text=payload.get("text") or kind, started=time.strftime("%H:%M:%S"))
        events = []

        def em(ev):  # every event to the pages and to the turn's trace (renders saved as files, not in the json)
            if STOP.is_set():
                raise Stopped("stopped by the user")
            emit(ev)
            if ev.get("type") != "render":
                events.append(dict(ev, t=round(time.time(), 3)))
        try:
            if kind == "prompt":
                chat.snapshot()
                r = mode_a3.run(chat.model(), chat.S, assets, payload["text"], task_id=f"chat_{chat.turn + 1}", on_event=em)
                chat.task = TaskConfig(**r["task"]) if r.get("task") else chat.task
                reply = chat.summary_text(r)
                em(dict(type="reply", text=reply))
                imgs = chat.renders()
                emit(dict(type="render", images=imgs))
                row = chat.record("prompt", payload["text"], events, reply, images=imgs)
                emit(dict(type="turns", turns=chat.turns, current=row["turn"]))
            elif kind == "check":
                res = chat.check(em)
                reply = (f"Check: {res['error']}" if "error" in res else
                         f"Check finished: status {res['status']} (do-nothing must fail, the expert must succeed). Expert video below.")
                em(dict(type="reply", text=reply))
                if "error" not in res:
                    row = chat.record("check", "/check", events, reply, video=res["video"])
                    emit(dict(type="turns", turns=chat.turns, current=row["turn"]))
            elif kind == "save":
                emit(dict(type="reply", text="Saved: " + ", ".join(f"{k} → {v}" for k, v in chat.save().items()) +
                          f" · every turn is also traced in {chat.trace_dir}"))
            elif kind == "open_turn":
                row = chat.open_turn(int(payload["turn"]))
                emit(dict(type="reply", text=f"Reopened turn {row['turn']} ({row['time']}): " + (f"task \"{row['task']}\" ({row['status']})" if row['task'] else "no task") +
                          (f"; added {', '.join(row['added'])}" if row['added'] else "") + ". New prompts continue from this scene; Undo goes back."))
                emit(dict(type="render", images=chat.renders()))
                emit(dict(type="turns", turns=chat.turns, current=row["turn"]))
            elif kind == "undo":
                emit(dict(type="reply", text="Undone: back to the scene before the last change." if chat.undo() else "Nothing to undo."))
                emit(dict(type="render", images=chat.renders()))
            elif kind == "reset":
                chat.set_scene(chat.scene)
                emit(dict(type="reply", text=f"Reset: the {chat.scene} scene as reconstructed, nothing added (the session's turns stay in the sidebar)."))
                emit(dict(type="render", images=chat.renders()))
            elif kind == "render":
                emit(dict(type="render", images=chat.renders()))
            elif kind == "composite":
                imgs, miss = chat.composite()
                if imgs:
                    warn = imgs.pop("_warning", None)
                    emit(dict(type="reply", text=("⚠ " + warn + ". " if warn else "") + "What the policy would see (exterior composite over the real plate; wrist view v2). Added objects are drawn with their model colours."))
                    emit(dict(type="render", images=imgs))
                else:
                    emit(dict(miss, type="assets_missing", builder=BUILD_ASSETS_CMD is not None))
            elif kind == "build_assets":
                emit(dict(type="reply", text=chat.build_assets(emit)))
            elif kind == "new":
                chat.new_session()
                emit(dict(type="hello", scenes=list(SCENES), models=MODELS, scene=chat.scene, model=chat.model_name, base=chat.S.list_scene()["base"],
                          turns=chat.turns, trace_dir=chat.trace_dir, new=True))
                emit(dict(type="render", images=chat.renders()))
            elif kind == "scene":
                chat.set_scene(payload["scene"])
                emit(dict(type="hello", scenes=list(SCENES), models=MODELS, scene=chat.scene, model=chat.model_name, base=chat.S.list_scene()["base"],
                          turns=chat.turns, trace_dir=chat.trace_dir))
                emit(dict(type="render", images=chat.renders()))
            elif kind == "model":
                chat.model_name = payload["model"]
                emit(dict(type="reply", text=f"Model: {chat.model_name}."))
        except Stopped:
            if kind == "prompt":
                chat.undo()  # a stopped prompt leaves the scene as it was
            STOP.clear()
            emit(dict(type="reply", text="Stopped. The scene is as it was before this request."))
        except Exception as e:  # shown in the chat with the traceback (observability O5), and traced
            import traceback
            tb = traceback.format_exc()
            emit(dict(type="error", text=f"{type(e).__name__}: {e}", traceback=tb))
        finally:
            chat.busy = False
            RUNNING.clear()
            emit(dict(type="idle"))

    try:
        while True:
            msg = await sock.receive_json()
            if msg.get("type") == "stop":
                if chat.busy:
                    STOP.set()
                    emit(dict(type="reply", text="Stopping: the current model or tool call finishes first, then the request stops."))
                continue
            if msg.get("type") == "stop_policy":
                p = POLICY.get("proc")
                if p is not None:
                    import signal
                    os.killpg(p.pid, signal.SIGTERM)  # the runner and, for groot, its policy server
                    emit(dict(type="reply", text=f"Stopping the {POLICY.get('policy')} run on turn {POLICY.get('turn')}."))
                continue
            if msg.get("type") == "run_policy":
                if chat.busy:
                    q.put_nowait(dict(type="error", text=f"wait for the current request ({RUNNING.get('text')}) to finish, then run the policy"))
                    continue
                why = chat.start_policy(msg.get("policy", "hold"), emit, ALLOW_GROOT)
                if why:
                    q.put_nowait(dict(type="policy_refused", text=why))
                continue
            if chat.busy:
                q.put_nowait(dict(type="error", text=f"still working on: {RUNNING.get('text')} (started {RUNNING.get('started')}); press Stop to cancel it"))
                q.put_nowait(dict(type="idle_if_not_busy"))
                continue
            threading.Thread(target=worker, args=(msg.get("type"), msg), daemon=True).start()
    except WebSocketDisconnect:
        SUBS.discard((loop, q))
        pumper.cancel()


def main():
    global chat, ALLOW_GROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--scene", default="orange", choices=list(SCENES))
    ap.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b", choices=MODELS)
    ap.add_argument("--new", action="store_true", help="start a new session instead of resuming the latest one")
    ap.add_argument("--allow_groot", action="store_true", help="let the Run policy button start GR00T (GPU): only with prim's go")
    a = ap.parse_args()
    ALLOW_GROOT = a.allow_groot
    chat = Chat(a.scene, a.model)
    if not a.new:
        import glob
        old = sorted(d for d in glob.glob(os.path.join(OUT, "traces", "[0-9]*")) if os.path.isdir(d) and d != chat.trace_dir
                     and glob.glob(os.path.join(d, "turn_[0-9][0-9][0-9].json")))  # only sessions with turns
        fresh = chat.trace_dir
        if old and chat.resume(old[-1]):
            print(f"resumed session {old[-1]} ({chat.turn} turns)", flush=True)
            if not os.listdir(fresh):
                os.rmdir(fresh)  # the empty folder made at start-up
    print(f"TaskGen chat on http://127.0.0.1:{a.port}  (scene {a.scene}, model {a.model})", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
