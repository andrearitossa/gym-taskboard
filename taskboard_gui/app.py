"""Task-board GUI: browser front-end + programmatic TCP port.

    python -m taskboard_gui [--http 8000] [--port 5555]

The browser (http://localhost:8000) shows a mouse-orbit 3D view, the robot
cameras, the task picker and the NIST score. Two operators:

* manual: keyboard drives the gripper in Cartesian space (IK);
* programmatic: a TCP port takes JSON lines and answers with the env response:
    {"cmd": "reset", "task": "TaskBoard1-Pegs-Assembly-v0", "seed": 0}
    {"cmd": "step", "action": [8 floats in -1..1]}
    {"cmd": "info"}
"""

import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from aiohttp import web
from PIL import Image

import gym_taskboard  # noqa: F401  (registers the environments)
from gym_taskboard.control import ik_action, tcp

STATIC = Path(__file__).parent / "static"
HZ = 20                                  # physics / control ticks per second
EVERY = {"main": 2, "hand": 4, "left": 4, "right": 4}  # render a view every N ticks (10 / 5 fps)
STEP, FINE, YAW = 0.004, 0.0008, np.radians(3)  # manual jog per tick
GRIP_RAMP = 0.25  # gripper command change per tick: full stroke in ~0.4 s
KEYS = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 1, 0), "d": (0, -1, 0), "r": (0, 0, 1), "f": (0, 0, -1)}


def tasks():
    return sorted(i for i in gym.registry if i.startswith("TaskBoard"))


def jpeg(img, quality=80):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def tolist(x):
    return {k: tolist(v) for k, v in x.items()} if isinstance(x, dict) else (
        x.tolist() if isinstance(x, np.ndarray) else (bool(x) if isinstance(x, np.bool_) else x))


class Session:
    def __init__(self, port):
        self.port, self.mode, self.clients = port, "manual", set()
        self.keys, self.remote_peer = set(), None
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 200, -35, 1.25
        self.cam.lookat[:] = [0.45, 0.02, 0.05]
        self.load(tasks()[0], seed=0)

    # -- env -----------------------------------------------------------------------
    def load(self, task, seed=None):
        if getattr(self, "task", None) != task:
            if hasattr(self, "env"):
                self.env.close()
            spec = gym.spec(task)
            self.env = spec.make(obs_type="state").unwrapped
            self.max_steps, self.task = spec.max_episode_steps, task
            self.views = {k: mujoco.Renderer(self.env.model, *s) for k, s in
                          (("main", (540, 860)), ("hand", (200, 266)), ("left", (200, 266)), ("right", (200, 266)))}
            for name, r in self.views.items():  # shadows only in the big view (a shadow pass ~ 25 ms)
                r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = name == "main"
        return self.reset(seed)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.steps, self.ret, self.reward, self.done = 0, 0.0, 0.0, False
        self.info, self.obs = info, obs
        self.grip, self.grip_goal, self.grip_state = -1.0, -1.0, "open"
        hand = self.env.data.xmat[self.env.model.body("panda/hand").id].reshape(3, 3)
        self.target, self.yaw = tcp(self.env).copy(), float(np.arctan2(hand[1, 0], hand[0, 0]))
        return self.response()

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(np.asarray(action, np.float32))
        self.steps += 1
        trunc = trunc or self.steps >= self.max_steps
        self.obs, self.info, self.reward, self.done = obs, info, r, term or trunc
        self.ret += r
        return self.response(r, term, trunc)

    def response(self, reward=0.0, terminated=False, truncated=False):
        return tolist({"obs": self.obs, "reward": reward, "terminated": terminated, "truncated": truncated,
                       "score": self.info["score"], "max_score": self.info["max_score"],
                       "parts": self.info["parts"], "step": self.steps})

    def manual_tick(self):
        """One env.step per tick of operator input (held keys or a moving gripper), like a remote step."""
        moving = self.grip_state in ("closing", "opening")
        if not (self.keys & {*KEYS, "q", "e"}) and not moving:
            return
        self.grip += float(np.clip(self.grip_goal - self.grip, -GRIP_RAMP, GRIP_RAMP))  # smooth ramp
        fine = "shift" in self.keys
        for k, v in KEYS.items():
            if k in self.keys:
                self.target += np.array(v) * (FINE if fine else STEP)
        self.target[2] = max(self.target[2], 0.0)
        self.yaw += YAW * (("q" in self.keys) - ("e" in self.keys)) * (0.25 if fine else 1)
        self.step(ik_action(self.env, self.target, self.yaw, self.grip))
        if moving and self.grip == self.grip_goal:
            fingers = self.env.data.qvel[self.env.v_robot[7:]]
            if np.abs(fingers).max() < 1e-3:  # fingers stopped: open, closed, or holding a part
                width = self.env.data.qpos[self.env.q_robot[7:]].sum()
                self.grip_state = "open" if self.grip_goal < 0 else ("holding" if width > 0.002 else "closed")

    # -- frames --------------------------------------------------------------------
    def frame(self, tick):
        d, out = self.env.data, {}
        for name, r in self.views.items():
            if tick % EVERY[name]:
                continue
            if name == "main":
                r.update_scene(d, camera=self.cam)
            else:
                r.update_scene(d, camera="panda/hand" if name == "hand" else name)
            out[name] = jpeg(r.render(), 85 if name == "main" else 75)
        return {"type": "frame", "images": out, "task": self.task, "mode": self.mode,
                "score": self.info["score"], "max_score": self.info["max_score"], "parts": tolist(self.info["parts"]),
                "step": self.steps, "max_steps": self.max_steps, "return": self.ret, "done": self.done,
                "grip": self.grip_state, "sim_time": self.steps * self.env.model.opt.timestep * self.env.frame_skip, "port": self.port, "peer": self.remote_peer}

    # -- browser messages ----------------------------------------------------------
    def on_message(self, msg):
        t = msg.get("type")
        if t == "keys":
            self.keys = set(msg["keys"])
        elif t == "gripper":  # {"action": "close" | "open"}
            close = msg["action"] == "close"
            self.grip_goal, self.grip_state = (1.0, "closing") if close else (-1.0, "opening")
        elif t == "load":
            self.load(msg["task"], msg.get("seed"))
        elif t == "reset":
            self.reset(msg.get("seed"))
        elif t == "mode":
            self.mode, self.keys = msg["mode"], set()
        elif t == "orbit":
            self.cam.azimuth -= msg["dx"] * 0.3
            self.cam.elevation = float(np.clip(self.cam.elevation - msg["dy"] * 0.3, -89, 5))
        elif t == "zoom":
            self.cam.distance = float(np.clip(self.cam.distance * (1 + msg["d"] * 0.001), 0.3, 3))


# -- programmatic operator: TCP, one JSON object per line ---------------------------
async def serve_remote(s: Session, reader, writer):
    peer = "%s:%d" % writer.get_extra_info("peername")[:2]
    s.remote_peer = peer
    try:
        while line := await reader.readline():
            try:
                req = json.loads(line)
                cmd = req.get("cmd")
                if s.mode != "programmatic":
                    res = {"error": "operator is manual - switch to Programmatic in the GUI"}
                elif cmd == "step":
                    res = s.step(req["action"])
                elif cmd == "reset":
                    res = s.load(req["task"], req.get("seed")) if "task" in req else s.reset(req.get("seed"))
                elif cmd == "info":
                    res = {"task": s.task, "tasks": tasks(), "action_dim": 8, **s.response()}
                else:
                    res = {"error": f"unknown cmd {cmd!r}"}
            except Exception as e:  # report, keep the connection
                res = {"error": repr(e)}
            writer.write((json.dumps(res) + "\n").encode())
            await writer.drain()
    finally:
        s.remote_peer = None
        writer.close()


async def main_loop(s: Session):
    tick = 0
    while True:
        t, tick = time.perf_counter(), tick + 1
        if s.mode == "manual" and not s.done:
            s.manual_tick()
        if s.clients:
            msg = json.dumps(s.frame(tick))
            for ws in list(s.clients):
                try:
                    await ws.send_str(msg)
                except ConnectionResetError:
                    s.clients.discard(ws)
        await asyncio.sleep(max(0.0, 1 / HZ - (time.perf_counter() - t)))


def make_app(s: Session):
    async def index(_):
        return web.FileResponse(STATIC / "index.html")

    async def socket(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        s.clients.add(ws)
        await ws.send_str(json.dumps({"type": "tasks", "tasks": tasks()}))
        async for msg in ws:
            try:
                s.on_message(json.loads(msg.data))
            except Exception as e:
                await ws.send_str(json.dumps({"type": "error", "error": repr(e)}))
        s.clients.discard(ws)
        return ws

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/ws", socket)])
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", type=int, default=8000)
    ap.add_argument("--port", type=int, default=5555, help="programmatic operator TCP port")
    args = ap.parse_args()
    s = Session(args.port)

    async def start(app):
        app["remote"] = await asyncio.start_server(lambda r, w: serve_remote(s, r, w), "0.0.0.0", args.port)
        app["loop"] = asyncio.create_task(main_loop(s))

    app = make_app(s)
    app.on_startup.append(start)
    print(f"GUI on http://localhost:{args.http}  ·  programmatic port tcp://localhost:{args.port}")
    web.run_app(app, port=args.http, print=None)


if __name__ == "__main__":
    main()
