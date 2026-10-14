"""Gymnasium environment: a Franka Panda assembling / disassembling a NIST task board."""

import copy

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from . import scene, scoring
from .boards import BOARDS
from .geometry import yaw_quat
from .parts import MM

ARM_STEP = 0.05  # max joint-target change per step [rad]


def _quat_mul(a, b):
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


class TaskBoardEnv(gym.Env):
    """
    Args:
        board: board name, e.g. "TaskBoard1".
        mode: "assembly" (kit -> board) or "disassembly" (board -> bin).
        parts: part names and/or competencies ("pegs", "gears", ...) to score;
            None = the whole board. Other parts start (and stay) assembled.
        obs_type: "state" (joints + object poses), "image" (joints + cameras), or "all".
        image_size: (height, width) of camera observations.
        randomize: random planar pose of board and kit/bin each reset (NIST protocol).
        control_hz: policy rate; physics runs at 1 kHz.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}
    CAMERAS = ("hand", "left", "right")

    def __init__(self, board="TaskBoard1", mode="assembly", parts=None, obs_type="state",
                 image_size=(128, 128), randomize=True, control_hz=20, render_mode=None):
        assert mode in ("assembly", "disassembly") and obs_type in ("state", "image", "all")
        self.board, self.mode, self.obs_type = BOARDS[board], mode, obs_type
        self.randomize, self.render_mode, self.image_size = randomize, render_mode, image_size
        self.parts = copy.deepcopy(self.board.PARTS)  # parts hold per-env state
        wanted = set(parts or [p.name for p in self.parts])
        self.active = [p for p in self.parts if p.name in wanted or p.competency in wanted]
        assert self.active, f"no parts match {parts}"

        self.model = scene.build(self.board, self.parts, mode).compile()
        self.data = mujoco.MjData(self.model)
        self.frame_skip = round(1 / (control_hz * self.model.opt.timestep))
        self.metadata["render_fps"] = control_hz
        self._bind()

        n = self.model
        lo, hi = n.actuator_ctrlrange[:7].T
        self._ctrl_lo, self._ctrl_hi = lo, hi
        self.action_space = spaces.Box(-1, 1, (8,), np.float32)
        obs = {"joints": spaces.Box(-np.inf, np.inf, (18,), np.float64)}
        if obs_type in ("state", "all"):
            obs["objects"] = spaces.Box(-np.inf, np.inf, (7 * (len(self.active) + 2),), np.float64)
        if obs_type in ("image", "all"):
            for c in self.CAMERAS:
                obs[c] = spaces.Box(0, 255, (*image_size, 3), np.uint8)
        self.observation_space = spaces.Dict(obs)
        self._renderer = self._viewer = None

    # -- model bookkeeping -------------------------------------------------------
    def _bind(self):
        m = self.model
        self.board_id, self.kit_id = m.body("board").id, m.body("kit").id
        self.board_mocap, self.kit_mocap = m.body_mocapid[self.board_id], m.body_mocapid[self.kit_id]
        arm = [m.joint(f"panda/joint{i}") for i in range(1, 8)]
        fingers = [m.joint(f"panda/finger_joint{i}") for i in (1, 2)]
        self.q_robot = np.array([j.qposadr[0] for j in arm + fingers])
        self.v_robot = np.array([j.dofadr[0] for j in arm + fingers])
        for p in self.parts:
            p.body_id = m.body(p.name).id
            p.fixture_id = m.body(f"{p.name}_fixture").id
            p.qadr = m.joint(p.name).qposadr[0]
            if p.coupling:
                p.coupling.bind(m)
        self.coupled = [p for p in self.parts if p.coupling]
        self.geom_root = m.body_rootid[m.geom_bodyid]

    def _rel_pose(self, p):
        """Pose of part `p` in its fixture frame."""
        d = self.data
        Rf = d.xmat[p.fixture_id].reshape(3, 3)
        R = d.xmat[p.body_id].reshape(3, 3)
        return Rf.T @ (d.xpos[p.body_id] - d.xpos[p.fixture_id]), Rf.T @ R

    # -- reset ---------------------------------------------------------------------------
    def _planar_pose(self, xy, z):
        rng = self.np_random
        dxy = rng.uniform(-0.02, 0.02, 2) if self.randomize else np.zeros(2)
        yaw = rng.uniform(-np.pi / 12, np.pi / 12) if self.randomize else 0.0
        return np.array([*(np.array(xy) + dxy), z]), np.array(yaw_quat(yaw))

    def _set_free(self, p, pos, quat):
        self.data.qpos[p.qadr:p.qadr + 3] = pos
        self.data.qpos[p.qadr + 3:p.qadr + 7] = quat

    def _assemble(self, p, board_pos, board_quat):
        """Put part p in its assembled pose on the board."""
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, board_quat)
        local = np.array([*p.xy, p.target_z])
        q = _quat_mul(board_quat, yaw_quat(p.yaw + p.target_yaw))
        self._set_free(p, board_pos + R.reshape(3, 3) @ local, q)

    def _drop_on_table(self, p, clearance=0.5 * MM):
        """Shift a free part so its lowest point is `clearance` above the table."""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        low = np.inf
        for gid in np.flatnonzero(m.geom_bodyid == p.body_id):
            c, h = m.geom_aabb[gid, :3], m.geom_aabb[gid, 3:]
            corners = c + h * np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
            R = d.geom_xmat[gid].reshape(3, 3)
            low = min(low, (d.geom_xpos[gid] + corners @ R.T)[:, 2].min())
        d.qpos[p.qadr + 2] += clearance - low

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.q_robot[:7]] = d.ctrl[:7] = scene.PANDA_HOME
        d.qpos[self.q_robot[7:]], d.ctrl[7] = 0.04, 255

        bpos, bquat = self._planar_pose(scene.BOARD_XY, scene.BOARD_Z)
        kpos, kquat = self._planar_pose(scene.KIT_XY, 0.0)
        d.mocap_pos[self.board_mocap], d.mocap_quat[self.board_mocap] = bpos, bquat
        d.mocap_pos[self.kit_mocap], d.mocap_quat[self.kit_mocap] = kpos, kquat

        Rk = np.empty(9)
        mujoco.mju_quat2Mat(Rk, kquat)
        mujoco.mj_kinematics(m, d)  # fixture poses for the couplings
        for p in self.parts:
            if p.coupling:
                p.coupling.release(m, d)
            if self.mode == "assembly" and p in self.active:
                x, y, yaw, lying = p.kit
                q = _quat_mul(kquat, yaw_quat(yaw))
                if lying:
                    q = _quat_mul(q, [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
                self._set_free(p, kpos + Rk.reshape(3, 3) @ np.array([x * MM, y * MM, 0.05]), q)
                self._drop_on_table(p)
            else:
                self._assemble(p, bpos, bquat)
                if p.coupling:
                    p.coupling.engage(m, d, 0.0, 0.0)
        mujoco.mj_forward(m, d)
        for _ in range(200):  # let everything settle (0.2 s)
            self._physics_step()
        self._prev_score = self._score()[0]
        return self._obs(), self._info()

    # -- step ------------------------------------------------------------------------------
    def _physics_step(self):
        mujoco.mj_step(self.model, self.data)
        for p in self.coupled:
            p.coupling.update(self.model, self.data, *self._rel_pose(p))

    def step(self, action):
        a = np.clip(action, -1, 1)
        d = self.data
        d.ctrl[:7] = np.clip(d.ctrl[:7] + ARM_STEP * a[:7], self._ctrl_lo, self._ctrl_hi)
        d.ctrl[7] = (1 - a[7]) / 2 * 255  # -1 = open, +1 = closed
        for _ in range(self.frame_skip):
            self._physics_step()
        score, max_score, _ = self._score()
        reward, self._prev_score = score - self._prev_score, score
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, score == max_score, False, self._info()

    # -- scoring ---------------------------------------------------------------------------
    def _touching_board(self):
        d = self.data
        g1, g2 = d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon]
        r1, r2 = self.geom_root[g1], self.geom_root[g2]
        return set(r2[r1 == self.board_id]) | set(r1[r2 == self.board_id])

    def _in_kit(self, p):
        d = self.data
        R = d.xmat[self.kit_id].reshape(3, 3)
        x = R.T @ (d.xpos[p.body_id] - d.xpos[self.kit_id])
        w, l, h = scene.BIN_SIZE
        return abs(x[0]) < w / 2 and abs(x[1]) < l / 2 and x[2] < h

    def status(self):
        """Per active part: NIST score-sheet flags."""
        touching, out = self._touching_board(), {}
        for p in self.active:
            inserted, seated = p.status(*self._rel_pose(p), self.data)
            on_board = p.body_id in touching
            if self.mode == "assembly":
                out[p.name] = {"inserted": inserted, "seated": seated, "placed": inserted or on_board}
            else:
                removed = not inserted and not on_board
                out[p.name] = {"removed": removed, "in_kit": removed and self._in_kit(p)}
        return out

    def _score(self):
        st = self.status()
        score = sum(scoring.part_score(s, self.mode) for s in st.values())
        return score, scoring.max_score(len(st), self.mode), st

    def _info(self):
        score, max_score, st = self._score()
        return {"score": score, "max_score": max_score, "completion": score / max_score, "parts": st}

    # -- observations / rendering ------------------------------------------------------------
    def _obs(self):
        d = self.data
        obs = {"joints": np.concatenate([d.qpos[self.q_robot], d.qvel[self.v_robot]])}
        if self.obs_type in ("state", "all"):
            ids = [p.body_id for p in self.active] + [self.board_id, self.kit_id]
            obs["objects"] = np.concatenate([np.r_[d.xpos[i], d.xquat[i]] for i in ids])
        if self.obs_type in ("image", "all"):
            r = self._get_renderer(*self.image_size)
            for c in self.CAMERAS:
                r.update_scene(d, camera=f"panda/{c}" if c == "hand" else c)
                obs[c] = r.render().copy()
        return obs

    def _get_renderer(self, h, w):
        if self._renderer is None or (self._renderer.height, self._renderer.width) != (h, w):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, h, w)
        return self._renderer

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            r = self._get_renderer(480, 640)
            r.update_scene(self.data, camera="left")
            return r.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        if self._viewer is not None:
            self._viewer.close()
