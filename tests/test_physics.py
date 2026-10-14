"""Mechanism checks without the robot: parts are placed / pushed with external wrenches."""

import numpy as np
import mujoco
import pytest

from gym_taskboard import TaskBoardEnv
from gym_taskboard.parts import MM


def make(parts, mode="assembly"):
    env = TaskBoardEnv(mode=mode, parts=parts, randomize=False)
    env.reset(seed=0)
    return env


def place(env, name, dz, yaw=0.0, dxy=(0, 0)):
    """Put part `name` over its fixture, `dz` above its assembled height."""
    p = next(p for p in env.parts if p.name == name)
    d = env.data
    Rf = d.xmat[p.fixture_id].reshape(3, 3)
    pos = d.xpos[p.fixture_id] + Rf @ np.array([*dxy, p.target_z + dz])
    q = np.empty(4)
    mujoco.mju_mulQuat(q, d.xquat[p.fixture_id], [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    d.qpos[p.qadr:p.qadr + 7] = [*pos, *q]
    d.qvel[env.model.joint(name).dofadr[0]:][:6] = 0
    mujoco.mj_forward(env.model, d)
    return p


def run(env, seconds, wrench=None, body=None):
    for _ in range(int(seconds / env.model.opt.timestep)):
        if wrench is not None:
            env.data.xfrc_applied[body] = wrench
        env._physics_step()
    env.data.xfrc_applied[:] = 0


def active_body(env, p):
    """The body that carries a coupled part: its threaded twin while engaged."""
    return env.model.body(f"{p.name}_threaded").id if p.coupling.engaged() else p.body_id


def flags(env, name):
    return env.status()[name]


def test_assembled_board_is_stable():
    env = make(None, mode="disassembly")
    run(env, 1.0)
    assert env._info()["score"] == 0  # nothing removed
    for p in env.parts:
        assert p.status(*env._rel_pose(p), env.data) == (True, True), p.name


@pytest.mark.parametrize("name", ["peg_round_4", "peg_round_16", "peg_rect_4", "peg_rect_16", "peg_rect_12"])
def test_peg_drops_into_hole(name):
    env = make([name])
    place(env, name, dz=40 * MM)
    run(env, 0.5)
    assert flags(env, name) == {"inserted": True, "seated": True, "placed": True}


def test_rect_peg_blocked_when_rotated():
    env = make(["peg_rect_16"])
    place(env, "peg_rect_16", dz=40 * MM, yaw=np.pi / 2)
    run(env, 0.5)
    assert not flags(env, "peg_rect_16")["inserted"]


def test_round_peg_misaligned_rests_on_plate():
    env = make(["peg_round_8"])
    place(env, "peg_round_8", dz=40 * MM, dxy=(6 * MM, 0))
    run(env, 0.5)
    f = flags(env, "peg_round_8")
    assert f["placed"] and not f["inserted"]


@pytest.mark.parametrize("name", ["nut_m4", "nut_m16"])
def test_nut_threads_down_and_back_off(name):
    env = make([name])
    p = place(env, name, dz=p_travel(env, name) - 0.2 * MM)
    run(env, 0.05)
    c = p.coupling
    assert c.engaged()
    tau = 0.02 * p.d / (4 * MM)
    run(env, 3.0, wrench=[0, 0, 0, 0, 0, -tau], body=active_body(env, p))  # clockwise = tighten
    assert flags(env, name)["seated"]
    for _ in range(300):  # loosen until the nut comes free
        run(env, 0.01, wrench=[0, 0, 0, 0, 0, tau], body=active_body(env, p))
        if not c.engaged():
            break
    assert not c.engaged()
    run(env, 0.5)  # free nut rests on the bolt tip
    assert np.isfinite(env.data.qacc).all()


def p_travel(env, name):
    return next(p for p in env.parts if p.name == name).coupling.travel


def test_nut_stays_put_without_torque():
    env = make(["nut_m8"])
    p = place(env, "nut_m8", dz=4 * MM)
    run(env, 0.05)
    s0 = p.coupling.slide(env.data)
    run(env, 1.0)
    assert abs(p.coupling.slide(env.data) - s0) < 0.05 * MM


def test_gear_train_ratio():
    env = make(["gears"], mode="disassembly")  # all three gears on their shafts
    g = {p.name: p for p in env.parts if p.competency == "gears"}
    def yaw(p):
        R = env.data.xmat[p.body_id].reshape(3, 3)
        return np.arctan2(R[1, 0], R[0, 0])

    prev = {n: yaw(p) for n, p in g.items()}
    w = dict.fromkeys(g, 0.0)  # accumulated rotation (backlash averages out)
    for _ in range(300):
        run(env, 0.01, wrench=[0, 0, 0, 0, 0, 0.02], body=g["gear_large"].body_id)
        for n, p in g.items():
            a = yaw(p)
            w[n] += (a - prev[n] + np.pi) % (2 * np.pi) - np.pi
            prev[n] = a
    assert w["gear_large"] > 0.1
    assert w["gear_medium"] / w["gear_large"] == pytest.approx(-60 / 40, rel=0.1)
    assert w["gear_small"] / w["gear_large"] == pytest.approx(60 / 20, rel=0.1)


def test_gear_drops_on_shaft_and_meshes():
    env = make(["gear_small"])
    p = place(env, "gear_small", dz=12 * MM, yaw=np.pi / 20)  # half a tooth: meshes with medium
    run(env, 1.0, wrench=[0, 0, -2, 0, 0, 0.002], body=p.body_id)
    assert flags(env, "gear_small")["seated"]


def test_bnc_bayonet_locks():
    env = make(["bnc"])
    p = place(env, "bnc", dz=1.5 * MM)
    run(env, 0.05)
    assert p.coupling.engaged()
    run(env, 2.0, wrench=[0, 0, 0, 0, 0, -0.05], body=active_body(env, p))
    assert flags(env, "bnc")["seated"]


@pytest.mark.parametrize("name", ["usb", "rj45", "waterproof", "dsub"])
def test_connector_push_in(name):
    env = make([name])
    p = place(env, name, dz=4 * MM)
    run(env, 1.0, wrench=[0, 0, -20, 0, 0, 0], body=p.body_id)
    assert flags(env, name)["seated"]
