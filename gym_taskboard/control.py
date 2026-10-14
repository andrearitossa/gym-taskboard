"""Cartesian helpers on top of the joint-space action (damped-least-squares IK)."""

import numpy as np
import mujoco

from .env import ARM_STEP

TCP = np.array([0, 0, 0.1034])  # fingertip centre in the hand frame


def tcp(env):
    hand = env.model.body("panda/hand").id
    return env.data.xpos[hand] + env.data.xmat[hand].reshape(3, 3) @ TCP


def ik_action(env, target, yaw, grip):
    """Joint-delta action that moves the TCP towards `target` with the gripper pointing down."""
    m, d = env.model, env.data
    hand = m.body("panda/hand").id
    R = d.xmat[hand].reshape(3, 3)
    pos = d.xpos[hand] + R @ TCP
    Rd = np.array([[np.cos(yaw), np.sin(yaw), 0], [np.sin(yaw), -np.cos(yaw), 0], [0, 0, -1]])
    err_r = 0.5 * (np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2]))
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, pos, hand)
    J = np.vstack([jp, jr])[:, env.v_robot[:7]]
    e = np.r_[np.clip(target - pos, -0.02, 0.02), np.clip(err_r, -0.2, 0.2)]
    dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
    q = d.qpos[env.q_robot[:7]]
    dq = q + dq - d.ctrl[:7]  # track from the current configuration
    return np.r_[np.clip(dq / ARM_STEP, -1, 1), grip]
