"""Scripted Panda demo: pick a round peg from the kit and insert it (damped-least-squares IK).

    python scripts/demo_peg.py [--peg peg_round_16] [--video out.mp4] [--view]
"""

import argparse

import numpy as np

from gym_taskboard import TaskBoardEnv
from gym_taskboard.control import ik_action, tcp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peg", default="peg_round_16")
    ap.add_argument("--yaw", type=float, default=np.pi / 2, help="gripper yaw (keep the hand clear of neighbours)")
    ap.add_argument("--video")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    env = TaskBoardEnv(parts=[args.peg], render_mode="human" if args.view else None)
    env.reset(seed=1)
    p = next(p for p in env.parts if p.name == args.peg)
    frames = []

    def go(target, grip, steps, yaw=0.0):
        """Servo to `target`; returns after convergence (2 mm) or `steps` steps."""
        for i in range(steps):
            _, _, done, _, info = env.step(ik_action(env, target, yaw, grip))
            if args.video:
                frames.append(_frame(env))
            if i >= 5 and np.linalg.norm(tcp(env) - target) < 0.002:
                break
        return info

    peg = env.data.xpos[p.body_id].copy()
    top = peg[2] + p.length / 2
    hole = env.data.xpos[p.fixture_id].copy()
    yaw = args.yaw
    go(peg + [0, 0, 0.15], -1, 100, yaw)                 # above the peg
    go(np.r_[peg[:2], top - 0.012], -1, 100, yaw)        # straddle the top
    go(np.r_[peg[:2], top - 0.012], 1, 20, yaw)         # close
    go(np.r_[peg[:2], 0.20], 1, 100, yaw)                # lift
    go(np.r_[hole[:2], 0.20], 1, 100, yaw)               # transfer
    lift = p.length - 0.012                         # TCP height above peg bottom
    go(np.r_[hole[:2], hole[2] + lift + 0.005], 1, 100, yaw)   # peg tip 5 mm above the hole
    go(np.r_[hole[:2], hole[2] + lift - 0.035], 1, 100, yaw)  # insert
    info = go(np.r_[hole[:2], hole[2] + lift - 0.035], -1, 20, yaw)  # release
    info = go(np.r_[hole[:2], 0.2], -1, 30, yaw)
    print(args.peg, info["parts"][args.peg], f"score {info['score']}/{info['max_score']}")
    if args.video:
        import imageio
        imageio.mimsave(args.video, frames, fps=env.metadata["render_fps"])
    env.close()


def _frame(env):
    r = env._get_renderer(480, 640)
    r.update_scene(env.data, camera="right")
    return r.render().copy()


if __name__ == "__main__":
    main()
