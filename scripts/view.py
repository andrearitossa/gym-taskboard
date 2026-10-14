"""Open the MuJoCo viewer on a task-board scene.

    python scripts/view.py [--mode assembly|disassembly] [--parts gears nut_m8 ...]
"""

import argparse
import time

import mujoco.viewer

from gym_taskboard import TaskBoardEnv

ap = argparse.ArgumentParser()
ap.add_argument("--board", default="TaskBoard1")
ap.add_argument("--mode", default="assembly")
ap.add_argument("--parts", nargs="*")
args = ap.parse_args()

env = TaskBoardEnv(board=args.board, mode=args.mode, parts=args.parts)
env.reset(seed=0)
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        t = time.perf_counter()
        env._physics_step()  # keeps the threaded couplings live
        viewer.sync()
        time.sleep(max(0.0, env.model.opt.timestep - (time.perf_counter() - t)))
