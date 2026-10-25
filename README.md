# gym-taskboard

MuJoCo + Gymnasium environments of the [NIST assembly task boards](https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly),
manipulated by a Franka Panda. Implemented: **Task Board #1** (peg insertion, gear meshing,
electrical connectors, nut threading).

![Task-board GUI: orbiting the scene and jogging the Panda by keyboard](https://raw.githubusercontent.com/andrearitossa/gym-taskboard/main/docs/images/gui.gif)

Requires Python 3.13 or newer.

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest tests                  # mechanism + API tests
.venv/bin/python scripts/view.py --mode disassembly   # interactive viewer
.venv/bin/python scripts/demo_peg.py              # scripted Panda inserts a peg
```

```python
import gymnasium as gym, gym_taskboard

env = gym.make("TaskBoard1-Assembly-v0", obs_type="all")   # or a competency / part subset
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info["score"], info["max_score"], info["parts"]["gear_small"])
```

Docs: [usage](https://github.com/andrearitossa/gym-taskboard/blob/main/docs/usage.md) · [architecture](https://github.com/andrearitossa/gym-taskboard/blob/main/docs/architecture.md) ·
[Task Board #1](https://github.com/andrearitossa/gym-taskboard/blob/main/docs/taskboard1.md) · [physics](https://github.com/andrearitossa/gym-taskboard/blob/main/docs/physics.md)

The browser GUI lives in the separate `taskboard_gui` package and uses the
environment as a client. Install its optional dependencies and launch it with:

```bash
uv pip install --python .venv/bin/python -e '.[gui]'
./gui.sh
# Or start the server directly:
MUJOCO_GL=egl .venv/bin/python -m taskboard_gui --http 8000 --port 5555
```

| Manual operator (keyboard IK jog) | Programmatic operator (JSON over TCP) |
|---|---|
| ![Manual mode](https://raw.githubusercontent.com/andrearitossa/gym-taskboard/main/docs/images/gui-manual.png) | ![Programmatic mode](https://raw.githubusercontent.com/andrearitossa/gym-taskboard/main/docs/images/gui-programmatic.png) |

Headless rendering: `MUJOCO_GL=egl`. The Panda model is from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0).

## License

MIT, see [LICENSE](https://github.com/andrearitossa/gym-taskboard/blob/main/LICENSE). The bundled Panda model keeps its Apache-2.0 license.
