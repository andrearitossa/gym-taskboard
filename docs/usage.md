# Usage

## Environment ids

`{Board}[-{Competency}]-{Mode}-v0`, e.g. `TaskBoard1-Assembly-v0`, `TaskBoard1-Gears-Disassembly-v0`.

| Mode | Start | Goal |
|---|---|---|
| `Assembly` | scored parts lie on the kit print, the rest are assembled | insert every scored part |
| `Disassembly` | the whole board is assembled | move scored parts into the bin |

Competencies: `Pegs`, `Gears`, `Connectors`, `Nuts`. Any subset of parts:
`gym.make("TaskBoard1-Assembly-v0", parts=["nut_m8", "usb"])`.

## Arguments (`TaskBoardEnv`)

| arg | default | meaning |
|---|---|---|
| `obs_type` | `"state"` | `"state"`, `"image"` or `"all"` |
| `image_size` | `(128, 128)` | camera resolution (H, W) |
| `randomize` | `True` | random planar pose of board and kit/bin each reset (±2 cm, ±15°), as the NIST protocol asks |
| `control_hz` | `20` | policy rate; physics runs at 1 kHz |

## Spaces

**Action** `Box(-1, 1, (8,))`: 7 joint-target deltas (×0.05 rad) + gripper (−1 open, +1 close, 70 N grasp force).

**Observation** `Dict`:

| key | shape | content |
|---|---|---|
| `joints` | 18 | Panda qpos (7 arm + 2 fingers) and qvel |
| `objects` (state) | 7·(N+2) | world pose (pos, quat wxyz) of each scored part, then the board and the kit/bin |
| `hand`, `left`, `right` (image) | H×W×3 | wrist camera, two side views |

## Reward and scoring

Reward is the increase of the NIST score sheet ([RAL-SI-2020-B-1](https://www.nist.gov/document/benchmarknistassemblytaskboard1v10)); the episode terminates at full score.

| Assembly (per part) | pts | Disassembly (per part) | pts |
|---|---|---|---|
| inserted / threaded | 3 | removed from the board | 2 |
| seated | 2 | placed in the bin | 1 |
| placed (touches the board) | 1 | | |

`info` holds `score`, `max_score`, `completion` and per-part flags. For trial
statistics, `gym_taskboard.scoring` has `success_probability(m, n, CL)` (the NIST
P_S bound) and `summarize(times, completions)` (mean, std, 95% CI).
