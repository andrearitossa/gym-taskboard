# Architecture

```
gym_taskboard/
  geometry.py      convex building blocks: boxes, prisms, rings (holes/bores), involute teeth, plate tiling
  parts.py         Part types: RoundPeg, RectPeg, Gear, Nut, Plug, DSub, BNC (+ HelicalCoupling)
  boards/board1.py Task Board #1: part table (positions, sizes, kit poses) + plate builder
  boards/__init__  BOARDS registry
  scene.py         MjSpec scene: table, Panda, board, kit/bin, cameras, contact settings
  env.py           TaskBoardEnv (Gymnasium): reset protocol, control, observations, scoring
  scoring.py       NIST score sheets, P_S bound, trial statistics
```

**Part** = one row of the NIST score sheet. It builds two things:
- its *fixture* on the board (hole, shaft, bolt, socket), in a frame at the part's board position;
- its *free body* (what the robot handles).

`status(p, R)` maps the free body's pose in the fixture frame to `(inserted, seated)`.
"Placed" is decided by the env from board contacts, so parts stay pure geometry plus a predicate.

**Board** = a module with `NAME`, `PARTS`, `COMPETENCIES` and `build_board(spec, body)`.
The board is a mocap body: rigidly held, but re-posed every reset.

## Adding a task board

1. Write `boards/boardN.py` with a `PARTS` table, reusing part types or adding new ones in `parts.py`.
2. Register it in `boards/__init__.py`. Gym ids are generated automatically.
3. Add mechanism tests in `tests/test_physics.py` (place the part, push it, check its flags).
