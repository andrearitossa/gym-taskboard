"""MuJoCo/Gymnasium environments for the NIST assembly task boards."""

from gymnasium.envs.registration import register

from .boards import BOARDS
from .env import TaskBoardEnv

# e.g. TaskBoard1-Assembly-v0, TaskBoard1-Gears-Disassembly-v0
for _board, _mod in BOARDS.items():
    for _mode in ("assembly", "disassembly"):
        for _task in (None, *_mod.COMPETENCIES):
            _id = "-".join(filter(None, (_board, _task and _task.capitalize(), _mode.capitalize())))
            register(id=f"{_id}-v0", entry_point="gym_taskboard.env:TaskBoardEnv",
                     max_episode_steps=2000 if _task is None else 500,
                     kwargs=dict(board=_board, mode=_mode, parts=_task and [_task]))
