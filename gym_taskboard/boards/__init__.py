"""Task-board registry. A board module defines NAME, PARTS, COMPETENCIES, build_board()."""

from . import board1

BOARDS = {board1.NAME: board1}
