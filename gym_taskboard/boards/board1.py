"""NIST Assembly Task Board #1: pegs, gears, connectors, nuts.

Layout in mm, board frame (origin = centre of the 384 x 384 mm plate top).
Positions come from the NIST CAD plate (GMC_Laser_Plate_Virtual), hole sizes
from the same CAD; part sizes from the NIST parts list (MISUMI numbers).
Kit poses follow the NIST kit-tray print (kitprint.pdf), in the kit frame.
"""

import numpy as np

from .. import geometry as g
from ..parts import (MM, ACRYLIC, PLATE_T, STANDOFF, FRICTION, RoundPeg, RectPeg,
                     Gear, Nut, Plug, DSub, BNC)

NAME = "TaskBoard1"
HALF = 192 * MM
Q = np.pi / 2

PARTS = [
    # pegs: round RGOCG*-50 in drilled holes, rectangular KET* in laser-cut slots
    RoundPeg("peg_round_4", (-0.52, 155.15), d=4, hole=4.2, kit=(146, -128, 0, True)),
    RoundPeg("peg_round_8", (-75.52, 155.15), d=8, hole=8.2, kit=(-136, -145, 0, False)),
    RoundPeg("peg_round_12", (-150.52, -69.85), d=12, hole=12.2, kit=(-6, -62, 0, False)),
    RoundPeg("peg_round_16", (74.48, -69.85), d=16, hole=16.2, kit=(76, 74, 0, False)),
    RectPeg("peg_rect_4", (-0.52, 5.15), size=(4, 4), hole=(4.11, 4.11), kit=(-4, 145, Q, True)),
    RectPeg("peg_rect_8", (74.48, 80.15), size=(8, 7), hole=(8.16, 7.15), kit=(-63, -93, Q, True)),
    RectPeg("peg_rect_12", (-150.52, 5.15), size=(12, 8), hole=(12.18, 8.14), yaw=Q, kit=(-141, -51, 0, True)),
    RectPeg("peg_rect_16", (-150.52, 155.15), size=(16, 10), hole=(16.25, 10.21), kit=(-93, 150, 0, False)),
    # gears GEABP1.0-{20,40,60}-10-B-10 on SPTR10-20 shafts (centre distances 30 / 50 mm);
    # phase = half a tooth so a gap of each gear faces a tooth of the medium gear
    Gear("gear_small", (29.48, -144.85), teeth=20, hub_d=16, phase=np.pi / 20, kit=(143, 61, 0, False)),
    Gear("gear_medium", (-0.52, -144.85), teeth=40, hub_d=30, kit=(70, 6, 0, False)),
    Gear("gear_large", (-50.52, -144.85), teeth=60, hub_d=35, phase=np.pi / 60, kit=(-122, 42, 0, False)),
    # connectors (female side fixed on the board, male part in the kit)
    BNC("bnc", (-0.52, 87.65), kit=(-67, 100, 0, False)),
    Plug("waterproof", (-75.52, 5.15), yaw=Q, nose=(44, 21, 12), body=(63, 32, 23),
         housing=(64, 34, 31.5), kit=(110, 140, 0, False)),
    DSub("dsub", (74.48, 5.15), kit=(-33, -6, 0, False)),
    Plug("usb", (-150.52, 80.15), nose=(12, 4.5, 12), body=(16, 8.5, 30),
         housing=(20, 14, 20), kit=(93, -127, Q, True)),
    Plug("rj45", (149.48, 155.15), yaw=Q, nose=(11.7, 8, 14), body=(11.7, 11, 25),
         housing=(42, 20, 17), kit=(-38, -154, 0, True)),
    # ISO coarse nuts on bolts threaded up through the plate (M4x16 ... M16x25)
    Nut("nut_m4", (-0.52, -69.85), d=4, pitch=0.7, af=7, m=3.2, protrusion=7, kit=(-148, 122, 0, False)),
    Nut("nut_m8", (-150.52, -144.85), d=8, pitch=1.25, af=13, m=6.5, protrusion=9, kit=(4, 68, 0, False)),
    Nut("nut_m12", (74.48, 155.15), d=12, pitch=1.75, af=19, m=10, protrusion=11, kit=(139, -11, 0, False)),
    Nut("nut_m16", (149.48, -144.85), d=16, pitch=2.0, af=24, m=13, protrusion=16, kit=(35, -111, 0, False)),
]

COMPETENCIES = ("pegs", "gears", "connectors", "nuts")


def build_board(spec, board):
    """Plate (tiled around the peg holes), standoffs and the gear plate."""
    keepouts = [k for p in PARTS for k in p.keepouts()]
    kw = dict(density=ACRYLIC, friction=FRICTION, rgba=(0.95, 0.95, 0.93, 1))
    for x0, y0, x1, y1 in g.plate_tiles(HALF, keepouts):
        g.box(board, (x1 - x0, y1 - y0, PLATE_T), ((x0 + x1) / 2, (y0 + y1) / 2, -PLATE_T / 2), **kw)
    for sx in (-1, 1):
        for sy in (-1, 1):
            g.cylinder(board, 5 * MM, STANDOFF, pos=[sx * 175 * MM, sy * 175 * MM, -PLATE_T - STANDOFF / 2],
                       rgba=(0.7, 0.7, 0.72, 1), contype=0, conaffinity=0)
    g.box(board, (120 * MM, 60 * MM, 5 * MM), (-10.52 * MM, -144.85 * MM, 2.5 * MM),
          friction=FRICTION, rgba=(0.75, 0.75, 0.78, 1))
    for p in PARTS:
        p.build_fixture(spec, p.fixture_body(spec, board))
