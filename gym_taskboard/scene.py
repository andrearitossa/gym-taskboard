"""Builds the MuJoCo scene: table, Franka Panda, task board, kit / bin, cameras."""

from pathlib import Path

import numpy as np
import mujoco

from . import geometry as g
from .parts import MM, PLATE_T, STANDOFF

PANDA_XML = Path(__file__).parent / "assets" / "franka_emika_panda" / "panda.xml"
PANDA_HOME = [-0.1728, -0.0981, 0.234, -2.3974, 0.0305, 2.3017, 0.8252]  # TCP at (0.48, 0.03, 0.25), pointing down

BOARD_XY = (0.50, -0.20)    # nominal centres on the table (robot base at origin)
KIT_XY = (0.45, 0.27)
BOARD_Z = PLATE_T + STANDOFF  # plate top above the table
BIN_SIZE = (0.30, 0.30, 0.08)

# Contact model. dt = 1 ms resolves 0.05-0.2 mm clearances; solref time
# constant = 4 dt (stiffest stable value, MuJoCo recommends >= 2 dt); solimp
# reaches 0.99 impedance within 0.5 mm so penetration stays well below clearances.
TIMESTEP = 1e-3
SOLREF = [4e-3, 1]
SOLIMP = [0.95, 0.99, 5e-4, 0.5, 2]


def _panda():
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    panda.delete(panda.key("home"))
    panda.delete(panda.light("top"))  # the scene has its own lights; a second shadow pass is costly
    # The real arm runs with gravity compensation (libfranka); without it the
    # position servos sag by ~1 cm.
    for body in panda.bodies:
        body.gravcomp = 1
    # Franka Hand: 70 N continuous grasp force. A stiff servo saturating at
    # 140 N tendon force (= 70 N squeeze per finger pair) replaces Menagerie's
    # kp = 100 N/m, which squeezes small parts with < 1 N.
    kp = 1e4
    grip = panda.actuator("actuator8")
    grip.gainprm[0], grip.biasprm[:3] = kp * 0.04 / 255, [0, -kp, -100]
    grip.forcerange = [-140, 140]
    # wrist camera on the hand's +x side, tilted 40 deg toward the fingers and rolled so the
    # fingertips sit at the bottom of the image with the grasped part in the centre
    q = np.array([0.0, 1, 0, 0])  # look along the hand's +z (the approach direction)
    for axis, deg in (([0, 1, 0], 40), ([0, 0, 1], -90)):
        r = np.r_[np.cos(np.radians(deg) / 2), np.sin(np.radians(deg) / 2) * np.array(axis)]
        mujoco.mju_mulQuat(q, q.copy(), r)
    panda.body("hand").add_camera(name="hand", pos=[0.08, 0, 0.05], quat=q, fovy=60)
    return panda


def build(board, parts, mode):
    """Return an MjSpec. `board` is a board module, `parts` its Part list."""
    spec = mujoco.MjSpec()
    spec.modelname = board.NAME
    o = spec.option
    o.timestep, o.integrator = TIMESTEP, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.cone, o.impratio = mujoco.mjtCone.mjCONE_ELLIPTIC, 10  # no-slip grasping
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    spec.visual.quality.shadowsize = 2048  # one shadow-casting light; 4096 maps dominate render time
    spec.stat.center, spec.stat.extent = [0.45, 0, 0.1], 0.8

    world = spec.worldbody
    world.add_light(pos=[0.5, 0, 1.5], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    world.add_light(pos=[0, 1, 1.5], dir=[0.3, -0.6, -1], diffuse=[0.3, 0.3, 0.3], castshadow=False)
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.2, 1.2, 0.01],
                   rgba=[0.55, 0.42, 0.3, 1], friction=[0.5, 0.005, 0.0001])

    # The board is a mocap body: fixed to the table but posed freely per episode.
    b = world.add_body(name="board", mocap=True, pos=[*BOARD_XY, BOARD_Z])
    board.build_board(spec, b)

    if mode == "assembly":
        kit = world.add_body(name="kit", mocap=True, pos=[*KIT_XY, 0])
        g.box(kit, (0.406, 0.406, 0.0005), (0, 0, 0.00025), rgba=[0.85, 0.93, 0.97, 1],
              contype=0, conaffinity=0)
    else:  # disassembly: parts go into an open bin
        kit = world.add_body(name="kit", mocap=True, pos=[*KIT_XY, 0])
        (w, d, h), t = BIN_SIZE, 0.005
        kw = dict(rgba=[0.3, 0.45, 0.7, 1])
        g.box(kit, (w, d, t), (0, 0, t / 2), **kw)
        for s in (-1, 1):
            g.box(kit, (w, t, h), (0, s * (d - t) / 2, h / 2), **kw)
            g.box(kit, (t, d, h), (s * (w - t) / 2, 0, h / 2), **kw)

    for p in parts:
        p.build(spec, world)

    g.merge_visuals(spec)  # before the Panda: it already has separate visual meshes
    spec.attach(_panda(), prefix="panda/", frame=world.add_frame())

    # front-diagonal close-ups: "left" on the kit / bin, "right" on the task board. Each tracks
    # its (randomly placed) target so the surface fills the frame every episode.
    for name, target, (x, y), side in (("left", "kit", KIT_XY, 1), ("right", "board", BOARD_XY, -1)):
        world.add_camera(name=name, pos=[x + 0.30, y + side * 0.28, 0.48], fovy=50,
                         mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody=target)

    for geom in spec.geoms:
        geom.solref, geom.solimp, geom.margin = SOLREF, SOLIMP, 0
    return spec
