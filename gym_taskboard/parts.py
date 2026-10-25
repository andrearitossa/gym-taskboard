"""Task-board parts: each part owns its board-side fixture and its free body.

Frames: the board frame has its origin at the centre of the plate's top
surface, z up. Each part has a *fixture frame* (board frame shifted to the
part's `xy` and rotated by `yaw`). `status(p, R)` gets the free body's pose in
that frame and returns (inserted, seated) per the NIST score sheet.
"""

import numpy as np
import mujoco

from . import geometry as g

MM = 1e-3
STEEL, ACRYLIC, POM, PLASTIC = 7850, 1180, 1410, 1500
FRICTION = [0.4, 0.005, 0.0001]  # metal/plastic sliding on metal/plastic
PLATE_T = 9 * MM       # acrylic plate thickness
STANDOFF = 20 * MM     # plate underside to table
TABLE_Z = -(PLATE_T + STANDOFF)  # table surface in board frame


def _tilt(R, symmetric=False):
    z = abs(R[2, 2]) if symmetric else R[2, 2]
    return np.arccos(np.clip(z, -1, 1))


class Part:
    competency = ""
    coupling = None
    target_z = 0.0      # free-body origin height when assembled (fixture frame)
    target_yaw = 0.0
    rgba = (0.6, 0.6, 0.6, 1)

    def __init__(self, name, xy, yaw=0.0, kit=(0, 0, 0, False)):
        self.name, self.xy, self.yaw = name, np.array(xy) * MM, yaw
        self.kit = kit  # (x mm, y mm, yaw, lying-on-side) in the kit frame

    # -- building --------------------------------------------------------
    def fixture_body(self, spec, board):
        return board.add_body(name=f"{self.name}_fixture", pos=[*self.xy, 0], quat=g.yaw_quat(self.yaw))

    def keepouts(self):
        return []

    def build(self, spec, world):
        """Free body of the part (the fixture is built by the board)."""
        body = world.add_body(name=self.name)
        body.add_freejoint(name=self.name)
        self.build_body(spec, body)
        for geom in body.geoms:
            geom.friction = FRICTION
            geom.rgba = self.rgba

    def build_fixture(self, spec, fx):
        pass

    def build_body(self, spec, body):
        raise NotImplementedError

    def status(self, p, R, data=None):
        raise NotImplementedError


# -- pegs -------------------------------------------------------------------

class Peg(Part):
    """50 mm peg through the plate; seated when it rests on the table."""
    competency, length = "pegs", 50 * MM
    target_z = TABLE_Z + 25 * MM
    rgba = (0.75, 0.75, 0.78, 1)
    wall = 8 * MM  # plate material kept around the hole (fixture geometry)

    def status(self, p, R, data=None):
        bottom = p[2] - abs(R[2, 2]) * self.length / 2
        aligned = _tilt(R, symmetric=True) < np.radians(5) and np.hypot(p[0], p[1]) < 2 * MM
        inserted = aligned and bottom < -2 * MM
        return inserted, inserted and bottom < TABLE_Z + 1.5 * MM


class RoundPeg(Peg):
    def __init__(self, name, xy, d, hole, **kw):
        super().__init__(name, xy, **kw)
        self.r, self.hole_r = d / 2 * MM, hole / 2 * MM

    def keepouts(self):
        h = self.hole_r + self.wall
        return [(self.xy[0] - h, self.xy[1] - h, self.xy[0] + h, self.xy[1] + h)]

    def build_fixture(self, spec, fx):
        g.ring(spec, fx, f"{self.name}_hole", self.hole_r, g.square(self.hole_r + self.wall),
               -PLATE_T, 0, n=32, density=ACRYLIC, rgba=(0.95, 0.95, 0.93, 1))

    def build_body(self, spec, body):
        g.cylinder(body, self.r, self.length, density=STEEL)


class RectPeg(Peg):
    def __init__(self, name, xy, size, hole, **kw):
        super().__init__(name, xy, **kw)
        self.size, self.hole = np.array(size) * MM, np.array(hole) * MM

    def keepouts(self):
        c, s = abs(np.cos(self.yaw)), abs(np.sin(self.yaw))
        w, d = self.hole + 2 * self.wall
        hx, hy = (c * w + s * d) / 2, (s * w + c * d) / 2
        return [(self.xy[0] - hx, self.xy[1] - hy, self.xy[0] + hx, self.xy[1] + hy)]

    def build_fixture(self, spec, fx):
        g.rect_frame(fx, self.hole, self.hole + 2 * self.wall, -PLATE_T, 0,
                     density=ACRYLIC, rgba=(0.95, 0.95, 0.93, 1))

    def build_body(self, spec, body):
        g.box(body, (*self.size, self.length), density=STEEL)

    def status(self, p, R, data=None):
        inserted, seated = super().status(p, R)
        yaw_err = abs((np.arctan2(R[1, 0], R[0, 0]) + np.pi / 2) % np.pi - np.pi / 2)
        ok = yaw_err < np.radians(5)  # the bar is symmetric under 180 deg
        return inserted and ok, seated and ok


# -- gears --------------------------------------------------------------------

class Gear(Part):
    """Module-1 spur gear (MISUMI GEABP1.0-z-10-B-10) on a 10 mm shaft.

    20 mm long: 10 mm hub at the bottom, 10 mm toothed face on top. Teeth are
    true involute convex meshes, so meshing with neighbours is a contact effect.
    """
    competency = "gears"
    rgba = (0.97, 0.97, 0.97, 1)
    plate_top = 5 * MM          # gear plate on the board
    shaft_h, length = 20 * MM, 20 * MM
    target_z = plate_top

    def __init__(self, name, xy, teeth, hub_d, phase=0.0, **kw):
        super().__init__(name, xy, **kw)
        self.teeth, self.hub_r, self.target_yaw = teeth, hub_d / 2 * MM, phase

    def build_fixture(self, spec, fx):
        g.cylinder(fx, 4.995 * MM, self.shaft_h, pos=[0, 0, self.plate_top + self.shaft_h / 2],
                   density=STEEL, friction=FRICTION, rgba=(0.7, 0.7, 0.72, 1))

    def build_body(self, spec, body):  # origin at the bottom face, on the axis
        bore, n = 5.05 * MM, 24
        g.ring(spec, body, f"{self.name}_hub", bore, g.circle(self.hub_r), 0, 10 * MM, n=n, density=POM)
        tooth, rf = g.gear_tooth(self.teeth)
        g.ring(spec, body, f"{self.name}_disc", bore, g.circle(rf), 10 * MM, 20 * MM, n=n, density=POM)
        for k in range(self.teeth):
            a = 2 * np.pi * k / self.teeth
            c, s = np.cos(a), np.sin(a)
            poly = [(c * x - s * y, s * x + c * y) for x, y in tooth]
            g.prism(spec, body, f"{self.name}_tooth{k}", poly, 10 * MM, 20 * MM, density=POM)

    def status(self, p, R, data=None):
        aligned = _tilt(R) < np.radians(5) and np.hypot(p[0], p[1]) < 1 * MM
        inserted = aligned and p[2] < self.plate_top + self.shaft_h - 2 * MM
        return inserted, inserted and p[2] < self.plate_top + 1 * MM


# -- helical coupling (threads, bayonet) ---------------------------------------

class HelicalCoupling:
    """Screw joint (threads, BNC bayonet) built from MuJoCo joints + a joint equality.

    MuJoCo has no screw joint, and a free body cannot be put on one. So the part
    exists twice: the free body, and a *threaded twin* (same geoms, same mass)
    on the fixture axis with a slide s and a hinge h tied by the equality
    s = a0 + pitch/(2 pi) h. When the free part sits coaxially on the thread
    start, the twin takes over (collisions switch to it) and the free body
    becomes a collision-less shadow that tracks it. Rotating the part advances it
    by exactly one pitch per turn; the plate / housing contact seats it, and
    unscrewing past s = travel hands the part back to its free body.
    """

    HARD = dict(solref=[4e-3, 1], solimp=[0.9999, 0.9999, 1e-3, 0.5, 2])  # near-rigid

    def __init__(self, part, pitch, travel, z0=0.0, friction=0.0):
        self.part, self.pitch, self.travel, self.z0, self.friction = part, pitch, travel, z0, friction
        self.k = pitch / (2 * np.pi)
        self.on = False

    def build(self, spec, fx):
        n = self.part.name
        t = fx.add_body(name=f"{n}_threaded", pos=[0, 0, self.z0])
        t.add_joint(name=f"{n}_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                    range=[-0.5 * MM, self.travel], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                    solref_limit=self.HARD["solref"], solimp_limit=self.HARD["solimp"])
        t.add_joint(name=f"{n}_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                    frictionloss=self.friction)  # thread friction
        self.part.build_body(spec, t)
        for geom in t.geoms:
            geom.friction, geom.rgba = FRICTION, [0, 0, 0, 0]  # the shadow is what gets drawn
        spec.add_equality(name=f"{n}_screw", type=mujoco.mjtEq.mjEQ_JOINT, name1=f"{n}_slide",
                          name2=f"{n}_hinge",
                          data=[0, self.k, 0, 0, 0] + [0] * (mujoco.mjNEQDATA - 5), **self.HARD)

    def bind(self, model):
        n = self.part.name
        self.eq = model.equality(f"{n}_screw").id
        js, jh = model.joint(f"{n}_slide"), model.joint(f"{n}_hinge")
        self.qs, self.qh, self.vs, self.vh = js.qposadr[0], jh.qposadr[0], js.dofadr[0], jh.dofadr[0]
        self.fixture = model.body(f"{n}_fixture").id
        self.free_geoms = np.flatnonzero(model.geom_bodyid == model.body(n).id)
        self.twin_geoms = np.flatnonzero(model.geom_bodyid == model.body(f"{n}_threaded").id)
        # Preserve both masks, including zeros on visual-only meshes. Enabling
        # a merged visual mesh would collide its convex hull and close the bore.
        self._collision_masks = [
            (ids, model.geom_contype[ids].copy(), model.geom_conaffinity[ids].copy())
            for ids in (self.twin_geoms, self.free_geoms)
        ]
        q = model.joint(n)
        self.fq, self.fv = q.qposadr[0], q.dofadr[0]

    def engaged(self, data=None):
        return self.on

    def slide(self, data):
        return data.qpos[self.qs]

    def _collide(self, model, twin):
        for (ids, contype, conaffinity), on in zip(self._collision_masks, (twin, not twin)):
            model.geom_contype[ids] = contype if on else 0
            model.geom_conaffinity[ids] = conaffinity if on else 0

    def _shadow(self, data):
        """Write the twin's pose / velocity into the free body."""
        Rf = data.xmat[self.fixture].reshape(3, 3)
        s, h, w = data.qpos[self.qs], data.qpos[self.qh], data.qvel[self.vh]
        q = np.empty(4)
        mujoco.mju_mulQuat(q, data.xquat[self.fixture], np.array([np.cos(h / 2), 0, 0, np.sin(h / 2)]))
        data.qpos[self.fq:self.fq + 3] = data.xpos[self.fixture] + Rf[:, 2] * (self.z0 + s)
        data.qpos[self.fq + 3:self.fq + 7] = q
        data.qvel[self.fv:self.fv + 3] = Rf[:, 2] * self.k * w   # world-frame linear
        data.qvel[self.fv + 3:self.fv + 6] = [0, 0, w]           # body-frame angular

    def engage(self, model, data, s, yaw, w=0.0):
        data.qpos[self.qs], data.qpos[self.qh] = s, yaw
        data.qvel[self.vh], data.qvel[self.vs] = w, self.k * w
        model.eq_data[self.eq, 0] = s - self.k * yaw
        self.on = True
        self._collide(model, True)
        self._shadow(data)

    def release(self, model, data):
        self._shadow(data)
        self.on = False
        self._collide(model, False)

    def update(self, model, data, p, R):
        """Called after every physics step with the free part's pose in the fixture frame."""
        if self.on:
            if self.slide(data) >= self.travel - 0.02 * self.pitch:
                self.release(model, data)
            else:
                self._shadow(data)
            return
        z = p[2] - self.z0
        hi, lo = self.travel - 0.1 * MM, self.travel - min(0.5 * self.pitch, 1 * MM)
        if lo <= z <= hi and np.hypot(p[0], p[1]) < 0.4 * MM and _tilt(R) < np.radians(3):
            w = data.qvel[self.fv + 5]
            self.engage(model, data, z, np.arctan2(R[1, 0], R[0, 0]), w)


# -- nuts -----------------------------------------------------------------------

class Nut(Part):
    """ISO hex nut threaded onto a bolt fixed upright in the plate."""
    competency = "nuts"
    rgba = (0.35, 0.35, 0.38, 1)

    def __init__(self, name, xy, d, pitch, af, m, protrusion, **kw):
        super().__init__(name, xy, **kw)
        self.d, self.af, self.m, self.prot = d * MM, af * MM, m * MM, protrusion * MM
        self.coupling = HelicalCoupling(self, pitch * MM, self.prot, friction=0.002 * d / 8)

    def build_fixture(self, spec, fx):
        g.cylinder(fx, self.d / 2, self.prot, pos=[0, 0, self.prot / 2], density=STEEL,
                   friction=FRICTION, rgba=(0.2, 0.2, 0.22, 1))
        self.coupling.build(spec, fx)

    def build_body(self, spec, body):  # origin at the bottom face, on the axis
        g.ring(spec, body, self.name, self.d / 2 + 0.1 * MM, g.hexagon(self.af), 0, self.m, n=12, density=STEEL)

    def status(self, p, R, data=None):
        c = self.coupling
        engaged = c.engaged()
        return engaged and c.slide(data) < c.travel - c.pitch, engaged and c.slide(data) < 0.3 * MM


# -- connectors -------------------------------------------------------------------

class Plug(Part):
    """Male nose into a socket cavity (USB-A, RJ45, waterproof MCON)."""
    competency = "connectors"
    rgba = (0.15, 0.15, 0.17, 1)
    clearance = 0.15 * MM

    def __init__(self, name, xy, nose, body, housing, **kw):
        super().__init__(name, xy, **kw)
        self.nose, self.bsize, self.housing = (np.array(v) * MM for v in (nose, body, housing))
        self.depth = self.nose[2]
        self.target_z = self.housing[2] - self.depth

    def build_fixture(self, spec, fx):
        hw, hd, hh = self.housing
        cav = self.nose[:2] + 2 * self.clearance
        floor = hh - self.depth
        kw = dict(density=PLASTIC, friction=FRICTION, rgba=(0.1, 0.1, 0.1, 1))
        g.box(fx, (hw, hd, floor), (0, 0, floor / 2), **kw)
        g.rect_frame(fx, cav, (hw, hd), floor, hh, **kw)

    def build_body(self, spec, body):  # origin at the nose tip
        g.box(body, self.nose, (0, 0, self.nose[2] / 2), density=PLASTIC)
        g.box(body, self.bsize, (0, 0, self.nose[2] + self.bsize[2] / 2), density=PLASTIC)

    def status(self, p, R, data=None):
        yaw_err = abs((np.arctan2(R[1, 0], R[0, 0]) + np.pi / 2) % np.pi - np.pi / 2)
        aligned = _tilt(R) < np.radians(5) and np.hypot(p[0], p[1]) < 1.5 * MM and yaw_err < np.radians(5)
        inserted = aligned and p[2] < self.housing[2] - 2 * MM
        return inserted, inserted and p[2] < self.target_z + 0.5 * MM


class DSub(Part):
    """D-sub 25 (shell size B): plug shell slides over the board's female D shell."""
    competency = "connectors"
    rgba = (0.15, 0.15, 0.17, 1)
    housing = np.array([44, 55, 22]) * MM        # 3D-printed DB holder
    post_top, post_h = 38.38 * MM, 7.24 * MM     # female D: long side, short side
    post_z = 6 * MM                              # how far the D shell protrudes
    wall, clearance = 0.8 * MM, 0.2 * MM

    @property
    def target_z(self):
        return self.housing[2]

    def _d(self, top, h):
        b = top - 2 * h * np.tan(np.radians(10))  # 10 deg D-sub side angle
        return [(-top / 2, h / 2), (-b / 2, -h / 2), (b / 2, -h / 2), (top / 2, h / 2)]

    def build_fixture(self, spec, fx):
        hw, hd, hh = self.housing
        kw = dict(density=PLASTIC, friction=FRICTION, rgba=(0.1, 0.1, 0.1, 1))
        g.box(fx, (hw, hd, hh), (0, 0, hh / 2), **kw)
        # D shell long axis along fixture y (the holder's long side)
        poly = [(y, x) for x, y in self._d(self.post_top, self.post_h)]
        g.prism(spec, fx, f"{self.name}_post", poly, hh, hh + self.post_z,
                rgba=(0.75, 0.75, 0.78, 1), friction=FRICTION)

    def build_body(self, spec, body):  # origin at the shell's bottom rim
        top, h, t, c = self.post_top + 2 * self.clearance, self.post_h + 2 * self.clearance, self.wall, self.clearance
        L = self.post_z
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = self._d(top, h)
        kw = dict(density=STEEL)
        g.prism(spec, body, f"{self.name}_w0", [(y0, x0), (y3, x3), (y3 + t, x3), (y0 + t, x0)], 0, L, **kw)
        g.prism(spec, body, f"{self.name}_w1", [(y1, x1), (y2, x2), (y2 - t, x2), (y1 - t, x1)], 0, L, **kw)
        g.prism(spec, body, f"{self.name}_w2", [(y0, x0), (y1, x1), (y1, x1 - t), (y0 + t, x0 - t)], 0, L, **kw)
        g.prism(spec, body, f"{self.name}_w3", [(y3, x3), (y2, x2), (y2, x2 + t), (y3 + t, x3 + t)], 0, L, **kw)
        g.box(body, (16 * MM, 53 * MM, 30 * MM), (0, 0, L + 15 * MM), density=PLASTIC)

    def status(self, p, R, data=None):
        yaw_err = abs(np.arctan2(R[1, 0], R[0, 0]))  # D shape: one orientation only
        aligned = _tilt(R) < np.radians(5) and np.hypot(p[0], p[1]) < 1 * MM and yaw_err < np.radians(5)
        inserted = aligned and p[2] < self.housing[2] + self.post_z - 1.5 * MM
        return inserted, inserted and p[2] < self.housing[2] + 0.5 * MM


class BNC(Part):
    """BNC: plug sleeve over the jack post, then a quarter-turn bayonet lock."""
    competency = "connectors"
    rgba = (0.8, 0.8, 0.82, 1)
    flange, post_r, post_h = 2 * MM, 4.8 * MM, 12 * MM
    sleeve_in, sleeve_out, sleeve_len = 4.9 * MM, 7.05 * MM, 10 * MM
    lock_travel = 2 * MM

    def __init__(self, name, xy, **kw):
        super().__init__(name, xy, **kw)
        seat = self.flange + self.post_h - self.sleeve_len  # sleeve rim height when locked
        self.target_z = seat
        # 90 deg turn = lock_travel  ->  pitch = 4 * lock_travel
        self.coupling = HelicalCoupling(self, 4 * self.lock_travel, self.lock_travel, z0=seat, friction=0.005)

    def build_fixture(self, spec, fx):
        kw = dict(density=STEEL, friction=FRICTION, rgba=(0.8, 0.8, 0.82, 1))
        g.box(fx, (17.5 * MM, 17.5 * MM, self.flange), (0, 0, self.flange / 2), **kw)
        g.cylinder(fx, self.post_r, self.post_h, pos=[0, 0, self.flange + self.post_h / 2], **kw)
        self.coupling.build(spec, fx)

    def build_body(self, spec, body):  # origin at the sleeve rim
        g.ring(spec, body, self.name, self.sleeve_in, g.circle(self.sleeve_out), 0, self.sleeve_len,
               n=24, density=STEEL)
        g.cylinder(body, self.sleeve_out, 11 * MM, pos=[0, 0, self.sleeve_len + 5.5 * MM], density=PLASTIC)

    def status(self, p, R, data=None):
        c = self.coupling
        engaged = c.engaged()
        return engaged, engaged and c.slide(data) < 0.2 * MM
