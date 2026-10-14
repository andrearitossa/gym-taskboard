"""Collision-geometry builders on top of MjSpec.

MuJoCo collides convex shapes only, so every concave feature (holes, bores,
cavities, gear teeth) is assembled from convex pieces: primitives where
possible, small convex meshes otherwise. All sizes are in metres.
"""

import numpy as np
import mujoco

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
MESH = mujoco.mjtGeom.mjGEOM_MESH


def yaw_quat(yaw):
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def box(body, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), **kw):
    """Box from full extents `size`."""
    return body.add_geom(type=BOX, size=[s / 2 for s in size], pos=pos, quat=quat, **kw)


def cylinder(body, radius, height, pos=(0, 0, 0), **kw):
    return body.add_geom(type=CYL, size=[radius, height / 2, 0], pos=pos, **kw)


def prism(spec, body, name, polygon, z0, z1, **kw):
    """Convex prism extruded from a 2D convex polygon between heights z0 and z1."""
    if spec.mesh(name) is None:  # meshes are shared by identical bodies
        verts = [(x, y, z) for z in (z0, z1) for x, y in polygon]
        spec.add_mesh(name=name, uservert=np.ravel(verts).tolist())
    return body.add_geom(type=MESH, meshname=name, **kw)


def ring(spec, body, name, r_in, outer, z0, z1, n=24, **kw):
    """Annulus with a polygonal bore whose flats sit exactly at `r_in`.

    `outer(theta)` returns the outer boundary radius along the ray at `theta`;
    the ring is split into `n` convex quads along those rays.
    """
    rv = r_in / np.cos(np.pi / n)  # bore vertices so that the flats are at r_in
    th = 2 * np.pi * np.arange(n + 1) / n
    for i in range(n):
        a, b = th[i], th[i + 1]
        poly = [(rv * np.cos(a), rv * np.sin(a)), (rv * np.cos(b), rv * np.sin(b)),
                (outer(b) * np.cos(b), outer(b) * np.sin(b)),
                (outer(a) * np.cos(a), outer(a) * np.sin(a))]
        prism(spec, body, f"{name}_{i}", poly, z0, z1, **kw)


def circle(r):
    return lambda th: r


def square(half):
    """Outer boundary of an axis-aligned square (rays every 45 deg hit corners)."""
    return lambda th: half / np.cos((th + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def hexagon(across_flats):
    """Hex nut boundary with flats facing +-x (corners at 30 + k*60 deg)."""
    return lambda th: across_flats / 2 / np.cos((th + np.pi / 6) % (np.pi / 3) - np.pi / 6)


def rect_frame(body, inner, outer, z0, z1, pos=(0, 0), quat=(1, 0, 0, 0), **kw):
    """Rectangular hole: 4 boxes filling `outer` minus a centred `inner` opening."""
    (iw, id_), (ow, od) = inner, outer
    c, s = quat[0], quat[3]
    yaw = 2 * np.arctan2(s, c)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    h, zc = z1 - z0, (z0 + z1) / 2
    pieces = [((ow, (od - id_) / 2), (0, (od + id_) / 4)),
              ((ow, (od - id_) / 2), (0, -(od + id_) / 4)),
              (((ow - iw) / 2, id_), ((ow + iw) / 4, 0)),
              (((ow - iw) / 2, id_), (-(ow + iw) / 4, 0))]
    for (w, d), off in pieces:
        x, y = R @ off + pos
        box(body, (w, d, h), (x, y, zc), quat, **kw)


def gear_tooth(z, module=1e-3, backlash=5e-5, pressure=np.radians(20), n=8):
    """Convex 2D involute tooth outline centred on +x, from root to tip circle."""
    rp, rb = module * z / 2, module * z / 2 * np.cos(pressure)
    ra, rf = rp + module, rp - 1.25 * module
    inv = lambda a: np.tan(a) - a
    half_pitch = (np.pi * module / 2 - backlash) / (2 * rp) + inv(pressure)
    pts = []
    for r in np.linspace(max(rb, rf), ra, n):
        psi = half_pitch - inv(np.arccos(min(rb / r, 1.0)))
        pts.append((r, psi))
    root = rf - 0.2 * module  # overlap with the gear disc so there is no seam
    pts.insert(0, (root, pts[0][1]))
    left = [(r * np.cos(p), r * np.sin(p)) for r, p in pts]
    right = [(r * np.cos(-p), r * np.sin(-p)) for r, p in reversed(pts)]
    return left + right, rf


def plate_tiles(half, keepouts):
    """Cover the square [-half, half]^2 minus axis-aligned `keepouts` with few rectangles.

    Returns (xmin, ymin, xmax, ymax) tuples. Grid sweep + greedy merging.
    """
    xs = sorted({-half, half, *[k[0] for k in keepouts], *[k[2] for k in keepouts]})
    ys = sorted({-half, half, *[k[1] for k in keepouts], *[k[3] for k in keepouts]})
    free = lambda x, y: not any(k[0] <= x <= k[2] and k[1] <= y <= k[3] for k in keepouts)
    rows = []  # per row: list of (x0, x1) free spans
    for j in range(len(ys) - 1):
        yc, spans, start = (ys[j] + ys[j + 1]) / 2, [], None
        for i in range(len(xs) - 1):
            ok = free((xs[i] + xs[i + 1]) / 2, yc)
            if ok and start is None:
                start = xs[i]
            if not ok and start is not None:
                spans.append((start, xs[i])); start = None
        if start is not None:
            spans.append((start, xs[-1]))
        rows.append(spans)
    tiles, open_ = [], {}  # span -> y start, merged while identical spans continue
    for j, spans in enumerate(rows + [[]]):
        for sp in list(open_):
            if sp not in spans:
                tiles.append((sp[0], open_.pop(sp), sp[1], ys[j]))
        for sp in spans:
            open_.setdefault(sp, ys[j])
    return tiles


def _piece_mesh(spec, geom):
    """Triangle soup (verts, faces) of a box / cylinder / prism geom, in its body frame."""
    if geom.type == BOX:
        sx, sy, sz = geom.size
        poly, z0, z1 = [(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)], -sz, sz
    elif geom.type == CYL:
        r, h = geom.size[0], geom.size[1]
        th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        poly, z0, z1 = list(zip(r * np.cos(th), r * np.sin(th))), -h, h
    else:
        v = np.reshape(spec.mesh(geom.meshname).uservert, (-1, 3))
        n = len(v) // 2
        poly, z0, z1 = [tuple(p) for p in v[:n, :2]], v[0, 2], v[n, 2]
    poly = np.array(poly)
    x, y = poly.T
    if np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) < 0:  # make it counter-clockwise
        poly = poly[::-1]
    n = len(poly)
    verts = np.array([(px, py, z) for z in (z0, z1) for px, py in poly])
    faces = [(0, i + 1, i) for i in range(1, n - 1)] + [(n, n + i, n + i + 1) for i in range(1, n - 1)]
    faces += [f for i in range(n) for f in ((i, (i + 1) % n, n + (i + 1) % n), (i, n + (i + 1) % n, n + i))]
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(geom.quat, float))
    return verts @ R.reshape(3, 3).T + geom.pos, np.array(faces)


def merge_visuals(spec):
    """Draw each body's convex pieces as one mesh per colour.

    Collision needs many small convex geoms; drawing them one by one costs a
    draw call each (slow on software / translated GL). The pieces are moved to
    the hidden collision group 3 and a single non-colliding visual mesh is added.
    """
    for body in spec.bodies:
        groups = {}
        for geom in body.geoms:
            if geom.group == 0 and geom.rgba[3] > 0 and geom.type in (BOX, CYL, MESH):
                groups.setdefault(tuple(geom.rgba), []).append(geom)
        for k, (rgba, geoms) in enumerate(groups.items()):
            if len(geoms) < 2:
                continue
            verts, faces, off = [], [], 0
            for geom in geoms:
                v, f = _piece_mesh(spec, geom)
                verts.append(v), faces.append(f + off)
                off += len(v)
                geom.group = 3
            name = f"{body.name}_visual{k}"
            spec.add_mesh(name=name, uservert=np.concatenate(verts).ravel().tolist(),
                          userface=np.concatenate(faces).ravel().tolist())
            body.add_geom(type=MESH, meshname=name, rgba=rgba, contype=0, conaffinity=0, group=0,
                          density=0, mass=0)
