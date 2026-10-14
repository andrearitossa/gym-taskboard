"""Collision switching, orientation and bin-containment regressions."""

import mujoco
import numpy as np
import pytest

from gym_taskboard import TaskBoardEnv, scene
from gym_taskboard.boards.board1 import PARTS
from gym_taskboard.parts import MM


def test_coupling_preserves_collision_masks_through_reset_engage_release():
    with TaskBoardEnv(randomize=False) as env:
        m, d = env.model, env.data
        originals = {}
        for p in env.coupled:
            c = p.coupling
            # Exercise distinct bit masks as well as visual-only zero masks.
            m.geom_contype[c.free_geoms[0]] = 2
            m.geom_conaffinity[c.free_geoms[0]] = 5
            m.geom_contype[c.twin_geoms[0]] = 4
            m.geom_conaffinity[c.twin_geoms[0]] = 3
            c.bind(m)
            originals[p.name] = [
                (ids, m.geom_contype[ids].copy(), m.geom_conaffinity[ids].copy())
                for ids in (c.free_geoms, c.twin_geoms)
            ]
            visual = c.free_geoms[m.geom_group[c.free_geoms] == 0]
            assert len(visual) > 0  # merged visuals must actually be exercised
            assert not m.geom_contype[visual].any()
            assert not m.geom_conaffinity[visual].any()

        def check(p, engaged):
            for (ids, contype, affinity), on in zip(originals[p.name], (not engaged, engaged)):
                np.testing.assert_array_equal(m.geom_contype[ids], contype if on else np.zeros_like(contype))
                np.testing.assert_array_equal(m.geom_conaffinity[ids], affinity if on else np.zeros_like(affinity))

        for _ in range(2):
            env.reset(seed=0)
            for p in env.coupled:
                c = p.coupling
                check(p, False)
                c.engage(m, d, 0.0, 0.0)
                check(p, True)
                c.release(m, d)
                check(p, False)


@pytest.mark.parametrize('part', [p for p in PARTS if p.coupling is None], ids=lambda p: p.name)
def test_seating_orientation_respects_part_symmetry(part):
    pos = np.array([0, 0, part.target_z])
    assert part.status(pos, np.eye(3)) == (True, True)
    flipped = np.diag([1, -1, -1])
    expected = part.competency == 'pegs'
    assert part.status(pos, flipped) == (expected, expected)


def test_upside_down_parts_do_not_engage_threads_or_bayonet():
    with TaskBoardEnv(randomize=False) as env:
        env.reset(seed=0)
        for p in env.coupled:
            c = p.coupling
            pos = np.array([0, 0, c.z0 + c.travel - 0.2 * MM])
            c.update(env.model, env.data, pos, np.diag([1, -1, -1]))
            assert not c.engaged(), p.name
            c.update(env.model, env.data, pos, np.eye(3))
            assert c.engaged(), p.name


@pytest.mark.parametrize('yaw', [0.0, np.pi / 4])
def test_bin_requires_full_containment_in_bin_frame(yaw):
    with TaskBoardEnv(mode='disassembly', parts=['peg_round_4'], randomize=False) as env:
        env.reset(seed=0)
        p, m, d = env.active[0], env.model, env.data
        d.mocap_quat[env.kit_mocap] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        R = d.xmat[env.kit_id].reshape(3, 3).copy()
        origin = d.xpos[env.kit_id].copy()

        def check(local_pos, expected, horizontal=False):
            q = d.xquat[env.kit_id].copy()
            if horizontal:
                mujoco.mju_mulQuat(q, q.copy(), np.array([np.sqrt(0.5), 0, np.sqrt(0.5), 0]))
            env._set_free(p, origin + R @ np.array(local_pos), q)
            mujoco.mj_forward(m, d)
            assert env.status()[p.name] == {'removed': True, 'in_kit': expected}
            assert env._info()['score'] == (3 if expected else 2)

        check((0, 0, -1), False)                       # below the entire bin
        check((0, 0, 0.10), False)                     # above the entire bin
        check((0, 0, 0.02), False)                     # origin inside, bottom through floor
        check((0, 0, 0.07), False)                     # protrudes over rim
        check((0.16, 0, 0.035), False)                 # outside side wall
        check((0.144, 0, 0.035), False)                # origin inside, body overlaps wall
        check((-0.144, 0, 0.035), False)
        check((0, 0.144, 0.035), False)
        check((0, -0.144, 0.035), False)
        check((0.13, 0, 0.035), False, horizontal=True) # rotated body protrudes through wall
        check((0, 0, 0.04), True)                      # contained, without support contact
        check((0, 0, scene.BIN_WALL + 0.025), True)     # bottom on the floor
        check((0, 0, scene.BIN_WALL + 0.025 - 0.05 * MM), True)  # contact tolerance
        for _ in range(300):
            env._physics_step()
        assert env.status()[p.name] == {'removed': True, 'in_kit': True}  # physically resting in bin
