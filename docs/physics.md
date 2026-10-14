# Physics modelling

The aim is real contact mechanics wherever MuJoCo can resolve it, and exact kinematic constraints where it cannot.

## Contacts
- **Convex decomposition.** MuJoCo collides convex shapes only. So every hole, bore and cavity is built from convex pieces:
  - round holes: 32 convex quads whose flats sit exactly at the hole radius;
  - slots and cavities: 4 boxes;
  - the plate: rectangles tiled around the holes;
  - gear teeth: one convex involute prism per tooth.
  Clearances are therefore the real ones (0.05–0.25 mm), and meshing, jamming and misalignment are emergent.
- **Solver settings.** `dt = 1 ms`, `implicitfast`, elliptic friction cones with `impratio = 10` (stable, no-slip grasps), multi-CCD (default).
- **Contact stiffness.** All geoms use `solref = (4 ms, 1)` (≥ 2·dt for stability) and `solimp = (0.95, 0.99, 0.5 mm)`. Measured penetration is ≈ 0 mm in grasps and sub-0.1 mm in fits.
- **Friction.** Parts use μ = 0.4 (metal/plastic). The Panda finger pads keep μ = 1; MuJoCo takes the max of the pair.

## Threads and the bayonet (`HelicalCoupling`)
MuJoCo has no screw joint, and a free body cannot be attached to one. Each nut (and the BNC plug) therefore has a
*threaded twin* on the fixture axis, with a slide `s` and a hinge `h` tied by the joint equality `s = a0 + pitch/2π · h`.
- **Engage.** When the free nut is coaxial (< 0.4 mm, < 3°) on the first half-pitch of the bolt, the twin takes over: collisions move to the twin, and the free body becomes a collision-less shadow that tracks it. Observations stay continuous.
- **Seat.** The nut advances exactly one pitch per turn and is stopped by its contact with the plate, so bearing friction and clamping are physical. Thread friction is a hinge `frictionloss`, so nuts do not back-drive under gravity.
- **Release.** Unscrewing past the bolt tip hands the nut back to its free body, with matching velocity.

Two lessons from building this:
- **Constraint softness scales with the inverse mass of the DOFs a constraint acts on directly.** A massless follower welded to the nut gave millimetres of sag, which is why the twin carries the real mass.
- **Hard limits must not take the clamping load.** A frictionless joint limit at the seat jammed the nut. The slide limit therefore sits 0.5 mm below the plate as a safety stop only.

## Robot
- **Arm.** Menagerie Panda with gravity compensation on all links, as the real arm runs under libfranka. Without it the position servos sag by about 1 cm.
- **Gripper.** A stiff servo saturating at 140 N tendon force, matching the Franka Hand's 70 N continuous grasp. Menagerie's kp = 100 N/m squeezes small parts with less than 1 N, and they slip.

## Validation (`tests/test_physics.py`)
Covered by tests:
- pegs drop into their holes;
- rotated or offset pegs are blocked;
- nuts thread down under torque, seat, unscrew and release, and hold still without torque;
- the three-gear train turns at the 60:40:20 ratio;
- a dropped gear meshes;
- the BNC locks;
- connectors push in.

`scripts/demo_peg.py` has the Panda pick and seat pegs with IK.
