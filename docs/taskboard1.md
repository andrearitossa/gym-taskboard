# Task Board #1

Sources (NIST): [assembly instructions & parts list](https://www.nist.gov/document/assemblyinstructionsv4docx),
[CAD](https://www.nist.gov/document/task-board-1-cad-zip), [kit print](https://www.nist.gov/document/kitprintpdf),
[benchmark sheet](https://www.nist.gov/document/benchmarknistassemblytaskboard1v10), [protocol](https://www.nist.gov/document/protocolnistassemblytaskboardsv10).

The board is a 384 × 384 × 9 mm acrylic plate on 20 mm standoffs. Hole positions and sizes come from the CAD plate
(`GMC_Laser_Plate_Virtual`), cross-checked against the NIST photos. There are 20 scored parts:

| Competency | Part (`name`) | Real part | Modelled as |
|---|---|---|---|
| pegs | `peg_round_{4,8,12,16}` | RGOCG Ø4–16 × 50 mm | cylinder in a Ø+0.2 mm through-hole; seated on the table |
| pegs | `peg_rect_{4,8,12,16}` | KET 4×4, 8×7, 12×8, 16×10 × 50 mm | box in a slot +0.1–0.25 mm |
| gears | `gear_{small,medium,large}` | GEABP1.0-{20,40,60}-10-B-10 | involute module-1 teeth, Ø10.1 bore on Ø9.99 shafts, centre distances 30 / 50 mm |
| connectors | `bnc` | BNC-R / BNCP | sleeve over a Ø9.6 post, then a 90° bayonet lock (2 mm travel) |
| connectors | `usb` | USB-A | 12 × 4.5 nose in a 12.3 × 4.8 cavity, 12 mm deep |
| connectors | `rj45` | RJ45 | 11.7 × 8 nose, 14 mm deep |
| connectors | `waterproof` | MCON-E10 | 44 × 21 nose, 12 mm deep |
| connectors | `dsub` | D-sub 25 (shell B) | plug D-shell over the board's D-shell (10° sides, 0.2 mm clearance) |
| nuts | `nut_m{4,8,12,16}` | ISO hex nuts | threaded onto fixed bolts, coarse pitch 0.7 / 1.25 / 1.75 / 2.0 mm |

The kit poses follow the NIST kit print: gears and nuts lie flat; some pegs stand and others lie down; plugs lie on their side.

## Simplifications
- All geometry is procedural, built from real dimensions. The NIST STL meshes are not used.
- Thread flanks, connector pins, latches and set screws are not modelled. Connectors are held by friction only.
- Cables (USB, RJ45) are not simulated; only the plug body is.
- Gears are mounted hub-down. Bolts and female connectors are rigid parts of the board.
