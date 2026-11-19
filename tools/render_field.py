#!/usr/bin/env python3
"""Render the map-driven sizing-field PLYs as pictures, on this Mac, with no GL context.

The app writes three coloured PLYs per config when MAP_TEL_DEBUG_PLY is on
(baseline_comparison.cc), into <config>/sizing_field/:

    tel_01_jump.ply     the raw detector -- per-vertex max over incident edges of
                        sqrt(log(s0/s0')^2 + log(s1/s1')^2), the prescribed field's
                        discontinuity. Magenta = no jump, white = at/above 4x threshold.
    tel_02_phi.ply      the diffused seam indicator, one implicit heat step at t = w^2,
                        normalised to [0,1]. Magenta = cold, white = hot. THIS is the
                        field whose thickness w sets.
    tel_03_sizing.ply   the resulting target edge length after min() with the curvature
                        field. Magenta = floor (refine hard), white = max_edge_length.

Those open in any mesh viewer as they are. This script is for looking at them without
one, and for putting several side by side: it reuses the rasterizer in render_trace.py,
so the camera, shading and output match the optimization animations.

USAGE
    .venv/bin/python tools/render_field.py <config_dir>              # all three, in a row
    .venv/bin/python tools/render_field.py <file.ply> [more.ply ...] # specific ones
    options: --azim --elev --width --height --wire -o OUT.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_trace import Camera, render_panel, pick_font, BG, GRID, INK  # noqa: E402


def read_colored_ply(path):
    """ASCII PLY with x y z red green blue per vertex -> (V, F, face_rgb in [0,1]).

    write_colored_ply_flat splits every face into its own three vertices so the colours
    stay flat, so a face's colour is just its first vertex's.
    """
    with open(path, "r") as fh:
        if fh.readline().strip() != "ply":
            raise SystemExit(f"{path}: not a PLY")
        n_v = n_f = 0
        props = []
        elem = None
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "element":
                elem = t[1]
                if elem == "vertex":
                    n_v = int(t[2])
                elif elem == "face":
                    n_f = int(t[2])
            elif t[0] == "property" and elem == "vertex":
                props.append(t[-1])
            elif t[0] == "end_header":
                break
        idx = {name: i for i, name in enumerate(props)}
        for need in ("x", "y", "z", "red", "green", "blue"):
            if need not in idx:
                raise SystemExit(f"{path}: vertex property '{need}' missing")

        V = np.empty((n_v, 3))
        C = np.empty((n_v, 3))
        for i in range(n_v):
            t = fh.readline().split()
            V[i] = (float(t[idx["x"]]), float(t[idx["y"]]), float(t[idx["z"]]))
            C[i] = (float(t[idx["red"]]), float(t[idx["green"]]), float(t[idx["blue"]]))
        F = np.empty((n_f, 3), dtype=np.int64)
        for i in range(n_f):
            t = fh.readline().split()
            if int(t[0]) != 3:
                raise SystemExit(f"{path}: face {i} is not a triangle")
            F[i] = (int(t[1]), int(t[2]), int(t[3]))
    return V, F, C[F[:, 0]] / 255.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", type=Path, nargs="+",
                    help="a config dir (uses its sizing_field/) or explicit .ply files")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--azim", type=float, default=35.0)
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--width", type=int, default=760, help="width of ONE panel")
    ap.add_argument("--height", type=int, default=680, help="height of ONE panel")
    ap.add_argument("--wire", action="store_true", help="overlay the input mesh edges")
    args = ap.parse_args()

    plys = []
    for p in args.inputs:
        if p.is_dir():
            d = p / "sizing_field" if (p / "sizing_field").is_dir() else p
            found = sorted(d.glob("*.ply"))
            if not found:
                raise SystemExit(f"no .ply in {d} -- is MAP_TEL_DEBUG_PLY on, and did "
                                 f"the run have MAP_DRIVEN_TEL on with a prescribed field?")
            plys += found
        else:
            plys.append(p)

    head = 34
    total_w = args.width * len(plys)
    total_h = head + args.height
    canvas = Image.new("RGB", (total_w, total_h), BG)
    dr = ImageDraw.Draw(canvas)
    font_hd, font_sm = pick_font(17), pick_font(13)

    for k, ply in enumerate(plys):
        V, F, rgb = read_colored_ply(ply)
        cam = Camera(V.min(axis=0), V.max(axis=0), args.azim, args.elev,
                     args.width, args.height)
        panel = render_panel(V, F, cam, (70, 70, 74) if args.wire else None,
                             face_rgb=rgb)
        canvas.paste(Image.fromarray(panel), (k * args.width, head))
        dr.text((k * args.width + 12, 9), ply.stem, font=font_hd, fill=INK)
        if k:
            dr.line([k * args.width, head, k * args.width, total_h], fill=GRID)

    dr.text((total_w - 230, 13), "magenta = low   white = high", font=font_sm,
            fill=(140, 140, 146))

    out = args.out or (plys[0].parent / "sizing_field.png")
    canvas.save(out)
    print(f"Wrote {out}  ({len(plys)} panel(s), {total_w}x{total_h})")


if __name__ == "__main__":
    main()
