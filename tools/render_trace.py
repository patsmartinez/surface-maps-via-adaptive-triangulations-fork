#!/usr/bin/env python3
"""Render the per-iteration trace of baseline_comparison as an animation. Mac-local.

WHY THIS EXISTS INSTEAD OF A SCREENSHOT LOOP
The app's own render path (screenshot_config, src/SurfaceMaps/Viewer/) needs glow, which
wants OpenGL 4.2-4.5; macOS caps desktop GL at 4.1, so ISM_HAS_VIEWER is off on Apple
Silicon and there is no in-app way to draw a frame here. trace_step() already dumps the
one thing a renderer needs -- the lifted common triangulation on each input mesh, once
per optimizer step -- so this rasterizes those OBJs itself. numpy + Pillow, nothing else.

WHAT YOU ARE LOOKING AT
Each frame is the common triangulation T lifted onto mesh A (left, magenta wireframe) and
onto mesh B (right, teal). The surfaces barely move -- the lift keeps T on the fixed input
geometry -- so everything you see change is the map: vertices sliding across the surface,
splits adding detail, collapses removing it, flips rewiring. The strips underneath plot
the trace's own metrics with a cursor on the current frame, so a jump in the geometry can
be read against the number it moved.

Grey vertical bands mark the remeshing callbacks (kind contains "remesh"); everything
between them is continuous Newton. That separation is the whole point of the two-callback
trace: it shows whether a change came from a gradient step or from one remeshing pass.

INPUT
  baseline_comparison.cc:  TRACE_ITERATIONS = true, TRACE_WRITE_MESHES = true
  produces, per config:    <out>/<config>/trace/fNNNN_<kind>_{A,B}.obj
                           <out>/<config>/trace.csv

USAGE
  <venv>/bin/python tools/render_trace.py <config_dir> [-o DIR] [--azim 35] [--elev 20]
                    [--width 900] [--height 760] [--stride 1] [--fps 12] [--gif-scale 0.5]

  <config_dir> is the directory holding trace/ and trace.csv, e.g.
  build-headless/output/baseline_comparison_l_shape/AonB_prescribed_fps16_.../
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- Palette: the app's own colours, so a frame reads like a viewer screenshot. --------
BG = (255, 255, 255)
SURFACE = np.array([0.88, 0.88, 0.90])
MAGENTA = (200, 30, 160)      # T on mesh A, matches view_wireframe(..., MAGENTA)
TEAL = (0, 150, 160)          # T on mesh B
INK = (40, 40, 44)
GRID = (218, 218, 222)
BAND = (240, 240, 243)        # remeshing bands in the metric strips

FRAME_RE = re.compile(r"^f(\d+)_(.+)_([AB])\.obj$")


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------
def read_obj(path):
    """Minimal OBJ reader: v + f only, fan-triangulated, 1-based indices, a/b/c stripped."""
    verts, faces = [], []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def unique_edges(F):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def read_trace_csv(path):
    """-> (header list, dict column -> np.array). Non-numeric columns stay as str lists."""
    if not path.exists():
        return [], {}
    rows = [ln.rstrip("\n").split(",") for ln in open(path) if ln.strip()]
    head, body = rows[0], rows[1:]
    cols = {}
    for j, name in enumerate(head):
        vals = [r[j] if j < len(r) else "" for r in body]
        try:
            cols[name] = np.array([float(v) for v in vals])
        except ValueError:
            cols[name] = vals
    return head, cols


def discover_frames(trace_dir):
    """-> sorted [(frame_index, kind)] for which BOTH the A and B obj exist."""
    found = {}
    for p in trace_dir.iterdir():
        m = FRAME_RE.match(p.name)
        if m:
            found.setdefault((int(m.group(1)), m.group(2)), set()).add(m.group(3))
    return sorted(k for k, sides in found.items() if sides == {"A", "B"})


# --------------------------------------------------------------------------------------
# Camera: fixed for the whole animation, or the map's motion drowns in camera motion
# --------------------------------------------------------------------------------------
class Camera:
    """Orthographic. Fixed pose and fixed scale, both derived once from a bounding box.

    Orthographic on purpose: with perspective, a vertex sliding toward the viewer changes
    size as well as position, and the thing being measured here is position.
    """

    def __init__(self, bb_min, bb_max, azim, elev, w, h, fill=0.88):
        self.center = 0.5 * (bb_min + bb_max)
        a, e = math.radians(azim), math.radians(elev)
        # Camera sits along +d from the centre; d also orders depth (larger dot = nearer).
        self.d = np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
        up_world = np.array([0.0, 1.0, 0.0])
        right = np.cross(up_world, self.d)
        if np.linalg.norm(right) < 1e-9:                      # looking straight down
            right = np.cross(np.array([0.0, 0.0, 1.0]), self.d)
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.d, self.right)
        self.w, self.h = w, h

        corners = np.array([[x, y, z] for x in (bb_min[0], bb_max[0])
                            for y in (bb_min[1], bb_max[1])
                            for z in (bb_min[2], bb_max[2])]) - self.center
        ex = np.abs(corners @ self.right).max()
        ey = np.abs(corners @ self.up).max()
        self.scale = min(0.5 * w * fill / max(ex, 1e-12), 0.5 * h * fill / max(ey, 1e-12))
        self.diag = float(np.linalg.norm(bb_max - bb_min))

    def project(self, V):
        q = V - self.center
        px = 0.5 * self.w + (q @ self.right) * self.scale
        py = 0.5 * self.h - (q @ self.up) * self.scale
        depth = -(q @ self.d)                                  # smaller = nearer
        return np.stack([px, py], axis=1), depth


# --------------------------------------------------------------------------------------
# Rasterizer
# --------------------------------------------------------------------------------------
def render_panel(V, F, cam, edge_color, edge_bias_frac=0.0015):
    """Flat-shaded z-buffered surface with its own edges drawn on top, depth-tested."""
    h, w = cam.h, cam.w
    img = np.empty((h, w, 3), dtype=np.float32)
    img[:] = np.asarray(BG, dtype=np.float32) / 255.0
    zbuf = np.full((h, w), np.inf)
    if len(F) == 0:
        return (img * 255).astype(np.uint8)

    P, D = cam.project(V)

    # Flat Lambert from a light just off the view axis, two-sided (|cos|) so a stray
    # inward-facing normal shades like its neighbours instead of going black.
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(nl, 1e-300)
    light = cam.d + 0.35 * cam.right + 0.30 * cam.up
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.abs(n @ light)

    x, y = P[:, 0], P[:, 1]
    base = SURFACE.astype(np.float32)
    for i in range(len(F)):
        a, b, c = F[i]
        x0, y0, x1, y1, x2, y2 = x[a], y[a], x[b], y[b], x[c], y[c]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        xmin = max(int(math.floor(min(x0, x1, x2))), 0)
        xmax = min(int(math.ceil(max(x0, x1, x2))), w - 1)
        ymin = max(int(math.floor(min(y0, y1, y2))), 0)
        ymax = min(int(math.ceil(max(y0, y1, y2))), h - 1)
        if xmin > xmax or ymin > ymax:
            continue
        gx, gy = np.meshgrid(np.arange(xmin, xmax + 1) + 0.5,
                             np.arange(ymin, ymax + 1) + 0.5)
        w0 = ((x1 - gx) * (y2 - gy) - (x2 - gx) * (y1 - gy)) / area
        w1 = ((x2 - gx) * (y0 - gy) - (x0 - gx) * (y2 - gy)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        dd = w0 * D[a] + w1 * D[b] + w2 * D[c]
        zsub = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        upd = inside & (dd < zsub)
        if not upd.any():
            continue
        zsub[upd] = dd[upd]
        img[ymin:ymax + 1, xmin:xmax + 1][upd] = base * shade[i]

    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    _draw_edges(img, zbuf, P, D, unique_edges(F), edge_color,
                bias=edge_bias_frac * cam.diag)
    return img


def _draw_edges(img, zbuf, P, D, E, color, bias):
    """All edges in one vectorized pass, hidden parts rejected against the surface z-buffer.

    The bias is in world units along the view axis: without it an edge loses the depth
    test against its own two faces and the wireframe comes out stippled.
    """
    h, w = img.shape[:2]
    x0, y0, d0 = P[E[:, 0], 0], P[E[:, 0], 1], D[E[:, 0]]
    x1, y1, d1 = P[E[:, 1], 0], P[E[:, 1], 1], D[E[:, 1]]
    nsamp = np.maximum(np.abs(x1 - x0), np.abs(y1 - y0)).astype(np.int64) + 1
    nsamp = np.clip(nsamp, 1, 8192)
    total = int(nsamp.sum())
    if total == 0:
        return
    offs = np.concatenate(([0], np.cumsum(nsamp)[:-1]))
    k = np.arange(total) - np.repeat(offs, nsamp)
    t = k / np.maximum(np.repeat(nsamp, nsamp) - 1, 1)
    px = np.repeat(x0, nsamp) + t * np.repeat(x1 - x0, nsamp)
    py = np.repeat(y0, nsamp) + t * np.repeat(y1 - y0, nsamp)
    pd = np.repeat(d0, nsamp) + t * np.repeat(d1 - d0, nsamp)
    ix = np.rint(px).astype(np.int64)
    iy = np.rint(py).astype(np.int64)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy, pd = ix[ok], iy[ok], pd[ok]
    if ix.size == 0:
        return
    vis = pd <= zbuf[iy, ix] + bias
    img[iy[vis], ix[vis]] = color


# --------------------------------------------------------------------------------------
# Metric strips
# --------------------------------------------------------------------------------------
def pick_font(size):
    for cand in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/Library/Fonts/Arial.ttf"):
        if os.path.exists(cand):
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_strip(dr, box, frames, values, cursor, label, kinds, logscale=True):
    """One metric against frame index, log y by default, grey bands on remeshing frames."""
    x0, y0, x1, y1 = box
    dr.rectangle(box, fill=(252, 252, 253), outline=GRID)
    good = np.isfinite(values)
    if logscale:
        good &= values > 0
    if good.sum() < 2:
        dr.text((x0 + 6, y0 + 4), f"{label}: no data", font=pick_font(12), fill=INK)
        return

    v = np.log10(values[good]) if logscale else values[good]
    f = frames[good]
    fmin, fmax = frames.min(), frames.max()
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-12:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    def sx(fr):
        return x0 + (x1 - x0) * (fr - fmin) / max(fmax - fmin, 1)

    def sy(val):
        return y1 - (y1 - y0 - 2) * (val - vmin) / (vmax - vmin) - 1

    # Remeshing passes as bands, so a jump can be attributed before reading any number.
    for fr, kd in zip(frames, kinds):
        if "remesh" in kd:
            px = sx(fr)
            dr.rectangle([px - 1, y0 + 1, px + 1, y1 - 1], fill=BAND)

    dr.line([(sx(a), sy(b)) for a, b in zip(f, v)], fill=(70, 95, 200), width=2)
    cx = sx(cursor)
    dr.line([cx, y0 + 1, cx, y1 - 1], fill=(220, 60, 60), width=2)

    cur = values[frames == cursor]
    cur_s = f"{cur[0]:.3e}" if cur.size and np.isfinite(cur[0]) else "-"
    font = pick_font(13)
    txt = f"{label}   {cur_s}"
    tw = dr.textlength(txt, font=font)
    dr.rectangle([x0 + 2, y0 + 2, x0 + 12 + tw, y0 + 20], fill=(252, 252, 253))
    dr.text((x0 + 7, y0 + 3), txt, font=font, fill=INK)
    small = pick_font(11)
    hi, lo = (f"1e{vmax:.1f}", f"1e{vmin:.1f}") if logscale else (f"{vmax:.3g}", f"{vmin:.3g}")
    dr.text((x1 - 54, y0 + 3), hi, font=small, fill=(140, 140, 146))
    dr.text((x1 - 54, y1 - 15), lo, font=small, fill=(140, 140, 146))


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config_dir", type=Path, help="dir holding trace/ and trace.csv")
    ap.add_argument("-o", "--out", type=Path, default=None, help="default <config_dir>/anim")
    ap.add_argument("--azim", type=float, default=35.0)
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--width", type=int, default=900, help="width of ONE panel")
    ap.add_argument("--height", type=int, default=760, help="height of ONE panel")
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--gif-scale", type=float, default=0.5, help="0 to skip the gif")
    ap.add_argument("--no-strips", action="store_true", help="geometry only, no metric plots")
    args = ap.parse_args()

    trace_dir = args.config_dir / "trace"
    if not trace_dir.is_dir():
        sys.exit(f"no trace/ in {args.config_dir}\n"
                 f"Set TRACE_ITERATIONS and TRACE_WRITE_MESHES in baseline_comparison.cc "
                 f"and re-run the app.")
    keys = discover_frames(trace_dir)
    if not keys:
        sys.exit(f"{trace_dir} holds no complete fNNNN_<kind>_{{A,B}}.obj pair. "
                 f"TRACE_WRITE_MESHES was probably off.")
    keys = keys[::max(1, args.stride)]

    _, cols = read_trace_csv(args.config_dir / "trace.csv")
    out = args.out or (args.config_dir / "anim")
    png_dir = out / "frames"
    png_dir.mkdir(parents=True, exist_ok=True)

    # One camera per side, fixed for the whole animation. Per side because A and B are
    # different shapes (that is the point of the pair) and a shared scale would leave the
    # larger one clipped or the smaller one tiny; fixed over time because any camera that
    # re-fits per frame turns a sliding vertex into a moving camera and the animation
    # stops meaning anything. Fitted to the union of the first and last frame, so a mesh
    # that grows under splits never walks out of view halfway through.
    cams = {}
    for side in "AB":
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for fr, kd in {keys[0], keys[-1]}:
            V, _ = read_obj(trace_dir / f"f{fr:04d}_{kd}_{side}.obj")
            lo, hi = np.minimum(lo, V.min(axis=0)), np.maximum(hi, V.max(axis=0))
        cams[side] = Camera(lo, hi, args.azim, args.elev, args.width, args.height)

    strip_h = 0 if args.no_strips else 74
    n_strips = 0 if args.no_strips else 2
    head_h = 34
    total_w = 2 * args.width
    total_h = head_h + args.height + n_strips * strip_h

    font_hd = pick_font(18)
    font_sm = pick_font(13)
    gif_frames = []
    tag = args.config_dir.name

    for n, (fr, kd) in enumerate(keys):
        VA, FA = read_obj(trace_dir / f"f{fr:04d}_{kd}_A.obj")
        VB, FB = read_obj(trace_dir / f"f{fr:04d}_{kd}_B.obj")
        panel_a = render_panel(VA, FA, cams["A"], MAGENTA)
        panel_b = render_panel(VB, FB, cams["B"], TEAL)

        canvas = Image.new("RGB", (total_w, total_h), BG)
        canvas.paste(Image.fromarray(panel_a), (0, head_h))
        canvas.paste(Image.fromarray(panel_b), (args.width, head_h))
        dr = ImageDraw.Draw(canvas)
        dr.line([args.width, head_h, args.width, head_h + args.height], fill=GRID)

        label = ""
        if isinstance(cols.get("label"), list) and isinstance(cols.get("frame"), np.ndarray):
            hit = np.nonzero(cols["frame"] == fr)[0]
            if hit.size:
                label = cols["label"][hit[0]]
        nv = int(cols["n_verts_T"][cols["frame"] == fr][0]) if "n_verts_T" in cols and (cols["frame"] == fr).any() else -1
        nf = int(cols["n_faces_T"][cols["frame"] == fr][0]) if "n_faces_T" in cols and (cols["frame"] == fr).any() else -1

        dr.text((10, 7), f"{tag}", font=font_hd, fill=INK)
        right = f"frame {fr}   {kd}  {label}" + (f"   |V(T)| {nv}  |F(T)| {nf}" if nv > 0 else "")
        dr.text((total_w - 10 - dr.textlength(right, font=font_sm), 12), right,
                font=font_sm, fill=(90, 90, 96))
        dr.text((12, head_h + 6), "T on A", font=font_sm, fill=MAGENTA)
        dr.text((args.width + 12, head_h + 6), "T on B", font=font_sm, fill=TEAL)

        if not args.no_strips and isinstance(cols.get("frame"), np.ndarray):
            kinds = cols.get("kind", [""] * len(cols["frame"]))
            y = head_h + args.height
            for key, lab, logy in (("corr_mean", "corr_mean", True),
                                   ("drift_max_deg", "drift_max (deg)", True)):
                if key in cols and isinstance(cols[key], np.ndarray):
                    draw_strip(dr, (0, y, total_w - 1, y + strip_h - 1),
                               cols["frame"], cols[key], fr, lab, kinds, logscale=logy)
                y += strip_h

        canvas.save(png_dir / f"{n:05d}.png")
        if args.gif_scale > 0:
            gif_frames.append(canvas.resize(
                (int(total_w * args.gif_scale), int(total_h * args.gif_scale)),
                Image.LANCZOS))
        if (n + 1) % 20 == 0 or n + 1 == len(keys):
            print(f"  {n + 1}/{len(keys)} frames", flush=True)

    print(f"PNGs: {png_dir}")
    if gif_frames:
        gif = out / "anim.gif"
        gif_frames[0].save(gif, save_all=True, append_images=gif_frames[1:],
                           duration=int(1000 / args.fps), loop=0, optimize=True)
        print(f"GIF:  {gif}  ({len(gif_frames)} frames, {gif.stat().st_size / 1e6:.1f} MB)")

    if shutil.which("ffmpeg"):
        mp4 = out / "anim.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", str(png_dir / "%05d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        str(mp4)], check=True)
        print(f"MP4:  {mp4}")
    else:
        print("MP4:  skipped (no ffmpeg; `brew install ffmpeg` to also get anim.mp4)")


if __name__ == "__main__":
    main()
