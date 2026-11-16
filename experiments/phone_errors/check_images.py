"""Check 2: write warped images for a few perturbations as PNG, and measure the image shift against what the matrices predict."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from pe_common import *
DEG = np.pi / 180
PNG = f"{OUT}/png"; os.makedirs(PNG, exist_ok=True)

def luma(x):
    Y = np.zeros((256, 512), np.float32); x = x.astype(np.float32)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = x[0], x[1], x[2], x[3]; return Y
def view(i, cam, vw, f, rpy):
    return c_warp(src_frame(cam, i), c_matrix(K_of(f), rpy, vw))
def rgbish(x):   # grey from luma, brightened for the night footage
    Y = luma(x); return np.clip(Y * 255 / max(Y.max(), 1) , 0, 255).astype(np.uint8)
def with_lines(g, vw):
    im = np.stack([g] * 3, -1).copy(); row = int(round(47.6 if vw == 0 else 0.5 * (256 + 47.6)))   # model-frame principal point row (luma coords)
    im[row, :, :] = [255, 0, 0]; im[:, 256, :] = [0, 160, 255]; return im

def shift_est(a, b, rng=70):
    """integer (dy, dx) such that b(y,x) ~ a(y-dy, x-dx), by minimum mean abs difference over the overlap (gradient images to ignore brightness)"""
    ga = np.hypot(*np.gradient(a)); gb = np.hypot(*np.gradient(b)); best = (1e18, 0, 0)
    for dy in range(-rng, rng + 1, 1):
        for dx in (0,):
            pass
    def cost(dy, dx):
        y0, y1 = max(0, dy), 256 + min(0, dy); x0, x1 = max(0, dx), 512 + min(0, dx)
        A = ga[y0 - dy:y1 - dy, x0 - dx:x1 - dx]; B = gb[y0:y1, x0:x1]; return np.abs(A - B).mean() / (np.abs(A).mean() + 1e-6)
    cy = min(range(-rng, rng + 1), key=lambda d: cost(d, 0)); cx = min(range(-rng, rng + 1), key=lambda d: cost(0, d))
    return cy, cx

def expected_shift(f, rpy_new, vw, pt=(672.0, 380.0)):
    """where a fixed source pixel `pt` lands in the 512x256 luma view, new minus baseline (dst = M^-1 src)"""
    def dst(rpy):
        m = c_matrix(K_of(f), rpy, vw).reshape(3, 3).astype(np.float64); p = np.linalg.solve(m, np.array([pt[0], pt[1], 1.0])); return p[:2] / p[2]
    return dst(rpy_new) - dst(RPY0)

cases = [("pitch+2", F_NARROW, F_NARROW, lambda r: r + np.array([0, 2 * DEG, 0], np.float32)),
         ("pitch-2", F_NARROW, F_NARROW, lambda r: r - np.array([0, 2 * DEG, 0], np.float32)),
         ("yaw+2", F_NARROW, F_NARROW, lambda r: r + np.array([0, 0, 2 * DEG], np.float32)),
         ("roll+2", F_NARROW, F_NARROW, lambda r: r + np.array([2 * DEG, 0, 0], np.float32))]
print("Shift of image content in the narrow model view (512x256 luma coords), frames 100/300/500: measured (dy,dx) from pixels (only the axis that the perturbation moves is meaningful; the other component is an unconstrained search and is noise) vs predicted from the warp matrix for the source centre")
for name, f, _, fn in cases:
    ms = []
    for i in (100, 300, 500):
        ms.append(shift_est(luma(view(i, 0, 0, f, RPY0)), luma(view(i, 0, 0, f, fn(RPY0).astype(np.float32)))))
    e = expected_shift(f, fn(RPY0).astype(np.float32), 0)
    print("  %-8s measured (dy,dx) per frame %s   predicted (dy,dx) = (%.1f, %.1f)" % (name, ms, e[1], e[0]))
i = 300
rows = []
for nm, rpy in (("baseline", RPY0), ("pitch+2deg", RPY0 + np.array([0, 2 * DEG, 0], np.float32)), ("pitch-2deg", RPY0 - np.array([0, 2 * DEG, 0], np.float32)), ("yaw+2deg", RPY0 + np.array([0, 0, 2 * DEG], np.float32))):
    rows.append(with_lines(rgbish(view(i, 0, 0, F_NARROW, rpy.astype(np.float32))), 0))
Image.fromarray(np.concatenate([np.concatenate(rows[:2], 1), np.concatenate(rows[2:], 1)], 0)).save(f"{PNG}/calib_narrow_view.png")
z = np.zeros(3, np.float32); f63 = (W / 2) / np.tan(63 * DEG / 2)
tiles = [[with_lines(rgbish(view(i, 0, 0, F_NARROW, RPY0)), 0), with_lines(rgbish(view(i, 1, 1, F_WIDE, RPY0)), 1)],
         [with_lines(rgbish(view(i, 1, 0, F_WIDE, RPY0)), 0), with_lines(rgbish(view(i, 1, 1, F_WIDE, RPY0)), 1)],
         [with_lines(rgbish(view(i, 0, 0, F_NARROW, RPY0)), 0), with_lines(rgbish(view(i, 0, 1, F_NARROW, RPY0)), 1)],
         [with_lines(rgbish(view(i, 0, 0, f63, z)), 0), with_lines(rgbish(view(i, 0, 1, f63, z)), 1)]]
Image.fromarray(np.concatenate([np.concatenate(t, 1) for t in tiles], 0)).save(f"{PNG}/onecamera_views.png")
print("PNGs written to", PNG, "(rows: baseline narrow|wide; one_wide; one_narrow; ios recipe on narrow frames; red line = model principal-point row, blue = centre column)")
