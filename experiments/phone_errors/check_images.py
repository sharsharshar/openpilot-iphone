"""Check 2: write warped images for a few perturbations as PNG, and measure the image shift against what the matrices predict.
Works on any clip (frame size and focal lengths from pe_common). Also writes more_views_a.png / more_views_b.png for the "More clips" cases
(built from the other camera, iOS recipe, grey, frozen) and prints the fraction of each model view that falls outside the source frame."""
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

def expected_shift(f, rpy_new, vw, pt=None):
    """where a fixed source pixel `pt` lands in the 512x256 luma view, new minus baseline (dst = M^-1 src)"""
    pt = pt or (W / 2.0, H / 2.0)
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

# ---- "More clips" cases: every tile is (narrow model view | wide model view), frame i; red line = principal-point row, blue = centre column
def tile(a_, b_): return np.concatenate([with_lines(rgbish(a_), 0), with_lines(rgbish(b_), 1)], 1)
fz = lambda cam, vw, f: view(0, cam, vw, f, RPY0)       # frame 0: what a frozen input shows at every step
A = [tile(view(i, 0, 0, F_NARROW, RPY0), view(i, 1, 1, F_WIDE, RPY0)),                 # baseline
     tile(view(i, 0, 0, F_NARROW, RPY0), view(i, 0, 1, F_NARROW, RPY0)),               # one_narrow: wide view built from narrow-camera pixels
     tile(view(i, 1, 0, F_WIDE, RPY0), view(i, 1, 1, F_WIDE, RPY0))]                   # one_wide: narrow view built from wide-camera pixels
B = [tile(view(i, 0, 0, f63, z), view(i, 0, 1, f63, z)),                                # iOS recipe on narrow-camera frames
     tile(view(i, 1, 0, f63, z), view(i, 1, 1, f63, z)),                                # iOS recipe on wide-camera frames
     tile(fz(0, 0, F_NARROW), view(i, 1, 1, F_WIDE, RPY0)),                             # narrow frozen (frame 0) | wide live
     tile(view(i, 0, 0, F_NARROW, RPY0), fz(1, 1, F_WIDE))]                             # narrow live | wide frozen (frame 0)
Image.fromarray(np.concatenate(A, 0)).save(f"{PNG}/more_views_a.png"); Image.fromarray(np.concatenate(B, 0)).save(f"{PNG}/more_views_b.png")
print("PNGs more_views_a.png (rows: baseline; one_narrow; one_wide) and more_views_b.png (rows: iOS recipe on narrow frames; iOS recipe on wide frames; narrow frozen|wide live; narrow live|wide frozen); frame %d, all with the logged rpy except the iOS rows (rpy 0, f for 63 deg)" % i)
print("camera config %s: %dx%d, f narrow %.1f wide %.1f; iOS recipe f %.1f" % (CAM_KEY, W, H, F_NARROW, F_WIDE, f63))
print("fraction of the model view outside the source frame (edge pixels replicated), logged rpy unless noted:")
for nm, f, r, vw in (("baseline narrow view (narrow cam)", F_NARROW, RPY0, 0), ("baseline wide view (wide cam)", F_WIDE, RPY0, 1), ("wide view from NARROW camera (one_narrow)", F_NARROW, RPY0, 1),
                     ("narrow view from WIDE camera (one_wide)", F_WIDE, RPY0, 0), ("iOS recipe, narrow view, rpy 0", f63, z, 0), ("iOS recipe, wide view, rpy 0", f63, z, 1)):
    print("  %-45s %.3f" % (nm, oob_fraction(f, r, vw)))

# ---- geometric agreement of the "built from the other camera" views with the real ones (model-view luma coords, 512x256).
# Brute-force search of the integer shift (dy, dx) that best aligns gradient-magnitude images of the two views (blurred, central crop, at half size),
# frames 100/300/500. A pure resolution difference should give (0, 0); a non-zero shift means the two cameras disagree about where the same
# scene point is, under the assumed intrinsics and the one shared calibration.
def shift2d(a, b, rng=40, crop=(slice(16, 112), slice(64, 192))):
    def prep(x):
        x = luma(x); x = x.reshape(128, 2, 256, 2).mean((1, 3)); x = (x + np.roll(x, 1, 0) + np.roll(x, -1, 0) + np.roll(x, 1, 1) + np.roll(x, -1, 1)) / 5
        gy, gx = np.gradient(x); return np.hypot(gx, gy)
    ga, gb = prep(a), prep(b); A = ga[crop]; A = (A - A.mean()) / (A.std() + 1e-6); best = (-2, 0, 0)
    for dy in range(-rng // 2, rng // 2 + 1):
        for dx in range(-rng // 2, rng // 2 + 1):
            B = gb[crop[0].start + dy:crop[0].stop + dy, crop[1].start + dx:crop[1].stop + dx]
            if B.shape != A.shape: continue
            B = (B - B.mean()) / (B.std() + 1e-6); c = float((A * B).mean())
            if c > best[0]: best = (c, 2 * dy, 2 * dx)
    return best
print("Where does the same scene point land? shift (dy, dx) in model-view luma pixels (positive = the one-camera view is shifted down / right of the true view), with normalised correlation; central crop")
for nm, cam, vw, f in (("narrow view built from WIDE camera (one_wide)", 1, 0, F_WIDE), ("wide view built from NARROW camera (one_narrow)", 0, 1, F_NARROW)):
    res = []
    for fi in (100, 300, 500):
        truth = view(fi, vw, vw, F_NARROW if vw == 0 else F_WIDE, RPY0)     # true camera of that view
        c, dy, dx = shift2d(truth, view(fi, cam, vw, f, RPY0)); res.append("(%d, %d) corr %.2f" % (dy, dx, c))
    print("  %-50s frames 100/300/500: %s" % (nm, "; ".join(res)))
