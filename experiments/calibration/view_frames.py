"""Warped narrow frame i before recovery (zero angles), after recovery (angles in use at the end of clL_zero) and with the logged angles; the model-frame
principal point row (y = 47.6 of 256 in the 512x256 luma frame) is drawn as a red line. Usage: CAL_CLIP_DIR=<clip> python -I view_frames.py <frame> <out.png>"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cal_common as cc
from PIL import Image, ImageDraw
i = int(sys.argv[1]); r = np.load(f"{cc.RES}/clL_zero.npz"); rec = r["rpy_used"][-1]
def luma(rpy):
    x = cc.warp_pair(i, rpy)[0].astype(np.uint8); Y = np.zeros((256, 512), np.uint8)
    Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2] = x[0], x[1], x[2], x[3]; return Y
panels = [luma(np.zeros(3)), luma(rec), luma(cc.RPY_LOG[i])]
im = Image.fromarray(np.concatenate(panels, 0)).convert("RGB"); d = ImageDraw.Draw(im)
for k, t in enumerate(("zero angles", "after recovery (clL_zero end): %s deg" % np.round(np.degrees(rec), 2), "logged: %s deg" % np.round(np.degrees(cc.RPY_LOG[i]), 2))):
    d.line([0, k * 256 + 47.6, 511, k * 256 + 47.6], fill=(255, 0, 0)); d.line([256, k * 256, 256, k * 256 + 255], fill=(255, 0, 0)); d.text((5, k * 256 + 5), t, fill=(255, 255, 0))
im.save(sys.argv[2]); print("diff recovered vs logged warp, mean abs grey:", np.abs(panels[1].astype(int) - panels[2]).mean(), "zero vs logged:", np.abs(panels[0].astype(int) - panels[2]).mean())
