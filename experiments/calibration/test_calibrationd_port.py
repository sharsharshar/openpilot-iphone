"""Checks of the Calibrator port. (1) The cases of upstream selfdrive/locationd/test/test_calibrationd.py (basics, low speed, yaw rate, std rejects,
auto reset) run against the port; (2) the euler helpers against upstream transformations.py if its path is given in UPSTREAM_TRANSFORMATIONS.
Usage: CAL_CLIP_DIR=<any clip dir with calibration/cal_log.pkl> python -I test_calibrationd_port.py"""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # -I removes the script directory from sys.path
import numpy as np
import cal_common as cc
from cal_common import Calibrator, DEFAULTS, HEIGHT_INIT, RECAL, CAL

BLOCK_SIZE, INPUTS_NEEDED, INPUTS_WANTED = DEFAULTS["BLOCK_SIZE"], DEFAULTS["INPUTS_NEEDED"], DEFAULTS["INPUTS_WANTED"]
MIN_SPEED_FILTER, MAX_YAW_RATE_FILTER = DEFAULTS["MIN_SPEED_FILTER"], DEFAULTS["MAX_YAW_RATE_FILTER"]
MAX_ALLOWED_PITCH_SPREAD, MAX_ALLOWED_YAW_SPREAD = DEFAULTS["MAX_ALLOWED_PITCH_SPREAD"], DEFAULTS["MAX_ALLOWED_YAW_SPREAD"]

def process_messages(c, cam_odo_calib, cycles, cam_odo_speed=MIN_SPEED_FILTER + 1, carstate_speed=MIN_SPEED_FILTER + 1, cam_odo_yr=0.0, cam_odo_speed_std=1e-3, cam_odo_height_std=1e-3):
    for _ in range(cycles):
        c.handle_v_ego(carstate_speed)
        c.handle_cam_odom([cam_odo_speed, np.sin(cam_odo_calib[2]) * cam_odo_speed, -np.sin(cam_odo_calib[1]) * cam_odo_speed], [0.0, 0.0, cam_odo_yr], [0.0, 0.0, 0.0],
                          [cam_odo_speed_std] * 3, [0.0, 0.0, HEIGHT_INIT.item()], [cam_odo_height_std] * 3)

def mk(): return Calibrator()
c = mk(); process_messages(c, [0, 0, 0], BLOCK_SIZE * (INPUTS_WANTED + 1)); assert c.valid_blocks == INPUTS_WANTED; np.testing.assert_allclose(c.rpy, np.zeros(3)); np.testing.assert_allclose(c.height, HEIGHT_INIT)
c = mk(); process_messages(c, [0, 0, 0], BLOCK_SIZE * INPUTS_WANTED, cam_odo_speed=MIN_SPEED_FILTER - 1); process_messages(c, [0, 0, 0], BLOCK_SIZE * INPUTS_WANTED, carstate_speed=MIN_SPEED_FILTER - 1); assert c.valid_blocks == 0
c = mk(); process_messages(c, [0, 0, 0], BLOCK_SIZE * INPUTS_WANTED, cam_odo_yr=MAX_YAW_RATE_FILTER); assert c.valid_blocks == 0
c = mk(); process_messages(c, [0, 0, 0], BLOCK_SIZE * INPUTS_WANTED, cam_odo_speed_std=1e3); assert c.valid_blocks == INPUTS_NEEDED
c = mk(); process_messages(c, [0, 0, 0], BLOCK_SIZE * INPUTS_WANTED, cam_odo_height_std=1e3); assert c.valid_blocks == INPUTS_NEEDED
c = mk(); process_messages(c, [0, 0, 0], BLOCK_SIZE * (INPUTS_NEEDED + 1)); assert c.valid_blocks == INPUTS_NEEDED; np.testing.assert_allclose(c.rpy, 0, atol=1e-3)
process_messages(c, [0, MAX_ALLOWED_PITCH_SPREAD * 0.9, MAX_ALLOWED_YAW_SPREAD * 0.9], BLOCK_SIZE + 10); assert c.valid_blocks == INPUTS_NEEDED + 1 and c.cal_status == CAL
for ang in ([0, MAX_ALLOWED_PITCH_SPREAD * 1.1, 0.0], [0, 0.0, MAX_ALLOWED_YAW_SPREAD * 1.1]):
    c = mk(); process_messages(c, [0, 0, 0], BLOCK_SIZE * (INPUTS_NEEDED + 1)); assert c.valid_blocks == INPUTS_NEEDED
    process_messages(c, ang, BLOCK_SIZE + 10); assert c.valid_blocks == 0 and c.cal_status == RECAL; np.testing.assert_allclose(c.rpy, ang, atol=1e-2)
print("upstream test_calibrationd cases pass on the port")
# Toy closed loop: the 'model' reports the residual between a true angle triple and the angles currently in use (its output lives in the calibrated frame, see RESULTS.md).
# The estimate then converges to the truth. If the report ignored the angles in use (constant report) the estimate would keep adding the same angle and run away.
off = np.array([0, 0.02, -0.03]); v = MIN_SPEED_FILTER + 1
c = mk()
for _ in range(BLOCK_SIZE * 8):
    c.handle_v_ego(v); res = off - c.get_smooth_rpy()
    c.handle_cam_odom([v, np.sin(res[2]) * v, -np.sin(res[1]) * v], [0, 0, 0], [0, 0, 0], [1e-3] * 3, [0, 0, 1.22], [1e-3] * 3)
np.testing.assert_allclose(c.rpy, off, atol=2e-3); print("toy closed loop converges to the true angles:", np.round(c.rpy, 4))
c = mk(); process_messages(c, off, BLOCK_SIZE * 8); print("toy open loop (report ignores the angles in use): estimate", np.round(c.rpy, 4), "vs true", off, "-> runs away, as expected")
up = os.environ.get("UPSTREAM_TRANSFORMATIONS")
if up:
    spec = importlib.util.spec_from_file_location("upt", up); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    rng = np.random.default_rng(0); worst = 0
    for _ in range(200):
        e = rng.uniform(-0.3, 0.3, 3)
        worst = max(worst, np.abs(m.euler2rot_single(e) - cc.rot_from_euler(e)).max(), np.abs(m.rot2euler_single(m.euler2rot_single(e)) - cc.euler_from_rot(cc.rot_from_euler(e))).max())
    print("euler helpers vs upstream transformations.py: max abs diff", worst)
# Effective weights of moving_avg_with_linear_decay over one block (value after the last sample as a linear function of the samples in the block)
for B in (100, 20):
    w = np.zeros(B)
    for j in range(B):
        x = np.zeros(B); x[j] = 1; v = 0.0
        for i in range(B): v = cc.moving_avg_with_linear_decay(v, x[i], i, float(B))
        w[j] = v
    print("BLOCK_SIZE %d: weights sum %.4f, effective number of samples 1/sum(w^2) = %.1f, peak weight at sample %d, weight share of last quarter %.2f, of first half %.4f" % (B, w.sum(), 1 / (w ** 2).sum(), w.argmax(), w[-B // 4:].sum(), w[:B // 2].sum()))
