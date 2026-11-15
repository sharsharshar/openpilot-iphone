"""Pure numpy/tinygrad oracle for opcore. Functions are copied from upstream, file named in each comment.
openpilot: https://github.com/commaai/openpilot (MIT). tinygrad: https://github.com/tinygrad/tinygrad (MIT)."""
import numpy as np
from oracle_parser import Parser
from upstream_constants import ModelConstants, Plan

DT_MDL = 0.05                       # openpilot/common/realtime.py
LONG_SMOOTH_SECONDS = 0.3           # openpilot/selfdrive/modeld/modeld.py
LAT_SMOOTH_SECONDS = 0.0            # same
MIN_LAT_CONTROL_SPEED = 0.3         # same
MIN_SPEED = 1.0                     # selfdrive/controls/lib/drive_helpers.py
MIN_STABLE_DELAY = 0.3

# output layout given by the task (start, end)
SLICES = dict(meta=(0, 55), desire_pred=(55, 87), pose=(87, 99), wide_from_device_euler=(99, 105),
              road_transform=(105, 117), lane_lines=(117, 645), lane_lines_prob=(645, 653),
              road_edges=(653, 917), lead=(917, 1061), lead_prob=(1061, 1064), hidden_state=(1064, 1576),
              plan=(1576, 2566), desire_state=(2566, 2574))

# ---- drive_helpers.py (verbatim logic) ----
def should_stop(v_ego, a_target): return bool(v_ego < 0.3 and a_target < 0.1)
def smooth_value(val, prev_val, tau, dt=DT_MDL):
  alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1
  return alpha * val + (1 - alpha) * prev_val
def get_accel_from_plan(speeds, accels, t_idxs, action_t=DT_MDL):
  if len(speeds) == len(t_idxs):
    v_now = speeds[0]; a_now = accels[0]
    if action_t < MIN_STABLE_DELAY:
      v_target = v_now + (action_t / MIN_STABLE_DELAY) * (np.interp(MIN_STABLE_DELAY, t_idxs, speeds) - v_now)
    else:
      v_target = np.interp(action_t, t_idxs, speeds)
    a_target = 2 * (v_target - v_now) / (action_t) - a_now
  else:
    a_target = 0.0
  return a_target
def curv_from_psis(psi_target, psi_rate, vego, action_t):
  vego = np.clip(vego, MIN_SPEED, np.inf)
  return 2*(psi_target / (vego * action_t)) - psi_rate / vego
def get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
  if action_t < MIN_STABLE_DELAY:
    psi_target = (action_t / MIN_STABLE_DELAY) * np.interp(MIN_STABLE_DELAY, t_idxs, yaws)
  else:
    psi_target = np.interp(action_t, t_idxs, yaws)
  return curv_from_psis(psi_target, yaw_rates[0], vego, action_t)

def parse(row):
  """row: float32[2576] -> dict of parsed arrays (upstream Parser on slices)."""
  row = np.asarray(row, np.float32)
  outs = {k: row[np.newaxis, a:b].copy() for k, (a, b) in SLICES.items()}
  return Parser().parse_outputs(outs)

# modeld.py get_action_from_model, 'action' not in model_output branch; prev = (curv, accel)
def get_action(outs, prev, lat_t, long_t, v_ego):
  v_ego = max(v_ego, 0.)             # modeld.py: v_ego = max(vEgo, 0.)
  plan = outs['plan'][0]
  acc = get_accel_from_plan(plan[:, Plan.VELOCITY][:, 0], plan[:, Plan.ACCELERATION][:, 0], ModelConstants.T_IDXS, action_t=long_t)
  curv = get_curvature_from_plan(plan[:, Plan.T_FROM_CURRENT_EULER][:, 2], plan[:, Plan.ORIENTATION_RATE][:, 2],
                                 ModelConstants.T_IDXS, v_ego, lat_t)
  stop = should_stop(v_ego, acc)
  acc = smooth_value(acc, prev[1], LONG_SMOOTH_SECONDS)
  curv = smooth_value(curv, prev[0], LAT_SMOOTH_SECONDS) if v_ego > MIN_LAT_CONTROL_SPEED else prev[0]
  return float(curv), float(acc), bool(stop)

def result_fields(outs, curv, acc, stop):
  """Flatten parsed outputs into the op_result_t field layout (as modeld/fill_model_msg index them)."""
  plan = outs['plan'][0]
  return dict(desired_curvature=curv, desired_accel=acc, should_stop=int(stop),
              plan_pos=plan[:, Plan.POSITION], plan_vel=plan[:, Plan.VELOCITY],
              lane_lines=outs['lane_lines'][0],                 # (4,33,2) y,z
              lane_line_probs=outs['lane_lines_prob'][0, 1::2],  # fill_model_msg.py: [0,1::2]
              road_edges=outs['road_edges'][0],
              lead_prob=float(outs['lead_prob'][0, 0]),
              lead_xyv=outs['lead'][0, 0, 0, :3])               # hypothesis 0, t=0: x,y,v

# ---- openpilot/common/transformations/{orientation,camera,model}.py ----
def rot_from_euler(e):   # transformations.py euler2rot_single (Rz@Ry@Rx)
  phi, theta, psi = e
  cx, sx, cy, sy, cz, sz = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
  Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
  Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
  return Rz @ Ry @ Rx
device_frame_from_view_frame = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
view_frame_from_device_frame = device_frame_from_view_frame.T
MEDMODEL_CY = 47.6
medmodel_intrinsics = np.array([[910.0, 0, 256.0], [0, 910.0, MEDMODEL_CY], [0, 0, 1.0]])
sbigmodel_intrinsics = np.array([[455.0, 0, 256.0], [0, 455.0, 0.5 * (256 + MEDMODEL_CY)], [0, 0, 1.0]])
def _view_from_calib0(): return np.hstack((view_frame_from_device_frame.dot(rot_from_euler([0, 0, 0])), [[0], [0], [0]]))
calib_from_medmodel = np.linalg.inv(np.dot(medmodel_intrinsics, _view_from_calib0())[:, :3])
calib_from_sbigmodel = np.linalg.inv(np.dot(sbigmodel_intrinsics, _view_from_calib0())[:, :3])
def get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame=False):
  calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
  device_from_calib = rot_from_euler(device_from_calib_euler)
  camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
  return camera_from_calib @ calib_from_model

# ---- tinygrad examples/openpilot/compile_warp.py (real tinygrad, CPU) ----
_tg = None
def tinygrad_warp(frame_flat, M, cam_w, cam_h, stride, y_height, uv_height, model_w=512, model_h=256):
  """Runs upstream make_frame_prepare on CPU. frame_flat: uint8 buffer of >= stride*(y_height+uv_height) bytes."""
  global _tg
  import os; os.environ.setdefault("DEV", "CPU")
  from tinygrad import Tensor, Device, Context
  def warp_perspective_tinygrad(src_flat, M_inv, dst_shape, src_shape, stride_pad, border_fill_val=None):  # verbatim
    w_dst, h_dst = dst_shape; h_src, w_src = src_shape
    x = Tensor.arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst).reshape(-1)
    y = Tensor.arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst).reshape(-1)
    src_x = M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2]
    src_y = M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2]
    src_w = M_inv[2, 0] * x + M_inv[2, 1] * y + M_inv[2, 2]
    src_x = src_x / src_w; src_y = src_y / src_w
    x_nn_clipped = Tensor.round(src_x).clip(0, w_src - 1).cast('int')
    y_nn_clipped = Tensor.round(src_y).clip(0, h_src - 1).cast('int')
    return src_flat[y_nn_clipped * (w_src + stride_pad) + x_nn_clipped]
  def frames_to_tensor(frames):  # verbatim
    H = (frames.shape[0] * 2) // 3; W = frames.shape[1]
    return Tensor.cat(frames[0:H:2, 0::2], frames[1:H:2, 0::2], frames[0:H:2, 1::2], frames[1:H:2, 1::2],
                      frames[H:H+H//4].reshape((H//2, W//2)), frames[H+H//4:H+H//2].reshape((H//2, W//2)), dim=0).reshape((6, H//2, W//2))
  uv_offset = stride * y_height; stride_pad = stride - cam_w
  input_frame = Tensor(np.ascontiguousarray(frame_flat))
  M_inv = Tensor(np.asarray(M, np.float32))
  M_inv_uv = M_inv * Tensor([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]])
  uv = input_frame[uv_offset:uv_offset + uv_height * stride].reshape(uv_height, stride)
  with Context(SPLIT_REDUCEOP=0):
    y = warp_perspective_tinygrad(input_frame[:cam_h*stride], M_inv, (model_w, model_h), (cam_h, cam_w), stride_pad).realize()
    u = warp_perspective_tinygrad(uv[:cam_h//2, :cam_w:2].flatten(), M_inv_uv, (model_w//2, model_h//2), (cam_h//2, cam_w//2), 0).realize()
    v = warp_perspective_tinygrad(uv[:cam_h//2, 1:cam_w:2].flatten(), M_inv_uv, (model_w//2, model_h//2), (cam_h//2, cam_w//2), 0).realize()
  yuv = y.cat(u).cat(v).reshape((model_h * 3 // 2, model_w))
  return frames_to_tensor(yuv).numpy()
