"""Extract the calibration-related fields of a clip's rlog (cameraOdometry, extrinsicsCalibration / liveCalibration, carState, camera
frame ids, device type) with pycapnp into <clip_dir>/calibration/cal_log.pkl.
Usage: python -I extract_cal_log.py <clip_dir> <cereal_dir>      (clip_dir holds rlog.zst or rlog.bz2; schemas as in experiments/real_footage/extract_log.py)
"""
import capnp, zstandard, bz2, os, sys, pickle, numpy as np
S, CER = sys.argv[1], sys.argv[2]
os.makedirs(S + "/calibration", exist_ok=True)
capnp.remove_import_hook()
log = capnp.load(CER + "/log.capnp", imports=[CER, CER + "/inc"])
if os.path.exists(S + "/rlog.zst"): raw = zstandard.ZstdDecompressor().stream_reader(open(S + "/rlog.zst", "rb")).read()
else: raw = bz2.BZ2File(S + "/rlog.bz2").read()
d = dict(odo=[], ext=[], live=[], cs=[], nar=[], dev=None, init=None, sensor=None)
def cal(e): return dict(rpy=list(e.rpyCalib), wide=list(e.wideFromDeviceEuler) if hasattr(e, "wideFromDeviceEuler") else [], status=str(e.calStatus), perc=int(e.calPerc),
                        valid_blocks=int(e.validBlocks), spread=list(e.rpyCalibSpread), height=list(e.height))
for m in log.Event.read_multiple_bytes(raw):
    w = m.which(); t = m.logMonoTime
    if w == "cameraOdometry":
        c = m.cameraOdometry
        d["odo"].append(dict(t=t, fid=int(c.frameId), valid=bool(m.valid), trans=list(c.trans), rot=list(c.rot), trans_std=list(c.transStd), rot_std=list(c.rotStd),
                             wide=list(c.wideFromDeviceEuler), wide_std=list(c.wideFromDeviceEulerStd), rt=list(c.roadTransformTrans), rt_std=list(c.roadTransformTransStd)))
    elif w == "extrinsicsCalibration": d["ext"].append(dict(t=t, valid=bool(m.valid), **cal(m.extrinsicsCalibration)))
    elif w == "liveCalibration": d["live"].append(dict(t=t, valid=bool(m.valid), **cal(m.liveCalibration)))
    elif w == "carState": d["cs"].append((t, m.carState.vEgo))
    elif w == "narrowRoadCameraState": c = m.narrowRoadCameraState; d["nar"].append((t, int(c.frameId))); d["sensor"] = str(c.sensor)
    elif w == "deviceState" and d["dev"] is None: d["dev"] = str(m.deviceState.deviceType)
    elif w == "initData": d["init"] = str(m.initData.deviceType)
pickle.dump(d, open(S + "/calibration/cal_log.pkl", "wb"))
print("odo %d ext %d live %d cs %d nar %d dev %s init %s sensor %s" % (len(d["odo"]), len(d["ext"]), len(d["live"]), len(d["cs"]), len(d["nar"]), d["dev"], d["init"], d["sensor"]))
if d["odo"]: print("odo0", d["odo"][0]); print("odo frame ids", d["odo"][0]["fid"], d["odo"][-1]["fid"], "valid frac", np.mean([o["valid"] for o in d["odo"]]))
for k in ("ext", "live"):
    if d[k]: print(k, "first", d[k][0]); print(k, "last", d[k][-1])
