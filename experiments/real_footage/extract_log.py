"""Parse rlog.zst of the model-replay segment with pycapnp (schema files log.capnp, car.capnp, deprecated.capnp,
custom.capnp in <scratch>/cereal, plus <scratch>/cereal/inc/include/c++.capnp) and pickle the fields used.
Usage: python extract_log.py <scratch_dir> [cereal_dir]   (expects rlog.zst or rlog.bz2 in <scratch_dir>, writes log.pkl;
cereal_dir defaults to <scratch_dir>/cereal; older logs (before extrinsicsCalibration) fall back to liveCalibration.rpyCalib)
"""
import capnp, zstandard, bz2, os, sys, numpy as np, pickle
S=sys.argv[1]; CER=sys.argv[2] if len(sys.argv)>2 else S+'/cereal'
capnp.remove_import_hook()
log = capnp.load(CER+'/log.capnp', imports=[CER, CER+'/inc'])
if os.path.exists(S+'/rlog.zst'): raw = zstandard.ZstdDecompressor().stream_reader(open(S+'/rlog.zst','rb')).read()
else: raw = bz2.BZ2File(S+'/rlog.bz2').read()
d = dict(live=[], cs=[], ext=[], nar=[], wide=[], model=[], dm=[], dev=None, sensor=None, wsensor=None)
for m in log.Event.read_multiple_bytes(raw):
    w = m.which(); t = m.logMonoTime
    if w=='carState': d['cs'].append((t, m.carState.vEgo))
    elif w=='extrinsicsCalibration':
        e=m.extrinsicsCalibration; d['ext'].append((t, list(e.rpyCalib), list(e.wideFromDeviceEuler), str(e.calStatus), e.calPerc, list(e.height)))
    elif w=='liveCalibration':
        e=m.liveCalibration; d['live'].append((t, list(e.rpyCalib), [], str(e.calStatus), e.calPerc, list(e.height)))
    elif w=='narrowRoadCameraState':
        c=m.narrowRoadCameraState; d['nar'].append((t,c.frameId,c.timestampSof,c.timestampEof)); d['sensor']=str(c.sensor)
    elif w=='wideRoadCameraState':
        c=m.wideRoadCameraState; d['wide'].append((t,c.frameId,c.timestampSof,c.timestampEof)); d['wsensor']=str(c.sensor)
    elif w=='deviceState' and d['dev'] is None: d['dev']=str(m.deviceState.deviceType)
    elif w=='initData': d['init']=str(m.initData.deviceType)
    elif w=='modelV2':
        mv=m.modelV2
        d['model'].append(dict(t=t, fid=mv.frameId, fidx=mv.frameIdExtra, vx=list(mv.velocity.x), px=list(mv.position.x), py=list(mv.position.y),
            curv=mv.action.desiredCurvature, acc=mv.action.desiredAcceleration, lp=list(mv.laneLineProbs), lead=[l.prob for l in mv.leadsV3], leadx=[l.x[0] for l in mv.leadsV3],
            yaw=list(mv.orientation.z), has_action=(mv.action.desiredCurvature!=0 or mv.action.desiredAcceleration!=0)))
if not d['ext']: d['ext']=d['live']; print('no extrinsicsCalibration in log: using liveCalibration.rpyCalib')
pickle.dump(d, open(S+'/log.pkl','wb'))
print('dev',d['dev'],d.get('init'),'sensors',d['sensor'],d['wsensor'])
print(len(d['cs']),len(d['ext']),len(d['nar']),len(d['wide']),len(d['model']))
print('ext0',d['ext'][0]); print('extN',d['ext'][-1])
n=np.array([x[1] for x in d['nar']]); w=np.array([x[1] for x in d['wide']])
print('nar fid',n[:3],n[-3:],'wide fid',w[:3],w[-3:],'equal',np.array_equal(n,w))
if len(n)==len(w): sofd=np.array([x[2] for x in d['nar']])-np.array([x[2] for x in d['wide']]); print('sof diff ms',sofd.min()/1e6,sofd.max()/1e6)
else: print('narrow and wide have different frame counts', len(n), len(w))
print('model fid',d['model'][0]['fid'],d['model'][-1]['fid'], d['model'][0]['fidx'])
