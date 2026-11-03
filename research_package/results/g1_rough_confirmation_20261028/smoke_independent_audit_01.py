"""Independent CPU-only engineering audit for the four confirmation smoke children."""
from pathlib import Path
import hashlib, json
import numpy as np

O=Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()

def digest(arrays):
    h=hashlib.sha256()
    for key,value in sorted(arrays.items()):
        a=np.ascontiguousarray(value)
        if a.dtype.hasobject or not np.isfinite(a).all(): raise AssertionError(f"nonfinite {key}")
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def main():
    protocol=sha(O/'prospective_protocol.json')
    p=read(O/'prospective_protocol.json')
    assert p['status']=='LOCKED_BEFORE_R_ENVIRONMENT_SAMPLING'
    for entry in p['identity']['files']:
        assert sha(entry['path'])==entry['sha256'], entry['path']
    methods=('RE_live','RE_frozen','RC_clock','FC_clock')
    checked=[]
    for method in methods:
        folder=O/'smoke'/f'smoke_{method}_terrain20264901'
        manifest=read(folder/'manifest.json'); row=read(folder/'episodes.json')[0]
        assert manifest['status']=='COMPLETED' and manifest['transitions']==10
        assert manifest['PPO_updates']==manifest['Adam_steps']==0
        assert manifest['protocol_sha256']==protocol
        raw_path=folder/row['raw_path']; assert sha(raw_path)==row['raw_sha256']
        with np.load(raw_path,allow_pickle=False) as saved: raw=dict(saved)
        assert len(raw['time'])==10 and not raw['terminated'].any() and not raw['timed_out'].any()
        assert np.isfinite(raw['phase_live']).all() and np.isfinite(raw['phase_used']).all()
        assert raw['phase_live'].shape==raw['phase_used'].shape==(10,2)
        if method=='RE_frozen':
            np.testing.assert_allclose(raw['phase_used'],np.broadcast_to(raw['phase_used'][0],raw['phase_used'].shape),atol=1e-7)
        initial=read(folder/'steady_initial.json')
        feet=set(manifest['foot_geom_ids']); terrain=set(manifest['terrain_geom_ids'])
        assigned=set(initial['assigned'][0]); xlo,xhi,ylo,yhi=initial['bounds'][0]
        flags=[]
        for k,(g0,g1) in enumerate(raw['points_geom']):
            assert (g0 in feet and g1 in terrain) or (g1 in feet and g0 in terrain)
            assert raw['points_dist'][k] < raw['points_includemargin'][k]
            assert np.any(raw['points_efc_address'][k]>=0)
            geom=g1 if g0 in feet else g0
            x,y=raw['points_pos'][k,:2]
            flags.append(bool(geom in assigned and xlo+.25<x<xhi-.25 and ylo+.25<y<yhi-.25))
        np.testing.assert_array_equal(flags,raw['points_interior'])
        assert int(raw['foot_contact_total'].sum())==len(flags)
        assert any(flags)
        phys=manifest['steady_physics']; assert sha(folder/phys['path'])==phys['sha256']
        with np.load(folder/phys['path'],allow_pickle=False) as saved: assert digest(dict(saved))==phys['array_sha256']==initial['physical_sha256']
        checked.append(dict(method=method,raw_sha256=row['raw_sha256'],effective_eval=10,contact_points=len(flags)))
    result=dict(status='PASS',protocol_sha256=protocol,checked=checked,total_eval=40,PPO_updates=0,Adam_steps=0,
        scope='independent CPU engineering arithmetic after smoke; no efficacy selection and no formal scientific audit')
    (O/'smoke_independent_audit_01.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
