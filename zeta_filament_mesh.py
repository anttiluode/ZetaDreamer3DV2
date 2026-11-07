"""Sub-voxel complex-zero intersections for Zeta Dreamer.

Each RGB complex channel is piecewise-linearly interpolated over six tetrahedra
per voxel cube. Re(Z)=Im(Z)=0 intersects a tetrahedron in a line segment, whose
two endpoints are solved exactly on triangular faces using barycentric
coordinates. This does NOT dilate grid cells and it does not pretend that
the resulting piecewise-linear curves are infinite-resolution neural outputs.

The returned compact Uint16 segment buffer is displayed as thin screen-space
ribbons by the browser (2 pixels by default). Format per segment:
[x0,y0,z0,x1,y1,z1,channel,65535], each unsigned 16-bit integer.
"""
from __future__ import annotations

import base64
import numpy as np

_CORNERS = np.array(
    [(0,0,0), (0,0,1), (0,1,1), (0,1,0),
     (1,0,0), (1,0,1), (1,1,1), (1,1,0)], dtype=np.int32)
_TETS = ((0,1,2,6), (0,2,3,6), (0,3,7,6),
         (0,7,4,6), (0,4,5,6), (0,5,1,6))
_FACES = ((0,1,2), (0,1,3), (0,2,3), (1,2,3))


def _face_roots(values: np.ndarray, positions: np.ndarray):
    """Locate complex zero on triangles, vectorized over candidate cells.

    values shape [N,3,2], positions [N,3,3] in (z,y,x) grid units.
    """
    z0,z1,z2 = values[:,0],values[:,1],values[:,2]
    d1=z1-z0;d2=z2-z0
    det=d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0]
    ok=np.abs(det)>1e-10
    denom=np.where(ok,det,1.0)
    u=(-z0[:,0]*d2[:,1]+d2[:,0]*z0[:,1])/denom
    v=(-d1[:,0]*z0[:,1]+z0[:,0]*d1[:,1])/denom
    tol=1e-5
    ok &= (u>=-tol)&(v>=-tol)&(u+v<=1+tol)
    point=positions[:,0]+u[:,None]*(positions[:,1]-positions[:,0])+v[:,None]*(positions[:,2]-positions[:,0])
    return ok,point


def extract_vortex_segments(raw: np.ndarray, *, max_segments: int = 400000,
                            chunk_cubes: int = 12000) -> tuple[str,dict]:
    """Find sub-grid segments for all 3 complex output channels.

    raw is [Z,Y,X,3,2] float array; returns base64 packed uint16 and statistics.
    Batch operations keep memory bounded even when exporting a 128^3 volume.
    The output is limited deterministically if it exceeds max_segments.
    """
    raw=np.asarray(raw,dtype=np.float32)
    if raw.ndim!=5 or raw.shape[-2:]!=(3,2) or min(raw.shape[:3])<2:
        raise ValueError("Expected complex field [z,y,x,3,2]")
    nz,ny,nx=raw.shape[:3]
    count=(nz-1)*(ny-1)*(nx-1)
    results=[]
    counts=[0,0,0]
    for offset in range(0,count,chunk_cubes):
        ids=np.arange(offset,min(count,offset+chunk_cubes))
        z=ids//((ny-1)*(nx-1))
        y=(ids//(nx-1))%(ny-1)
        x=ids%(nx-1)
        start=np.stack((z,y,x),axis=-1)
        cp=start[:,None,:]+_CORNERS[None,:,:]
        values=raw[cp[:,:,0],cp[:,:,1],cp[:,:,2]]
        for channel in range(3):
            f=values[:,:,channel,:]
            for tet in _TETS:
                zz=f[:,tet,:]
                # A continuous real/imag zero is impossible in this tetrahedron
                # if either component has strictly the same sign at all vertices.
                viable=(zz[:,:,0].min(axis=1)<=0)&(zz[:,:,0].max(axis=1)>=0)
                viable&=(zz[:,:,1].min(axis=1)<=0)&(zz[:,:,1].max(axis=1)>=0)
                if not np.any(viable):
                    continue
                v=zz[viable];p=cp[viable][:,tet,:].astype(np.float32)
                face_ok=[];face_pts=[]
                for face in _FACES:
                    mask,pts=_face_roots(v[:,face,:],p[:,face,:])
                    face_ok.append(mask);face_pts.append(pts)
                good=np.stack(face_ok,axis=1)
                # Tangencies at tetrahedral vertices/edges can have 3+ face
                # hits; retain only unambiguous two-point intersections.
                exact=good.sum(axis=1)==2
                if not np.any(exact):
                    continue
                pairs=np.stack(face_pts,axis=1)[exact]
                which=np.argsort(~good[exact],axis=1)[:,:2]
                endpoints=np.take_along_axis(pairs,which[:,:,None],axis=1)
                length=np.linalg.norm(endpoints[:,1]-endpoints[:,0],axis=1)
                endpoints=endpoints[length>1e-5]
                if endpoints.size==0:
                    continue
                # Convert grid index (z,y,x) to WebGL normalized (x,y,z).
                normalized=(endpoints[...,::-1]/np.array([nx-1,ny-1,nz-1],dtype=np.float32))*2-1
                # Exclude segments completely outside the viewer's spherical volume.
                mid=normalized.mean(axis=1)
                normalized=normalized[(mid*mid).sum(axis=1)<=1.06]
                if not len(normalized):
                    continue
                u=np.clip(np.rint((normalized+1)*32767.5),0,65535).astype(np.uint16)
                line=np.empty((len(u),8),dtype=np.uint16)
                line[:,:6]=u.reshape(-1,6)
                line[:,6]=channel*32767
                line[:,7]=65535
                results.append(line)
                counts[channel]+=len(line)
    if results:
        output=np.concatenate(results,axis=0)
    else:
        output=np.zeros((0,8),dtype=np.uint16)
    total=len(output)
    if total>max_segments:
        chosen=np.linspace(0,total-1,max_segments,dtype=np.int64)
        output=output[chosen]
    return base64.b64encode(output.astype('<u2',copy=False).tobytes()).decode('ascii'),{
        'filament_segment_count':int(len(output)),
        'filament_uncapped_count':int(total),
        'filament_by_channel':counts,
        'filament_method':'piecewise-linear Re(Z)=Im(Z)=0 in tetrahedra; sub-voxel segments',
    }
