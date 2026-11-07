"""Laplace-equation control for ZetaDreamer's sampled complex field.

Complex Dirichlet data are the first and last sampled (x,y) planes.
Reflective (Neumann) conditions hold at the x/y side walls.
This is a comparison interior, not a reconstruction of unknown ground truth.
"""
from __future__ import annotations
import base64
import numpy as np

def _reflect2(a: np.ndarray) -> np.ndarray:
    if min(a.shape[:2]) < 3:
        raise ValueError("Need at least 3 x/y samples")
    a = np.concatenate((a, a[-2:0:-1, ...]), axis=0)
    return np.concatenate((a, a[:, -2:0:-1, ...]), axis=1)

def harmonic_slab(raw: np.ndarray, xy_width: float = 2., depth: float = 2.) -> np.ndarray:
    """Solve (∂x²+∂y²+∂z²)H=0 for each Re/Im component in [z,y,x,3,2].

    Even FFT extension implements Neumann x/y walls. Nonzero Fourier
    modes interpolate with sinh ratios; DC interpolates linearly.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 5 or raw.shape[-2:] != (3, 2) or min(raw.shape[:3]) < 3:
        raise ValueError("Expected [z,y,x,3,2], dimensions >=3")
    if not np.isfinite(raw).all() or xy_width <= 0 or depth <= 0:
        raise ValueError("Finite field and positive lengths required")
    nz,ny,nx=raw.shape[:3]
    h,w=2*(ny-1),2*(nx-1)
    a=np.fft.rfft2(_reflect2(raw[0]),axes=(0,1))
    b=np.fft.rfft2(_reflect2(raw[-1]),axes=(0,1))
    ky=np.fft.fftfreq(h,d=xy_width/(ny-1))
    kx=np.fft.rfftfreq(w,d=xy_width/(nx-1))
    k=2*np.pi*np.hypot(ky[:,None],kx[None,:])
    denominator=-np.expm1(-2*k*depth)
    denominator[0,0]=1.
    out=np.empty_like(raw)
    for j,z in enumerate(np.linspace(0.,depth,nz)):
        wa=np.exp(-k*z)*(-np.expm1(-2*k*(depth-z)))/denominator
        wb=np.exp(-k*(depth-z))*(-np.expm1(-2*k*z))/denominator
        wa[0,0],wb[0,0]=1.-z/depth,z/depth
        freq=wa[...,None,None]*a+wb[...,None,None]*b
        out[j]=np.fft.irfft2(freq,s=(h,w),axes=(0,1))[:ny,:nx].astype(np.float32)
    out[0],out[-1]=raw[0],raw[-1]
    return out

def reference_payload(raw: np.ndarray, scale: float, *, quantized: bool=False) -> dict:
    """Encode harmonic Re/Im using the original field's signed RGBA8 scale."""
    if not np.isfinite(scale) or scale<=0:
        raise ValueError("Positive finite complex packing scale required")
    ref=harmonic_slab(raw)
    base=np.sqrt(np.mean(np.square(raw,dtype=np.float64),axis=(1,2,3,4)))
    err=np.sqrt(np.mean(np.square(raw-ref,dtype=np.float64),axis=(1,2,3,4)))
    curve=np.where(base>1e-10,err/np.maximum(base,1e-10),0.)
    data=np.rint(np.clip(ref/scale*.5+.5,0.,1.)*255).astype(np.uint8)
    harm0=data[...,:2,:].reshape(*raw.shape[:3],4)
    harm1=np.zeros((*raw.shape[:3],4),dtype=np.uint8)
    harm1[...,:2]=data[...,2,:]
    return {
        "harm0":base64.b64encode(harm0.tobytes(order="C")).decode("ascii"),
        "harm1":base64.b64encode(harm1.tobytes(order="C")).decode("ascii"),
        "harmonic_rms_curve":[round(float(v),5) for v in curve],
        "harmonic_ref":"Laplace Dirichlet slab, first/last sampled complex planes, Neumann x/y edges",
        "harmonic_quantized_input":bool(quantized),
    }
