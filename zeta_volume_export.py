"""Sample the REAL complex ZetaDreamerNet into an offline WebGL2 navigable sphere.

Uses the user's trained model, not synthesized arbitrary wave interference.
Coordinates: x,y in [-1,+1], t in [t_min,t_max] mapped to normalized z in [-1,+1].
The resulting HTML stores three 64^3 RGBA8 volumes with no server/CDN dependency.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from zeta_filament_mesh import extract_vortex_segments

TEMPLATE = Path(__file__).with_name('zeta_volume_viewer_template.html')
RIEMANN_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178,
    40.918720, 43.327073, 48.005151, 49.773832, 52.970321,
    56.446248, 59.347044, 60.831779, 65.112544, 67.079811,
    69.546402, 72.067158, 75.704691, 77.144840, 79.337375,
    82.910381, 84.735493, 87.425275, 88.809112
]


def _phase_winding_mask(field: np.ndarray) -> tuple[np.ndarray, dict]:
    """Detect plaquettes with nonzero integer phase winding, across 3 orientations.

    A candidate 3D phase-vortex filament intersects a plaquette if the total
    unwrapped phase increment around its four edges equals ±2π. Discrete
    sampling can miss or alias vortices when changes exceed π per cell.
    """
    if field.ndim != 5 or field.shape[-2:] != (3, 2):
        raise ValueError('Expected complex field [z, y, x, RGB, real/imag]')
    phase = np.arctan2(field[..., 1], field[..., 0])
    # One global mask, populated by all RGB complex channels and orientations.
    result = np.zeros(field.shape[:3], dtype=np.float32)
    stats = {'xy': 0, 'xz': 0, 'yz': 0}

    def wrap(d):
        return (d + np.pi) % (2 * np.pi) - np.pi

    for a, b, name in ((1, 2, 'xy'), (0, 2, 'xz'), (0, 1, 'yz')):
        base = [slice(None)] * 4
        base[a] = slice(0, -1)
        base[b] = slice(0, -1)
        plus_a = base.copy(); plus_a[a] = slice(1, None)
        plus_b = base.copy(); plus_b[b] = slice(1, None)
        plus_ab = plus_a.copy(); plus_ab[b] = slice(1, None)
        p00, p10, p11, p01 = [phase[tuple(s)] for s in (base, plus_a, plus_ab, plus_b)]
        total = wrap(p10 - p00) + wrap(p11 - p10) + wrap(p01 - p11) + wrap(p00 - p01)
        winding = np.abs(np.rint(total / (2 * np.pi)))
        has_vortex = np.any(winding >= 1.0, axis=-1).astype(np.float32)
        stats[name] = int(np.count_nonzero(has_vortex))
        for corner in (base, plus_a, plus_ab, plus_b):
            result[tuple(corner[:3])] = np.maximum(result[tuple(corner[:3])], has_vortex)

    # No morphological expansion: a phase-winding mask is useful as a
    # diagnostic but should not be mistaken for a resolved filament surface.
    return result, stats


def _to_rgba_u8(channels: np.ndarray) -> np.ndarray:
    return np.clip(np.round(channels * 255), 0, 255).astype(np.uint8)


def evaluate_volume(model, t_min: float, t_max: float, size: int = 64,
                    batch: int = 8192,
                    progress: Callable[[int, int], None] | None = None) -> tuple[np.ndarray, dict]:
    """Return raw Z(z,y,x,color,realimag), and sampling metadata."""
    if not (16 <= size <= 128):
        raise ValueError('Grid size must be in range 16..128')
    if not (np.isfinite(t_min) and np.isfinite(t_max) and t_max > t_min):
        raise ValueError('t_max must be greater than t_min')
    device = next(model.parameters()).device
    grid = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(grid, grid, indexing='ij')
    coords = torch.from_numpy(np.stack((xx, yy), axis=-1).reshape(-1, 2)).to(device)
    zs = np.linspace(t_min, t_max, size, dtype=np.float64)
    raw = np.empty((size, size, size, 3, 2), np.float32)

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for iz, t in enumerate(zs):
            parts = []
            for i in range(0, coords.shape[0], batch):
                out = model.forward_complex(coords[i:i + batch], float(t))
                parts.append(out.detach().cpu().numpy().astype(np.float32))
            raw[iz] = np.concatenate(parts, axis=0).reshape(size, size, 3, 2)
            if progress:
                progress(iz + 1, size)
    if was_training:
        model.train()
    return raw, {'t_min': t_min, 't_max': t_max, 'size': size}


def pack_volume(raw: np.ndarray) -> tuple[dict, dict]:
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite model output encountered')
    mag = np.sqrt(np.sum(raw**2, axis=-1) + 1e-8).clip(0, 1)
    # Gradient per voxel in each dimension. Normalized to a robust percentile.
    d0, d1, d2 = np.gradient(mag, axis=(0, 1, 2))
    grad = np.sqrt(np.sum(d0*d0 + d1*d1 + d2*d2, axis=-1))
    grad_scale = float(max(np.quantile(grad, .98), 1e-6))
    edge = np.clip(grad / grad_scale, 0, 1)
    lum = np.clip(mag @ np.array([.23, .66, .11], dtype=np.float32), 0, 1)
    vortex, stats = _phase_winding_mask(raw)

    # The global 99.9th percentile balances color-channel saturation and
    # zero-crossing precision across the volume. Zero is encoded at 127.5.
    scale = float(max(np.quantile(np.abs(raw), .999), .02))
    signed = np.clip(raw / scale * .5 + .5, 0, 1)
    tex_rgb = _to_rgba_u8(np.concatenate([mag, edge[..., None]], axis=-1))
    tex_c0 = _to_rgba_u8(signed[..., :2, :].reshape(*raw.shape[:3], 4))
    tex_c1 = _to_rgba_u8(np.concatenate((signed[..., 2, :], vortex[..., None], lum[..., None]), axis=-1))

    meta = {'edge_norm_percentile98': grad_scale, 'complex_scale_percentile999': scale,
            'vortex_plaquettes': stats, 'vortex_voxel_count': int(np.count_nonzero(vortex > .5)),
            'mean_rgb': [round(float(x), 5) for x in mag.mean((0, 1, 2))]}
    return {
        'rgb': base64.b64encode(tex_rgb.tobytes(order='C')).decode('ascii'),
        'c0': base64.b64encode(tex_c0.tobytes(order='C')).decode('ascii'),
        'c1': base64.b64encode(tex_c1.tobytes(order='C')).decode('ascii'),
    }, meta


def export_world(model, path: str | Path, t_min: float = 75.0, t_max: float = 85.0,
                 size: int = 64, anchors: list[float] | None = None,
                 progress: Callable[[int, int], None] | None = None) -> dict:
    raw, sampling = evaluate_volume(model, t_min, t_max, size=size, progress=progress)
    packed, meta = pack_volume(raw)
    from zeta_harmonic import reference_payload
    harmonic = reference_payload(raw, meta['complex_scale_percentile999'])
    # Unlike the legacy 3x3 dilated plaquette mask, these are actual
    # sub-grid line segments obtained by solving Re(Z)=Im(Z)=0.
    segments, seg_meta = extract_vortex_segments(raw)
    data = {**packed, **harmonic, 'segments': segments, 'size': size, 't0': t_min, 't1': t_max,
            'anchors': [float(x) for x in (anchors or [])],
            'zeros': RIEMANN_ZEROS}
    template = TEMPLATE.read_text('utf-8')
    replacements = {
        '__WORLD_TITLE__': f't={t_min:.2f}..{t_max:.2f}',
        '__GRID_SIZE__': str(size),
        '__T_START__': f'{t_min:.2f}',
        '__T_END__': f'{t_max:.2f}',
        '__ANCHOR_LABEL__': ', '.join(f'{v:.2f}' for v in (anchors or [])) or 'no anchors recorded',
        '__EMBEDDED_DATA__': json.dumps(data, separators=(',', ':'))
    }
    for k, v in replacements.items():
        template = template.replace(k, v)
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template, encoding='utf-8')
    return {'path': str(path), 'bytes': path.stat().st_size, **sampling, **meta, **seg_meta}
