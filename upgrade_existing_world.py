"""Upgrade an already-exported Zeta Dreamer 3D HTML without retraining.

The original standalone worlds contain the three complex RGB channels packed
into RGBA8 volume textures. Decode their near-zero complex components,
reconstruct a piecewise-linear Re(Z)=Im(Z)=0 filament mesh, and insert both
the original textures and new sub-voxel curves into the current renderer.

Usage:
    python3.13 upgrade_existing_world.py zeta_world_75_85.html
    python3.13 upgrade_existing_world.py old1.html old2.html

Writes <original>_sharp.html by default and never replaces the source file.
The recovered complex values are 8-bit quantized; regenerating with the
latest Python model and a 96^3 or higher sampling grid gives better fidelity.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from zeta_filament_mesh import extract_vortex_segments

TEMPLATE = Path(__file__).with_name('zeta_volume_viewer_template.html')


def read_payload(html: str) -> dict:
    marker = 'const EMBEDDED = '
    offset = html.find(marker)
    if offset == -1:
        raise ValueError('No Zeta Dreamer 3D volume found in the HTML')
    payload, _ = json.JSONDecoder().raw_decode(html[offset + len(marker):].lstrip())
    if not all(k in payload for k in ('rgb', 'c0', 'c1', 'size', 't0', 't1')):
        raise ValueError('The original HTML volume payload is missing required fields')
    return payload


def unpack_complex(payload: dict) -> np.ndarray:
    n = int(payload['size'])
    if n < 2 or n > 256:
        raise ValueError('Unsupported 3D grid size')
    nbytes = n*n*n*4
    a = np.frombuffer(base64.b64decode(payload['c0']), dtype=np.uint8)
    b = np.frombuffer(base64.b64decode(payload['c1']), dtype=np.uint8)
    if len(a) != nbytes or len(b) != nbytes:
        raise ValueError('Corrupted complex volume texture')
    c0 = a.reshape(n,n,n,4).astype(np.float32)/127.5-1
    c1 = b.reshape(n,n,n,4).astype(np.float32)/127.5-1
    raw = np.zeros((n,n,n,3,2),dtype=np.float32)
    raw[...,0,:]=c0[...,:2]
    raw[...,1,:]=c0[...,2:4]
    raw[...,2,:]=c1[...,:2]
    return raw


def upgrade(source: str | Path, target: str | Path | None = None) -> dict:
    path = Path(source).expanduser().resolve()
    payload = read_payload(path.read_text('utf-8'))
    if 'segments' not in payload:
        segments, statistics = extract_vortex_segments(unpack_complex(payload))
        payload['segments'] = segments
    else:
        statistics = {
            'filament_segment_count': len(base64.b64decode(payload['segments'])) // 16,
            'filament_method': 'reused sub-voxel mesh from original payload',
        }
    if 'harm0' not in payload or 'harm1' not in payload:
        from zeta_harmonic import reference_payload
        payload.update(reference_payload(unpack_complex(payload), 1.0, quantized=True))
    template = TEMPLATE.read_text('utf-8')
    anchors=payload.get('anchors',[])
    values={
        '__WORLD_TITLE__': f"sharpened t={payload['t0']:g}..{payload['t1']:g}",
        '__GRID_SIZE__': str(payload['size']),
        '__T_START__': f"{payload['t0']:g}",
        '__T_END__': f"{payload['t1']:g}",
        '__ANCHOR_LABEL__': ', '.join(f'{float(t):.2f}' for t in anchors) or 'not recorded',
        '__EMBEDDED_DATA__': json.dumps(payload,separators=(',',':'))
    }
    for key,val in values.items():
        template=template.replace(key,val)
    output=Path(target).expanduser().resolve() if target is not None else path.with_name(path.stem+'_sharp.html')
    if output == path:
        raise ValueError('Refusing to overwrite the input HTML')
    output.write_text(template,'utf-8')
    return {'source':str(path),'output':str(output),'bytes':output.stat().st_size,**statistics}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',nargs='+',help='Previously exported Zeta Dreamer 3D HTML files')
    args=parser.parse_args()
    for filename in args.html:
        details=upgrade(filename)
        print(f"Sharpened {details['source']} -> {details['output']}")
        print(f"  complex-zero segments: {details['filament_segment_count']:,}")


if __name__=='__main__':
    main()
