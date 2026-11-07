"""Make a standalone V2 HTML world from the included genuine example_model.pt.

Example (Windows):
    python3.13 build_example_world.py --size 96 --t0 79.3 --t1 80.9

Uses the trained original network, not fabricated shader geometry.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from zeta_dreamer_3d import ZetaDreamerNet
from zeta_volume_export import export_world

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",default="example_model.pt")
    p.add_argument("--output",default="zeta_v2_example.html")
    p.add_argument("--size",type=int,default=96)
    p.add_argument("--t0",type=float,default=79.3)
    p.add_argument("--t1",type=float,default=80.9)
    args=p.parse_args()
    if args.size not in (32,48,64,80,96,128):
        p.error("--size should be 32, 48, 64, 80, 96, or 128")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=True)
    if checkpoint.get("format")!="zeta_dreamer3d_v1":
        raise ValueError("Not a Zeta Dreamer 3D model checkpoint")
    model=ZetaDreamerNet(
        n_arms=int(checkpoint["n_arms"]),hidden=128,
        f0=float(checkpoint["f0"]),sigma=float(checkpoint["sigma"]),
        t_scale=float(checkpoint["t_scale"])).to(device)
    model.load_state_dict(checkpoint["state"],strict=True)
    model.eval()
    print(f"Exporting sampled complex field on {device}: {args.size}³ points")
    report=export_world(model,Path(args.output),args.t0,args.t1,args.size,
                        anchors=checkpoint.get("anchors",[]),
                        progress=lambda i,n:print(f"\rSampling {i}/{n}",end="",flush=True))
    print()
    print("Open offline HTML:",report["path"])
    print("Complex filament segments:",report["filament_segment_count"])

if __name__=="__main__":
    main()
