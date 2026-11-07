# ZETA DREAMER 3D V2 — THE PHASE OBSERVATORY

A **real trained complex neural field**, a walkable phase world, and an explicit mathematical control for its invented interior.

V2 retains the complete [ZetaDreamer3D](https://github.com/anttiluode/ZetaDreamer3D) original **32 prime-frequency arms and three complex hidden layers of width 128**. It evaluates the actual trained complex network \`Z(x,y,t) ∈ ℂ³\` rather than replacing it with procedural fog. The thin vortex lines are intersections of **Re(Z)=Im(Z)=0** in a piecewise-linear tetrahedral interpolant of the sampled field.

## Immediate use: sharpen an old exported HTML

Already have an exported world, e.g. \`E:\DocsHouse\1051\zeta_world_75_85.html\`? From this repository:

\`\`\`powershell
python3.13 -m pip install numpy
python3.13 upgrade_existing_world.py "E:\DocsHouse\1051\zeta_world_75_85.html"
\`\`\`

This writes \`zeta_world_75_85_sharp.html\`, leaving the original untouched. V2 extracts true sub-grid filament segments if they were missing and computes a harmonic control from the *existing quantized complex texture*, no retraining required.

For a higher-fidelity world, install \`torch numpy Pillow opencv-python\`, run \`python3.13 zeta_dreamer_3d.py\`, load your saved checkpoint or train new pictures, then choose **ENTER 3D WORLD** and 96³ or 128³ sampling. **No sampler can recover fine detail discarded from a previous 64³ export.**

## Nine distinct viewing modes

1. **Hybrid geometry** — RGB edges and authentic complex-zero filaments.
2. **RGB gradient sheets** — magnitude gradients, not phase topology.
3. **RGB image clouds** — the network's visible image field.
4. **Complex nodal sheets** — Re(Z)=0 and Im(Z)=0.
5. **Vortex filaments** — sub-voxel intersections of the two zero surfaces.
6. **Interference residue** — \`Rθ=2 Re(exp(iθ) conj(A) B)\` from selected complex channels, optional t displacement, phase rotation and gain.
7. **Phase current** — \`J=Im(conj(Z)∇Z)\` from finite differences, with x/y/t components encoded in RGB.
8. **Harmonic control** — a *known* Laplace-equation field from the first and last sampled complex slices.
9. **Learned − harmonic** — complex discrepancy from that reference. Inspect the per-slice relative RMS curve to see where the learned extension differs.

## The mathematical reference

For each Re/Im component the reference solves

\`\`\`math
(∂x² + ∂y² + ∂z²)H = 0,
H(x,y,0) = Z(x,y,t₀),   H(x,y,L) = Z(x,y,t₁).
\`\`\`

The lateral x/y walls have Neumann boundary conditions; the solver applies an even FFT extension. For lateral wave number \`k>0\`:

\`\`\`math
Ĥ(k,z) = sinh(k(L-z))/sinh(kL) Ẑ₀(k) + sinh(kz)/sinh(kL) Ẑ₁(k).
\`\`\`

For \`k=0\`, the interpolation is linear. The implementation uses stable exponential ratios.

**Crucial limitations:** this is a *comparison*, not the true image reconstruction, not a learned weight-space predictor, and not an AdS/CFT dual. Both selected bounding planes can be outside the original training anchors. The network's image magnitude does not uniquely determine complex phase, and neither do two sampled complex planes determine the network's interior. The control is useful precisely because its assumptions are explicit.

## Controls and cost

WASD fly, R/F rise/fall, mouse-look on the 3D canvas, G center, O outside, Q/E yaw, Shift fast, 1–9 modes, H hide interface, P screenshot. **The left-side scrollbar works independently**; the wheel changes flight speed only over the canvas. Lower resolution and ray count for performance. A V2 export adds two RGBA8 3D textures for the reference; these cost memory but retain measurable complex information.

The exported standalone HTML works offline in a WebGL2-enabled Edge/Chrome; Python is needed only to create/export it. A texture-sampled neural field is *not* executing the full MLP per ray step. Sub-grid filaments and phase current are properties of the piecewise-sampled representation.

Run numerical tests:

\`\`\`powershell
python3.13 -m unittest discover -s tests -p "test_harmonic_v2.py" -v
\`\`\`

See [original project](https://github.com/anttiluode/ZetaDreamer3D) for its synthetic sample checkpoints and full GUI details. MIT license retained.
