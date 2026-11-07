"""
THE ZETA DREAMER
================
PerceptionLab / Antti Luode

One network. Any number of images. All indexed by position t on the Riemann critical line.

The idea:
    - Each image is assigned a position t on the real line (which corresponds to
      imaginary part s = 1/2 + it of the Riemann zeta function).
    - The prime arm encoding at position t:
          phase_k = f0 * ln(p_k) * (spatial_projection + t_scale * t)
      means each arm rotates at its own irrational rate as t increases.
    - The network learns to decode the prime arm interference pattern at each t
      into the corresponding image.
    - Between training t values, the network generates images whose structure
      is determined entirely by how the prime arms interfere at that t.

The prediction:
    The transition between stored images follows prime-interference dynamics.
    The Riemann zeros (t ≈ 14.134, 21.022, 25.010, ...) are positions where
    the Dirichlet partial sum has specific mathematical structure.
    Whether the generated images are anomalously structured there is an
    open empirical question — this tool lets you explore it.

The honest framing:
    This is a continuous generative manifold indexed by a number-theoretic axis.
    It is not proof of anything about RH. It is a new way to organize
    and interpolate between stored representations using prime mathematics.

Controls:
    Load Image → assign t value → Train → Sweep t → watch the dream

pip install torch numpy Pillow opencv-python
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
from threading import Thread
import cv2
import os
import time
import webbrowser
from pathlib import Path
from zeta_volume_export import export_world
from upgrade_existing_world import upgrade as upgrade_exported_html

# ─────────────────────────────────────────────────────────────────
# KNOWN RIEMANN ZEROS (imaginary parts of first 25 non-trivial zeros)
# ─────────────────────────────────────────────────────────────────

RIEMANN_ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918720, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809112,
]

T_MIN, T_MAX = 0.0, 90.0

# ─────────────────────────────────────────────────────────────────
# PRIMES
# ─────────────────────────────────────────────────────────────────

def sieve(n):
    s = np.ones(n + 1, dtype=bool); s[0] = s[1] = False
    for i in range(2, int(n**0.5) + 1):
        if s[i]: s[i*i::i] = False
    return np.where(s)[0]

PRIMES = sieve(3000)

# ─────────────────────────────────────────────────────────────────
# ZETA ENCODING  — encodes (x, y, t) via prime arms
# ─────────────────────────────────────────────────────────────────

class ZetaEncoding(nn.Module):
    """
    Each prime p_k contributes one arm:
        phase = f0 * ln(p_k) * (x*cos(θ_k) + y*sin(θ_k)) + t_freq_k * t
    
    The spatial part varies with position in the image.
    The temporal part rotates at rate t_freq_k = f0 * ln(p_k) * t_scale
    as t moves along the critical line.
    
    Because ln(p_k) values are Q-linearly independent, arms at different
    primes rotate at incommensurable rates — the pattern never exactly repeats.
    """
    def __init__(self, n_arms=32, f0=5.0, sigma=0.5, t_scale=0.3):
        super().__init__()
        self.n_arms  = n_arms
        primes = PRIMES[:n_arms]
        log_p  = np.array([np.log(float(p)) for p in primes])

        # Spatial frequency: f0 * ln(p) * p^{-sigma}
        sf = torch.tensor(
            [f0 * lp * float(p)**(-sigma) for lp, p in zip(log_p, primes)],
            dtype=torch.float32)

        # Temporal frequency: f0 * ln(p) * t_scale  (same log structure, different scale)
        tf = torch.tensor(
            [f0 * lp * t_scale for lp in log_p],
            dtype=torch.float32)

        # Orientations: uniformly tiled over [0, π)
        angles = torch.tensor(
            [k * np.pi / n_arms for k in range(n_arms)], dtype=torch.float32)

        self.register_buffer('sf',  sf)
        self.register_buffer('tf',  tf)
        self.register_buffer('cos_a', torch.cos(angles))
        self.register_buffer('sin_a', torch.sin(angles))
        self.out_dim = 2 * n_arms

    def forward(self, coords, t):
        """
        coords : (..., 2)  xy in [-1,1]
        t      : scalar float or tensor
        """
        x = coords[..., 0:1]; y = coords[..., 1:2]
        proj  = x * self.cos_a + y * self.sin_a        # (..., n_arms)
        phase = proj * self.sf                          # spatial part
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(float(t), dtype=torch.float32, device=coords.device)
        phase = phase + t * self.tf                     # add temporal part
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

    def dirichlet_tip(self, t, coord_val=0.25):
        """Return the tip of the Dirichlet partial sum at a given t."""
        # Evaluate at a single reference point
        ref = torch.tensor([[coord_val, coord_val]], dtype=torch.float32,
                           device=self.sf.device)
        with torch.no_grad():
            enc = self.forward(ref, t)
            n   = enc.shape[-1] // 2
            # Sum the complex arms
            real_sum = enc[0, :n].sum().item()
            imag_sum = enc[0, n:].sum().item()
        return real_sum, imag_sum

# ─────────────────────────────────────────────────────────────────
# COMPLEX LINEAR LAYER
# ─────────────────────────────────────────────────────────────────

class ComplexLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.fr = nn.Linear(i, o, bias=False)
        self.fi = nn.Linear(i, o, bias=False)
        self.br = nn.Parameter(torch.zeros(o))
        self.bi = nn.Parameter(torch.zeros(o))
        nn.init.xavier_normal_(self.fr.weight, gain=0.2)
        nn.init.xavier_normal_(self.fi.weight, gain=0.2)

    def forward(self, z):
        r, i = z[..., 0], z[..., 1]
        return torch.stack([
            self.fr(r) - self.fi(i) + self.br,
            self.fr(i) + self.fi(r) + self.bi
        ], dim=-1)

# ─────────────────────────────────────────────────────────────────
# ZETA DREAMER NETWORK
# ─────────────────────────────────────────────────────────────────

class ZetaDreamerNet(nn.Module):
    def __init__(self, n_arms=32, hidden=128, f0=5.0, sigma=0.5, t_scale=0.3):
        super().__init__()
        self.enc = ZetaEncoding(n_arms, f0, sigma, t_scale)
        n = n_arms
        self.l1  = ComplexLinear(n, hidden)
        self.l2  = ComplexLinear(hidden, hidden)
        self.l3  = ComplexLinear(hidden, hidden)
        self.out = ComplexLinear(hidden, 3)

    def forward_complex(self, coords, t):
        """Three genuinely complex decoder channels before RGB magnitude.

        IMPORTANT: compute phase from these components, not RGB sqrt(re²+im²).
        """
        enc = self.enc(coords, t)
        n = enc.shape[-1] // 2
        z = torch.stack([enc[..., :n], enc[..., n:]], dim=-1)
        z = torch.tanh(self.l1(z))
        z = torch.tanh(self.l2(z))
        z = torch.tanh(self.l3(z))
        return self.out(z)

    def forward(self, coords, t):
        out = self.forward_complex(coords, t)
        return torch.sqrt(out[..., 0]**2 + out[..., 1]**2 + 1e-8)

# ─────────────────────────────────────────────────────────────────
# COLOURS / CONSTANTS
# ─────────────────────────────────────────────────────────────────

DARK = '#05070a'; BLUE = '#7aafdd'; RED = '#ff5555'; GOLD = '#ffcc44'
ZERO_COLOR = '#ff3333'     # colour for Riemann zero markers
IMG_COLOR  = '#44ff88'     # colour for stored-image markers
RES_TRAIN  = 96            # training resolution
RES_VIEW   = 160           # preview resolution
BATCH      = 4096

# ─────────────────────────────────────────────────────────────────
# TIMELINE WIDGET
# ─────────────────────────────────────────────────────────────────

class Timeline(tk.Canvas):
    """
    A horizontal timeline from T_MIN to T_MAX showing:
      - Red tick marks at each Riemann zero
      - Green tick marks at stored-image t values
      - A draggable playhead
    """
    HEIGHT = 70

    def __init__(self, parent, on_seek, **kw):
        super().__init__(parent, height=self.HEIGHT, bg='#000',
                         highlightthickness=0, **kw)
        self.on_seek    = on_seek
        self.t_val      = 0.0
        self.img_ts     = []        # list of (t, label)
        self._dragging  = False

        self.bind('<ButtonPress-1>',   self._click)
        self.bind('<B1-Motion>',       self._drag)
        self.bind('<ButtonRelease-1>', lambda e: setattr(self, '_dragging', False))
        self.bind('<Configure>',       lambda e: self.redraw())

    def _t_to_x(self, t):
        w = self.winfo_width()
        return max(0, min(w, (t - T_MIN) / (T_MAX - T_MIN) * w))

    def _x_to_t(self, x):
        w = self.winfo_width()
        return float(np.clip(x / w * (T_MAX - T_MIN) + T_MIN, T_MIN, T_MAX))

    def _click(self, e):
        self._dragging = True
        self.t_val = self._x_to_t(e.x)
        self.on_seek(self.t_val)
        self.redraw()

    def _drag(self, e):
        if self._dragging:
            self.t_val = self._x_to_t(e.x)
            self.on_seek(self.t_val)
            self.redraw()

    def set_t(self, t):
        self.t_val = float(t)
        self.redraw()

    def set_images(self, img_ts):
        self.img_ts = img_ts
        self.redraw()

    def redraw(self):
        self.delete('all')
        w = max(self.winfo_width(), 10)
        h = self.HEIGHT

        # Background
        self.create_rectangle(0, 0, w, h, fill='#0a0c10', outline='')

        # Axis line
        self.create_line(0, h//2, w, h//2, fill='#223', width=1)

        # Riemann zeros
        for z in RIEMANN_ZEROS:
            if T_MIN <= z <= T_MAX:
                x = self._t_to_x(z)
                self.create_line(x, 8, x, h-8, fill=ZERO_COLOR, width=1)
                self.create_text(x, 6, text=f'{z:.1f}', fill=ZERO_COLOR,
                                 font=('JetBrains Mono', 5), anchor='n')

        # Image markers
        for t, label in self.img_ts:
            x = self._t_to_x(t)
            self.create_polygon(x, h-14, x-6, h-6, x+6, h-6,
                                fill=IMG_COLOR, outline='')
            self.create_text(x, h-4, text=label[:6], fill=IMG_COLOR,
                             font=('JetBrains Mono', 6), anchor='s')

        # Playhead
        px = self._t_to_x(self.t_val)
        self.create_polygon(px, 2, px-7, 16, px+7, 16,
                            fill=GOLD, outline='')
        self.create_line(px, 16, px, h-2, fill=GOLD, width=2)
        self.create_text(px, h//2 - 6, text=f't={self.t_val:.2f}',
                         fill=GOLD, font=('JetBrains Mono', 7))

        # Zero legend
        self.create_rectangle(4, h-16, 14, h-6, fill=ZERO_COLOR, outline='')
        self.create_text(18, h-11, text='Riemann zero', fill=ZERO_COLOR,
                         font=('JetBrains Mono', 6), anchor='w')
        self.create_rectangle(100, h-16, 110, h-6, fill=IMG_COLOR, outline='')
        self.create_text(114, h-11, text='stored image', fill=IMG_COLOR,
                         font=('JetBrains Mono', 6), anchor='w')

# ─────────────────────────────────────────────────────────────────
# DIRICHLET SPIRAL CANVAS
# ─────────────────────────────────────────────────────────────────

class SpiralCanvas(tk.Canvas):
    HEIGHT = 90

    def __init__(self, parent, **kw):
        super().__init__(parent, height=self.HEIGHT, bg='#000',
                         highlightthickness=0, **kw)
        self._arms   = []
        self._t_text = ''

    def draw(self, enc_module, t, device):
        self.delete('all')
        w = max(self.winfo_width(), 10); h = self.HEIGHT
        cx, cy = w // 2, h // 2

        n_arms = enc_module.n_arms
        primes = PRIMES[:n_arms]
        sf     = enc_module.sf.cpu().numpy()
        tf     = enc_module.tf.cpu().numpy()
        ca     = enc_module.cos_a.cpu().numpy()
        sa     = enc_module.sin_a.cpu().numpy()

        # Reference point: centre of image
        t_f    = float(t)
        scale  = min(w, h) * 0.32

        import matplotlib.cm as cm
        colors = cm.plasma(np.linspace(0.05, 0.95, n_arms))

        x_acc, y_acc = cx, cy
        prev_x, prev_y = cx, cy

        # Draw each arm as a segment from current tip
        for k in range(n_arms):
            phase = 0.0 * sf[k] + t_f * tf[k]   # at coord=0
            amp   = 1.0 / (n_arms * 0.12)
            dx    = amp * np.cos(phase) * scale
            dy    = amp * np.sin(phase) * scale * 0.7

            nx, ny = x_acc + dx, y_acc + dy
            r_, g_, b_, _ = colors[k]
            col = f'#{int(r_*255):02x}{int(g_*255):02x}{int(b_*255):02x}'
            self.create_line(x_acc, y_acc, nx, ny, fill=col, width=1)
            x_acc, y_acc = nx, ny

        # Tip marker
        self.create_oval(x_acc-3, y_acc-3, x_acc+3, y_acc+3,
                         fill='white', outline='')
        self.create_text(10, 10, text=f'Dirichlet spiral  t={t_f:.3f}',
                         fill=BLUE, font=('JetBrains Mono', 7), anchor='nw')

        # Distance from origin
        dist = np.sqrt((x_acc-cx)**2 + (y_acc-cy)**2)
        self.create_text(w-8, 10, text=f'|tip|={dist/scale:.3f}',
                         fill=GOLD, font=('JetBrains Mono', 7), anchor='ne')

        # Mark if near a Riemann zero
        near_zero = any(abs(t_f - z) < 0.5 for z in RIEMANN_ZEROS)
        if near_zero:
            self.create_text(w//2, h-8, text='★ near Riemann zero',
                             fill=ZERO_COLOR, font=('JetBrains Mono', 7), anchor='s')

# ─────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────

class ZetaDreamerApp:
    def __init__(self, root):
        self.root     = root
        self.root.title('Zeta Dreamer 3D · PerceptionLab')
        self.root.geometry('1450x960')
        self.root.configure(bg=DARK)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')

        # State
        self.stored   = []          # list of {t, img_np, coords, pixels, label}
        self.model    = None
        self.opt      = None
        self.is_train = False
        self.is_play  = False
        self.play_dir = 1.0         # +1 or -1
        self.cur_t    = 0.0
        self.loaded_anchors = []  # restored from a saved checkpoint
        self.export_busy = False

        self.n_arms   = 32
        self.f0       = 5.0
        self.sigma    = 0.5
        self.t_scale  = 0.3

        self._build_model()
        self._setup_ui()

    # ── MODEL ─────────────────────────────────────────────────────

    def _build_model(self):
        self.model = ZetaDreamerNet(
            n_arms=self.n_arms, hidden=128,
            f0=self.f0, sigma=self.sigma,
            t_scale=self.t_scale).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=0.003)

    # ── UI ────────────────────────────────────────────────────────

    def _setup_ui(self):
        # ── LEFT PANEL: fixed-width, split into top/scroll/bottom ──
        left = tk.Frame(self.root, bg='#0d1117', width=290)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        left.pack_propagate(False)

        # Fixed top
        top_fixed = tk.Frame(left, bg='#0d1117')
        top_fixed.pack(fill=tk.X, side=tk.TOP)

        tk.Label(top_fixed, text='ZETA DREAMER 3D', bg='#0d1117', fg=BLUE,
                 font=('JetBrains Mono', 11, 'bold')).pack(pady=(8, 1))
        tk.Label(top_fixed, text='images on the Riemann critical line',
                 bg='#0d1117', fg='#334455',
                 font=('JetBrains Mono', 7)).pack(pady=(0, 6))

        btn_row = tk.Frame(top_fixed, bg='#0d1117')
        btn_row.pack(fill=tk.X, padx=6, pady=2)
        self._btn(btn_row, '+ IMAGE', self._load_image).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self._btn(btn_row, '+ FOLDER', self._load_folder,
                  bg='#0d1a2a', fg='#ff8c3c').pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self._btn(top_fixed, '✦ ENTER 3D WORLD', self._export_3d_dialog,
                  bg='#30402e', fg='#ffd898').pack(fill=tk.X, padx=8, pady=(8, 2))
        self._btn(top_fixed, '↻ SHARPEN OLD 3D HTML', self._upgrade_old_html,
                  bg='#1c3038', fg='#88e4dd').pack(fill=tk.X, padx=8, pady=(2, 6))

        # Fixed bottom (controls — packed from bottom so they never disappear)
        bot_fixed = tk.Frame(left, bg='#0d1117')
        bot_fixed.pack(fill=tk.X, side=tk.BOTTOM, padx=0, pady=0)

        checkpoints = tk.Frame(bot_fixed, bg='#0d1117')
        checkpoints.pack(fill=tk.X, padx=8, pady=2)
        self._btn(checkpoints, 'SAVE MODEL', self._save_model, bg='#1c3038', fg='#88e4dd').pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self._btn(checkpoints, 'LOAD MODEL', self._load_model, bg='#1c3038', fg='#88e4dd').pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self._btn(bot_fixed, 'SAVE FRAME', self._save_frame).pack(
            fill=tk.X, padx=8, pady=2)
        self._btn(bot_fixed, 'RENDER VIDEO', self._render_video,
                  bg='#1a0a2a', fg='#cc88ff').pack(
            fill=tk.X, padx=8, pady=2)

        self.lbl_loss = tk.Label(bot_fixed, text='idle', bg='#0d1117', fg='#446644',
                                 font=('JetBrains Mono', 8))
        self.lbl_loss.pack(pady=2)

        self._btn(bot_fixed, '▶ START DREAMING', self._toggle_train,
                  bg='#003300', fg='#7add7a').pack(fill=tk.X, padx=8, pady=4)

        pf = tk.LabelFrame(bot_fixed, text='MODEL PARAMS', bg='#0d1117', fg='#334',
                           font=('JetBrains Mono', 7))
        pf.pack(fill=tk.X, padx=6, pady=4)
        self.sl_arms   = self._sl(pf, 'N arms', 4, 64, 32,  l=155, integer=True)
        self.sl_f0     = self._sl(pf, 'f0',     1, 20,  5,  l=155)
        self.sl_tscale = self._sl(pf, 't scale',0.05,1.5,0.3,l=155)
        self._btn(pf, 'REBUILD', self._rebuild).pack(fill=tk.X, padx=4, pady=3)

        # Scrollable image list in the middle
        mid = tk.Frame(left, bg='#0d1117')
        mid.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        list_canvas = tk.Canvas(mid, bg='#0d1117', highlightthickness=0)
        list_scroll = tk.Scrollbar(mid, orient=tk.VERTICAL,
                                   command=list_canvas.yview, width=14)
        list_canvas.configure(yscrollcommand=list_scroll.set)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.img_list_frame = tk.Frame(list_canvas, bg='#0d1117')
        _win = list_canvas.create_window((0, 0), window=self.img_list_frame, anchor='nw')

        def _scroll_cfg(e):
            list_canvas.configure(scrollregion=list_canvas.bbox('all'))

        def _width_cfg(e):
            list_canvas.itemconfigure(_win, width=e.width)

        self.img_list_frame.bind('<Configure>', _scroll_cfg)
        list_canvas.bind('<Configure>', _width_cfg)

        def _mw(e):
            list_canvas.yview_scroll(int(-1*(e.delta/120)), 'units')

        list_canvas.bind('<Enter>', lambda e: list_canvas.bind_all('<MouseWheel>', _mw))
        list_canvas.bind('<Leave>', lambda e: list_canvas.unbind_all('<MouseWheel>'))

        # Right area
        right = tk.Frame(self.root, bg=DARK)
        right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Dream canvas
        self.dream_lbl = tk.Label(right, bg='#000')
        self.dream_lbl.pack(expand=True, fill=tk.BOTH, padx=6, pady=6)

        # Playback controls
        ctrl = tk.Frame(right, bg=DARK)
        ctrl.pack(fill=tk.X, padx=6)
        self._btn(ctrl, '◀◀', lambda: self._step(-2.0)).pack(side=tk.LEFT, padx=2)
        self._btn(ctrl, '◀',  lambda: self._step(-0.2)).pack(side=tk.LEFT, padx=2)
        self.play_btn = self._btn(ctrl, '▶ PLAY', self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        self._btn(ctrl, '▶',  lambda: self._step(0.2)).pack(side=tk.LEFT, padx=2)
        self._btn(ctrl, '▶▶', lambda: self._step(2.0)).pack(side=tk.LEFT, padx=2)

        # Speed slider
        self.sl_speed = tk.Scale(ctrl, from_=0.01, to=1.0, resolution=0.01,
                                 orient=tk.HORIZONTAL, bg=DARK, fg=GOLD,
                                 label='play speed', length=120, highlightthickness=0)
        self.sl_speed.set(0.1)
        self.sl_speed.pack(side=tk.LEFT, padx=10)

        # Jump-to-zero buttons
        for z in RIEMANN_ZEROS[:8]:
            b = tk.Button(ctrl, text=f'{z:.1f}', bg='#220000', fg=ZERO_COLOR,
                          font=('JetBrains Mono', 7), relief='flat',
                          command=lambda zz=z: self._jump(zz))
            b.pack(side=tk.LEFT, padx=1)

        # Timeline
        tl_frame = tk.LabelFrame(right, text='CRITICAL LINE  (red = Riemann zeros  |  green = stored images)',
                                 bg=DARK, fg='#334', font=('JetBrains Mono', 7))
        tl_frame.pack(fill=tk.X, padx=6, pady=4)
        self.timeline = Timeline(tl_frame, self._on_seek)
        self.timeline.pack(fill=tk.X, padx=4, pady=4)

        # Dirichlet spiral
        sp_frame = tk.LabelFrame(right, text='PRIME ARMS  (Dirichlet spiral at current t)',
                                 bg=DARK, fg='#334', font=('JetBrains Mono', 7))
        sp_frame.pack(fill=tk.X, padx=6, pady=4)
        self.spiral = SpiralCanvas(sp_frame)
        self.spiral.pack(fill=tk.X, padx=4, pady=4)

    # ── IMAGE MANAGEMENT ──────────────────────────────────────────

    def _next_t(self):
        """Suggest next t: spread after the last stored image, avoiding zeros."""
        n = len(self.stored)
        if n == 0: return 5.0
        last_t = max(e['t'] for e in self.stored)
        # Step by ~7, nudge away from nearest Riemann zero
        t = last_t + 7.0
        while any(abs(t - z) < 1.0 for z in RIEMANN_ZEROS):
            t += 0.7
        return min(t, T_MAX - 3.0)

    def _add_entry(self, path, t_val):
        """Core: load one image file and add at t_val."""
        try:
            img   = Image.open(path).convert('RGB').resize((RES_TRAIN, RES_TRAIN))
            arr   = np.array(img, dtype=np.float32) / 255.0
            label = os.path.splitext(os.path.basename(path))[0][:10]
            h, w, _ = arr.shape
            ys, xs = np.meshgrid(np.linspace(-1,1,h), np.linspace(-1,1,w), indexing='ij')
            coords = torch.tensor(np.stack([xs,ys],-1).astype(np.float32)).to(self.device)
            pixels = torch.tensor(arr, dtype=torch.float32).to(self.device)
            thumb  = ImageTk.PhotoImage(img.resize((48, 38)))
            entry  = dict(t=t_val, img_np=arr, coords=coords.view(-1,2),
                          pixels=pixels.view(-1,3), label=label, thumb=thumb)
            self.stored.append(entry)
            self._add_img_row(entry)
            self._update_timeline()
            return True
        except Exception as ex:
            print(f'Skip {path}: {ex}')
            return False

    def _load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.tif *.webp')])
        if not path: return
        default_t = self._next_t()
        t_val = self._ask_float(
            f'Position on critical line\n'
            f'Riemann zeros: {", ".join(str(round(z,1)) for z in RIEMANN_ZEROS[:8])}',
            default_t)
        if t_val is None: return
        self._add_entry(path, t_val)

    def _load_folder(self):
        """Load all images from a folder, auto-assigning t values."""
        folder = filedialog.askdirectory(title='Select folder of images')
        if not folder: return

        exts = {'.png','.jpg','.jpeg','.bmp','.tif','.webp'}
        paths = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ])
        if not paths:
            self.lbl_loss.config(text='No images found in folder'); return

        n = len(paths)
        # Ask for t range
        t_start = self._next_t()
        t_end_default = min(t_start + max(7.0 * n, 10.0), T_MAX - 3.0)

        d = tk.Toplevel(self.root)
        d.title(f'Load {n} images from folder')
        d.configure(bg=DARK); d.grab_set()

        tk.Label(d, text=f'Found {n} images.\nAssign t values from → to:',
                 bg=DARK, fg=BLUE, font=('JetBrains Mono', 9),
                 justify=tk.CENTER).pack(padx=20, pady=10)

        row = tk.Frame(d, bg=DARK); row.pack(pady=4)
        tk.Label(row, text='t start:', bg=DARK, fg=GOLD,
                 font=('JetBrains Mono', 8)).pack(side=tk.LEFT, padx=4)
        v_start = tk.StringVar(value=str(round(t_start, 1)))
        tk.Entry(row, textvariable=v_start, width=7, bg='#1a2a3a', fg=GOLD,
                 font=('JetBrains Mono', 9)).pack(side=tk.LEFT)
        tk.Label(row, text='  t end:', bg=DARK, fg=GOLD,
                 font=('JetBrains Mono', 8)).pack(side=tk.LEFT, padx=4)
        v_end = tk.StringVar(value=str(round(t_end_default, 1)))
        tk.Entry(row, textvariable=v_end, width=7, bg='#1a2a3a', fg=GOLD,
                 font=('JetBrains Mono', 9)).pack(side=tk.LEFT)

        result = [None]

        def ok():
            try:
                ts = float(v_start.get()); te = float(v_end.get())
                result[0] = (ts, te); d.destroy()
            except ValueError: pass

        tk.Button(d, text=f'LOAD {n} IMAGES', command=ok,
                  bg='#003300', fg='#7add7a',
                  font=('JetBrains Mono', 10), relief='flat',
                  padx=16, pady=6).pack(pady=10)
        d.bind('<Return>', lambda e: ok())
        d.wait_window()

        if result[0] is None: return
        t_s, t_e = result[0]
        t_vals = np.linspace(t_s, t_e, n) if n > 1 else [t_s]

        loaded = 0
        for path, tv in zip(paths, t_vals):
            if self._add_entry(path, float(tv)):
                loaded += 1
        self.lbl_loss.config(text=f'Loaded {loaded}/{n} images from folder')

    def _add_img_row(self, entry):
        row = tk.Frame(self.img_list_frame, bg='#101820', relief='flat')
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, image=entry['thumb'], bg='#101820').pack(side=tk.LEFT, padx=3)
        info = tk.Frame(row, bg='#101820')
        info.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(info, text=entry['label'], bg='#101820', fg=BLUE,
                 font=('JetBrains Mono', 8)).pack(anchor='w')
        t_var = tk.StringVar(value=str(round(entry['t'], 2)))
        t_entry = tk.Entry(info, textvariable=t_var, width=7,
                           bg='#1a2a3a', fg=GOLD, font=('JetBrains Mono', 8))
        t_entry.pack(anchor='w')

        def _update_t(*a):
            try:
                entry['t'] = float(t_var.get())
                self._update_timeline()
            except ValueError:
                pass

        t_var.trace_add('write', _update_t)

        def _jump_to():
            self._jump(entry['t'])

        tk.Button(row, text='go', bg='#002200', fg=IMG_COLOR,
                  font=('JetBrains Mono', 7), relief='flat',
                  command=_jump_to).pack(side=tk.RIGHT, padx=3)

    def _update_timeline(self):
        self.timeline.set_images([(e['t'], e['label']) for e in self.stored])

    # ── TRAINING ──────────────────────────────────────────────────

    def _toggle_train(self):
        if not self.stored:
            self.lbl_loss.config(text='Load images first'); return
        self.is_train = not self.is_train
        if self.is_train:
            Thread(target=self._train_loop, daemon=True).start()

    def _train_loop(self):
        step = 0
        while self.is_train:
            total_loss = 0.0
            # Train on each stored image in random order
            indices = np.random.permutation(len(self.stored))
            for idx in indices:
                entry = self.stored[idx]
                t_val = float(entry['t'])
                n_px  = entry['coords'].shape[0]
                bi    = torch.randint(0, n_px, (BATCH,), device=self.device)
                bx, by = entry['coords'][bi], entry['pixels'][bi]

                self.opt.zero_grad()
                pred = self.model(bx, t_val)
                loss = nn.MSELoss()(pred, by)
                loss.backward()
                self.opt.step()
                total_loss += loss.item()

            step += 1
            if step % 20 == 0:
                avg = total_loss / max(1, len(self.stored))
                self.root.after(0, self.lbl_loss.config,
                                {'text': f'step {step}  loss={avg:.5f}'})
                self.root.after(0, self._render_current)

    # ── RENDERING ─────────────────────────────────────────────────

    def _render_current(self):
        self._render_t(self.cur_t)

    def _render_t(self, t):
        with torch.no_grad():
            r = RES_VIEW
            ys, xs = np.meshgrid(np.linspace(-1,1,r), np.linspace(-1,1,r), indexing='ij')
            coords = torch.tensor(np.stack([xs,ys],-1).astype(np.float32)).to(self.device)
            pred   = self.model(coords, float(t))
            img    = (pred.cpu().numpy().clip(0,1)*255).astype(np.uint8)
            img    = cv2.resize(img, (560, 480), interpolation=cv2.INTER_LANCZOS4)

        # Overlay t value and zero proximity
        pil = Image.fromarray(img)
        d   = ImageDraw.Draw(pil)
        near_zero = [(z, abs(t-z)) for z in RIEMANN_ZEROS if abs(t-z) < 1.5]
        if near_zero:
            z, dist = min(near_zero, key=lambda x: x[1])
            text = f'★ near Riemann zero {z:.3f}  (Δ={dist:.3f})'
            d.rectangle([0, 0, 560, 22], fill=(80, 0, 0))
            d.text((8, 4), text, fill=(255, 60, 60))
        else:
            d.rectangle([0, 0, 560, 22], fill=(5, 7, 10))
            d.text((8, 4), f't = {t:.4f}', fill=(120, 175, 221))

        tk_img = ImageTk.PhotoImage(pil)
        self.dream_lbl.config(image=tk_img); self.dream_lbl.image = tk_img
        self.spiral.draw(self.model.enc, t, self.device)
        self.timeline.set_t(t)

    # ── NAVIGATION ────────────────────────────────────────────────

    def _on_seek(self, t):
        self.cur_t = t
        self._render_t(t)

    def _step(self, dt):
        self.cur_t = float(np.clip(self.cur_t + dt, T_MIN, T_MAX))
        self._render_t(self.cur_t)

    def _jump(self, t):
        self.cur_t = float(t)
        self._render_t(t)
        self.timeline.set_t(t)

    def _toggle_play(self):
        self.is_play = not self.is_play
        self.play_btn.config(text='■ STOP' if self.is_play else '▶ PLAY')
        if self.is_play:
            Thread(target=self._play_loop, daemon=True).start()

    def _play_loop(self):
        import time
        while self.is_play:
            speed = float(self.sl_speed.get())
            self.cur_t += speed * self.play_dir
            if self.cur_t >= T_MAX: self.play_dir = -1.0
            if self.cur_t <= T_MIN: self.play_dir =  1.0
            self.cur_t = float(np.clip(self.cur_t, T_MIN, T_MAX))
            self.root.after(0, self._render_t, self.cur_t)
            time.sleep(0.08)

    # ── SAVE ──────────────────────────────────────────────────────

    def _save_frame(self):
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg')])
        if not path: return
        with torch.no_grad():
            r = 256
            ys, xs = np.meshgrid(np.linspace(-1,1,r), np.linspace(-1,1,r), indexing='ij')
            coords = torch.tensor(np.stack([xs,ys],-1).astype(np.float32)).to(self.device)
            pred   = self.model(coords, float(self.cur_t))
            img    = (pred.cpu().numpy().clip(0,1)*255).astype(np.uint8)
        Image.fromarray(img).save(path)
        print(f'Saved t={self.cur_t:.4f} to {path}')

    def _render_video(self):
        if not self.stored:
            self.lbl_loss.config(text='Load images first'); return
        
        # FIX 2: Stop training so weights don't shift mid-render
        if self.is_train:
            self.is_train = False
            self.lbl_loss.config(text='Training stopped for video render')

        # FIX 3: Clamp to trained region exactly
        ts_min = min(e['t'] for e in self.stored)
        ts_max = max(e['t'] for e in self.stored)

        d = tk.Toplevel(self.root)
        d.title('Render Video'); d.configure(bg=DARK); d.grab_set()

        tk.Label(d, text='ZETA VIDEO', bg=DARK, fg='#88ccff',
                font=('JetBrains Mono', 12, 'bold')).pack(pady=(12,2))

        grid = tk.Frame(d, bg=DARK); grid.pack(padx=20, pady=4)

        def pr(label, default, r):
            tk.Label(grid, text=label, bg=DARK, fg=GOLD,
                    font=('JetBrains Mono', 8), width=14, anchor='e'
            ).grid(row=r, column=0, padx=6, pady=3)

            v = tk.StringVar(value=str(default))
            tk.Entry(grid, textvariable=v, width=10,
                    bg='#1a2a3a', fg=GOLD,
                    font=('JetBrains Mono', 9)
            ).grid(row=r, column=1, padx=6, pady=3)
            return v
        
        # Use more frames and introduce lock-in slow down parameter
        v_ts     = pr('t start', round(ts_min, 1), 0)
        v_te     = pr('t end',   round(ts_max, 1), 1)
        v_w      = pr('width',   512, 2)
        v_h      = pr('height',  512, 3)
        v_fps    = pr('fps',     30,  4)
        v_frames = pr('frames',  1500, 5) # Consolidated duplicate v_frames
        v_speed  = pr('lock-in slow %', 10, 6) # e.g. drop to 10% speed near images

        res = [None]

        def ok():
            try:
                res[0] = dict(
                    t_start=float(v_ts.get()),
                    t_end=float(v_te.get()),
                    W=int(v_w.get()),
                    H=int(v_h.get()),
                    fps=int(v_fps.get()),
                    frames=int(v_frames.get()),
                    speed=float(v_speed.get())
                )
                d.destroy()
            except:
                pass

        tk.Button(d, text='RENDER VIDEO', command=ok,
                bg='#002244', fg='#88ccff',
                font=('JetBrains Mono', 11, 'bold'),
                relief='flat', padx=20, pady=8).pack(pady=12)

        d.bind('<Return>', lambda e: ok())
        d.wait_window()

        if res[0] is None: return

        Thread(target=self._video_thread, args=(res[0],), daemon=True).start()

    def _video_thread(self, cfg):
        stored_ts = [e['t'] for e in self.stored]
        t_start, t_end = cfg['t_start'], cfg['t_end']
        frames = cfg['frames']
        
        # Minimum speed modifier when exactly on top of an image
        min_speed_pct = max(0.01, cfg['speed'] / 100.0) 

        # CDF Warp trick: create a trajectory that slows down around images
        if frames <= 1:
            t_vals = [t_start]
        else:
            # 1. Calculate raw linear path
            raw_t = np.linspace(t_start, t_end, max(10000, frames * 2))
            speeds = []
            
            # 2. Calculate local speed based on distance to nearest stored image
            for rt in raw_t:
                dist = min([abs(rt - st) for st in stored_ts] + [999.0])
                # Speed scales linearly with distance, capping at 1.0
                s = max(min_speed_pct, min(1.0, dist / 1.0)) 
                speeds.append(s)
            
            # 3. Integrate speed into "required time" to build the CDF
            speeds = np.array(speeds)
            time_req = 1.0 / speeds 
            cum_time = np.cumsum(time_req)
            cum_time -= cum_time[0]
            if cum_time[-1] > 0:
                cum_time /= cum_time[-1] # normalize 0 to 1
            
            # 4. Map linear frames backwards through the inverse CDF to get warped t-values
            t_vals = np.interp(np.linspace(0, 1, frames), cum_time, raw_t)

        W, H = cfg['W'], cfg['H']

        # FIX 1: Match live preview resolution!
        # Neural fields are not resolution-invariant without special tricks
        r = RES_VIEW 
        ys, xs = np.meshgrid(
            np.linspace(-1,1,r),
            np.linspace(-1,1,r),
            indexing='ij'
        )

        coords = torch.tensor(
            np.stack([xs, ys], -1).astype(np.float32)
        ).to(self.device)

        # Ask save path
        path = filedialog.asksaveasfilename(
            defaultextension='.mp4',
            filetypes=[('MP4 video','*.mp4')]
        )
        if not path: return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, cfg['fps'], (W, H))

        pw = tk.Toplevel(self.root)
        pw.title('Rendering video...')
        pw.configure(bg=DARK)
        pw.geometry('400x100')

        tk.Label(pw, text='Rendering frames...',
                bg=DARK, fg='#88ccff',
                font=('JetBrains Mono', 10)).pack(pady=10)

        pv = tk.DoubleVar(value=0)
        ttk.Progressbar(pw, variable=pv,
                        maximum=len(t_vals), length=360).pack()

        start = time.time()

        with torch.no_grad():
            for i, t in enumerate(t_vals):

                pred = self.model(coords, float(t))
                frame = (pred.cpu().numpy().clip(0,1)*255).astype(np.uint8)

                # Resize *after* querying model, exactly like preview rendering does
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LANCZOS4)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                writer.write(frame)

                if i % 10 == 0:
                    self.root.after(0, pv.set, i)

        writer.release()
        pw.destroy()

        dt = time.time() - start

        self.lbl_loss.config(
            text=f'Video saved ({cfg["frames"]} frames in {dt:.1f}s)'
        )

        print(f'Video saved → {path}')

    def _rebuild(self):
        self.is_train = False
        self.n_arms   = int(self.sl_arms.get())
        self.f0       = float(self.sl_f0.get())
        self.t_scale  = float(self.sl_tscale.get())
        # Recompute all coordinate tensors (they don't change, but model does)
        self._build_model()
        self.lbl_loss.config(text=f'Rebuilt: {self.n_arms} arms')

    # ── TRAINED WORLD / CHECKPOINT TOOLS ───────────────────────────

    def _anchor_ts(self):
        return [float(e['t']) for e in self.stored] or list(self.loaded_anchors)

    def _save_model(self):
        path = filedialog.asksaveasfilename(
            title='Save Zeta Dreamer weights', defaultextension='.pt',
            initialfile='zeta_dreamer3d_model.pt', filetypes=[('Torch checkpoint', '*.pt')])
        if not path:
            return
        try:
            self.is_train = False
            payload = dict(
                format='zeta_dreamer3d_v1', state=self.model.state_dict(),
                n_arms=int(self.n_arms), f0=float(self.f0), sigma=float(self.sigma),
                t_scale=float(self.t_scale), anchors=self._anchor_ts())
            torch.save(payload, path)
            self.lbl_loss.config(text='Saved model: '+Path(path).name)
        except Exception as e:
            messagebox.showerror('Save failed', str(e))

    def _load_model(self):
        path = filedialog.askopenfilename(
            title='Load Zeta Dreamer weights', filetypes=[('Torch checkpoint', '*.pt')])
        if not path:
            return
        try:
            self.is_train = False
            obj = torch.load(path, map_location=self.device, weights_only=True)
            if obj.get('format') != 'zeta_dreamer3d_v1':
                raise ValueError('This is not a Zeta Dreamer 3D checkpoint')
            self.n_arms = int(obj['n_arms'])
            self.f0 = float(obj['f0'])
            self.sigma = float(obj['sigma'])
            self.t_scale = float(obj['t_scale'])
            self.sl_arms.set(self.n_arms)
            self.sl_f0.set(self.f0)
            self.sl_tscale.set(self.t_scale)
            self._build_model()
            self.model.load_state_dict(obj['state'])
            self.loaded_anchors = [float(x) for x in obj.get('anchors', [])]
            self.stored.clear()  # source images were not saved in the checkpoint
            for child in self.img_list_frame.winfo_children():
                child.destroy()
            self._update_timeline()
            self.timeline.set_images([(t, 'saved') for t in self.loaded_anchors])
            self.lbl_loss.config(text='Loaded model: '+Path(path).name)
            self._render_current()
        except Exception as e:
            messagebox.showerror('Load failed', str(e))

    def _upgrade_old_html(self):
        """Repair an earlier standalone WebGL viewer without retraining."""
        if self.export_busy:
            messagebox.showinfo('Export', 'Another 3D operation is running')
            return
        path = filedialog.askopenfilename(
            title='Choose an older Zeta Dreamer 3D HTML world',
            filetypes=[('Standalone HTML worlds', '*.html')])
        if not path:
            return
        self.export_busy = True
        self.lbl_loss.config(text='Rebuilding thin filaments from saved HTML…')
        def worker():
            try:
                info = upgrade_exported_html(path)
                def done():
                    self.export_busy = False
                    self.lbl_loss.config(text=f'Upgraded: {info["filament_segment_count"]:,} thin segments')
                    webbrowser.open(Path(info['output']).resolve().as_uri())
                    messagebox.showinfo('World upgraded',
                        'A sharper copy was saved next to the original.\n'
                        'No retraining was needed.\n\n'
                        + Path(info['output']).name)
                self.root.after(0, done)
            except Exception as e:
                def fail(msg=str(e)):
                    self.export_busy = False
                    self.lbl_loss.config(text='Upgrade failed')
                    messagebox.showerror('Upgrade failed', msg)
                self.root.after(0, fail)
        Thread(target=worker, daemon=True).start()

    def _export_3d_dialog(self):
        if self.export_busy:
            messagebox.showinfo('Export', 'A 3D export is already running')
            return
        if not self.stored and not self.loaded_anchors:
            if not messagebox.askyesno('Untrained model',
                'No trained image positions are recorded. Export an untrained network anyway?'):
                return
        d = tk.Toplevel(self.root)
        d.title('Make a walkable Zeta Dreamer volume')
        d.configure(bg=DARK)
        d.geometry('500x345')
        d.grab_set()
        tk.Label(d, text='ENTER THE COMPLEX VOLUME', bg=DARK, fg=GOLD,
                 font=('Consolas', 13, 'bold')).pack(pady=(15, 3))
        tk.Label(d, text='Each image is an (x,y) slice. Choose the third-coordinate range.',
                 bg=DARK, fg=BLUE, font=('Consolas', 9)).pack(pady=(0, 15))
        t0 = tk.StringVar(value='79.30')
        t1 = tk.StringVar(value='80.90')
        grid = tk.StringVar(value='96')
        for label, var in [('t start', t0), ('t end', t1), ('grid (32 / 64 / 96 / 128)', grid)]:
            r = tk.Frame(d, bg=DARK); r.pack(fill=tk.X, padx=55, pady=3)
            tk.Label(r, text=label, bg=DARK, fg='#aabccf', width=24,
                     anchor='w', font=('Consolas', 9)).pack(side=tk.LEFT)
            tk.Entry(r, textvariable=var, width=10, bg='#15283b', fg=GOLD).pack(side=tk.RIGHT)

        controls = tk.Frame(d, bg=DARK); controls.pack(fill=tk.X, padx=20, pady=12)
        def stored():
            ts = self._anchor_ts()
            if not ts:
                return
            t0.set(f'{min(ts)-1.5:.2f}'); t1.set(f'{max(ts)+1.5:.2f}')
        self._btn(controls, 'TRAINED RANGE', stored).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        self._btn(controls, 'AROUND SLIDER',
                  lambda: (t0.set(f'{self.cur_t-.8:.3f}'),
                           t1.set(f'{self.cur_t+.8:.3f}'))).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        self._btn(controls, 'FAR 79.3..80.9',
                  lambda: (t0.set('79.30'), t1.set('80.90'))).pack(
                      side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        def begin():
            try:
                aa, bb = float(t0.get()), float(t1.get())
                size = int(grid.get())
                if not (np.isfinite(aa) and np.isfinite(bb) and bb > aa):
                    raise ValueError('t end must be greater than t start')
                if size not in (32, 48, 64, 80, 96, 128):
                    raise ValueError('Choose grid size 32, 48, 64, 80, 96 or 128')
            except ValueError as e:
                messagebox.showerror('Invalid range', str(e), parent=d)
                return
            path = filedialog.asksaveasfilename(
                title='Save walkable world (self-contained HTML)',
                defaultextension='.html',
                initialfile=f'zeta_world_{aa:g}_{bb:g}.html',
                filetypes=[('HTML world', '*.html')], parent=d)
            if not path:
                return
            d.destroy()
            self.is_train = False
            self.export_busy = True
            self.lbl_loss.config(text='Preparing 3D volume…')
            anchors = self._anchor_ts()
            def begin_thread():
                def progress(i,n):
                    if i % 4 == 0 or i == n:
                        self.root.after(0, lambda i=i, n=n:
                            self.lbl_loss.config(text=f'Sampling neural volume: {i}/{n} slices'))
                def worker():
                    try:
                        info = export_world(self.model, path, aa, bb,
                                            size=size, anchors=anchors, progress=progress)
                        def done():
                            self.export_busy = False
                            self.lbl_loss.config(text=f'3D world ready: {size}³ / {info.get("filament_segment_count",0):,} lines')
                            print('ZETA 3D EXPORT:', info)
                            webbrowser.open(Path(path).resolve().as_uri())
                            messagebox.showinfo('3D WORLD READY',
                                'Your world is saved as a single HTML file.\n'
                                'It can be reopened by double-clicking.\n\n'
                                'Click the 3D view and use W A S D to fly.\n'
                                'Press G to return to the center; O for the outside view.')
                        self.root.after(0, done)
                    except Exception as e:
                        def fail(msg=str(e)):
                            self.export_busy = False
                            self.lbl_loss.config(text='Export failed')
                            messagebox.showerror('3D export failed', msg)
                        self.root.after(0, fail)
                Thread(target=worker, daemon=True).start()
            # End the active training iteration before sampling model weights.
            self.root.after(800, begin_thread)

        self._btn(d, 'BUILD WORLD AND STEP INSIDE', begin,
                  bg='#244a42', fg='#bcffee').pack(fill=tk.X, padx=35, pady=3)
        d.bind('<Escape>', lambda e: d.destroy())

    # ── HELPERS ───────────────────────────────────────────────────

    def _btn(self, parent, text, cmd, bg='#0d1a2a', fg=BLUE):
        return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                         activebackground='#1a3a5a', activeforeground=fg,
                         relief=tk.FLAT, padx=8, pady=5,
                         font=('JetBrains Mono', 8), cursor='hand2')

    def _sl(self, parent, label, mn, mx, val, l=120, integer=False):
        f = tk.Frame(parent, bg='#0d1117'); f.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(f, text=label, bg='#0d1117', fg='#445566',
                 font=('JetBrains Mono', 7)).pack(side=tk.LEFT)
        s = tk.Scale(f, from_=mn, to=mx, resolution=1 if integer else 0.05,
                     orient=tk.HORIZONTAL, bg='#0d1117', fg=BLUE,
                     highlightthickness=0, length=l, troughcolor='#0d1a2a',
                     font=('JetBrains Mono', 7))
        s.set(val); s.pack(side=tk.LEFT); return s

    def _ask_float(self, prompt, default):
        """Simple modal dialog to get a float."""
        d = tk.Toplevel(self.root)
        d.title('Enter t value')
        d.configure(bg=DARK)
        d.grab_set()
        tk.Label(d, text=prompt, bg=DARK, fg=BLUE,
                 font=('JetBrains Mono', 9), wraplength=320).pack(padx=20, pady=10)
        var = tk.StringVar(value=str(round(default, 2)))
        e = tk.Entry(d, textvariable=var, bg='#1a2a3a', fg=GOLD,
                     font=('JetBrains Mono', 11), width=12, justify='center')
        e.pack(pady=8)
        e.focus_set()

        result = [None]

        def ok():
            try:
                result[0] = float(var.get())
                d.destroy()
            except ValueError:
                pass

        tk.Button(d, text='SET t', command=ok, bg='#003300', fg=IMG_COLOR,
                  font=('JetBrains Mono', 10), relief='flat',
                  padx=16, pady=6).pack(pady=8)
        e.bind('<Return>', lambda e: ok())
        d.wait_window()
        return result[0]


# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    root = tk.Tk()
    ZetaDreamerApp(root)
    root.mainloop()