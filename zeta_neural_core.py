"""Headless exact model definitions from original ZetaDreamer3D.

Use this for trained checkpoint inference without Tkinter or OpenCV.
"""
import torch
import torch.nn as nn
import numpy as np

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
