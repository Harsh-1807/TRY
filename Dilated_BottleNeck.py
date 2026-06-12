# -*- coding: utf-8 -*-
"""
CorrDiff Network — v11  (D2M-separate + VarMap-separate + DilatedBottleneck)
=============================================================================
Changes from v10:
  * var_map and d2m are now fully SEPARATE paths in BOTH Regressor and UNet.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ────────────────────────────────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────────────────────────────────
def _g(ch, mx=32):
    for g in range(min(mx, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1

# ────────────────────────────────────────────────────────────────────────────
# Building blocks
# ────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.p = nn.AdaptiveAvgPool2d(1)
        self.f = nn.Sequential(
            nn.Linear(ch, max(4, ch // r), bias=False), nn.ReLU(inplace=True),
            nn.Linear(max(4, ch // r), ch, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.f(self.p(x).view(b, c)).view(b, c, 1, 1)


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        b, c, h, w = x.shape
        yg = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xg = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return self.conv(torch.cat([x, yg, xg], dim=1))


class FourierFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(channels, channels, 2) * 0.02)
        self.norm = nn.GroupNorm(_g(channels), channels)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(self.norm(x))
        weight = torch.view_as_complex(self.complex_weight)
        out_fft = torch.einsum('bchw,cd->bdhw', x_fft, weight)
        return x + self.proj(torch.fft.irfft2(out_fft, s=(H, W)))


class ResConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_g(ic), ic), nn.SiLU(),
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.GroupNorm(_g(oc), oc), nn.SiLU(),
            nn.Conv2d(oc, oc, 3, padding=1))
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
        self.se = SEBlock(oc)

    def forward(self, x):
        return self.skip(x) + self.se(self.net(x))


class BnAttn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        while ch % heads != 0 and heads > 1:
            heads -= 1
        self.h = heads
        self.s = 8
        self.scale = (ch // heads) ** -0.5
        self.norm = nn.GroupNorm(_g(ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        s = min(self.s, H)
        lo = F.adaptive_avg_pool2d(self.norm(x), (s, s))
        qkv = self.qkv(lo).reshape(B, 3, self.h, C // self.h, s * s)
        q, k, v = qkv.unbind(1)
        a = torch.einsum('bhdn,bhdm->bhnm', q * self.scale, k).softmax(-1)
        o = torch.einsum('bhnm,bhdm->bhdn', a, v).reshape(B, C, s, s)
        return x + F.interpolate(self.proj(o), (H, W), mode='bilinear', align_corners=False)


class TemporalAttention(nn.Module):
    def __init__(self, ch, T=4, heads=4):
        super().__init__()
        while ch % heads != 0 and heads > 1:
            heads -= 1
        self.T = T
        self.h = heads
        self.scale = (ch // heads) ** -0.5
        self.norm = nn.GroupNorm(_g(ch), ch)
        self.qkv = nn.Linear(ch, ch * 3, bias=False)
        self.proj = nn.Linear(ch, ch)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, T=None):
        T = T or self.T
        BT, C, H, W = x.shape
        if BT % T != 0:
            return x
        B = BT // T
        tok = F.adaptive_avg_pool2d(self.norm(x), 1).view(BT, C).view(B, T, C)
        qkv = self.qkv(tok).reshape(B, T, 3, self.h, C // self.h)
        q, k, v = qkv.unbind(2)
        a = torch.einsum('bthd,bshd->bths', q * self.scale, k).softmax(-1)
        out = torch.einsum('bths,bshd->bthd', a, v).reshape(B, T, C)
        out = self.proj(out).view(BT, C, 1, 1)
        return x + self.gate.tanh() * out


class DilatedBottleneck(nn.Module):
    """ASPP-style multi-rate dilated bottleneck. Zero-init → identity at start."""
    def __init__(self, in_channels, mid_channels=512):
        super().__init__()
        rates = [1, 2, 4, 8]
        branch_ch = mid_channels // len(rates)
        self.branches = nn.ModuleList()
        for rate in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_ch, kernel_size=3,
                          padding=rate, dilation=rate, bias=False),
                nn.GroupNorm(_g(branch_ch), branch_ch),
                nn.SiLU(),
            ))
        self.project = nn.Sequential(
            nn.Conv2d(branch_ch * len(rates), in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_g(in_channels), in_channels),
        )
        nn.init.zeros_(self.project[0].weight)

    def forward(self, x):
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        return x + self.project(out)

# ════════════════════════════════════════════════════════════════════════════
# REGRESSOR  (Stage 1)
# ════════════════════════════════════════════════════════════════════════════
class CorrDiffRegressor(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=64,
                 channel_mult=(1, 2, 4), num_blocks=2, global_dim=2,
                 topo_channels=3, d2m_channels=1, use_d2m=True, **kw):
        super().__init__()
        cms = list(channel_mult)
        st = base_channels
        emb = base_channels * 2
        self.use_d2m = use_d2m

        self.g_mlp = nn.Sequential(
            nn.Linear(global_dim, emb), nn.SiLU(), nn.Linear(emb, st)
        ) if global_dim > 0 else None

        self.r_stem = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            CoordConv2d(in_channels, st, 3, padding=1),
            nn.GroupNorm(_g(st), st), nn.SiLU())

        self.t_stem = nn.Sequential(
            CoordConv2d(topo_channels, st, 3, padding=1),
            nn.GroupNorm(_g(st), st), nn.SiLU())

        if use_d2m:
            self.d_stem = nn.Sequential(
                CoordConv2d(d2m_channels, st, 3, padding=1),
                nn.GroupNorm(_g(st), st), nn.SiLU())

        self.r_enc = nn.ModuleList()
        self.t_enc = nn.ModuleList()
        self.r_dn  = nn.ModuleList()
        self.t_dn  = nn.ModuleList()
        self.sk_ch = []
        rc = tc = st

        for li, m in enumerate(cms):
            oc = base_channels * m
            rb = nn.ModuleList()
            tb = nn.ModuleList()
            for _ in range(num_blocks):
                rb.append(ResConv(rc, oc))
                tb.append(ResConv(tc, oc))
                rc = tc = oc
            self.r_enc.append(rb)
            self.t_enc.append(tb)
            self.sk_ch.append(rc + tc)
            last = (li == len(cms) - 1)
            self.r_dn.append(nn.Identity() if last else nn.Conv2d(rc, rc, 4, 2, 1))
            self.t_dn.append(nn.Identity() if last else nn.Conv2d(tc, tc, 4, 2, 1))

        bn = base_channels * cms[-1]
        self.bn_proj = nn.Conv2d(rc + tc, bn, 1)
        self.bn_attn = BnAttn(bn, max(1, bn // 64))
        self.bn_se   = SEBlock(bn)
        self.bn_mid  = ResConv(bn, bn)

        self.d_ups = nn.ModuleList()
        self.d_blk = nn.ModuleList()
        dc = bn

        for li, m in reversed(list(enumerate(cms))):
            oc = base_channels * m
            sc = self.sk_ch[li]
            self.d_ups.append(
                nn.Identity() if li == len(cms) - 1 else
                nn.Sequential(nn.ConvTranspose2d(dc, dc, 4, 2, 1),
                               nn.GroupNorm(_g(dc), dc), nn.SiLU()))
            blks = nn.ModuleList()
            ic2 = dc + sc
            for _ in range(num_blocks):
                blks.append(ResConv(ic2, oc))
                ic2 = oc
            self.d_blk.append(blks)
            dc = oc

        self.out = nn.Sequential(
            nn.GroupNorm(_g(dc), dc), nn.SiLU(),
            nn.Conv2d(dc, out_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, topo, global_features=None, d2m=None):
        r = self.r_stem(x)
        t = self.t_stem(topo)

        if self.use_d2m and d2m is not None:
            d2m_aligned = F.interpolate(d2m, size=r.shape[-2:],
                                        mode='bilinear', align_corners=False)
            r = r + self.d_stem(d2m_aligned)

        if self.g_mlp is not None and global_features is not None:
            gs = self.g_mlp(global_features)[:, :, None, None]
            r = r + gs
            t = t + gs

        rs, ts = [], []
        for li in range(len(self.r_enc)):
            for rb, tb in zip(self.r_enc[li], self.t_enc[li]):
                r = rb(r)
                t = tb(t)
            rs.append(r); ts.append(t)
            r = self.r_dn[li](r)
            t = self.t_dn[li](t)

        f = self.bn_proj(torch.cat([r, t], 1))
        f = self.bn_attn(f); f = self.bn_se(f); f = self.bn_mid(f)
        d = f

        for li, (up, blks) in enumerate(zip(self.d_ups, self.d_blk)):
            lv = len(self.r_enc) - 1 - li
            d = up(d)
            d = torch.cat([d, torch.cat([rs[lv], ts[lv]], 1)], 1)
            for b in blks:
                d = b(d)

        return self.out(d)

# ════════════════════════════════════════════════════════════════════════════
# UNET  (Stage 2)
# ════════════════════════════════════════════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, ic, oc, ec, down=False, up=False,
                 use_topo=True, topo_channels=3, dropout=0.1):
        super().__init__()
        self.rs = None
        if down: self.rs = nn.Conv2d(ic, ic, 4, 2, 1)
        if up:   self.rs = nn.ConvTranspose2d(ic, ic, 4, 2, 1)
        self.n1 = nn.GroupNorm(_g(ic), ic)
        self.c1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.ep = nn.Linear(ec, oc)
        self.use_topo = use_topo
        if use_topo:
            self.topo_proj = nn.Conv2d(topo_channels, oc * 2, kernel_size=3, padding=1)
            nn.init.zeros_(self.topo_proj.weight)
            nn.init.zeros_(self.topo_proj.bias)
        self.n2 = nn.GroupNorm(_g(oc), oc)
        self.dropout = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.se = SEBlock(oc)
        self.sk = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x, e, topo=None):
        if self.rs: x = self.rs(x)
        h = self.c1(F.silu(self.n1(x))) + self.ep(F.silu(e))[:, :, None, None]
        h_norm = self.n2(h)
        
        if self.use_topo and topo is not None:
            t_res = F.interpolate(topo, size=h.shape[-2:], mode='bilinear', align_corners=False)
            gamma, beta = self.topo_proj(t_res).chunk(2, dim=1)
            h_norm = h_norm * (1 + torch.tanh(gamma)) + beta
            
        h_norm = self.dropout(h_norm)
        return self.se(self.c2(F.silu(h_norm))) + self.sk(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=128,
                 channel_mult=(1, 2, 2, 4), num_res_blocks=2, dropout=0.1,
                 num_blocks=None, global_dim=2, use_bottleneck_attention=True,
                 topo_channels=3, use_d2m=True, d2m_channels=1,
                 use_var_map=True, var_map_channels=1,
                 temporal_frames=3, **kw):
        super().__init__()
        nrb = num_blocks if num_blocks else num_res_blocks
        ec  = base_channels * 4
        self.topo_channels  = topo_channels
        self.use_d2m        = use_d2m
        self.use_var_map    = use_var_map
        self.temporal_frames = temporal_frames

        head_in = in_channels

        self.t_emb = nn.Sequential(
            nn.Linear(base_channels, ec), nn.SiLU(), nn.Linear(ec, ec))
        self.g_mlp = nn.Sequential(
            nn.Linear(global_dim, ec), nn.SiLU()) if global_dim else None

        if use_d2m:
            self.d_stem = nn.Sequential(
                CoordConv2d(d2m_channels, base_channels, 3, padding=1),
                nn.GroupNorm(_g(base_channels), base_channels), nn.SiLU())
            self.d_gate = nn.Parameter(torch.zeros(1))

        if use_var_map:
            self.v_stem = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                CoordConv2d(var_map_channels, base_channels, 3, padding=1),
                nn.GroupNorm(_g(base_channels), base_channels), nn.SiLU())
            self.v_gate = nn.Parameter(torch.zeros(1))

        self.head = CoordConv2d(head_in, base_channels, 3, padding=1)
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        ch = base_channels
        sk = []

        for m in channel_mult:
            oc = base_channels * m
            for _ in range(nrb):
                self.downs.append(ResBlock(ch, oc, ec,
                                           topo_channels=topo_channels, dropout=dropout))
                ch = oc; sk.append(ch)
            self.downs.append(ResBlock(ch, ch, ec, down=True,
                                       topo_channels=topo_channels, dropout=dropout))
            sk.append(ch)

        self.m1 = ResBlock(ch, ch, ec, topo_channels=topo_channels, dropout=dropout)
        self.ma = BnAttn(ch, max(1, ch // 64)) if use_bottleneck_attention else nn.Identity()
        self.fft = FourierFilter(ch)
        mid_ch = min(512, ch)
        self.dilated_bottleneck = DilatedBottleneck(in_channels=ch, mid_channels=mid_ch)
        self.m2 = ResBlock(ch, ch, ec, topo_channels=topo_channels, dropout=dropout)

        for m in reversed(channel_mult):
            oc = base_channels * m
            self.ups.append(ResBlock(ch + sk.pop(), oc, ec, up=True,
                                     topo_channels=topo_channels, dropout=dropout))
            ch = oc
            for _ in range(nrb):
                self.ups.append(ResBlock(ch + sk.pop(), oc, ec,
                                         topo_channels=topo_channels, dropout=dropout))
                ch = oc

        self.out = nn.Sequential(
            nn.GroupNorm(_g(ch), ch), nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1))

    def _temb(self, t):
        half = self.t_emb[0].in_features // 2
        freq = torch.exp(
            torch.arange(half, device=t.device) * (-math.log(10000) / (half - 1)))
        e = t.unsqueeze(1) * freq.unsqueeze(0) * 2 * math.pi
        return self.t_emb(torch.cat([e.sin(), e.cos()], -1))

    def forward(self, x, t, topo=None, global_features=None,
                cfg_drop=None, d2m=None, var_map=None, T=None):
        T = T or self.temporal_frames
        emb = self._temb(t)

        if global_features is not None and self.g_mlp:
            gf = global_features.clone()
            if cfg_drop is not None:
                gf[cfg_drop] = 0.
            emb = emb + self.g_mlp(gf)

        x_in    = x.clone()
        topo_in = topo.clone() if topo is not None else None

        if cfg_drop is not None and cfg_drop.any():
            x_in[cfg_drop, 1:] = 0.
            if topo_in is not None:
                topo_in[cfg_drop] = 0.

        h = self.head(x_in)

        if self.use_d2m and d2m is not None:
            d2m_in = d2m.clone()
            if cfg_drop is not None and cfg_drop.any():
                d2m_in[cfg_drop] = 0.
            d2m_res = F.interpolate(d2m_in, size=h.shape[-2:],
                                    mode='bilinear', align_corners=False)
            h = h + self.d_gate.tanh() * self.d_stem(d2m_res)

        if self.use_var_map and var_map is not None:
            vm_in = var_map.clone()
            if cfg_drop is not None and cfg_drop.any():
                vm_in[cfg_drop] = 0.
            h = h + self.v_gate.tanh() * self.v_stem(vm_in)

        sk = [h]
        for layer in self.downs:
            h = layer(h, emb, topo_in)
            sk.append(h)

        h = self.m1(h, emb, topo_in)
        h = self.ma(h) if isinstance(self.ma, BnAttn) else h
        h = self.fft(h)
        h = self.dilated_bottleneck(h)
        h = self.m2(h, emb, topo_in)

        for layer in self.ups:
            h = torch.cat([h, sk.pop()], 1)
            h = layer(h, emb, topo_in)

        return self.out(h)

# ════════════════════════════════════════════════════════════════════════════
# Flow Matching
# ════════════════════════════════════════════════════════════════════════════
class FlowMatching:
    def __init__(self, n_steps=6, cfg_scale=2.5):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale

    def get_train_sample(self, x1):
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        alpha = torch.distributions.Beta(1.5, 1.5).sample((B,)).to(x1.device)
        t = alpha.view(B, 1, 1, 1)
        x_t = (1 - t) * x0 + t * x1
        v_target = x1 - x0
        return x_t, alpha, v_target

    @torch.no_grad()
    def sample(self, model, x_cond, topo, global_features=None,
               d2m=None, var_map=None, cfg_scale=None, T=1):
        cfg = cfg_scale or self.cfg_scale
        B, _, H, W = x_cond.shape
        x = torch.randn(B, 1, H, W, device=x_cond.device)
        dt = 1.0 / self.n_steps

        for i in range(self.n_steps):
            t_vec = torch.full((B,), i * dt, device=x.device)
            x_in  = torch.cat([x, x_cond], dim=1)
            v_c   = model(x_in, t_vec, topo=topo, global_features=global_features,
                          d2m=d2m, var_map=var_map, T=T)
            if cfg > 1.0:
                x_unc = torch.cat([x, torch.zeros_like(x_cond)], dim=1)
                mask  = torch.ones(B, dtype=torch.bool, device=x.device)
                v_u   = model(x_unc, t_vec, topo=topo, global_features=global_features,
                              d2m=d2m, var_map=var_map, cfg_drop=mask, T=T)
                v = v_u + cfg * (v_c - v_u)
            else:
                v = v_c
            x = x + dt * v
            x = torch.clamp(x, min=-1.0, max=7.0)

        return x

    @staticmethod
    def loss(v_pred, v_target):
        return F.mse_loss(v_pred, v_target)

# ════════════════════════════════════════════════════════════════════════════
# Physics Guidance
# ════════════════════════════════════════════════════════════════════════════
class PhysicsGuide:
    # Drop the threshold way down so it only catches absolute desert-level dry days
    DRY_THRESH_LOG = -10.0 

    @staticmethod
    def apply(pred_log, coarse_log, enforce_mass=True, enforce_dry=False): # Default to False!
        pred = pred_log.clamp(min=0.)
        
        # 1. Soften the Dry Enforcer (Or just rely on the default False)
        if enforce_dry:
            coarse_mean = coarse_log.mean(dim=[-2, -1], keepdim=True)
            dry_mask = coarse_mean < PhysicsGuide.DRY_THRESH_LOG
            pred = pred * (~dry_mask).float()
            
        # 2. Relax the Mass Enforcer to preserve Spatial Variance
        if enforce_mass:
            pred_phys   = torch.expm1(pred.clamp(0))
            coarse_phys = torch.expm1(coarse_log.clamp(0))
            
            coarse_up   = F.interpolate(coarse_phys, size=pred.shape[-2:],
                                        mode='bilinear', align_corners=False)
            
            target_mean = coarse_up.mean(dim=[-2, -1], keepdim=True).clamp(1e-6)
            pred_mean   = pred_phys.mean(dim=[-2, -1], keepdim=True).clamp(1e-6)
            
            # TIGHTEN the clamp. 
            # Allow max 50% reduction or 300% increase to preserve structural peaks.
            scale = (target_mean / pred_mean).clamp(0.5, 3.0) 
            
            pred = torch.log1p((pred_phys * scale).clamp(0))
            
        return pred

# ════════════════════════════════════════════════════════════════════════════
# QDM
# ════════════════════════════════════════════════════════════════════════════
class QDM:
    def __init__(self, n_quantiles=1000, clip_min=0.):
        self.n = n_quantiles
        self.clip = clip_min
        self.q = torch.linspace(0., 1., n_quantiles)
        self.cm = self.co = None
        self._fitted = False

    def fit(self, mp, op):
        mp_flat = mp.flatten().float().cpu()
        op_flat = op.flatten().float().cpu()
        max_elements = 15000000
        if op_flat.numel() > max_elements:
            indices  = torch.randperm(op_flat.numel())[:max_elements]
            op_sampled = op_flat[indices]; mp_sampled = mp_flat[indices]
        else:
            op_sampled = op_flat; mp_sampled = mp_flat
        self.cm = torch.quantile(mp_sampled, self.q)
        self.co = torch.quantile(op_sampled, self.q)
        self._fitted = True
        i95 = int(.95 * self.n); i99 = int(.99 * self.n)
        print(f"[QDM] P95 model={self.cm[i95]:.2f} obs={self.co[i95]:.2f} | "
              f"P99 model={self.cm[i99]:.2f} obs={self.co[i99]:.2f}")

    @torch.no_grad()
    def apply(self, pred):
        assert self._fitted
        dev = pred.device
        cm  = self.cm.to(dev); co = self.co.to(dev)
        x   = pred.flatten().float()
        idx = torch.searchsorted(cm.contiguous(),
                                 x.contiguous()).clamp(0, self.n - 1)
        # Additive delta: co[idx] - cm[idx], not ratio
        delta = (co[idx] - cm[idx]).clamp(-5.0, 7.0)   # cap delta to avoid tail explosion
        return (x + delta).clamp(self.clip).reshape(pred.shape).to(pred.dtype)

    def save(self, p):
        torch.save({"cm": self.cm, "co": self.co, "n": self.n, "clip": self.clip}, p)

    @classmethod
    def load(cls, p):
        d = torch.load(p, map_location="cpu")
        q = cls(d["n"], d["clip"])
        q.cm = d["cm"]; q.co = d["co"]; q._fitted = True
        return q
