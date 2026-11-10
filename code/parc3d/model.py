"""Encoder-processor-decoder with local latent tokens (in the spirit of UPT / AROMA).

  Encoder     any point cloud (N cells, any mesh) -> S supernodes, each aggregating its k
              nearest cells -> transformer over supernodes -> L latent tokens by cross-attention
              (fixed count, independent of the mesh)
  Decoder     any query point -> Fourier embedding -> cross-attention to the L tokens -> u v w p
              (each query is decoded independently: no dependence on which other points are asked)
  Propagator  tokens at t -> tokens at t + dt under the known inflow c(t), c(t + dt)
"""
import math

import torch
import torch.nn as nn


class FourierEmbed(nn.Module):
    def __init__(self, n_freq, dim):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(n_freq, dtype=torch.float32))
        self.mlp = nn.Sequential(nn.Linear(3 + 6 * n_freq, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):                                   # x (..., 3) in [-1, 1]
        xf = (x[..., None] * self.freqs).flatten(-2)
        return self.mlp(torch.cat([x, xf.sin(), xf.cos()], -1))


class Block(nn.Module):
    """Pre-norm transformer block; cross-attention when kv is given."""

    def __init__(self, dim, heads, cross=False, mlp_ratio=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.nkv = nn.LayerNorm(dim) if cross else None
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x, kv=None):
        h = self.n1(x)
        kv = h if self.nkv is None else self.nkv(kv)
        x = x + self.attn(h, kv, kv, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Encoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        d, h = m["dim"], m["heads"]
        self.n_super, self.k, self.rel_scale = m["n_supernodes"], m["k_neighbors"], m["rel_scale"]
        # HW3 density-tempered supernodes: 0 = uniform over cells (baseline), 1 = uniform in space
        self.alpha = float(m.get("supernode_alpha", 0.0))
        self.n_ref, self.k_density = 2048, 8
        self.embed = FourierEmbed(m["n_freq"], d)
        self.point = nn.Sequential(nn.Linear(d + 4, d), nn.GELU(), nn.Linear(d, d))
        self.msg = nn.Sequential(nn.Linear(d + 3, d), nn.GELU(), nn.Linear(d, d))
        self.super_pos = FourierEmbed(m["n_freq"], d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(m["enc_blocks"]))
        self.queries = nn.Parameter(torch.randn(m["n_latent"], d) * 0.02)
        self.cond = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.pool = Block(d, h, cross=True)
        self.latent_blocks = nn.ModuleList(Block(d, h) for _ in range(m["latent_blocks"]))

    def forward(self, x, r, c):
        """x (B, N, 3) normalized coords, r (B, N, 4) normalized residual fields, c (B, 6).
        Rows are in random order, so the first S rows are a random supernode subset."""
        B, N, _ = x.shape
        S, k = min(self.n_super, N), min(self.k, N)
        h = self.point(torch.cat([self.embed(x), r], -1))                      # (B, N, d)
        xs = x[:, :S] if self.alpha == 0 else self.pick_supernodes(x, S)
        with torch.autocast(x.device.type, enabled=False):                      # exact kNN in fp32
            nbr = torch.cdist(xs.float(), x.float()).topk(k, largest=False).indices    # (B, S, k)
        b = torch.arange(B, device=x.device)[:, None, None]
        rel = (x[b, nbr] - xs[:, :, None]) * self.rel_scale
        t = self.msg(torch.cat([h[b, nbr], rel], -1)).max(dim=2).values + self.super_pos(xs)
        for blk in self.blocks:
            t = blk(t)
        z = self.queries[None].expand(B, -1, -1) + self.cond(c)[:, None]
        z = self.pool(z, t)
        for blk in self.latent_blocks:
            z = blk(z)
        return z                                                                # (B, L, d)

    @torch.no_grad()
    def pick_supernodes(self, x, S):
        """Density-tempered choice: P(point i) ~ s_i^(3 alpha), s_i = local cell spacing
        (distance to the 8th nearest of a uniform subset). The input rows are uniform over
        cells, so alpha = 0 follows the mesh's cell density and alpha = 1 cancels it
        (density ~ s^-3), i.e. supernodes spread evenly in space whatever the mesh."""
        B, N, _ = x.shape
        with torch.autocast(x.device.type, enabled=False):
            ref = x[:, :min(N, self.n_ref)].float()
            kk = min(self.k_density + 1, ref.shape[1])
            s = torch.cdist(x.float(), ref).topk(kk, largest=False).values[..., -1]
        w = s.clamp_min(1e-6) ** (3.0 * self.alpha)
        gen = None if self.training else torch.Generator(device=x.device).manual_seed(0)
        idx = torch.multinomial(w, S, replacement=False, generator=gen)
        return x[torch.arange(B, device=x.device)[:, None], idx]


class Decoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        d, h = m["dim"], m["heads"]
        self.embed = FourierEmbed(m["n_freq"], d)
        self.blocks = nn.ModuleList(Block(d, h, cross=True) for _ in range(m["dec_blocks"]))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))

    def forward(self, z, xq):                                                   # xq (B, Q, 3)
        q = self.embed(xq)
        for blk in self.blocks:
            q = blk(q, z)
        return self.head(q)


class AutoEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.encoder, self.decoder = Encoder(m), Decoder(m)


class Propagator(nn.Module):
    def __init__(self, m):
        super().__init__()
        d, h = m["dim"], m["heads"]
        self.cond = nn.Sequential(nn.Linear(12, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(m["prop_blocks"]))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)   # starts as identity

    def forward(self, z, c0, c1):
        h = torch.cat([self.cond(torch.cat([c0, c1], -1))[:, None], z], 1)
        for blk in self.blocks:
            h = blk(h)
        return z + self.out(h[:, 1:])
