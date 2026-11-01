"""
Variants of YourModel used by cv.py. YourModelDistBias: v2 of the SAME geometry bet (a refinement, not a new bet;
not run). TokensNoAttn: YourModel without attention (tag B6).

YourModelDistBias = YourModel (geom PE) whose attention logits get an extra  -softplus(lambda_h) * d_ij,
d_ij = Euclidean distance between electrodes i and j (normalised to max 1), one learnable lambda per head per block,
0 bias on the CLS row/column. Geometry then CONSTRAINS attention toward scalp neighbourhoods instead of only being added
to the tokens. Same constructor/initialisation as YourModel for a given seed, so the bias is the only difference.
dist_perm permutes the distance matrix (control: right distance values, wrong neighbours).
"""
import os
import torch

torch.set_num_threads(int(os.environ.get("NTHREADS", 4)))

import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from main import Block, YourModel


class DistBlock(Block):
    def forward(self, z, bias):                                   # bias: (h, N, N), already -softplus(lam)*d
        B, N, D = z.shape
        q, k, v = (self.qkv(self.ln1(z)).view(B, N, 3, self.h, D // self.h)
                   .permute(2, 0, 3, 1, 4))
        a = (q @ k.transpose(-2, -1) / math.sqrt(D // self.h) + bias).softmax(-1)
        z = z + self.drop(self.proj((self.adrop(a) @ v).transpose(1, 2).reshape(B, N, D)))
        return z + self.drop(self.ff(self.ln2(z))), a


class YourModelDistBias(YourModel):
    def __init__(self, n_chan, n_time, n_classes=2, pos=None, dist_perm=None, **kw):
        super().__init__(n_chan, n_time, n_classes, pos=pos, pe="geom", **kw)
        P = torch.as_tensor(np.asarray(pos), dtype=torch.float32)
        d = torch.cdist(P, P)
        d = d / d.max()
        if dist_perm is not None:
            idx = torch.as_tensor(dist_perm)
            d = d[idx][:, idx]
        full = torch.zeros(n_chan + 1, n_chan + 1)
        full[1:, 1:] = d                                          # CLS row/column: no bias
        self.register_buffer("dist", full)
        for blk in self.blocks:
            blk.__class__ = DistBlock                             # same weights/init as YourModel's Block
        self.lam = nn.Parameter(torch.zeros(len(self.blocks), self.blocks[0].h))

    def forward(self, x, return_embed=False, return_attn=False):
        B, C, T = x.shape
        h = self.temporal(x.reshape(B * C, 1, T))
        h = torch.log(self.pool(h ** 2) + 1e-6)
        tok = self.embed(h.flatten(1)).view(B, C, -1) + self.pos_mlp(self.xyz)
        z = torch.cat([self.cls.expand(B, -1, -1), tok], 1)
        for i, blk in enumerate(self.blocks):
            z, a = blk(z, -F.softplus(self.lam[i])[:, None, None] * self.dist)
        emb = self.norm(z[:, 0])
        logits = self.head(emb)
        if return_attn:
            return logits, a
        return (logits, emb) if return_embed else logits


class TokensNoAttn(YourModel):
    """B6 control: YourModel (geom) with the attention sublayer removed from every block. Same filter bank, log
    band-power tokens, xyz PE and per-token pre-LN feed-forward residual blocks; readout = mean over electrode tokens
    (no CLS, which is meaningless without attention). Answers: does attention add anything over the token features?"""

    def __init__(self, n_chan, n_time, n_classes=2, pos=None, **kw):
        super().__init__(n_chan, n_time, n_classes, pos=pos, pe="geom", **kw)   # same init as YourModel(geom)
        for blk in self.blocks:
            del blk.qkv, blk.proj, blk.ln1, blk.adrop
        del self.cls

    def forward(self, x, return_embed=False, return_attn=False):
        assert not return_attn, "TokensNoAttn has no attention"
        B, C, T = x.shape
        h = self.temporal(x.reshape(B * C, 1, T))
        h = torch.log(self.pool(h ** 2) + 1e-6)
        z = self.embed(h.flatten(1)).view(B, C, -1) + self.pos_mlp(self.xyz)
        for blk in self.blocks:
            z = z + blk.drop(blk.ff(blk.ln2(z)))
        emb = self.norm(z.mean(1))
        logits = self.head(emb)
        return (logits, emb) if return_embed else logits
