"""Opt-in model variants for the ablation experiments.

The original model.py is intentionally untouched.  This module subclasses
the published model and adds two independently switchable changes:
ETF scale/bias calibration and a small residual feature adapter.
"""
import torch
import torch.nn as nn
from model import model_cifar as _BaseModel


class model_cifar(_BaseModel):
    def __init__(self, args, n_classes, out_dim=256):
        super().__init__(args, n_classes, out_dim=out_dim)
        self.use_etf_improved = bool(getattr(args, "use_etf_improved", False))
        self.use_adapter = bool(getattr(args, "use_adapter", False))
        dim = self.encoder[-1].out_features
        if self.use_adapter:
            hidden = max(16, dim // 4)
            self.adapter = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim)
            )
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
            self.adapter_gate = nn.Parameter(torch.tensor(0.0))
        if self.use_etf_improved:
            self.etf_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        h = self.backbone(x)
        h = torch.flatten(h, 1)
        features = self.encoder(h)
        if self.use_adapter:
            features = features + self.adapter_gate.tanh() * self.adapter(features)
        output = self.proxy(features)
        if self.use_etf_improved:
            # ETF is applied to the feature proxy later in the original loss
            # (``output @ ETF``), so calibrate its feature magnitude here.
            output = self.etf_scale.clamp_min(0.05) * output
        return h, features, output
