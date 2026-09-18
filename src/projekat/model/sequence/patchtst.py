"""PatchTST (Nie et al. 2023), small architecture: the sequence is split
into patches, projected to an embedding, passed through a transformer
encoder, and pooled to a single scalar prediction. Deliberately small
(few layers/heads/dim) given panel-scale (not tick-scale) training data,
per the thesis's own stated risk mitigation.

Stochastic -- 5-seed repetition required (runner.py). Depends on grid,
estimator, and jump_test since the sequence includes C_d/J_d."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from projekat.model.backtransform import residual_correction
from projekat.model.features import SEQUENCE_LENGTH, build_sequences
from projekat.model.registry import register_model
from projekat.model.targets import TARGET_COL

_N_FEATURES = 6  # RV_d, C_d, J_d, RS_pos, RS_neg, r_d
_PATCH_LEN = 10
_D_MODEL = 32
_N_HEADS = 4
_N_LAYERS = 2
_EPOCHS = 30
_LR = 1e-3
_BATCH_SIZE = 64


class _PatchTSTNet(nn.Module):
    def __init__(self, seq_len: int = SEQUENCE_LENGTH, n_features: int = _N_FEATURES, patch_len: int = _PATCH_LEN):
        super().__init__()
        n_patches = seq_len // patch_len
        self.patch_len = patch_len
        self.n_patches = n_patches
        self.input_proj = nn.Linear(patch_len * n_features, _D_MODEL)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, _D_MODEL))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=_D_MODEL, nhead=_N_HEADS, dim_feedforward=_D_MODEL * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=_N_LAYERS)
        self.head = nn.Linear(_D_MODEL * n_patches, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, f = x.shape
        x = x[:, : self.n_patches * self.patch_len, :]
        x = x.reshape(b, self.n_patches, self.patch_len * f)
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        x = x.reshape(b, -1)
        return self.head(x).squeeze(-1)


class _FittedPatchTST:
    def __init__(self, net: _PatchTSTNet, backtransform_correction: float):
        self._net = net
        self.backtransform_correction = backtransform_correction

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        self._net.eval()
        with torch.no_grad():
            x = torch.tensor(sequences, dtype=torch.float32)
            return self._net(x).numpy()


def _targets_for_sequences(df: pd.DataFrame, origin_idx: np.ndarray, horizon: int) -> np.ndarray:
    # sort order must match build_sequences exactly (symbol, then date) so
    # origin_idx lines up; grouping by symbol keeps the horizon-ahead shift
    # from crossing into a different symbol's future RV.
    df_sorted = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    rv_by_symbol = df_sorted.groupby("symbol")["RV_d"]
    future_sum = pd.Series(0.0, index=df_sorted.index)
    any_nan = pd.Series(False, index=df_sorted.index)
    for i in range(1, horizon + 1):
        shifted = rv_by_symbol.shift(-i)
        any_nan = any_nan | shifted.isna()
        future_sum = future_sum + shifted.fillna(0.0)
    y = future_sum / horizon
    y[any_nan] = np.nan
    log_y = np.log(y.clip(lower=1e-12))
    return log_y.to_numpy()[origin_idx]


class _PatchTST:
    name = "patchtst"
    stochastic = True
    depends_on = frozenset({"grid", "estimator", "jump_test"})
    is_sequence_model = True
    predicts_levels = False

    def fit(self, fit_train: pd.DataFrame, val_fold: pd.DataFrame, horizon: int, *, seed: int | None = None):
        torch.manual_seed(seed or 0)

        sequences, origin_idx = build_sequences(fit_train)
        targets = _targets_for_sequences(fit_train, origin_idx, horizon)
        valid = ~np.isnan(targets)
        sequences, targets = sequences[valid], targets[valid]

        net = _PatchTSTNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=_LR)
        loss_fn = nn.MSELoss()

        X = torch.tensor(sequences, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)
        n = len(X)
        net.train()
        for _epoch in range(_EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, _BATCH_SIZE):
                idx = perm[i : i + _BATCH_SIZE]
                optimizer.zero_grad()
                pred = net(X[idx])
                loss = loss_fn(pred, y[idx])
                loss.backward()
                optimizer.step()

        # validation-fold backtransform correction, out-of-fold
        val_sequences, val_origin = build_sequences(val_fold)
        if len(val_sequences) > 0:
            val_targets = _targets_for_sequences(val_fold, val_origin, horizon)
            valid_val = ~np.isnan(val_targets)
            net.eval()
            with torch.no_grad():
                val_pred = net(torch.tensor(val_sequences[valid_val], dtype=torch.float32)).numpy()
            correction = residual_correction(val_targets[valid_val], val_pred) if valid_val.any() else 0.0
        else:
            correction = 0.0

        return _FittedPatchTST(net, correction)

    def build_sequences(self, df: pd.DataFrame):
        return build_sequences(df)

    def targets_for_sequences(self, df: pd.DataFrame, origin_idx: np.ndarray, horizon: int) -> np.ndarray:
        return _targets_for_sequences(df, origin_idx, horizon)


patchtst = _PatchTST()
register_model(patchtst)
