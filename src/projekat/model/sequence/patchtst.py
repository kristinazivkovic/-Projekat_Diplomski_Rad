"""PatchTST (Nie et al. 2023), small architecture: the 60-day sequence of
six channels is split into non-overlapping patches (length 10, stride 10,
so N = 6 patches), each patch is flattened ACROSS ALL CHANNELS and
projected to the embedding (channel mixing -- a deliberate departure from
the original's channel independence: the RV forecast is meant to see
C, J and the semivariances, which is why the model depends on grid,
estimator and jump_test), passed through a transformer encoder, and the
flattened encoder output feeds one linear head per horizon's network.

Deliberately small (2 layers, 4 heads, d_model 32) given panel-scale
training data. Hyperparameters fixed, never tuned.

TRAINING (Model.refit = "yearly"): retrained FROM SCRATCH on the first
trading day of each year, on all rows strictly before that day, pooled
over every symbol (one network per horizon). Inputs are logs, except r_d
(signed, natural units) and the jump channel ln(1+J), which is
STANDARDIZED with the mean/sd of the training rows (frozen for the year).
Early stopping: the last config.PATCHTST_EARLY_STOP_SHARE (10%) of the
training dates form the stopping set; training rows whose target reaches
into that set are purged, so the two never share a realized RV. Adam, lr
1e-3, batch 64, at most 30 epochs, patience 5, best weights restored.
MSE on the log target.

FORECASTING: every day t, the 60-day sequence ending at t (never crossing
into another symbol). The correction is the same trailing out-of-sample
one every log model gets (walkforward.py).

Stochastic -- repeated for config.PATCHTST_SEEDS (3 seeds)."""

from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from projekat.config import PATCHTST_EARLY_STOP_SHARE, PATCHTST_PATIENCE
from projekat.model.features import JUMP_CHANNEL, SEQUENCE_COLUMNS, SEQUENCE_LENGTH, log1p_jump, sequences_at
from projekat.model.registry import register_model
from projekat.model.targets import target_columns

_N_FEATURES = len(SEQUENCE_COLUMNS)  # RV_d, C_d, J_d, RS_pos, RS_neg, r_d
_PATCH_LEN = 10                      # stride = patch length (no overlap)
_D_MODEL = 32
_N_HEADS = 4
_N_LAYERS = 2
_MAX_EPOCHS = 30
_LR = 1e-3
_BATCH_SIZE = 64

# GPU when available (CUDA torch build); PROJEKAT_DEVICE=cpu forces the CPU.
# Architecture, epochs, batch size and learning rate are identical on both.
_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() and os.getenv("PROJEKAT_DEVICE", "").lower() != "cpu" else "cpu"
)


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


def _predict_net(net: _PatchTSTNet, seqs: np.ndarray) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for lo in range(0, len(seqs), 4096):
            x = torch.tensor(seqs[lo: lo + 4096], dtype=torch.float32, device=_DEVICE)
            out.append(net(x).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def _train_net(x_fit, y_fit, x_stop, y_stop, *, seed: int) -> _PatchTSTNet:
    torch.manual_seed(seed)
    net = _PatchTSTNet().to(_DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=_LR)
    loss_fn = nn.MSELoss()
    X = torch.tensor(x_fit, dtype=torch.float32, device=_DEVICE)
    y = torch.tensor(y_fit, dtype=torch.float32, device=_DEVICE)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = len(X)

    best_loss, best_state, stale = np.inf, None, 0
    for _epoch in range(_MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=generator).to(_DEVICE)
        for i in range(0, n, _BATCH_SIZE):
            idx = perm[i: i + _BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(net(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
        if len(x_stop) == 0:
            continue
        stop_loss = float(np.mean((_predict_net(net, x_stop) - y_stop) ** 2))
        if stop_loss < best_loss:
            best_loss, best_state, stale = stop_loss, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= PATCHTST_PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def _standardize_jump(seqs: np.ndarray, mean: float, sd: float) -> np.ndarray:
    seqs = seqs.copy()
    seqs[:, :, JUMP_CHANNEL] = (seqs[:, :, JUMP_CHANNEL] - mean) / sd
    return seqs


def early_stopping_split(train: pd.DataFrame, horizon: int, usable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fit rows, stopping rows) for one horizon. The stopping set is the
    last PATCHTST_EARLY_STOP_SHARE of the training DATES; a fit row whose
    target is realized on or after the first stopping date is purged, so
    fit and stopping rows never share a realized RV."""
    dates = train["date"]
    unique_dates = np.sort(dates.unique())
    if len(unique_dates) == 0:
        return np.zeros(len(train), bool), np.zeros(len(train), bool)
    cut = unique_dates[min(int(len(unique_dates) * (1 - PATCHTST_EARLY_STOP_SHARE)), len(unique_dates) - 1)]
    log_col, _, end_col = target_columns(horizon)
    base = usable & np.isfinite(train[log_col].to_numpy(dtype=float))
    stop_rows = base & (dates >= cut).to_numpy()
    fit_rows = base & (dates < cut).to_numpy() & (train[end_col] < cut).to_numpy()
    return fit_rows, stop_rows


def jump_standardization(train: pd.DataFrame) -> tuple[float, float]:
    """(mean, sd) of ln(1+J) over the training rows -- frozen for the year."""
    j = log1p_jump(train["J_d"]).to_numpy(dtype=float)
    j = j[np.isfinite(j)]
    if len(j) == 0 or j.std() == 0:
        return (float(j.mean()) if len(j) else 0.0), 1.0
    return float(j.mean()), float(j.std())


class _FittedPatchTST:
    def __init__(self, nets: dict[int, _PatchTSTNet | None], j_mean: float, j_sd: float):
        self._nets = nets
        self._j_mean = j_mean
        self._j_sd = j_sd

    def predict(self, history: pd.DataFrame, origin_idx: np.ndarray, horizons) -> dict[int, np.ndarray]:
        seqs, ok = sequences_at(history, origin_idx)
        seqs = _standardize_jump(seqs[ok], self._j_mean, self._j_sd)
        out = {}
        for h in horizons:
            pred = np.full(len(origin_idx), np.nan)
            net = self._nets.get(h)
            if net is not None and ok.any():
                pred[ok] = _predict_net(net, seqs)
            out[h] = pred
        return out


class _PatchTST:
    name = "patchtst"
    stochastic = True
    predicts_levels = False
    refit = "yearly"
    correction = "log"
    depends_on = frozenset({"grid", "estimator", "jump_test"})

    def fit(self, train: pd.DataFrame, horizons, *, seed: int | None = None) -> _FittedPatchTST:
        seed = 0 if seed is None else int(seed)
        train = train.sort_values(["symbol", "date"]).reset_index(drop=True)

        # the jump channel's standardization comes from the training rows only
        j_mean, j_sd = jump_standardization(train)
        seqs, ok = sequences_at(train, np.arange(len(train)))
        seqs = _standardize_jump(seqs, j_mean, j_sd)

        nets: dict[int, _PatchTSTNet | None] = {}
        for h in horizons:
            log_col, _, _ = target_columns(h)
            y = train[log_col].to_numpy(dtype=float)
            fit_rows, stop_rows = early_stopping_split(train, h, ok)
            if fit_rows.sum() < _BATCH_SIZE:
                nets[h] = None
                continue
            nets[h] = _train_net(seqs[fit_rows], y[fit_rows], seqs[stop_rows], y[stop_rows], seed=seed)
        return _FittedPatchTST(nets, j_mean, j_sd)


patchtst = _PatchTST()
register_model(patchtst)
