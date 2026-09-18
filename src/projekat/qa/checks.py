"""Quality assurance for the assembled panel: hard assertions that must
never fire on real data, counted diagnostics that are reported but never
fatal, and the Level-4 jump-detection characterization.

The split between "hard assertion" and "diagnostic" below is deliberate and
each choice is backed by a measurement -- see each function's docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from projekat.config import (
    JUMP_DAY_SHARE_BOUNDS,
    MIN_A2_A3_DIVERGENCE,
    MIN_INTRADAY_OBS,
    RQ_HALF_DAY_FLAG_PERCENTILE,
)


class QAFailure(RuntimeError):
    """A hard assertion failed. Distinguished from ordinary exceptions so
    callers can tell a real measurement-layer defect apart from a
    diagnostic worth reporting but not fatal."""


@dataclass
class QAReport:
    hard_failures: list[str] = field(default_factory=list)
    diagnostics: dict[str, float] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if self.hard_failures:
            raise QAFailure("; ".join(self.hard_failures))


# ── hard assertions ──────────────────────────────────────────────────────


def assert_semivariance_identity(df: pd.DataFrame, *, tolerance: float = 1e-9) -> None:
    """RS_pos + RS_neg == RV_d, exactly, by construction of the estimators.
    This is a genuine invariant (unlike C_d + J_d == RV_d, which holds by
    construction of the decomposition formula and tests nothing)."""
    diff = (df["RS_pos"] + df["RS_neg"] - df["RV_d"]).abs()
    bad = diff > tolerance
    if bad.any():
        raise QAFailure(f"RS_pos + RS_neg != RV_d on {bad.sum()} rows (max diff {diff.max():.2e})")


def assert_j_zero_when_not_significant(df: pd.DataFrame, *, tolerance: float = 1e-12) -> None:
    """One-directional implication: test not significant => J == 0.

    The converse (J == 0 => test not significant) does NOT hold and must
    NEVER be asserted: when the test IS significant but the estimator
    exceeds RV that day (max(RV-est, 0) clamps to zero), J is legitimately
    zero despite a significant test. Verified this is common, not an edge
    case: BPV exceeds RV on 42.9% of days (AAPL, 5m, 2015). A biconditional
    assertion here would halt the pipeline on the first real symbol.
    """
    not_significant = ~df["jump_flag"].astype(bool)
    bad = not_significant & (df["J_d"].abs() > tolerance)
    if bad.any():
        raise QAFailure(f"J_d != 0 on {bad.sum()} days where jump_flag is False")


def assert_no_split_date_unadjusted() -> None:
    """The split-adjustment hard gate itself lives in
    corporate.verify_split_adjustment (which raises UnadjustedDataError
    directly) and is invoked from build_panel.py's main pipeline. This is
    a no-arg audit-trail marker only -- it does not re-check anything -- so
    QAReport has one place recording that the check ran."""
    return None


def assert_bns_jump_share_in_range(jump_day_share: float, symbol: str, design_id: str) -> None:
    """Hard-asserted for `bns` ONLY. Other A3 variants are deliberately
    more/less conservative by design (naive is a stated quality floor and
    may exceed 20%; lm_fdr is conservative by construction and may sit
    near 3%) -- asserting the 5-15% band on every variant would suppress
    exactly the variation Factor A3 exists to create. Call this only when
    design.jump_test == "bns"."""
    lo, hi = JUMP_DAY_SHARE_BOUNDS
    if not (lo <= jump_day_share <= hi):
        raise QAFailure(
            f"{symbol}/{design_id}: bns jump-day share {jump_day_share:.1%} outside [{lo:.0%},{hi:.0%}]"
        )


def assert_bar_counts_match_calendar(actual_counts: pd.Series, expected_counts: pd.Series) -> None:
    mismatched = actual_counts.gt(expected_counts * 1.05)  # small tolerance for rounding
    if mismatched.any():
        raise QAFailure(f"{mismatched.sum()} days exceed their calendar-expected bar count by >5%")


def assert_min_intraday_obs(n_obs: pd.Series, *, context: str = "") -> None:
    """Regression check on the GRID-WIDE floor enforced upstream in
    data/clean.py's has_sufficient_coverage -- a day-grid combination
    with fewer than MIN_INTRADAY_OBS (20) usable returns is dropped for
    the ENTIRE grid before it ever reaches any A2 estimator, not
    filtered per-estimator downstream (an earlier version applied the
    floor inside panel.py per estimator call, which left bpv-based
    designs with more usable days than medrv/trbpv-based designs at the
    same grid -- an unbalanced factorial design that corrupts the
    per-factor ANOVA variance shares). This assertion should therefore
    never fire on any design once the upstream floor is working: it
    exists to catch a regression, not to filter live data."""
    too_thin = n_obs < MIN_INTRADAY_OBS
    if too_thin.any():
        raise QAFailure(
            f"{context}: {too_thin.sum()} day(s) below MIN_INTRADAY_OBS={MIN_INTRADAY_OBS} "
            f"reached the panel -- the grid-wide floor in data/clean.py did not catch them "
            f"(min n_obs seen: {int(n_obs.min())})"
        )


# ── counted diagnostics (never fatal) ───────────────────────────────────


def diagnostic_negative_rv_minus_bpv_share(rv: pd.Series, bpv: pd.Series) -> float:
    """Expected ~50% under no jumps (RV - BPV approximately symmetric
    noise around zero); measured 42.9% on AAPL/5m/2015. A shortfall below
    50% is attributable to the jump contribution pushing the difference
    upward -- this is understanding, not a defect, and must never be a
    hard assertion (see bpv.py's docstring)."""
    return float(((rv - bpv) < 0).mean())


def diagnostic_significant_but_nonpositive_gap_share(df: pd.DataFrame, robust_col: str) -> float:
    """Share of days where the jump test IS significant but RV - est <= 0
    (so J clamps to zero despite significance). Measures how often A3
    flags a day A2 doesn't corroborate -- a direct read on disagreement
    between the two factors."""
    significant = df["jump_flag"].astype(bool)
    gap = df["RV_d"] - df[robust_col]
    return float((significant & (gap <= 0)).mean()) if significant.any() else 0.0


def diagnostic_jump_day_share(df: pd.DataFrame) -> float:
    return float(df["jump_flag"].mean())


def diagnostic_rq_half_vs_full(df: pd.DataFrame) -> dict[str, float]:
    """Half-day vs full-day RQ distribution, per the plan's measured
    finding that half-day RQ runs 0.05-0.12x the full-day median (a fourth
    moment is unstable at the ~14-42 observations a half-day carries at
    15m/5m grids -- 15m half-days, n=14, are excluded outright by
    MIN_INTRADAY_OBS before reaching this diagnostic). Days that DO reach
    here are flagged, never dropped."""
    full = df.loc[~df["is_half_day"], "RQ"]
    half = df.loc[df["is_half_day"], "RQ"]
    if len(full) == 0 or len(half) == 0:
        return {"full_median": float("nan"), "half_median": float("nan"), "ratio": float("nan")}
    full_median = float(full.median())
    half_median = float(half.median())
    return {
        "full_median": full_median,
        "half_median": half_median,
        "ratio": half_median / full_median if full_median else float("nan"),
    }


def flag_rq_outlier_days(df: pd.DataFrame) -> pd.Series:
    """Days whose RQ exceeds the 99th percentile of the FULL-day
    distribution -- flagged for review, never dropped (dropping half-days
    would undo the gain of retaining them with their correct bar count)."""
    full_rq = df.loc[~df["is_half_day"], "RQ"]
    if len(full_rq) == 0:
        return pd.Series(False, index=df.index)
    threshold = full_rq.quantile(RQ_HALF_DAY_FLAG_PERCENTILE)
    return df["RQ"] > threshold


def diagnostic_ordinary_jump_on_split_dates(jump_days: set[str], split_dates: list[str]) -> list[str]:
    return [d for d in split_dates if d in jump_days]


def diagnostic_a2_a3_divergence(jump_flags_a: pd.Series, jump_flags_b: pd.Series) -> float:
    """Share of days where two A3 variants (holding grid/estimator fixed)
    disagree. Must exceed MIN_A2_A3_DIVERGENCE (5%) or Factor A3 has
    nothing to measure -- report alongside the Jaccard matrix."""
    return float((jump_flags_a.astype(bool) != jump_flags_b.astype(bool)).mean())


def jaccard_index(flags_a: pd.Series, flags_b: pd.Series) -> float:
    a, b = flags_a.astype(bool), flags_b.astype(bool)
    union = (a | b).sum()
    if union == 0:
        return float("nan")
    return float((a & b).sum() / union)


def jaccard_matrix(jump_flags_by_variant: dict[str, pd.Series]) -> pd.DataFrame:
    names = list(jump_flags_by_variant)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            out.loc[a, b] = jaccard_index(jump_flags_by_variant[a], jump_flags_by_variant[b])
    return out


def cohens_kappa(flags_a: pd.Series, flags_b: pd.Series) -> float:
    """Agreement between two A3 variants CORRECTED FOR CHANCE.

    Jaccard counts only days at least one variant flagged, so it ignores
    the (large) set of days both agree are non-jumps; two variants that
    each flag ~5% of days at random still score a non-trivial Jaccard.
    Kappa = (p_observed - p_chance) / (1 - p_chance) subtracts exactly the
    agreement expected from their marginal rates, which is the right
    question for Factor A3: do these tests agree ABOUT THE SAME DAYS, or
    just flag similarly many days? Report both -- they answer different
    questions and Jaccard alone overstates agreement.

    NOT precision/recall: there is no ground truth for a 'real' jump, so
    neither variant can be treated as the label."""
    a, b = flags_a.astype(bool).to_numpy(), flags_b.astype(bool).to_numpy()
    n = len(a)
    if n == 0:
        return float("nan")

    p_observed = float((a == b).mean())
    p_chance = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    if p_chance >= 1.0:
        # both variants constant and identical -- agreement is total but
        # chance-corrected agreement is undefined (0/0)
        return float("nan")
    return (p_observed - p_chance) / (1.0 - p_chance)


def cohens_kappa_matrix(jump_flags_by_variant: dict[str, pd.Series]) -> pd.DataFrame:
    """Chance-corrected companion to jaccard_matrix -- report side by side."""
    names = list(jump_flags_by_variant)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            out.loc[a, b] = cohens_kappa(jump_flags_by_variant[a], jump_flags_by_variant[b])
    return out


# ── jump-component diagnostics for ttm_c_plus_j ─────────────────────────


def jump_component_diagnostics(
    j_true: pd.Series | np.ndarray,
    j_hat: pd.Series | np.ndarray,
    *,
    trailing_window: int = 22,
) -> dict[str, float]:
    """Diagnostics for the J-component forecast of ttm_c_plus_j.

    MSE alone conflates two different abilities -- predicting WHETHER a
    jump occurs and predicting HOW BIG it is -- because J = 0 on most
    days, so a forecaster that simply predicts zero everywhere scores
    well on MSE while having no skill at jump size at all. This reports
    the pieces separately:

      mse                    raw MSE of J_hat against J_true
      mse_vs_zero            MSE of the trivial J_hat = 0 reference
      mse_vs_trailing_mean   MSE of a trailing-mean reference
      skill_vs_zero          1 - mse/mse_vs_zero (<=0 means no better than zero)
      skill_vs_trailing      1 - mse/mse_vs_trailing_mean
      clipping_rate          share of days the zero-clip actually bound
                             (raw J_hat < 0), i.e. how often the model
                             wanted to predict a negative jump
      rank_corr_on_jump_days Spearman rank correlation of J_hat vs J_true
                             computed ONLY on days where J_true > 0 --
                             this is the jump-SIZE question, isolated from
                             the jump-OCCURRENCE question that dominates
                             every whole-sample metric
      n_jump_days            how many days that correlation is based on

    Expect ttm_j to sit near zero throughout, giving skill_vs_zero near 0
    and a weak rank correlation. That is a MEASUREMENT supporting the CHAR
    rationale (jumps are near-unpredictable from their own history), not a
    bug to fix."""
    j_true = np.asarray(j_true, dtype=float)
    j_hat_raw = np.asarray(j_hat, dtype=float)
    j_hat = np.clip(j_hat_raw, 0.0, None)

    valid = ~np.isnan(j_true) & ~np.isnan(j_hat)
    j_true, j_hat, j_hat_raw = j_true[valid], j_hat[valid], j_hat_raw[valid]
    if len(j_true) == 0:
        return {"n_obs": 0}

    mse = float(np.mean((j_true - j_hat) ** 2))
    mse_zero = float(np.mean(j_true**2))

    trailing = pd.Series(j_true).shift(1).rolling(trailing_window, min_periods=1).mean().to_numpy()
    trailing = np.nan_to_num(trailing, nan=0.0)
    mse_trailing = float(np.mean((j_true - trailing) ** 2))

    jump_days = j_true > 0
    n_jump_days = int(jump_days.sum())
    if n_jump_days >= 3 and np.std(j_hat[jump_days]) > 0:
        from scipy import stats as _stats

        rank_corr = float(_stats.spearmanr(j_true[jump_days], j_hat[jump_days]).statistic)
    else:
        rank_corr = float("nan")

    return {
        "n_obs": len(j_true),
        "mse": mse,
        "mse_vs_zero": mse_zero,
        "mse_vs_trailing_mean": mse_trailing,
        "skill_vs_zero": float(1 - mse / mse_zero) if mse_zero > 0 else float("nan"),
        "skill_vs_trailing": float(1 - mse / mse_trailing) if mse_trailing > 0 else float("nan"),
        "clipping_rate": float((j_hat_raw < 0).mean()),
        "rank_corr_on_jump_days": rank_corr,
        "n_jump_days": n_jump_days,
    }


def level4_summary(df: pd.DataFrame) -> dict[str, float]:
    jump_days = df.loc[df["jump_flag"].astype(bool)]
    mean_jump_size = float(jump_days["J_d"].mean()) if len(jump_days) else float("nan")
    j_share_of_rv = float(df["J_d"].sum() / df["RV_d"].sum()) if df["RV_d"].sum() else float("nan")
    return {
        "jump_day_share": diagnostic_jump_day_share(df),
        "mean_jump_size": mean_jump_size,
        "j_share_of_rv": j_share_of_rv,
    }
