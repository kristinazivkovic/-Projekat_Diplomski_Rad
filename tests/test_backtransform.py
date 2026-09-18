"""Non-parametric Jensen back-transform correction: on Gaussian log-residuals
it must recover the known sigma^2/2 value (the case where the Gaussian
formula happens to be right); on a skewed residual sample it must NOT match
sigma^2/2, since that's exactly the case where assuming normality would
silently mis-correct."""

from __future__ import annotations

import numpy as np
import pytest

from projekat.model.backtransform import backtransform, residual_correction


def test_correction_recovers_sigma2_over_2_for_gaussian_residuals():
    """residual = y_true_log - y_pred_log ~ N(0, sigma^2) => E[exp(residual)]
    = exp(sigma^2/2), so the non-parametric correction ln(mean(exp(resid)))
    should converge to sigma^2/2 for a large Gaussian sample."""
    rng = np.random.default_rng(0)
    sigma = 0.3
    n = 200_000
    resid = rng.normal(0.0, sigma, n)
    y_pred_log = np.zeros(n)
    y_true_log = y_pred_log + resid

    correction = residual_correction(y_true_log, y_pred_log)
    assert correction == pytest.approx(sigma**2 / 2, rel=0.02)


def test_correction_diverges_from_gaussian_formula_for_skewed_residuals():
    """A right-skewed residual distribution (e.g. log-residuals with a fat
    right tail) has the same variance as a comparison Gaussian sample but a
    different E[exp(residual)] -- the non-parametric correction must track
    the true mean(exp(residual)), not the Gaussian sigma^2/2 approximation,
    so the two should disagree by more than sampling noise can explain."""
    rng = np.random.default_rng(1)
    n = 200_000

    # right-skewed: exponential residuals, shifted to mean zero. Variance is
    # known exactly (scale^2 for a standard exponential shifted to mean 0).
    scale = 0.5
    resid = rng.exponential(scale, n) - scale
    sigma2 = scale**2  # Var(Exponential(scale)) = scale^2

    y_pred_log = np.zeros(n)
    y_true_log = resid

    correction = residual_correction(y_true_log, y_pred_log)
    gaussian_formula_value = sigma2 / 2

    # the non-parametric correction must differ meaningfully from the
    # Gaussian sigma^2/2 formula on this skewed sample
    assert abs(correction - gaussian_formula_value) > 0.03


def test_backtransform_applies_additive_correction_in_log_space():
    log_y_hat = np.array([0.0, 1.0, 2.0])
    correction = 0.5
    y_hat = backtransform(log_y_hat, correction)
    assert y_hat == pytest.approx(np.exp(log_y_hat + correction))


def test_zero_correction_is_a_no_op():
    log_y_hat = np.array([0.0, 1.0, -1.0])
    assert backtransform(log_y_hat, 0.0) == pytest.approx(np.exp(log_y_hat))
