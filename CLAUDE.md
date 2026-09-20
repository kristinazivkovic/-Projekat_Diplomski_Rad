# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Code supporting a diplomski rad (thesis): **"Dizajn merenja naspram izbora modela u predviđanju realizovane volatilnosti"** — does forecast accuracy for realized volatility come more from *measurement design* choices (sampling grid, jump-robust estimator, jump test) or from *model architecture*? `docs/obrazlozenje-teme-diplomskog-rada.md` is the original research proposal (hypotheses H1-H5); it is **outdated** and not a source for design decisions or citations — the code, `config.py`'s inline rationale and this file are authoritative.

The experiment is factorial: every (measurement design) x (model) combination is evaluated across horizons and market regimes, then variance in forecast loss is decomposed by ANOVA to attribute it to model choice vs. measurement choice.

## Commands

All commands run from the repository root using `uv` (package manager + task runner; see `pyproject.toml` / `uv.lock`). The shell is Windows (PowerShell / Git Bash).

```bash
# install / sync dependencies (torch comes from the PyTorch cu128 index -- see pyproject.toml)
uv sync

# run the full test suite / one file / one test
uv run pytest
uv run pytest tests/test_jumps.py
uv run pytest tests/test_jumps.py::test_name -v
```

The full pipeline, in order (each step is resumable; re-running skips finished work):

```bash
# 1. download 1m bars for config.UNIVERSE (parallel, quota-aware; --dry-run lists what is missing)
uv run python scripts/fetch_data.py

# 2. build the panel: every symbol x all 36 designs (~40 s per symbol), plus
#    panel_diagnostics.csv / panel_exclusions.parquet / panel_corporate.csv beside it
uv run python -m projekat.build_panel --symbols <config.UNIVERSE...> --start 2010-01-01 --end 2025-12-31 \
    --grids 1m 5m 15m --estimators bpv medrv trbpv --jump-tests naive bns lm lm_fdr \
    --out data/processed/v3/panel.parquet

# combination count + measured timing (first and last forecast block per model) before a long run
uv run python scripts/run_econometric.py --panel data/processed/v3/panel.parquet --models fast --dry-run --benchmark

# 3. forecasts: N parallel shards into ONE run dir (--models fast = everything but patchtst)
#    measured: 246 combinations in ~2.5 h on 18 shards; ARFIMA is the long pole (~1.8 h per combination)
uv run python scripts/run_parallel.py --run-dir results/run_v3 --panel data/processed/v3/panel.parquet --models fast
uv run python scripts/run_parallel.py --run-dir results/run_v3 --panel data/processed/v3/panel.parquet --models patchtst

# 4. evaluation: every level 1-4 artefact into <run-dir>/report/  (~6.5 h, ~45 GB peak)
uv run python scripts/run_report.py --run-dir results/run_v3 --panel data/processed/v3/panel.parquet

# 5. WI-12 diagnostics (H5 placebo: 100 permutations; re-estimation frequency) -- re-run walk-forwards
uv run python scripts/run_diagnostics.py --panel data/processed/v3/panel.parquet --out results/run_v3/report --workers 8

# the thesis PDF (XeLaTeX + biber via MiKTeX)
cd rad; .\build.ps1
```

`build_panel` without `--out` writes `panel_mini.parquet`, and without the factor lists builds only `5m__bpv__bns` — pass both for the real panel. `src/main.py` just prints the `LSE_KEY` env var — a smoke test for `.env` loading, not a pipeline entry point.

Tests (`tests/`) use synthetic, seeded fixtures (`tests/conftest.py`) with **no network access** — they never hit the vault or FRED.

## Data source & credentials

Intraday 1-minute bars come from the **LSE vault** (`lse-data` package) via `projekat.data.vault.fetch_1m`, authenticated with `LSE_KEY` in `.env` (loaded via `python-dotenv`). Downloads are cached under `data/raw/1m/` as Parquet with a JSON provenance manifest (hash, row count, fetch time) per file; re-fetching is a no-op unless `force=True`. `data/` is gitignored — entirely regenerable from the vault + FRED (VIX).

Vault facts that shape `data/vault.py` and `scripts/fetch_data.py`:

- `/export` **silently truncates at ~2.5M rows** (valid parquet, correct manifest, data just stops), so long ranges are fetched in 8-year chunks (`fetch_1m_chunked`), each its own cache entry; `load_1m_range` stitches them.
- The quota (HTTP 429) limits export-job **creation**, not downloads. `fetch_data.py` runs chunks in parallel behind one global gate (spacing + escalating backoff shared by all workers) and persists each `job_id` under `data/raw/1m/.jobs/`, so a crash resumes the job instead of spending quota on a new one.
- `fetch_1m` deliberately bypasses `client.history()`: that writes `stocks_{SYM}_1m.parquet` regardless of range, so two chunks of one symbol would share a `.part` file and a resumed download could append another job's bytes. Each chunk gets its own temp dir under `data/raw/1m/.tmp/`.

**Universe:** `config.UNIVERSE` is **28 symbols**. Two were selected and then excluded during QA (`config.EXCLUDED_SYMBOLS`); both exclusions belong in the thesis text (`rad/poglavlja/03-metodologija.tex`):

- **MT** — vault prices mix adjusted and unadjusted levels *within the day* for 2014-03..2016-03 (fake ~23% 5-minute returns on 496 days); caught by the split-adjustment gate.
- **NVDA** — prices are back-adjusted for the 4:1 (2021) and 10:1 (2024) splits, so 2014-2016 trades at ~0.46 USD rounded to the cent: an effective tick of ~2.2%. Measured on the built panel: the only 65 days in the whole panel with `RV_d = 0` (on 5m/15m the price does not move all day), a median RV(1m)/RV(5m) of 2.2-2.5 for 2014-2016 against 1.14 panel-wide, and in evaluation **99.77% of the entire QLIKE mass** (the naive model's mean QLIKE was 14,161 with NVDA and 2.32 without). RV there measures rounding, not volatility. `qa/panel_diagnostics.py` now reports both signals (`zero_rv_days`, `rv_ratio_1m_over_5m`) so a symbol like this is visible before a run rather than after it.

## Architecture

The pipeline is two sequential layers with a **frozen schema contract** between them (`measure/schema.py`, `PANEL_SCHEMA`) — treat that file as an API boundary, not an implementation detail.

```
data/  →  measure/  →  (panel parquet)  →  model/  →  evaluation/  →  results/
```

### 1. Measurement layer (`measure/`, orchestrated by `build_panel.py`)

Per symbol: fetch 1m bars (`data/vault.py`) → clean/filter each day against the real exchange calendar (`data/clean.py`, `data/calendar.py`) → resample to a grid and compute log-returns (`measure/grids.py`) → compute realized measures and jump decomposition per day (`measure/panel.py`) → roll HAR aggregates (`measure/rolling.py`) → validate against the schema (`measure/schema.py`) → write `data/processed/*.parquet`.

**`MeasurementDesign`** (`measure/design.py`) is the experiment's design space as a first-class, hashable object: `(grid, estimator, jump_test, threshold)` → `design_id` (e.g. `5m__bpv__bns`; `trbpv` designs carry a `__t{threshold}` suffix, e.g. `5m__trbpv__bns__t3`), which becomes the ANOVA factor key downstream. `evaluation/anova.py::parse_design_id` recovers `grid`/`estimator`/`jump_test` from it (relying on `MeasurementDesign.id`'s fixed field order; it ignores the threshold suffix, which is pinned, not a factor) — use it everywhere, never split `design_id` ad hoc. (`results/store.py` has its own filename parser, which splits on `key=` boundaries for the same reason.)

**Registry pattern (critical, applies to both `measure/` and `model/`):** `measure/panel.py` resolves A2 estimators and A3 jump tests **by name only** from `measure/registry.py`'s `ESTIMATORS`/`JUMP_TESTS` dicts — it must never import a concrete estimator/jump-test module. Modules under `measure/estimators/` and `measure/jumps/` self-register via `@register_estimator` / `@register_jump_test` decorators on import. **Adding a new estimator or jump test = one new file with the decorator, zero edits to `panel.py` or the registry.** The same pattern applies in `model/`: `model/runner.py` resolves models by name from `model/registry.py`'s `MODELS` dict; `model/baselines/`, `model/econometric/`, `model/trees/`, `model/sequence/`, `model/foundation/` submodules self-register via `@register_model` (so `MODELS` is empty until those packages are imported — callers import them for registration, e.g. `model/sharded.py`).

Three experimental factors, all pinned in `config.py` with the reasoning for each fixed choice documented inline there (read it before touching any of these):

- **A1 — sampling grid**: `1m` / `5m` / `15m` (`GRIDS`). Not `10m` — no native 10m timeframe in the vault.
- **A2 — jump-robust estimator**: `bpv`, `medrv`, `trbpv` (`measure/estimators/`), selected via `ESTIMATORS`.
- **A3 — jump test**: `naive`, `bns`, `lm`, `lm_fdr` (`measure/jumps/`), selected via `JUMP_TESTS`. `lm` and `lm_fdr` need whole-history context (trailing K-window for the Lee-Mykland statistic, trailing W-day window for Benjamini-Hochberg FDR) computed up front in `panel.py::_lm_day_stats` (estimator-independent, so `build_symbol_design(..., lm_cache=)` shares it across the three estimators), which also returns the per-day `local_sigma` slice so `panel.py` can call `JUMP_TESTS["lm"]`/`JUMP_TESTS["lm_fdr"]` directly, unlike `naive`/`bns` which are purely per-day. `naive`'s sigma comes from the `robust` (jump-robust) estimator of the design, never from `np.std(r)` over the whole day — the latter would include the tested jump in its own denominator. `panel.py` passes the design's estimator NAME to every jump test (`estimator=`).
- **Significance levels** (`config.py`): BNS one-sided at `BNS_ALPHA = 0.001` (p = 1 - Phi(Z)); LM Gumbel critical value at `LM_CONFIDENCE = 0.999`; LM-FDR Benjamini-Hochberg at `LM_FDR_Q = 0.05` over the trailing W days, with the Gumbel day p-value `1 - exp(-exp(-(max|L| - C_n)/S_n))`. Consequence measured on AAPL+BAC: LM-FDR flags MORE days than LM (5m 37.8% vs 19.4%) — with ~19% of days below p = 0.001 the BH threshold at q = 0.05 rises to ~0.02; the old assumption that `lm_fdr` is the conservative variant does not hold at these levels.
- **BNS theta is per estimator**: each estimator declares `bns_theta` in its own file (`bpv`/`trbpv` 0.609 = (pi/2)^2 + pi - 5, `medrv` 0.96); `bns.py::theta_for` reads it from the registry by name and raises for an estimator without one. TPQ is unchanged.

Non-obvious invariants worth knowing before editing this layer:

- **One day set for every grid.** `data/clean.py::clean_day_with_reason` decides per day ONCE, independent of the grid: early-close sessions are dropped outright (`DROP_HALF_DAYS`; a 15m half day gives only 13 returns), the 80% relative filter (`MIN_BAR_COVERAGE`) is measured on the **1m** bars all grids are built from, and the `MIN_INTRADAY_OBS` floor is checked against the number of returns **every configured grid** yields (`_n_grid_returns`, verified equal to `len(log_returns(resample_day(...)))`). So 1m/5m/15m share exactly the same days by construction (`panel_diagnostics.csv` records it per symbol). Never apply a coverage rule per estimator in `panel.py` — that would give designs at the same grid different day sets and unbalance the factorial design. Every excluded day carries a reason (`EXCLUSION_REASONS`, plus `no_vendor_bars`), written to `<panel stem>_exclusions.parquet`. (`qa/checks.py::assert_min_intraday_obs` is a regression guard, not a live filter.)
- `LM_LOCAL_WINDOW_K` is held fixed at **4 trading sessions** across all grids (not a literature-matched constant) specifically so only sampling frequency varies with A1, not local-window length too.
- Session windows come from the real NYSE calendar (`exchange_calendars`, `data/calendar.py`), not a fixed UTC clock window — a fixed window leaks thin post-close bars into early-close sessions.
- `RS_pos + RS_neg == RV_d` is a genuine hard-asserted invariant (`qa/checks.py`); `C_d + J_d == RV_d` is not (it holds by construction and asserts nothing).
- Overnight returns are never included in RV — each day's return array is built purely from that day's own prices (see `measure/panel.py::_concat_returns_in_order`).
- The split-adjustment hard gate (`data/corporate.py::verify_split_adjustment`) flags a series only when the close-to-close return on the event day is **closer to the unadjusted drop than to zero** and within 0.15 of it. The bare 0.15 band alone flagged ordinary days around small events (spin-offs of ~5%: PFE 2020-11-17, VZ 2010-07-02 were false positives); MT 2016-03-14 is a true positive. A corporate-actions FETCH failure skips the gate for that symbol but is recorded (`<panel stem>_corporate.csv`, counted in `panel_diagnostics.csv`), never silent.

### 2. Modelling layer (`model/`)

One **combination = (model, design, seed)** = one walk-forward pass (`model/walkforward.py::run_walk_forward`) producing a forecast for every trading day and every horizon; `model/runner.py::run_one` runs one by model NAME, `run_all` is the in-memory path the tests use. The real run goes through `model/sharded.py` (`enumerate_combinations` + `run_sharded`, one parquet per combination written immediately via `results/store.py`), driven by `scripts/run_econometric.py` and, in parallel, `scripts/run_parallel.py` (N `--shard I/N` processes, one thread each, into one run dir). Model groups for `--models` are `MODEL_GROUPS` in `sharded.py` (`fast` = everything except `patchtst`). `--dry-run --benchmark` times the first and last forecast block of one combination per model and extrapolates.

**Validation scheme: expanding window, daily forecasts** (`model/walkforward.py`; the old 11-window scheme with a validation fold, `model/windows.py`, is deleted — see `docs/changelog.md`):

- A forecast is made **every trading day t** from `FIRST_FORECAST_DATE` (2014-01-01) to `EVAL_END`. **2014 is a warm-up year**: its forecasts exist (`is_warmup = True`) only to supply the first out-of-sample errors; evaluation is 2015-01-01..2025-12-31 and `store.load_forecasts` drops warm-up rows by default.
- Every trained model is **re-estimated every 21 trading days** (`REFIT_EVERY_SESSIONS`, `Model.refit = "every_21"`) on ALL rows strictly before the refit date R, starting at 2010-01-01 (expanding, never rolling). `walkforward.training_frame` also blanks every target not realized before R (`target_end_h < R`), so a row at R-1 never trains on RV from R..R+21. PatchTST is retrained from scratch yearly (`"yearly"`); zero-shot models (`naive`, `ttm_*`) never train (`"never"`).
- **Inputs/state update daily** between refits: `FittedModel.predict(history, origin_idx, horizons)` sees only rows <= the block end and must use, for origin t, only rows <= t of the same symbol — HAR regressors of day t, the MEM recursion rolled through every day, ARFIMA filtered with d/ARMA fixed, TTM's 512-day context ending at t, PatchTST's 60-day sequence ending at t.
- **No validation fold.** The back-transform correction and the Mincer-Zarnowitz coefficients for day t come from the model's OWN out-of-sample errors whose target was **realized in the trailing 252 trading days strictly before t** (`CORRECTION_WINDOW_SESSIONS`; `backtransform.trailing_corrections` / `trailing_mincer_zarnowitz`, pooled over symbols per model x design x horizon x seed). Before the first realized error the correction is neutral (a = 0, m* = 1). `finalize_forecasts` writes `log_y_hat`, `raw_y_hat`, `backtransform_correction` / `combined_correction`, `y_hat`, `mz_intercept`, `mz_slope`, `y_hat_recalibrated`, `is_warmup`; `qlike_recalibrated` is reported beside raw `qlike`, never instead of it.
- `tests/test_walkforward.py` is the lookahead guard: every refit sees only rows and realized targets before R; the correction/MZ at t ignore every error realized on or after t; and perturbing ANY panel value after t leaves every model's forecast for t unchanged.

The correction is **non-parametric**: `a = ln(mean(exp(residual)))` (Duan's smearing), `y_hat = exp(log_y_hat + a)` — not the Gaussian `0.5*sigma^2`. Correction kinds are declared per model (`Model.correction`): `"log"` (the majority), `"ratio_of_sums"` for combined forecasts (`m* = sum(y)/sum(y_raw)`, never a mean of ratios), `"none"` for level models.

Models implement the `Model`/`FittedModel` protocols (`model/protocols.py`) and declare `depends_on: frozenset[str]` — the subset of `{"grid", "estimator", "jump_test"}` their inputs actually depend on. `runner.py::_representative_designs` uses this to fit a model once per distinct combination of only the factors it depends on (e.g. a grid-only model fits 3x, not up to 36x) — both for compute and so duplicated non-independent rows don't corrupt the ANOVA.

Models (17): `naive` (`model/baselines/`), `har`, `log_har`, `har_j`, `char`, `shar`, `harq`, `har_iv`, `arfima`, `mem` (`model/econometric/`), `lightgbm`/`xgboost` (`model/trees/`), `ttm_rv`, `ttm_c`, `ttm_c_plus_j`, `log_har_ttm` (`model/foundation/`), `patchtst` (`model/sequence/`, the one stochastic model — 3 seeds, `config.PATCHTST_SEEDS`). Each declares `name`, `stochastic`, `depends_on`, `predicts_levels`, `refit` and `correction` (`model/protocols.py`); there is no separate sequence-model path — sequences are internal to PatchTST's `predict`. Model details that are easy to get wrong:

- **HARQ** interacts ln(RV_d) with sqrt(RQ) **standardized on the training rows of each refit** (`features.harq_standardization`, frozen until the next refit), so the daily coefficient is `beta_d + beta_Q * z`.
- **J in log space is `ln(1 + J)`** (`features.log1p_jump`: HAR-J, the tree union, the PatchTST channel) — 0 exactly when J = 0, never `ln(max(J, eps))`.
- **ARFIMA**: local-Whittle bandwidth m = floor(n^0.65) (`lw_bandwidth`); u = (1-L)^d (x - mu) over the full expanding history, re-integrated with (1-L)^-d over the full history (exact inverse, no truncation). **Numerical guard**: the innovations come from inverting the MA polynomial, and with a near-unit MA root that inversion can amplify over thousands of observations — measured once as a log forecast of +55,848, i.e. `exp()` overflow to inf on 126 of 713k rows. A forecast more than `_MAX_LOG_DEVIATION` (10 in log RV) outside the observed range is a numerical failure, so it becomes NaN and is counted as a missing forecast downstream, never an inf that would own every mean it enters.
- **PatchTST**: patch 10, stride 10 (6 patches), each patch flattened ACROSS all six channels (channel mixing, a deliberate departure from the original's channel independence — why it depends on all three A factors); pooled over symbols; Adam 1e-3, batch 64, <= 30 epochs, early stopping (patience 5) on the last 10% of training dates with training targets reaching into that span purged (`early_stopping_split`); the ln(1+J) channel standardized with training mean/sd. Runs on the GPU when CUDA torch is present (`PROJEKAT_DEVICE=cpu` forces CPU; tests force CPU in `conftest.py`).
- Trees use `n_jobs=1` (parallelism comes from shards); hyperparameters are fixed everywhere, never tuned.

**Deliberately distinct positivity mechanisms**, never unified:

- **Log + non-parametric correction** (`log_har`, `shar`, `harq`, `har_iv`, `char`, `har_j`, `arfima`, `naive`, trees, PatchTST, `ttm_rv`, `ttm_c`): fit in log space, back-transformed with the trailing correction. The majority mechanism.
- **NNLS directly on levels** (`har.py`, the plain "HAR" — NOT `log_har`): non-negative coefficients on non-negative regressors, so no log and no correction (`correction = "none"`).
- **Structural positivity from a multiplicative form** (`mem.py`): `mu_t = omega + alpha*RV_{t-1} + beta*mu_{t-1}` is a linear combination of non-negative terms with non-negative coefficients and omega > 0, so `mu_t > 0` — no log, no correction.
- **Zero-clipping + one combined correction** (`ttm_c_plus_j`): `max(exp(C_hat), 0) + max(J_hat, 0)` in levels, then `m* = sum(y)/sum(y_raw)`.

Models with `predicts_levels = True` (`har`, `mem`, `ttm_c_plus_j`) return `y_{t,h}` in levels from `predict()`; their `log_y_hat` column is NaN. **`har` (NNLS, levels) and `log_har` (log+correction) are two separate, permanently distinct models** — a natural-seeming "just use one HAR" simplification would erase the ability to compare whether the positivity mechanism itself, not just the feature set, affects forecast accuracy. `log_har` — not `har` — is the headline benchmark for `evaluation/losses.py::relative_qlike`.

**Foundation models (`model/foundation/`)**: `ttm_rv` (grid), `ttm_c` / `ttm_c_plus_j` (grid, estimator, jump_test), and the `log_har_ttm` hybrid (grid). All are **zero-shot, deterministic, never trained** (`refit = "never"`; `fit()` does nothing, the correction comes from the trailing errors; `stochastic = False`, no seeds). Every day's forecast uses the **512-day context ending at that day** (`_ttm_base.contexts_at`, per symbol, left-padded with the first value when shorter); the output has 192 steps and the h-day forecast is the **mean of the first h steps**; one batched inference per (symbol, day) serves all horizons (`ttm_forecast_batch`, GPU when available). `log_har_ttm` uses the `ttm_rv` component. The checkpoint is **pinned exactly** in `config.py` (`TTM_MODEL_ID` + `TTM_REVISION`, currently the `512-192-r2` branch), never "TTM" generically — different releases have different pretraining datasets, so a contamination check against an unpinned model would prove nothing. A non-`ft` branch is pinned deliberately: the frequency-prefix-tuned variants require a `freq_token` input, which would add an unjustified hyperparameter to a zero-shot path.

The **pretraining contamination check** (`model/foundation/contamination.py`) is a first-class recorded artefact: it compares the pinned checkpoint's published cutoff (`config.TTM_PRETRAIN_CUTOFF`) against the test period and writes `results/ttm_contamination_check.json` **either way**, clean or not. Both entry points run it before producing any `ttm_*`/`log_har_ttm` forecast — `runner.py::run_all`, and `scripts/run_econometric.py` (shard 0, written into the run dir; `run_report.py` copies it into the report) — no such number is reported without the artefact. Days are counted as **XNYS trading sessions**, not calendar days. When contamination is found, the overlap is quantified and the model is **kept in the analysis, labelled contaminated** — never silently dropped and never silently treated as clean. **Current result: contaminated** — 2,264 of 2,766 evaluation trading days (81.9%, 2015-01-02..2023-12-29) precede the `512-192-r2` cutoff of 2024-01-01; only 2024-2025 is clean.

`ttm_c_plus_j` and `log_har_ttm` both take **one correction on the combined forecast**, never one per component (correcting `C` and `J` separately and summing leaves the RV forecast with no controlled bias): `correction = "ratio_of_sums"`, `m* = sum(y)/sum(y_raw)` over the trailing window. A ratio of sums, not a mean of ratios: after zero-clipping `y_raw` can be near zero, and one such day would own a mean of ratios. `log_har_ttm` combines **in log space, before the correction**, with weights **fixed at 0.5/0.5 and never optimized** (an optimized combination would make the hybrid's variance share reflect search effort rather than architecture). Weighting in levels would be a different estimator and must not be substituted. Expect the jump-component forecast inside `ttm_c_plus_j` ("`ttm_j`" in the diagnostics; not a registered model) to sit near zero throughout — that is a measurement supporting the CHAR rationale, not a bug to fix.

`arfima` and `mem` are both fit **independently per symbol** (a univariate long-memory/GARCH-style model cannot be pooled across symbols), re-estimated every 21 days, with the state rolled forward daily between refits. Both declare `depends_on = {"grid"}` (their input is RV_d alone). MEM's recursion is vectorized (`lfilter`) and starts at the symbol's first row with `mu_0` = the training mean, frozen at the refit.

Targets (`model/targets.py::add_targets`): `y_{t,h}` is the **h-day average** future RV (Corsi's own HAR construction), one column triple per horizon — `log_y_target_h{h}`, `y_target_h{h}`, and `target_end_h{h}`, the DATE the target is realized (the walk-forward's causality rules key on it). Targets are modelled in log, except the jump component inside `ttm_c_plus_j`, forecast in levels because `J = 0` is a legitimate and frequent value. Every shift/window groups by `symbol` first (`add_targets`, `features.sequences_at`) — a multi-symbol frame sorted by date alone would pull a "future" RV or a sequence window from another symbol.

Seed handling for stochastic models: the **primary** reported figure averages per-seed **losses**, not forecasts — `results/tables.py::collapse_seed_losses` (one row per forecast key, run in `build_all_reports` before any table or test); `runner.py::collapse_seeds` is a documented no-op that only marks where the averaging happens. Without the collapse, a 3-seed model would enter the ANOVA/VaR as three non-independent copies of each day. A seed-averaged-forecast ensemble exists as an explicitly separate, secondary figure and must never replace the primary one. `ensemble_forecast` averages the raw `log_y_hat` (pre-back-transform prediction) and each seed's `backtransform_correction` separately across seeds, then calls `backtransform()` exactly once on the two means — never `np.log(y_hat)` on the already-back-transformed per-seed forecast, which would bake each seed's own correction back in before averaging and double-count it.

### 3. Evaluation layer (`evaluation/`)

Four levels, matching the thesis's metric structure:

- **Level 1 — accuracy**: `evaluation/losses.py` — QLIKE (primary, robust to noisy RV proxy per Patton 2011), RMSE/MAE (secondary), relative QLIKE. **Relative QLIKE is a ratio of means, never a mean of ratios**: `mean(QLIKE_model) / mean(QLIKE_log_har)` over the pooled test period. The two are not interchangeable — a mean of ratios blows up whenever a single day's benchmark loss approaches zero, which happens. The baseline is `log_har` (`RELATIVE_QLIKE_BASELINE`), not `har`. QLIKE wins on disagreement with MAE, but a reversal is flagged only when **Diebold-Mariano significant at α=0.05 on both metrics** — flagging every mean-level reversal would report rounding noise as disagreement. Use `ranking_disagreement_table(daily_losses)` on the per-day frame, computed per `(horizon, design_id)` cell; `main_result_table` no longer carries this flag because it is already aggregated and cannot compute DM significance. `mae_median_variant` (correction `a = 0`) is the required robustness partner for MAE, since QLIKE/RMSE are optimized by the conditional mean and MAE by the conditional median.
- **Level 2 — significance & decomposition**: `evaluation/dm_test.py` (Diebold-Mariano — pass the actual forecast `h` through any caller; the HAC lag count depends on it. With 36 designs × 17 models, exhaustive pairwise DM is thousands of tests, so every reported family goes through **Benjamini-Hochberg FDR control** via `benjamini_hochberg_dm(..., family=...)` — BH is already this repo's convention in `measure/jumps/lm_fdr.py`. The family must be named explicitly and is recorded in the output's `dm_family` column; `DM_FAMILIES` declares the valid ones, and `dm_against_baseline` implements the pre-declared "every model vs `log_har`" contrast set. DM, MCS, the ranking table and H4 compare models **within a `(horizon, design_id)` cell**, but grid-only models (the `log_har` baseline among them) exist only on their representative design; `results/report.py::_iter_design_cells` shares their rows with every design of their grid **for these comparisons only**, one design at a time (the 12x copy never exists in memory) — valid because `y_true` is RV-based and identical across the designs of a grid — and **never for the ANOVA**), `evaluation/mcs.py` (Model Confidence Set, alpha 0.10 — membership is **not** a ranking of point losses; `results/tables.py::mcs_membership_table` aligns the loss series on `(date, symbol)` like DM does, never positionally; `model_confidence_set(..., horizon=...)` forwards `horizon` to `diebold_mariano` as `h`), `evaluation/mincer_zarnowitz.py` (regression of realized value on forecast with HAC SEs; `results/tables.py::mincer_zarnowitz_table` reports it over the evaluation period per `(model, design, horizon)`; the RECALIBRATION itself uses only the trailing coefficients computed in the walk-forward, so recalibrated QLIKE carries no lookahead).

  **`evaluation/anova.py` is the thesis's headline result.** The dependent variable is the **daily** QLIKE loss, one row per (model, design, horizon, symbol, day) — cell means would leave no error term. Non-obvious requirements, all enforced in code and explained at length in the module docstring:

  - **Fitted separately per horizon, never pooled.** `decompose_variance_*` return `{horizon: ANOVATable}`. Horizon is in the row key but is deliberately *not* a factor: h=1/5/22 have different target scales, so pooling dumps explainable variance into the error term and deflates every partial eta-squared — including the measurement share the thesis exists to report.
  - **`symbol` is an explicit blocking factor**, for the same reason: left in the residual it inflates the error term and shrinks all reported shares.
  - **Two decompositions over two DISJOINT row sets, neither duplicating rows** (`report.py::_split_by_dependence`, by `depends_on`): (a) `decompose_variance_design_sensitive` — design-sensitive models x 36 designs; factors model, grid, estimator, jump_test, regime, symbol + grid:estimator, grid:jump_test, model:jump_test; the only set where all three A factors vary, so the primary answer to H3. (b) `decompose_variance_grid_only` — grid-only models x their 3 representative designs; factors model, grid, regime, symbol. A grid-only model is NEVER copied across the 12 estimator × jump_test combinations in either.
  - **Type III SS with sum-to-zero contrasts, asserted not assumed.** The design is unbalanced even without copied rows (regime is not spread evenly over cells; models have different numbers of valid days, e.g. PatchTST needs 60 days of history) and carries three deliberate interactions, so Type II and Type III disagree on main effects. Type III is undefined without sum-to-zero coding, so the contrasts are written into the formula (`C(x, Sum)`).
  - **`regime` is a blocking factor like `symbol`**, not an experimental axis: one label per (symbol, day), from 5m RV_m, identical for every design (see stratification below), so it cannot absorb A1 variance.
  - **Partial eta-squared is reported as a descriptive variance share.** Designs derive from the same raw 1m bars, so 1m and 5m RV for the same day are highly correlated and the F-test's independence assumption does not hold. Any reported F/p must be read against **two-way (day × symbol) cluster-robust SEs** (Cameron-Gelbach-Miller, both dimensions simultaneously), returned in `ANOVATable.clustered`. `block_bootstrap_variance_shares` is the optional alternative inference path, with **block length max(h, 5)** (`anova.block_length`) — a 1-day block at h = 1 would be an i.i.d. bootstrap ignoring the persistence of losses.
  - **Skewness robustness is required, not optional.** `robustness_table` runs the same decomposition four ways side by side — raw daily QLIKE (primary), `log(QLIKE)`, within-day ranks across models, and raw with the top 1% of loss days dropped (the direct answer to "would this hold without March 2020?"). Stable shares across all four is a strong robustness claim; instability must be visible *before* the text is written, which is why this is a first-class table and not a buried diagnostic.
  - The two decompositions are returned **separately and never aggregated** into one headline number — that remains an open question with the advisor. `aggregate_headline_share` exists as a clearly marked hook that raises `NotImplementedError` so the decision has one obvious implementation site.
  - Interaction terms `grid:estimator`, `grid:jump_test`, `model:jump_test` are **structural, not incidental** — microstructure noise makes `RV - BPV` shift systematically with A1. Read the module docstring before touching them.
- **Level 3 — economic evaluation (VaR)**: `evaluation/var.py` — Kupiec, Christoffersen, Dynamic Quantile tests, Lopez loss. **Only h = 1, and the forecast made at t is compared with `r_d` of t+1 of the same symbol** (`r_d_next`, written by the walk-forward only when the symbol's next row is the next session). Comparing with `r_d(t)` — the old code did — tests the forecast against a return already known, whose RV is a model input. `r_d` is intraday (last minus first grid price of the day), never close-to-close, since RV excludes the overnight gap. The backtest pools the whole evaluation period and all symbols (a single 252-day window gives Kupiec no power), but Christoffersen and DQ build transitions/lags **within each symbol** (`groups=`), never across a symbol boundary. Zero breaches is a **rejection** (over-conservative VaR), not a pass: `kupiec_test` uses `xlogy`; `dynamic_quantile_test` returns NaN when the regression is rank-deficient. **H4**: `report.py::h4_kendall_table` — Kendall's tau between the model ranking by mean QLIKE and by Lopez loss, per design at h = 1 plus a pooled row.
- **Level 4 — jump-detection characterization**: `qa/checks.py` (Jaccard overlap between jump-test variants, jump-day share, placebo-test support for H5). No classification metrics (precision/recall) — there is no ground truth for what a "real" jump is. QLIKE is undefined at `J_d == 0` (division by zero/log of zero) — never apply `losses.py::qlike` to the jump component series; it is only ever computed on `(y_true, y_hat)`, the RV forecast, throughout the codebase. `assert_semivariance_identity` and `assert_min_intraday_obs` now actually run against the real built panel in `build_panel.py::main`, not just in tests; `corporate.verify_split_adjustment` similarly runs once per symbol in `build_panel_for_symbol` (non-fatal if the corporate-actions fetch itself fails, but a real `UnadjustedDataError` propagates and stops the build).

`evaluation/stratify.py` splits results by regime (calm/turbulent), jump-day presence, and horizon — averaging across the whole sample can hide a model winning in calm periods and losing in turbulent ones. **Regime is ONE label per (symbol, day), identical for every design**: `compute_regime_labels` compares the **5m** RV_m (`REGIME_GRID`) with its median (`REGIME_QUANTILE`) over the first training period (all rows before `FIRST_FORECAST_DATE`), computed once per run by shard 0 into `<run-dir>/regime_labels.parquet` + `regime_boundary.json`, and merged by `(symbol, date)` in `tables.attach_strata`. Tagging each row by its own RV_m would make 1m days "turbulent" more often than 15m days (noise-inflated RV) and leak A1 variance into the regime factor.

**WI-12 diagnostics** (`evaluation/diagnostics.py`, `scripts/run_diagnostics.py`, outside the main report): H5 placebo — `har_j` and `char` on 5m/bpv, h = 1, rerun on `N_PERMUTATIONS = 100` panels with the jump flag and J_d permuted within symbol (C_d and the J/C HAR aggregates recomputed from the permuted J_d), reporting the real dQLIKE vs `log_har` against the permutation distribution with p = (1 + #{perm <= real})/(1 + B); re-estimation frequency — the same walk-forward refit every 252/21/1 sessions. `_eval_qlike` applies the same scoreable-row filter as the report (`tables._scoreable`): without it a single zero-variance day turns every permutation's mean into NaN and the whole placebo into a constant p-value.

### 4. Results layer (`results/`, `scripts/run_report.py`)

- `results/store.py` — one run dir per run (`results/<run_id>/`): `tables/` (one parquet per (model, design, seed) = one walk-forward, every horizon and every day inside, written atomically), `index.parquet`, `manifest.json` (incl. the scheme constants), `failures.jsonl` (every failed combination is recorded, never silently dropped), `regime_labels.parquet`, `ttm_contamination_check.json`. Parallel shards **never write the index** (read-modify-write with no lock) and resume from table filenames instead (`completed_keys_from_files`); `run_report.py` rebuilds the index before loading. `store.load_forecasts` reads with pyarrow filters (one horizon, evaluation rows only, selected columns).
- `results/report.py::build_all_reports` is the **single caller** of every level 1-4 evaluation function; a new evaluation function without an entry there is dead code. It drops warm-up rows itself too. Daily losses are written as parquet, the rest as CSV.
- `scripts/run_report.py` builds the report **one horizon at a time** and concatenates the CSVs (an empty per-horizon CSV is a legitimate "nothing found"); the one cross-horizon step is re-applying Benjamini-Hochberg to the concatenated DM table, since the family is every reported comparison.
- `results/tables.py::attach_losses` scores only rows that CAN be scored and `unscoreable_rows_table` writes the rest as its own artefact (`unscoreable_rows.csv`), counted per (model, design, horizon, reason): `no_realized_target` (the last h days of the sample), `zero_target` (QLIKE is undefined at `y_true = 0`), `missing_forecast` (NaN) and `non_positive_or_infinite_forecast` (a clip to zero or an overflow). Left in, one such row turns a whole cell's mean into inf or NaN; dropped silently, the sample would shrink invisibly. A recalibrated forecast <= 0 gets a NaN `qlike_recalibrated`.
- `scripts/run_report.py --exclude-symbols` and `scripts/run_diagnostics.py --exclude-symbols` leave a symbol out of the EVALUATION only (recorded in `excluded_symbols.json`). This is a stopgap for a symbol excluded after its forecasts were produced: the cross-sectional models (HAR family, trees) were still fitted on it, so the clean fix is to rebuild the panel without it and re-run.

## Conventions to preserve

- **Never bypass the registries.** `measure/panel.py` and `model/runner.py` must resolve estimators/jump-tests/models by string name only; a new implementation is always "one new file with a decorator," never an edit to the resolver.
- **`measure/schema.py` (`PANEL_SCHEMA`) is the frozen contract** between the measurement and modelling layers — column names, dtypes, and units are fixed there. Treat a change to it as a breaking change requiring updates on both sides of the boundary.
- Fixed design constants in `config.py` (grids, K, W, TrBPV threshold, QA thresholds) carry inline rationale for *why* each was pinned rather than made variable — read the comment before changing a value, since several were pinned specifically to keep the factorial design balanced for ANOVA.
- Hard assertions vs. counted diagnostics in `qa/checks.py` are a deliberate split (see its module docstring) — a new QA check should default to a diagnostic unless it can never legitimately fire on real data.
- The code comments and docstrings are mixed Serbian (Latin script) and English; the thesis (`rad/`) is Serbian Cyrillic. Keep each file in the language it is already written in.

## Current state (2026-09-20)

**Done.** The validation scheme was replaced (expanding window, daily forecasts, no validation fold), the measurement layer revised (ln(1+J), per-estimator BNS theta, pinned significance levels, half days dropped from every grid, panel diagnostics) and the evaluation corrected (VaR against the next day's return at h = 1 only, regime labels from one grid, H4 Kendall tau, H5 placebo with 100 permutations, block length max(h, 5), contamination in trading days). `uv run pytest`: 222 passed. Details per item in `docs/changelog.md`.

**Runs on disk.**

- `data/processed/v2/panel.parquet` + `results/run_v2/` — 29 symbols, 246 combinations, full report in `results/run_v2/report/`. **Superseded: do not report these numbers.** NVDA carries 99.77% of the QLIKE mass (see the universe section), which drives every level-1 mean and flattens the raw-QLIKE ANOVA to ~0 for every factor. The `log` and `within_day_rank` robustness variants in that run are the only ones that still show structure (h = 1, design-sensitive: model 0.053 / 0.118, grid 0.019 / 0.000).
- The run is still useful as a check that the whole chain completes: 246/246 tables, zero failures, report 6.7 h, diagnostics 22 min.
- **`data/processed/v2/panel.parquet` no longer satisfies `PANEL_SCHEMA` and must not be reused.** `IV_d` (the design's jump-robust estimator value) was added to the schema so the C/J decomposition is reproducible from the panel alone — it is what `evaluation/diagnostics.py::h5_placebo_panel` needs to rebuild `J = 1{flag} * max(RV_d - IV_d, 0)` from a permuted indicator. Every v2 panel predates that column, so `validate_panel` rejects it and the H5 placebo raises instead of silently running on a different construction. Rebuild the panel (v3) rather than patching a column onto the old file.

**Pending — the clean re-run on 28 symbols (no NVDA).** Panel `data/processed/v3/panel.parquet`, forecasts + report in `results/run_v3/`; ~20 min panel, ~2.5 h forecasts (18 shards), ~6.5 h report, ~25 min diagnostics. The commands are in "Commands" above with the v3 paths. Expected level-1 QLIKE at h = 1, read off the v2 daily losses with NVDA removed: `har` 0.192, `xgboost` 0.190, `lightgbm` 0.192, `log_har` 0.210, `naive` 2.32.

**Still open after that.**

- **PatchTST has never been run** (`--models patchtst`, 36 designs x 3 seeds, yearly refits): benchmark it first (`--dry-run --benchmark`), it is the one model whose cost is not yet measured under the new scheme.
- **LM-FDR is more liberal than LM at the pinned levels** (5m: 39.0% vs 20.2% of days flagged; 1m: 73.6% vs 47.3%), because BH at q = 0.05 over a window where ~19% of days sit below p = 0.001 raises the threshold to ~0.02. `lm_fdr` being the conservative variant is no longer true and the thesis text must not claim it. TrBPV+BNS at 1m flags ~80% of days.
- **The report takes ~6.5 h**, dominated by the Type III ANOVA and the four-variant robustness table on ~17M rows per horizon, and peaks near the machine's commit limit. A chunked normal-equations ANOVA would fix both; not started.
- `aggregate_headline_share` still raises `NotImplementedError` — whether the two decompositions may be combined into one number is the open question with the advisor.

## History

The dated record of correctness fixes (what was wrong, how it was found, how it was verified) is in `docs/changelog.md`. This file describes the current state; where the two differ, this file wins.
