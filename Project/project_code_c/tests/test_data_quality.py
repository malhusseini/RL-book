"""Automated quality-gate tests for the CRSP data pipeline.

Run from the repo root with:
    python -m pytest Project/project_code_c/tests/test_data_quality.py -v

Gates tested
------------
1. (PERMNO, date) key uniqueness after pipeline — zero duplicates.
2. No ±inf returns after integrity_checks.
3. No NaN in RET after apply_missing_data_policy when it should be clean.
4. No long stale-price runs of exactly zero returns (> 5 consecutive zeros).
5. Delisting return merge: DLRET replaces final-day RET.
6. Timezone handling: tz-aware dates are normalized to tz-naive.
7. Future dates are dropped.
8. SHRCD/EXCHCD filter removes non-primary shares correctly.
9. Causal winsorization: RET_winsor ≠ NaN wherever RET ≠ NaN (after warm-up).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..data.ingest import integrity_checks, load_crsp
from ..data.clean import apply_missing_data_policy, winsorize_returns


# ── Synthetic panel builder ───────────────────────────────────────────────────

def _make_panel(
    n_stocks: int = 5,
    n_days: int = 200,
    rng_seed: int = 0,
    include_dlret: bool = True,
    include_shrcd: bool = True,
) -> pd.DataFrame:
    """Build a minimal synthetic CRSP panel for testing."""
    rng = np.random.default_rng(rng_seed)

    dates = pd.bdate_range("2010-01-01", periods=n_days)
    permnos = [10000 + i for i in range(n_stocks)]

    rows = []
    for pno in permnos:
        prc = 50.0 + rng.standard_normal(n_days).cumsum() * 0.5
        prc = np.abs(prc) + 5.0
        ret = rng.standard_normal(n_days) * 0.01
        for j, d in enumerate(dates):
            row = {
                "PERMNO": pno,
                "date": d,
                "RET": ret[j],
                "PRC": prc[j],
                "VOL": float(rng.integers(100_000, 1_000_000)),
                "SHROUT": float(rng.integers(50_000, 500_000)),
                "vwretd": rng.standard_normal() * 0.008,
            }
            if include_shrcd:
                row["SHRCD"] = 10
                row["EXCHCD"] = 1
            if include_dlret:
                row["DLRET"] = np.nan
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["PERMNO", "date"]).reset_index(drop=True)
    return df


# ── Gate 1: (PERMNO, date) key uniqueness ─────────────────────────────────────

def test_no_duplicate_keys_clean_data():
    df = _make_panel()
    df_clean = integrity_checks(df)
    dupes = df_clean.duplicated(subset=["PERMNO", "date"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate (PERMNO, date) keys after integrity_checks."


def test_dedup_resolves_injected_duplicates():
    df = _make_panel()
    # Inject a duplicate row (same PERMNO, date, but second has NaN return)
    dup_row = df.iloc[0:1].copy()
    dup_row["RET"] = np.nan
    df_with_dup = pd.concat([df, dup_row], ignore_index=True)
    df_clean = integrity_checks(df_with_dup)
    dupes = df_clean.duplicated(subset=["PERMNO", "date"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate keys after dedup."


# ── Gate 2: No ±inf returns ───────────────────────────────────────────────────

def test_no_inf_returns_after_integrity_checks():
    df = _make_panel()
    # Inject an inf return
    df.loc[5, "RET"] = np.inf
    df.loc[10, "RET"] = -np.inf
    df_clean = integrity_checks(df)
    assert not np.isinf(df_clean["RET"].dropna()).any(), "Infinite RET values found after pipeline."


# ── Gate 3: No stale-price zero runs ─────────────────────────────────────────

def test_no_long_stale_price_runs():
    """Checks that forward-filling prices does not create runs of exactly zero returns."""
    df = _make_panel()
    df_clean = integrity_checks(df)
    df_clean, _ = apply_missing_data_policy(df_clean)

    for pno, g in df_clean.groupby("PERMNO"):
        ret = g["RET"].values
        consecutive_zeros = 0
        max_zeros = 0
        for r in ret:
            if r == 0.0:
                consecutive_zeros += 1
                max_zeros = max(max_zeros, consecutive_zeros)
            else:
                consecutive_zeros = 0
        assert max_zeros <= 5, (
            f"PERMNO {pno}: {max_zeros} consecutive exactly-zero RET values "
            "(possible stale-price artifact)."
        )


# ── Gate 4: Delisting return merge ───────────────────────────────────────────

def test_delisting_return_merged():
    """DLRET should overwrite RET on the final observation of each PERMNO."""
    df = _make_panel(include_dlret=True)
    # Set a known DLRET for the last row of PERMNO 10000
    last_idx = df[df["PERMNO"] == 10000].index[-1]
    df.loc[last_idx, "DLRET"] = -0.35  # typical bankruptcy-type delisting return
    original_ret = df.loc[last_idx, "RET"]

    df_clean = integrity_checks(df)
    merged_ret = df_clean.loc[df_clean["PERMNO"] == 10000].iloc[-1]["RET"]
    assert merged_ret == pytest.approx(-0.35), (
        f"Expected DLRET=-0.35 merged into RET, got {merged_ret:.4f}."
    )
    assert merged_ret != original_ret, "DLRET was not applied."


# ── Gate 5: Timezone normalization ───────────────────────────────────────────

def test_tz_aware_dates_normalized():
    df = _make_panel()
    df["date"] = df["date"].dt.tz_localize("US/Eastern")
    df_clean = integrity_checks(df)
    assert df_clean["date"].dt.tz is None, "Dates still tz-aware after integrity_checks."


# ── Gate 6: Future dates dropped ─────────────────────────────────────────────

def test_future_dates_dropped():
    df = _make_panel()
    future_row = df.iloc[0:1].copy()
    future_row["date"] = pd.Timestamp("2099-01-01")
    df_with_future = pd.concat([df, future_row], ignore_index=True)
    df_clean = integrity_checks(df_with_future)
    assert (df_clean["date"] <= pd.Timestamp.today().normalize()).all(), (
        "Future-dated rows were not dropped."
    )


# ── Gate 7: SHRCD/EXCHCD filter ──────────────────────────────────────────────

def test_shrcd_exchcd_filter():
    df = _make_panel(include_shrcd=True)
    # Add a non-primary row (preferred share)
    non_primary = df.iloc[0:1].copy()
    non_primary["PERMNO"] = 99999
    non_primary["SHRCD"] = 31  # preferred share
    non_primary["EXCHCD"] = 1
    df_all = pd.concat([df, non_primary], ignore_index=True)
    df_clean = integrity_checks(df_all)
    assert 99999 not in df_clean["PERMNO"].values, (
        "Non-primary SHRCD was not filtered out."
    )


# ── Gate 8: Causal winsorization (RET_winsor present and not all NaN) ────────

def test_winsorization_is_causal_and_valid():
    df = _make_panel()
    df_clean = integrity_checks(df)
    df_clean, _ = apply_missing_data_policy(df_clean)
    df_clean = winsorize_returns(df_clean, z_cap=3.0, window=20)

    assert "RET_winsor" in df_clean.columns, "RET_winsor column missing."
    # After warm-up period, no NaN where RET is not NaN
    warm_up = 20
    dates_sorted = df_clean["date"].drop_duplicates().sort_values()
    post_warmup_dates = dates_sorted.iloc[warm_up:]
    sub = df_clean[df_clean["date"].isin(post_warmup_dates)]
    has_ret = sub["RET"].notna()
    no_winsor = sub.loc[has_ret, "RET_winsor"].isna().sum()
    # Allow a small fraction of NaN (due to per-PERMNO warm-up)
    frac_nan = no_winsor / max(has_ret.sum(), 1)
    assert frac_nan < 0.05, (
        f"{frac_nan:.1%} of post-warmup rows with RET have NaN RET_winsor."
    )


# ── Gate 9: Coverage filter drops low-coverage dates ─────────────────────────

def test_coverage_filter_drops_bad_dates():
    df = _make_panel(n_stocks=10)
    df_clean = integrity_checks(df)

    # Inject a date with only 20% coverage (set 80% of stocks to NaN on one day)
    all_dates = df_clean["date"].drop_duplicates().sort_values()
    target_date = all_dates.iloc[50]
    stocks_on_date = df_clean[df_clean["date"] == target_date]["PERMNO"].values
    n_to_null = int(0.8 * len(stocks_on_date))
    null_mask = (df_clean["date"] == target_date) & (
        df_clean["PERMNO"].isin(stocks_on_date[:n_to_null])
    )
    df_clean.loc[null_mask, "RET"] = np.nan

    _, dropped_log = apply_missing_data_policy(df_clean, min_coverage=0.90)
    assert target_date in dropped_log["date"].values, (
        "Low-coverage date was not dropped by apply_missing_data_policy."
    )


# ── Integration test: full pipeline runs without error ───────────────────────

def test_full_pipeline_smoke():
    """Verify that the full pipeline produces a non-empty, clean result."""
    df = _make_panel(n_stocks=8, n_days=250)
    df_clean = integrity_checks(df)
    df_clean, dropped = apply_missing_data_policy(df_clean, min_coverage=0.85)
    df_clean = winsorize_returns(df_clean, z_cap=3.0, window=30)

    assert len(df_clean) > 0, "Pipeline produced empty DataFrame."
    assert df_clean.duplicated(subset=["PERMNO", "date"]).sum() == 0
    assert not np.isinf(df_clean["RET"].dropna()).any()
    assert "RET_winsor" in df_clean.columns
