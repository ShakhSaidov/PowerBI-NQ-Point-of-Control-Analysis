"""
session_engine.py
==================
Loads raw 1-min OHLCV data and derives, for every RTH trading day:
    PH, PL, PC   - previous trading day's own RTH high / low / close
    ONH, ONL     - overnight session high / low (prior RTH close -> this
                   day's RTH open, gap-safe across weekends/holidays)
    PrevRange    - PH - PL  (per user spec: prior day's own range, NOT a
                   gapped true-range and NOT an average)
    PrevDayColor - 'Green' if prior day closed above its own open, else 'Red'

TRADING-DAY ASSIGNMENT ALGORITHM
---------------------------------
A "trading date" D is any calendar date (Eastern Time) for which the data
contains a bar at exactly 09:30:00 ET. We deliberately do NOT hardcode a
holiday calendar: the data is pre-vetted, so the presence/absence of a
09:30 bar IS the ground truth for whether the market opened that day. This
means holiday gaps, half-days-that-are-actually-full-holidays, etc. are all
handled correctly as long as the file is vetted as instructed -- if a date
has no 09:30 bar, it silently and correctly falls out of the trading
calendar and every subsequent bar during what would have been that day's
RTH hours is instead swept into the NEXT valid day's overnight bucket.

Every bar in the file is assigned to exactly one trading date, via two
cases:
  1. RTH bars: bar's own ET time is in [09:30, 16:00) AND its own ET date
     is a valid trading date -> trading_date = its own date.
  2. Everything else (pre-market, post-market, weekend, holiday-daytime,
     overnight): trading_date = the NEXT valid trading date whose 09:30
     open timestamp is strictly after this bar (found via a single
     vectorized searchsorted over the sorted array of session-open
     timestamps -- O(n log d) where d = number of trading days).

This correctly reproduces, with no special-casing: Friday 16:00 -> Monday
09:30 overnight spans, Friday 16:00 -> Tuesday 09:30 spans across a Monday
holiday (including Sunday evening AND the Monday holiday's own overnight
price action, since CME continues trading through most exchange holidays
even though the 09:30-16:00 RTH block itself doesn't open), and identical
behavior for any other gap length, without ever consulting a calendar.
"""
from datetime import time as dtime

import numpy as np
import pandas as pd

RTH_START = dtime(9, 30)
RTH_END = dtime(16, 0)
ET_TZ = "America/New_York"


def load_raw(path: str, start_date: str | None = None) -> pd.DataFrame:
    """Load the 1-min parquet file and convert its UTC index to ET wall-clock
    (tz-aware conversion done once, up front, so DST is handled correctly;
    the tz is then dropped since every downstream computation operates on ET
    wall-clock time directly)."""
    df = pd.read_parquet(path, engine="pyarrow")
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df = df.copy()
    df.index = idx
    df = df.sort_index()
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    if len(df) != n_before:
        print(f"  [load_raw] WARNING: dropped {n_before - len(df):,} duplicate-timestamp rows.")

    df.index = df.index.tz_convert(ET_TZ).tz_localize(None)
    df.index.name = "datetime"

    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input parquet is missing required columns: {missing}")

    return df


def assign_trading_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Adds 'trading_date' (Timestamp, midnight-normalized) and 'is_rth' /
    'is_overnight' boolean columns. Returns (annotated_df, valid_trading_dates)."""
    idx = df.index
    dates = idx.normalize()
    times = idx.time

    open_mask = np.array([t == RTH_START for t in times])
    valid_dates = pd.DatetimeIndex(sorted(pd.unique(dates[open_mask])))
    if len(valid_dates) == 0:
        raise ValueError(
            "No bars found at exactly 09:30:00 ET -- cannot establish a trading "
            "calendar. Check that the source timestamps are genuinely UTC."
        )

    open_ts_arr = (valid_dates + pd.Timedelta(hours=9, minutes=30)).values

    is_rth_time = np.array([(t >= RTH_START) and (t < RTH_END) for t in times])
    is_valid_date = dates.isin(valid_dates)
    own_day_mask = is_rth_time & is_valid_date

    pos = np.searchsorted(open_ts_arr, idx.values, side="left")
    has_next = pos < len(open_ts_arr)
    next_date = np.where(
        has_next,
        valid_dates.values[np.clip(pos, 0, len(valid_dates) - 1)],
        np.datetime64("NaT"),
    )

    trading_date = np.where(own_day_mask, dates.values, next_date)
    trading_date = pd.DatetimeIndex(trading_date)

    out = df.copy()
    out["trading_date"] = trading_date
    out["is_rth"] = own_day_mask
    out["is_overnight"] = (~own_day_mask) & (~trading_date.isna())

    n_dropped = int(trading_date.isna().sum())
    if n_dropped:
        print(
            f"  [assign_trading_dates] {n_dropped:,} bars at the tail of the file have "
            f"no known future session open (no 09:30 bar exists after them yet) and are "
            f"excluded -- this is expected for the last partial overnight session."
        )

    # ---- Sanity checks (fail loudly rather than silently miscompute) ----
    rth_bars = out[out["is_rth"]]
    bad = rth_bars.index.normalize().values != rth_bars["trading_date"].values
    assert not bad.any(), "BUG: an RTH bar was assigned to a trading_date other than its own calendar date."

    on_bars = out[out["is_overnight"]]
    if len(on_bars):
        # every overnight bar's timestamp must fall strictly before its
        # assigned trading date's own 09:30 open
        assigned_open = pd.DatetimeIndex(on_bars["trading_date"]) + pd.Timedelta(hours=9, minutes=30)
        assert (on_bars.index.values < assigned_open.values).all(), (
            "BUG: an overnight bar was assigned to a trading_date whose open it is not actually before."
        )

    return out, valid_dates


def build_daily_pocs(df: pd.DataFrame, valid_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per trading day (index = TradingDate) with PH/PL/PC/ONH/ONL,
    PrevRange, and PrevDayColor. The first day(s) in the file, which have no
    prior day to reference, are dropped (documented, not silently NaN'd)."""
    rth = df[df["is_rth"]]
    daily_rth = rth.groupby("trading_date").agg(
        rth_open=("open", "first"),
        rth_high=("high", "max"),
        rth_low=("low", "min"),
        rth_close=("close", "last"),
        rth_bar_count=("close", "size"),
    ).reindex(valid_dates)

    on = df[df["is_overnight"]]
    daily_on = on.groupby("trading_date").agg(
        onh=("high", "max"),
        onl=("low", "min"),
        on_bar_count=("close", "size"),
    ).reindex(valid_dates)

    daily_rth = daily_rth.sort_index()
    ph = daily_rth["rth_high"].shift(1)
    pl = daily_rth["rth_low"].shift(1)
    pc = daily_rth["rth_close"].shift(1)
    po = daily_rth["rth_open"].shift(1)
    prev_range = ph - pl

    out = pd.DataFrame({
        "PH": ph, "PL": pl, "PC": pc, "PrevOpen": po,
        "ONH": daily_on["onh"], "ONL": daily_on["onl"],
        "PrevRange": prev_range,
        "RTH_BarCount": daily_rth["rth_bar_count"],
        "ON_BarCount": daily_on["on_bar_count"],
    })
    out.index.name = "TradingDate"

    out["PrevDayColor"] = np.select(
        [out["PC"] > out["PrevOpen"], out["PC"] < out["PrevOpen"]],
        ["Green", "Red"],
        default="Flat",
    )

    n_before = len(out)
    out = out.dropna(subset=["PH", "PL", "PC", "ONH", "ONL"])
    n_dropped = n_before - len(out)
    if n_dropped:
        print(
            f"  [build_daily_pocs] Dropped {n_dropped} day(s) with no complete prior-day + "
            f"overnight context (always the first day(s) in the file, by definition)."
        )

    if (out["RTH_BarCount"] < 300).any():
        thin = out[out["RTH_BarCount"] < 300]
        print(
            f"  [build_daily_pocs] NOTE: {len(thin)} trading day(s) have <300 RTH 1-min bars "
            f"(a full session is 390) -- likely early closes (e.g. half-days). PH/PL/PC for "
            f"these days reflect whatever bars actually exist; verify this is expected."
        )

    return out