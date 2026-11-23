"""
build_pbi_dataset.py
====================
Generates a clean, normalized 5-table Star Schema in ./PowerBI_Data/
combining:
  - Canonical gap/holiday-safe session engine
  - PoC Hierarchy, Open Zone, and 1m/5m Candle Colors
  - Mutually exclusive structural regimes & prefix transitions
"""
import itertools
import os
import time

import numpy as np
import pandas as pd

from etl.session_engine import assign_trading_dates, build_daily_pocs, load_raw

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "../data/nq_1min.parquet"
OUTPUT_DIR = "./PowerBI_Data"
START_DATE = None          # e.g. "2015-01-01" or None for full history
WINDOW_START = "09:30"     # ET 24h
WINDOW_END = "11:30"       # ET 24h (half-open [09:30, 11:30))
USD_PER_POINT = 20.0       # NQ = $20/pt, MNQ = $2/pt
MAX_PREFIX_LEN = 4

POC_NAMES = ["PH", "PL", "PC", "ONH", "ONL"]
# ============================================================================


def _classify_wick_body(price: float, o: float, h: float, l: float, c: float) -> str:
    if price > max(o, c):
        return "UpperWick"
    if price < min(o, c):
        return "LowerWick"
    return "Body"


def _classify_color(o: float, c: float) -> str:
    return "Green" if c >= o else "Red"


def _time_bucket(mins: float) -> tuple[str, int]:
    if pd.isna(mins):
        return ("Not Hit", 7)
    if mins <= 1.0:
        return ("0-1m", 1)
    if mins <= 5.0:
        return ("1-5m", 2)
    if mins <= 15.0:
        return ("5-15m", 3)
    if mins <= 30.0:
        return ("15-30m", 4)
    if mins <= 60.0:
        return ("30-60m", 5)
    return ("60-120m", 6)


def _on_structure(ph: float, pl: float, onh: float, onl: float) -> str:
    if onl > ph:
        return "Gap Up (ONL > PH)"
    if onh < pl:
        return "Gap Down (ONH < PL)"
    if onl >= pl and onh <= ph:
        return "Inside Prev RTH"
    if onl < pl and onh > ph:
        return "Outside Prev RTH"
    if onl >= pl and onh > ph:
        return "Shift Up (Higher H & L)"
    return "Shift Down (Lower H & L)"


def _pc_position(pc: float, onh: float, onl: float) -> str:
    if pc > onh:
        return "Prev Close Above ONH"
    if pc < onl:
        return "Prev Close Below ONL"
    return "Prev Close Inside ON"


def _open_zone(open_price: float, sorted_pocs: list[tuple[str, float]]) -> str:
    if open_price > sorted_pocs[0][1]:
        return f"Above {sorted_pocs[0][0]}"
    if open_price < sorted_pocs[-1][1]:
        return f"Below {sorted_pocs[-1][0]}"
    for i in range(len(sorted_pocs) - 1):
        upper_poc, upper_val = sorted_pocs[i]
        lower_poc, lower_val = sorted_pocs[i + 1]
        if lower_val <= open_price <= upper_val:
            return f"Between {upper_poc} and {lower_poc}"
    return "Unknown"


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/4] Loading raw 1-min bars and computing canonical session levels...")
    raw_bars = load_raw(DATA_PATH, start_date=START_DATE)
    annotated, valid_dates = assign_trading_dates(raw_bars)
    daily = build_daily_pocs(annotated, valid_dates)

    print(f"[2/4] Preparing 1m & 5m bars for {WINDOW_START}-{WINDOW_END} ET...")
    rth = annotated[annotated["is_rth"]]
    win_1m = rth.between_time(WINDOW_START, WINDOW_END, inclusive="left")
    grouped_1m = {d: g for d, g in win_1m.groupby("trading_date")}

    # 09:30 and 16:00 are exact multiples of 5 mins from midnight, so origin='start_day'
    # aligns 5m bins cleanly to 09:30 without cross-day leakage.
    five = (
        rth[["open", "high", "low", "close"]]
        .resample("5min", origin="start_day")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna(subset=["open"])
    )
    win_5m = five.between_time(WINDOW_START, WINDOW_END, inclusive="left")
    grouped_5m = {d: g for d, g in win_5m.groupby(win_5m.index.normalize())}

    wh, wm = (int(x) for x in WINDOW_START.split(":"))

    day_rows = []
    touch_rows = []
    prefix_rows = []

    print("[3/4] Building Star Schema fact tables...")
    for date, pocs in daily.iterrows():
        day_1m = grouped_1m.get(date)
        if day_1m is None or day_1m.empty:
            continue
        day_5m = grouped_5m.get(date, pd.DataFrame(columns=["open", "high", "low", "close"]))

        open_anchor = pd.Timestamp(date) + pd.Timedelta(hours=wh, minutes=wm)
        open_price = float(day_1m.iloc[0]["open"])
        ph, pl, pc = float(pocs["PH"]), float(pocs["PL"]), float(pocs["PC"])
        onh, onl = float(pocs["ONH"]), float(pocs["ONL"])
        prev_range = float(pocs["PrevRange"])

        poc_prices = {"PH": ph, "PL": pl, "PC": pc, "ONH": onh, "ONL": onl}
        sorted_pocs = sorted(poc_prices.items(), key=lambda kv: kv[1], reverse=True)
        poc_hierarchy = " > ".join(k for k, _ in sorted_pocs)
        open_zone = _open_zone(open_price, sorted_pocs)

        per_poc = {}
        hit_times = {}

        for p in POC_NAMES:
            price = poc_prices[p]
            m1 = (day_1m["low"] <= price) & (day_1m["high"] >= price)
            if not m1.any():
                per_poc[p] = {
                    "Is_Hit": False,
                    "Is_Hit_Int": 0,
                    "Hit_Time_ET": pd.NaT,
                    "Mins_From_Open": np.nan,
                    "Time_Bucket": "Not Hit",
                    "Time_Bucket_Sort": 7,
                    "Physics_1m": "None",
                    "Color_1m": "None",
                    "Physics_5m": "None",
                    "Color_5m": "None",
                    "Max_Excursion_Pts": np.nan,
                    "Max_Excursion_USD": np.nan,
                    "Max_Excursion_Pct_PrevRange": np.nan,
                }
                continue

            ht = day_1m.index[m1][0]
            hit_times[p] = ht
            mins = (ht - open_anchor).total_seconds() / 60.0
            tb_label, tb_sort = _time_bucket(mins)

            c1 = day_1m.loc[ht]
            phys_1m = _classify_wick_body(price, float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"]))
            col_1m = _classify_color(float(c1["open"]), float(c1["close"]))

            phys_5m, col_5m = "None", "None"
            if not day_5m.empty:
                m5 = (day_5m["low"] <= price) & (day_5m["high"] >= price)
                if m5.any():
                    c5 = day_5m.loc[day_5m.index[m5][0]]
                    phys_5m = _classify_wick_body(price, float(c5["open"]), float(c5["high"]), float(c5["low"]), float(c5["close"]))
                    col_5m = _classify_color(float(c5["open"]), float(c5["close"]))

            after = day_1m.loc[ht:]
            exc_pts = float(max(abs(after["high"].max() - price), abs(price - after["low"].min())))
            exc_usd = exc_pts * USD_PER_POINT
            exc_pct = (exc_pts / prev_range) * 100.0 if prev_range > 0 else np.nan

            per_poc[p] = {
                "Is_Hit": True,
                "Is_Hit_Int": 1,
                "Hit_Time_ET": ht,
                "Mins_From_Open": round(mins, 2),
                "Time_Bucket": tb_label,
                "Time_Bucket_Sort": tb_sort,
                "Physics_1m": phys_1m,
                "Color_1m": col_1m,
                "Physics_5m": phys_5m,
                "Color_5m": col_5m,
                "Max_Excursion_Pts": round(exc_pts, 2),
                "Max_Excursion_USD": round(exc_usd, 2),
                "Max_Excursion_Pct_PrevRange": round(exc_pct, 2) if pd.notna(exc_pct) else np.nan,
            }

        hits_sorted = sorted(hit_times.items(), key=lambda kv: (kv[1], kv[0]))
        hit_order_map = {p: i + 1 for i, (p, _) in enumerate(hits_sorted)}
        times_only = [t for _, t in hits_sorted]
        had_ties = len(times_only) != len(set(times_only))

        first_hit_poc = hits_sorted[0][0] if hits_sorted else "NONE"
        hits_count = len(hits_sorted)
        hit_subset = "+".join(sorted(hit_times.keys())) if hit_times else "NONE"
        hit_sequence = " -> ".join(p for p, _ in hits_sorted) if hits_sorted else "NONE"

        if hits_count == 1:
            lone = first_hit_poc
            s_exc_pts = per_poc[lone]["Max_Excursion_Pts"]
            s_exc_usd = per_poc[lone]["Max_Excursion_USD"]
            s_exc_pct = per_poc[lone]["Max_Excursion_Pct_PrevRange"]
        else:
            s_exc_pts = s_exc_usd = s_exc_pct = np.nan

        t_date = date.date()
        day_rows.append({
            "TradingDate": t_date,
            "Prev_Day_Dir": pocs["PrevDayColor"],
            "Prev_Range": round(prev_range, 2),
            "RTH_Open_Price": round(open_price, 2),
            "PH_Price": round(ph, 2),
            "PL_Price": round(pl, 2),
            "PC_Price": round(pc, 2),
            "ONH_Price": round(onh, 2),
            "ONL_Price": round(onl, 2),
            "PoC_Hierarchy": poc_hierarchy,
            "Open_Zone": open_zone,
            "ON_Structure": _on_structure(ph, pl, onh, onl),
            "PC_Position": _pc_position(pc, onh, onl),
            "Hits_Count": hits_count,
            "First_Hit_PoC": first_hit_poc,
            "Hit_Subset_Pattern": hit_subset,
            "Hit_Sequence_Path": hit_sequence,
            "Had_Simultaneous_Ties": had_ties,
            "Single_Hit_Excursion_Pts": s_exc_pts,
            "Single_Hit_Excursion_USD": s_exc_usd,
            "Single_Hit_Excursion_Pct_PrevRange": s_exc_pct,
        })

        for p in POC_NAMES:
            info = per_poc[p]
            order = hit_order_map.get(p)
            touch_rows.append({
                "TradingDate": t_date,
                "PoC_Code": p,
                "PoC_Price": round(poc_prices[p], 2),
                "Is_Hit": info["Is_Hit"],
                "Is_Hit_Int": info["Is_Hit_Int"],
                "Hit_Order": order if order is not None else pd.NA,
                "Is_First_Hit": order == 1,
                "Hit_Time_ET": info["Hit_Time_ET"],
                "Mins_From_Open": info["Mins_From_Open"],
                "Time_Bucket": info["Time_Bucket"],
                "Time_Bucket_Sort": info["Time_Bucket_Sort"],
                "Physics_1m": info["Physics_1m"],
                "Color_1m": info["Color_1m"],
                "Physics_5m": info["Physics_5m"],
                "Color_5m": info["Color_5m"],
                "Max_Excursion_Pts": info["Max_Excursion_Pts"],
                "Max_Excursion_USD": info["Max_Excursion_USD"],
                "Max_Excursion_Pct_PrevRange": info["Max_Excursion_Pct_PrevRange"],
            })

        if hits_count > 0:
            path = [p for p, _ in hits_sorted]
            for k in range(1, min(MAX_PREFIX_LEN, len(path)) + 1):
                prefix = tuple(path[:k])
                s1, sk = prefix[0], prefix[-1]
                prefix_complete_mins = per_poc[sk]["Mins_From_Open"]
                ptb_label, ptb_sort = _time_bucket(prefix_complete_mins)
                remaining = [p for p in POC_NAMES if p not in prefix]
                for target in remaining:
                    t_hit = per_poc[target]["Is_Hit"]
                    t_mins = per_poc[target]["Mins_From_Open"] if t_hit else np.nan
                    prefix_rows.append({
                        "TradingDate": t_date,
                        "Prefix_Length": k,
                        "Prefix_Sequence": " -> ".join(prefix),
                        "First_Hit_PoC": s1,
                        "Last_Prefix_PoC": sk,
                        "Target_PoC": target,
                        "Target_Hit": t_hit,
                        "Target_Hit_Int": 1 if t_hit else 0,
                        "Prefix_Complete_Mins": prefix_complete_mins,
                        "Prefix_Time_Bucket": ptb_label,
                        "Prefix_Time_Bucket_Sort": ptb_sort,
                        "Target_Hit_Mins": t_mins,
                        "Transition_Minutes": round(t_mins - prefix_complete_mins, 2) if t_hit else np.nan,
                        "Entry_Physics_1m": per_poc[s1]["Physics_1m"],
                        "Entry_Color_1m": per_poc[s1]["Color_1m"],
                        "Entry_Physics_5m": per_poc[s1]["Physics_5m"],
                        "Entry_Color_5m": per_poc[s1]["Color_5m"],
                    })

    df_days = pd.DataFrame(day_rows)
    df_touches = pd.DataFrame(touch_rows)
    df_touches["Hit_Order"] = df_touches["Hit_Order"].astype("Int64")
    df_prefix = pd.DataFrame(prefix_rows)

    # Continuous Calendar Dimension
    all_dates = pd.date_range(start=df_days["TradingDate"].min(), end=df_days["TradingDate"].max(), freq="D")
    trading_set = set(df_days["TradingDate"])
    df_dim_date = pd.DataFrame({
        "Date": all_dates.date,
        "Year": all_dates.year,
        "Quarter": "Q" + all_dates.quarter.astype(str),
        "Month_Num": all_dates.month,
        "Month_Name": all_dates.strftime("%b"),
        "Year_Month": all_dates.strftime("%Y-%m"),
        "Weekday_Num": all_dates.dayofweek + 1,  # 1=Mon..7=Sun
        "Weekday": all_dates.strftime("%A"),
        "Is_Trading_Day": [d in trading_set for d in all_dates.date],
    })

    # PoC Dimension
    df_dim_poc = pd.DataFrame([
        {"PoC_Code": "PH",  "PoC_Name": "Previous RTH High",  "Session_Source": "Prior RTH", "Sort_Order": 1},
        {"PoC_Code": "PL",  "PoC_Name": "Previous RTH Low",   "Session_Source": "Prior RTH", "Sort_Order": 2},
        {"PoC_Code": "PC",  "PoC_Name": "Previous RTH Close", "Session_Source": "Prior RTH", "Sort_Order": 3},
        {"PoC_Code": "ONH", "PoC_Name": "Overnight High",     "Session_Source": "Overnight", "Sort_Order": 4},
        {"PoC_Code": "ONL", "PoC_Name": "Overnight Low",      "Session_Source": "Overnight", "Sort_Order": 5},
    ])

    print("[4/4] Writing Power BI Parquet tables...")
    tables = {
        "Dim_Date.parquet": df_dim_date,
        "Dim_PoC.parquet": df_dim_poc,
        "Fact_Trading_Days.parquet": df_days,
        "Fact_PoC_Touches.parquet": df_touches,
        "Fact_Prefix_Transitions.parquet": df_prefix,
    }
    for fname, tbl in tables.items():
        path = os.path.join(OUTPUT_DIR, fname)
        tbl.to_parquet(
            path,
            engine="pyarrow",
            index=False,
            coerce_timestamps="ms",
            allow_truncated_timestamps=True,
        )
        print(f"  -> {fname:35s} {len(tbl):>9,} rows")

    print(f"\nDone in {time.time() - t0:.2f}s. Output saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()