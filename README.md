# NQ Futures Session Analytics — Power BI Dashboard

An end-to-end financial analytics and business intelligence project analyzing historical **Nasdaq-100 E-mini Futures (NQ)** opening-session price behavior (**09:30–11:30 ET**, 2010–2026). Built with a **Python ETL pipeline** (`pandas`, `pyarrow`) feeding a **5-table Star Schema** in **Power BI Desktop** using **Power Query (M)** and **DAX**.

---

## Project Overview

For every valid Regular Trading Hours (**RTH: 09:30–16:00 ET**) session, the pipeline computes five pre-market reference levels (**Points of Concern / PoCs**): Previous RTH High (`PH`), Low (`PL`), Close (`PC`), Overnight High (`ONH`), and Overnight Low (`ONL`).

**Key Technical Highlights:**
* **Star Schema Data Model:** Normalized raw 1-minute OHLCV time-series data into 2 dimension tables (`Dim_Date`, `Dim_PoC`) and 3 grain-separated fact tables (`Fact_Trading_Days`, `Fact_PoC_Touches`, `Fact_Prefix_Transitions`) with single-direction `1:*` relationships.
* **Statistical Rigor in DAX:** Built a dedicated `_Measures` table featuring **95% Wilson Score Confidence Intervals**, conditional sequential transition probabilities, and sample-size reliability gating (`n >= 20`).
* **Interactive BI Experience:** Synchronized multi-page Global Filter Bar, Analytics Error Bars, conditional-formatting heatmaps, Top-N sequence filtering, Page Navigation tabs, and Bookmark filter resets.

---

## Dashboard Screenshots

### 1. Base Probabilities & Executive Overview
*Unconditional PoC hit rates with 95% Wilson CI error bars, first-hit share, multi-year stability (2010–2026), and weekday seasonality.*

![Page 1 - Base Probabilities](screenshots/1.%20Base%20Probabilities.png)

---

## Repository Structure

```text
├── NQ_PoC_Statistical_Analysis.pbix   # Power BI Desktop report & semantic model
├── etl/
│   ├── build_pbi_dataset.py           # Star Schema ETL builder
│   ├── sonnet_session_engine.py       # DST- & holiday-safe session alignment engine
│   └── requirements.txt
├── PowerBI_Data/                      # Pre-built Star Schema Parquet tables
└── screenshots/                       # Dashboard page exports
