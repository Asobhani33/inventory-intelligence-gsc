"""
kpi_engine.py
=============
Phase 3 — descriptive & diagnostic analytics. Reads the star schema built in
Phase 2 (`data/inventory_performance.db`) and computes the KPI, ABC/XYZ,
excess-inventory, and diagnostic-decomposition tables that Power BI and the
AI Advisor will both consume later. Nothing here is re-simulated — this
module only aggregates and classifies what Phase 2 already generated.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# Target Days-of-Supply by ABC class — an illustrative planning assumption
# (fast movers are reviewed/replenished more tightly than slow movers),
# documented in docs/SYNTHETIC_DATA_METHODOLOGY.md rather than presented as
# derived from real company policy.
TARGET_DOS_BY_ABC = {"A": 30, "B": 45, "C": 60}
EXCESS_MULTIPLIER = 2.0          # > 2x target DOS = excess
SLOW_MOVING_DEMAND_WINDOW = 90    # days
SLOW_MOVING_ZERO_DEMAND_SHARE = 0.80   # >=80% zero-demand days in the window
OBSOLETE_NO_DEMAND_WINDOW = 180  # days with zero demand + stock on hand


def load_star_schema(db_path: Path) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(db_path)
    tables = {}
    for name in ["fact_inventory", "fact_demand", "fact_orders", "dim_sku", "dim_warehouse", "dim_date"]:
        tables[name] = pd.read_sql(f"SELECT * FROM {name}", con, parse_dates=["date"] if name in
                                    ("fact_inventory", "fact_demand", "dim_date") else None)
    con.close()
    for k in ("fact_inventory", "fact_demand"):
        tables[k]["date"] = pd.to_datetime(tables[k]["date"])
    return tables


def compute_inventory_value(fact_inventory: pd.DataFrame, dim_sku: pd.DataFrame) -> pd.DataFrame:
    df = fact_inventory.merge(dim_sku[["sku", "unit_cost", "abc_class", "category"]], on="sku")
    df["inventory_value"] = df["on_hand_qty"] * df["unit_cost"]
    return df


def compute_observed_xyz(fact_demand: pd.DataFrame) -> pd.DataFrame:
    """XYZ from OBSERVED simulated demand (not the generator's own CV
    parameter) — this is what a real analyst would compute from data."""
    g = fact_demand.groupby(["sku", "warehouse_id"])["demand_qty"]
    stats = g.agg(mean_demand="mean", std_demand="std").reset_index()
    stats["cv"] = (stats["std_demand"] / stats["mean_demand"].replace(0, np.nan)).fillna(0)
    stats["xyz_class"] = np.select(
        [stats["cv"] < 0.5, stats["cv"] < 1.0],
        ["X", "Y"],
        default="Z",
    )
    return stats


def compute_service_level(fact_inventory: pd.DataFrame, fact_demand: pd.DataFrame) -> pd.DataFrame:
    """Volume-based fill rate: for each (sku, warehouse), service level =
    1 - (newly-unfulfilled demand / total demand), where 'newly
    unfulfilled' is the day-over-day INCREASE in the backorder balance
    (so we don't double-count an outstanding backorder every day it sits
    unresolved)."""
    df = fact_inventory[["date", "sku", "warehouse_id", "backorder_qty"]].sort_values(
        ["sku", "warehouse_id", "date"]).copy()
    df["prev_backorder"] = df.groupby(["sku", "warehouse_id"])["backorder_qty"].shift(1).fillna(0)
    df["new_unfulfilled"] = (df["backorder_qty"] - df["prev_backorder"]).clip(lower=0)

    demand_tot = fact_demand.groupby(["sku", "warehouse_id"])["demand_qty"].sum().rename("total_demand")
    unmet_tot = df.groupby(["sku", "warehouse_id"])["new_unfulfilled"].sum().rename("total_unmet")

    out = pd.concat([demand_tot, unmet_tot], axis=1).reset_index().fillna(0)
    out["service_level"] = 1 - (out["total_unmet"] / out["total_demand"].replace(0, np.nan))
    out["service_level"] = out["service_level"].fillna(1.0).clip(0, 1)
    return out


def compute_turnover_and_dos(fact_inventory: pd.DataFrame, fact_demand: pd.DataFrame,
                              dim_sku: pd.DataFrame, trailing_days: int = 90) -> pd.DataFrame:
    """As-of-latest-date snapshot: trailing-90-day average daily demand,
    current on-hand, Days of Supply, and an annualized turnover estimate."""
    last_date = fact_inventory["date"].max()
    window_start = last_date - pd.Timedelta(days=trailing_days - 1)

    recent_demand = fact_demand[fact_demand["date"] >= window_start]
    avg_daily_demand = recent_demand.groupby(["sku", "warehouse_id"])["demand_qty"].mean().rename("avg_daily_demand_90d")

    latest_inv = fact_inventory[fact_inventory["date"] == last_date][
        ["sku", "warehouse_id", "on_hand_qty"]].set_index(["sku", "warehouse_id"])

    avg_inv_90d = fact_inventory[fact_inventory["date"] >= window_start].groupby(
        ["sku", "warehouse_id"])["on_hand_qty"].mean().rename("avg_on_hand_90d")

    snap = latest_inv.join(avg_daily_demand).join(avg_inv_90d).reset_index().fillna(0)
    snap = snap.merge(dim_sku[["sku", "abc_class", "unit_cost", "category"]], on="sku")

    snap["days_of_supply"] = np.where(
        snap["avg_daily_demand_90d"] > 0,
        snap["on_hand_qty"] / snap["avg_daily_demand_90d"],
        np.inf,
    )
    annual_demand_units = snap["avg_daily_demand_90d"] * 365
    snap["inventory_turnover"] = np.where(
        snap["avg_on_hand_90d"] > 0,
        annual_demand_units / snap["avg_on_hand_90d"],
        0,
    )
    snap["target_dos"] = snap["abc_class"].map(TARGET_DOS_BY_ABC)
    return snap


def flag_excess_slow_obsolete(snapshot: pd.DataFrame, fact_demand: pd.DataFrame) -> pd.DataFrame:
    df = snapshot.copy()
    df["is_excess"] = (df["days_of_supply"] != np.inf) & (df["days_of_supply"] > EXCESS_MULTIPLIER * df["target_dos"])
    df.loc[df["days_of_supply"] == np.inf, "is_excess"] = df.loc[df["days_of_supply"] == np.inf, "on_hand_qty"] > 0

    last_date = fact_demand["date"].max()
    window_start = last_date - pd.Timedelta(days=SLOW_MOVING_DEMAND_WINDOW - 1)
    recent = fact_demand[fact_demand["date"] >= window_start]
    zero_share = recent.assign(is_zero=lambda d: d["demand_qty"] == 0).groupby(
        ["sku", "warehouse_id"])["is_zero"].mean().rename("zero_demand_share_90d")
    df = df.merge(zero_share.reset_index(), on=["sku", "warehouse_id"], how="left")
    df["zero_demand_share_90d"] = df["zero_demand_share_90d"].fillna(1.0)
    df["is_slow_moving"] = df["zero_demand_share_90d"] >= SLOW_MOVING_ZERO_DEMAND_SHARE

    obs_start = last_date - pd.Timedelta(days=OBSOLETE_NO_DEMAND_WINDOW - 1)
    recent_long = fact_demand[fact_demand["date"] >= obs_start]
    total_demand_long = recent_long.groupby(["sku", "warehouse_id"])["demand_qty"].sum().rename("demand_180d")
    df = df.merge(total_demand_long.reset_index(), on=["sku", "warehouse_id"], how="left")
    df["demand_180d"] = df["demand_180d"].fillna(0)
    df["is_obsolete"] = (df["demand_180d"] == 0) & (df["on_hand_qty"] > 0)

    df["excess_value"] = np.where(df["is_excess"], df["on_hand_qty"] * df["unit_cost"], 0.0)
    return df


def compute_monthly_diagnostic(inv_value_df: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month inventory value change, decomposed by warehouse and
    ABC class — answers 'why did inventory value move this month, and
    where.'"""
    df = inv_value_df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby(["month", "warehouse_id", "abc_class"])["inventory_value"].mean().reset_index()
    monthly = monthly.sort_values(["warehouse_id", "abc_class", "month"])
    monthly["prev_value"] = monthly.groupby(["warehouse_id", "abc_class"])["inventory_value"].shift(1)
    monthly["delta"] = monthly["inventory_value"] - monthly["prev_value"]
    return monthly


def build_all_kpi_tables(db_path: Path, out_dir: Path) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = load_star_schema(db_path)

    inv_value = compute_inventory_value(t["fact_inventory"], t["dim_sku"])
    xyz = compute_observed_xyz(t["fact_demand"])
    service = compute_service_level(t["fact_inventory"], t["fact_demand"])
    snapshot = compute_turnover_and_dos(t["fact_inventory"], t["fact_demand"], t["dim_sku"])
    snapshot = snapshot.merge(xyz[["sku", "warehouse_id", "cv", "xyz_class"]], on=["sku", "warehouse_id"], how="left")
    snapshot = snapshot.merge(service[["sku", "warehouse_id", "service_level"]], on=["sku", "warehouse_id"], how="left")
    snapshot = flag_excess_slow_obsolete(snapshot, t["fact_demand"])
    snapshot = snapshot.merge(t["dim_warehouse"][["warehouse_id", "region"]], on="warehouse_id", how="left")

    monthly_diag = compute_monthly_diagnostic(inv_value)

    tables = {"sku_warehouse_kpi_snapshot": snapshot, "monthly_inventory_diagnostic": monthly_diag}
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    return tables


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    build_all_kpi_tables(root / "data" / "inventory_performance.db", root / "data" / "processed" / "kpi_tables")
