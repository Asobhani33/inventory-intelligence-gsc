"""
powerbi_prep.py
================
Phase 6 — Power BI data prep. Power BI Desktop doesn't have a native SQLite
connector without an extra ODBC driver install, but it DOES read Parquet
natively (Get Data > Parquet, in current Power BI Desktop builds) — so this
gathers every table the dashboard needs, already split into clean-grain
Parquet files, into a single `data/powerbi/` folder. Point Power BI at that
one folder and every table below shows up ready to relate.

This intentionally drops two redundant tables rather than shipping
duplicates into Power BI:
- `kpi_sku_warehouse_snapshot` is a strict subset of `reco_replenishment`
  (which already carries KPI + health-score + reorder/transfer columns at
  the same sku x warehouse grain) — only the superset is exported.
- `predictive_health_scored` is likewise a subset of `reco_replenishment`.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pandas as pd

# Illustrative region center points, for the Phase 6 "Global Network" map
# visual only — the source datasets give a REGION per plant, never a real
# lat/long, so this is a documented visualization aid, not real facility
# geography (same honesty standard as every other assumption in this repo:
# see TARGET_DOS_BY_ABC in kpi_engine.py, REGION_SURCHARGE in
# optimization_engine.py).
REGION_CENTERS = {
    "North America": (39.8, -98.5),
    "Latin America": (-8.7, -55.4),
    "Europe": (50.1, 10.4),
    "Middle East & Africa": (20.0, 25.0),
    "South Asia": (21.0, 78.9),
    "Asia-Pacific": (15.0, 120.0),
}


def _seed_from(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


def build_warehouse_coords(dim_warehouse: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    rows = []
    for _, r in dim_warehouse.iterrows():
        center_lat, center_lon = REGION_CENTERS.get(r["region"], (0.0, 0.0))
        rng = np.random.default_rng(_seed_from("map_jitter", r["warehouse_id"]))
        lat = center_lat + rng.uniform(-6, 6)
        lon = center_lon + rng.uniform(-10, 10)
        rows.append({"warehouse_id": r["warehouse_id"], "region": r["region"],
                      "approx_lat": round(lat, 2), "approx_lon": round(lon, 2)})
    return pd.DataFrame(rows)

# table_name -> ("db" or "parquet", source)
SOURCES = {
    "dim_sku": ("db", "dim_sku"),
    "dim_warehouse": ("db", "dim_warehouse"),
    "dim_date": ("db", "dim_date"),
    "dim_replenishment_policy": ("db", "dim_replenishment_policy"),
    "fact_inventory": ("db", "fact_inventory"),
    "fact_demand": ("db", "fact_demand"),
    "fact_orders": ("db", "fact_orders"),
    "kpi_monthly_diagnostic": ("parquet", "data/processed/kpi_tables/monthly_inventory_diagnostic.parquet"),
    "predictive_forecast_test": ("parquet", "data/processed/forecast_test_predictions.parquet"),
    "reco_replenishment": ("parquet", "data/processed/recommendations/replenishment_recommendations.parquet"),
    "reco_transfers": ("parquet", "data/processed/recommendations/transfer_recommendations.parquet"),
    "reco_unmet_shortage": ("parquet", "data/processed/recommendations/unmet_shortage_fresh_po.parquet"),
    "reco_transfer_cost_matrix": ("parquet", "data/processed/recommendations/transfer_cost_matrix.parquet"),
}


def _clean_for_export(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if str(df[col].dtype) == "category":
            df[col] = df[col].astype(str)
    return df


def build_powerbi_folder(root: Path) -> dict[str, int]:
    out_dir = root / "data" / "powerbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = root / "data" / "inventory_performance.db"
    con = sqlite3.connect(db_path)

    row_counts = {}
    for table_name, (kind, source) in SOURCES.items():
        if kind == "db":
            df = pd.read_sql(f"SELECT * FROM {source}", con)
        else:
            df = pd.read_parquet(root / source)
        df = _clean_for_export(df)
        df.to_parquet(out_dir / f"{table_name}.parquet", index=False)
        row_counts[table_name] = len(df)

    con.close()

    dim_warehouse = pd.read_sql("SELECT * FROM dim_warehouse", sqlite3.connect(db_path))
    coords = build_warehouse_coords(dim_warehouse)
    coords.to_parquet(out_dir / "dim_warehouse_coords.parquet", index=False)
    row_counts["dim_warehouse_coords"] = len(coords)

    return row_counts


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    counts = build_powerbi_folder(root)
    print(f"Wrote {len(counts)} tables to data/powerbi/:")
    for name, n in counts.items():
        print(f"  {name}: {n} rows")
