"""
data_access.py
================
Phase 7 — loads every processed table the AI Advisor's tools query, once,
and caches them in memory. Nothing here talks to an LLM; this is the
grounding layer — nothing the Advisor says can go beyond what these
DataFrames actually contain.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import pandas as pd


def _find_project_root() -> Path:
    """Colab-aware, same pattern as every notebook in this repo."""
    try:
        import google.colab  # noqa: F401
        return Path("/content/drive/MyDrive/inventory-intelligence-gsc")
    except ImportError:
        here = Path(__file__).resolve()
        # src/ai_advisor/data_access.py -> project root is two levels up
        return here.parents[2]


@lru_cache(maxsize=1)
def load_all(root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = root or _find_project_root()
    proc = root / "data" / "processed"
    reco = proc / "recommendations"
    db_path = root / "data" / "inventory_performance.db"

    con = sqlite3.connect(db_path)
    dim_sku = pd.read_sql("SELECT * FROM dim_sku", con)
    dim_warehouse = pd.read_sql("SELECT * FROM dim_warehouse", con)
    con.close()

    tables = {
        "replenishment": pd.read_parquet(reco / "replenishment_recommendations.parquet"),
        "transfers": pd.read_parquet(reco / "transfer_recommendations.parquet"),
        "unmet_shortage": pd.read_parquet(reco / "unmet_shortage_fresh_po.parquet"),
        "transfer_cost_matrix": pd.read_parquet(reco / "transfer_cost_matrix.parquet"),
        "monthly_diagnostic": pd.read_parquet(proc / "kpi_tables" / "monthly_inventory_diagnostic.parquet"),
        "forecast_test": pd.read_parquet(proc / "forecast_test_predictions.parquet"),
        "dim_sku": dim_sku,
        "dim_warehouse": dim_warehouse,
    }
    return tables


def reload() -> None:
    """Call after re-running the pipeline notebooks, to pick up fresh data
    without restarting the Streamlit app."""
    load_all.cache_clear()
