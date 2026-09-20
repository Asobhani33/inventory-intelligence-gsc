"""
data_access.py
================
Phase 7 — loads every processed table the AI Advisor's tools query, once,
and caches them in memory. Nothing here talks to an LLM; this is the
grounding layer — nothing the Advisor says can go beyond what these
DataFrames actually contain.
"""
from __future__ import annotations

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

    # dim_sku / dim_warehouse used to come from data/inventory_performance.db
    # (a 130+MB SQLite file). That's too big for GitHub and isn't something
    # Streamlit Community Cloud can regenerate on deploy, so these two small
    # dimension tables are exported to parquet instead (see notebooks/07 or
    # the one-off export in the project's git history) and versioned
    # directly — everything the public app needs now lives as small parquet
    # files under data/processed/, no database file required.
    tables = {
        "replenishment": pd.read_parquet(reco / "replenishment_recommendations.parquet"),
        "transfers": pd.read_parquet(reco / "transfer_recommendations.parquet"),
        "unmet_shortage": pd.read_parquet(reco / "unmet_shortage_fresh_po.parquet"),
        "transfer_cost_matrix": pd.read_parquet(reco / "transfer_cost_matrix.parquet"),
        "monthly_diagnostic": pd.read_parquet(proc / "kpi_tables" / "monthly_inventory_diagnostic.parquet"),
        # Monthly-grain forecast test set (3 held-out months) — this is the
        # grain reported as the project's headline forecast-accuracy number
        # (WAPE / MASE on the PDF report and microsite), so get_forecast_accuracy
        # answers with the same number the report shows. The weekly-grain
        # model still exists and is trained in src/predictive_models.py /
        # notebooks/04_predictive_analytics.ipynb (forecast_test_predictions.parquet),
        # it's just not what the AI Advisor is wired to here.
        "forecast_test": pd.read_parquet(proc / "forecast_test_predictions_monthly.parquet"),
        "dim_sku": pd.read_parquet(proc / "dim_sku.parquet"),
        "dim_warehouse": pd.read_parquet(proc / "dim_warehouse.parquet"),
    }
    return tables


def reload() -> None:
    """Call after re-running the pipeline notebooks, to pick up fresh data
    without restarting the Streamlit app."""
    load_all.cache_clear()
