"""
tools.py
=========
Phase 7 — the AI Advisor's tool functions. Each one answers a narrow,
concrete question from the project's own processed tables and returns
plain JSON-serializable data — no free text, no invented numbers. The LLM
in agent.py can only ever talk about what these functions actually return.

`explain_health_score` is a rule-based decomposition of the Phase 4 health
score formula (recomputed from the row's own fields, using the same
weights as `kpi_engine.compute_health_score`), not a live SHAP call — the
Phase 4 models were trained in a notebook and never serialized to disk, so
there's no saved model to re-load here. Documented as such, not glossed
over: see docs/AI_ADVISOR_DESIGN.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import data_access as da

HEALTH_SCORE_WEIGHTS = {"dos": 0.20, "turnover": 0.15, "variability": 0.15,
                         "stockout": 0.25, "excess": 0.10, "service": 0.15}


def _df(name: str) -> pd.DataFrame:
    return da.load_all()[name]


def _row_to_records(df: pd.DataFrame, limit: Optional[int] = None) -> list[dict]:
    out = df.copy()
    if limit is not None:
        out = out.head(limit)
    for col in out.columns:
        if out[col].dtype == "object":
            continue
        if pd.api.types.is_bool_dtype(out[col]):
            continue
    return out.replace({np.nan: None}).to_dict(orient="records")


# --------------------------------------------------------------------------
# 1. Network-wide summary
# --------------------------------------------------------------------------

def get_network_summary() -> dict:
    """Overall health of the whole SKU x Warehouse network: total inventory
    value, total excess value, health tier counts, average service level."""
    r = _df("replenishment")
    return {
        "sku_warehouse_rows": int(len(r)),
        "total_inventory_value": round(float((r["on_hand_qty"] * r["unit_cost"]).sum()), 2),
        "total_excess_value": round(float(r["excess_value"].sum()), 2),
        "health_tier_counts": r["health_tier"].value_counts().to_dict(),
        "avg_service_level": round(float(r["service_level"].mean()), 4),
        "avg_inventory_turnover": round(float(r["inventory_turnover"].mean()), 2),
        "orders_now_count": int(r["needs_replenishment"].sum()),
        "excess_sku_count": int(r["is_excess"].sum()),
        "obsolete_sku_count": int(r["is_obsolete"].sum()),
    }


# --------------------------------------------------------------------------
# 2. Warehouse-level detail
# --------------------------------------------------------------------------

def get_warehouse_detail(warehouse_id: str) -> dict:
    """Inventory value, excess value, health tier breakdown, and the top 5
    excess SKUs for one warehouse."""
    r = _df("replenishment")
    sub = r[r["warehouse_id"] == warehouse_id]
    if sub.empty:
        w = _df("dim_warehouse")
        if warehouse_id in w["warehouse_id"].values:
            # A real warehouse in the network (real Brunel handling-cost/
            # capacity data exists for it) that simply has zero SKUs assigned
            # to it in ProductsPerPlant — not a data error, just an empty
            # warehouse. PLANT19 is the one case of this in the current data.
            return {"warehouse_id": warehouse_id,
                    "note": (f"{warehouse_id} is a real warehouse in the network (it has handling-cost "
                              "and capacity data), but no SKUs are assigned to it in the source data, so "
                              "it has no inventory, demand, or health metrics to report."),
                    "sku_count": 0, "total_inventory_value": 0.0}
        return {"error": f"No data for warehouse_id='{warehouse_id}'. "
                          f"Valid IDs: {sorted(r['warehouse_id'].unique().tolist())}"}
    top_excess = sub[sub["is_excess"]].sort_values("excess_value", ascending=False).head(5)
    return {
        "warehouse_id": warehouse_id,
        "region": sub["region"].iloc[0],
        "sku_count": int(len(sub)),
        "total_inventory_value": round(float((sub["on_hand_qty"] * sub["unit_cost"]).sum()), 2),
        "total_excess_value": round(float(sub["excess_value"].sum()), 2),
        "health_tier_counts": sub["health_tier"].value_counts().to_dict(),
        "avg_service_level": round(float(sub["service_level"].mean()), 4),
        "top_5_excess_skus": _row_to_records(
            top_excess[["sku", "category", "abc_class", "on_hand_qty", "unit_cost", "excess_value", "days_of_supply"]]
        ),
    }


# --------------------------------------------------------------------------
# 3. "Why did inventory change" — monthly diagnostic decomposition
# --------------------------------------------------------------------------

def why_is_inventory_high(warehouse_id: str, months_back: int = 3) -> dict:
    """Explains what's driving a warehouse's inventory value: recent
    month-over-month value change by ABC class, plus the specific excess
    SKUs currently sitting there."""
    m = _df("monthly_diagnostic")
    sub = m[m["warehouse_id"] == warehouse_id].sort_values("month")
    if sub.empty:
        return {"error": f"No monthly diagnostic data for warehouse_id='{warehouse_id}'."}
    recent = sub.tail(months_back * 3)  # 3 ABC classes per month
    warehouse_detail = get_warehouse_detail(warehouse_id)
    return {
        "warehouse_id": warehouse_id,
        "recent_monthly_value_by_abc_class": _row_to_records(
            recent[["month", "abc_class", "inventory_value", "delta"]]
        ),
        "current_excess_value": warehouse_detail.get("total_excess_value"),
        "top_5_excess_skus_driving_it": warehouse_detail.get("top_5_excess_skus"),
    }


# --------------------------------------------------------------------------
# 3b. Inventory value trend — network-wide OR one warehouse, up/down movers
# --------------------------------------------------------------------------

def get_inventory_trend(warehouse_id: Optional[str] = None, months_back: int = 6) -> dict:
    """Month-by-month total inventory value trend — for the whole network
    if warehouse_id is omitted, or for one warehouse if given. Also returns
    the single biggest month-over-month increase and decrease in the
    window, so this answers both 'why did inventory grow' and 'why did it
    shrink' — unlike why_is_inventory_high, which only looks at one
    warehouse's excess drivers. Prefer this tool for any general
    inventory-trend question; use why_is_inventory_high only when the user
    specifically wants the excess-SKU-level detail for one warehouse.

    Network-wide (warehouse_id omitted), the biggest mover is found at the
    WAREHOUSE level (each warehouse's own A+B+C classes summed first, then
    compared) — not on a single warehouse+class row, which would name a
    warehouse whose one class moved a lot even if its other classes moved
    the other way and mostly offset it. Scoped to one warehouse, the
    biggest mover is found at the ABC-class level instead, since warehouse
    is already fixed and class is the only remaining dimension to compare.
    """
    m = _df("monthly_diagnostic")
    sub = m if warehouse_id is None else m[m["warehouse_id"] == warehouse_id]
    if sub.empty:
        return {"error": f"No monthly diagnostic data for warehouse_id='{warehouse_id}'."}

    monthly_total = sub.groupby("month")["inventory_value"].sum().reset_index().sort_values("month")
    monthly_total["delta"] = monthly_total["inventory_value"].diff()
    monthly_total = monthly_total.tail(months_back)

    if warehouse_id is None:
        # Sum each warehouse's classes together FIRST, per month, so the
        # comparison below is one number per warehouse (its true net
        # change), not one number per warehouse+class segment.
        mover_basis = (
            sub.groupby(["month", "warehouse_id"])["inventory_value"].sum().reset_index()
        )
        mover_basis = mover_basis.sort_values(["warehouse_id", "month"])
        mover_basis["delta"] = mover_basis.groupby("warehouse_id")["inventory_value"].diff()
        mover_label = "whole warehouse (all ABC classes summed first, then compared across warehouses)"
    else:
        mover_basis = sub.sort_values(["abc_class", "month"]).copy()
        mover_basis["delta"] = mover_basis.groupby("abc_class")["inventory_value"].diff()
        mover_label = "one ABC class within this warehouse (warehouse is already fixed by the request)"

    recent_months = monthly_total["month"].tolist()
    detail = mover_basis[mover_basis["month"].isin(recent_months)].dropna(subset=["delta"])
    biggest_increase = detail.sort_values("delta", ascending=False).head(1)
    biggest_decrease = detail.sort_values("delta", ascending=True).head(1)

    return {
        "scope": warehouse_id or "whole network",
        "monthly_total_value_trend": _row_to_records(
            monthly_total.rename(columns={"inventory_value": "total_inventory_value"})
        ),
        "biggest_mover_grain": mover_label,
        "biggest_increase_in_window": _row_to_records(biggest_increase)[0] if len(biggest_increase) else None,
        "biggest_decrease_in_window": _row_to_records(biggest_decrease)[0] if len(biggest_decrease) else None,
    }


# --------------------------------------------------------------------------
# 3c. Consumption ($) value — network-wide or per-warehouse, estimated from
#     each SKU x Warehouse row's own trailing 90-day average daily demand.
#     (No raw daily transaction table is shipped with the deployed app — only
#     the aggregated 90-day figure kpi_engine already computed — so this is
#     an explicit estimate, not a literal sum of one calendar month's actual
#     transactions, and says so via method_note.)
# --------------------------------------------------------------------------

def get_consumption_value(warehouse_id: Optional[str] = None) -> dict:
    """Estimated monthly consumption ($) value — network-wide, or for one
    warehouse — derived from each row's trailing-90-day average daily demand
    x unit cost x ~30.44 days. Use this whenever the user asks how much
    inventory is being consumed/used per month, in dollars, by warehouse."""
    r = _df("replenishment")
    sub = r if warehouse_id is None else r[r["warehouse_id"] == warehouse_id]
    if sub.empty:
        return {"error": f"No data for warehouse_id='{warehouse_id}'."}
    sub = sub.copy()
    sub["monthly_consumption_value"] = sub["avg_daily_demand_90d"] * sub["unit_cost"] * 30.44
    by_warehouse = (sub.groupby("warehouse_id")["monthly_consumption_value"].sum()
                     .round(2).reset_index().sort_values("monthly_consumption_value", ascending=False))
    return {
        "scope": warehouse_id or "whole network",
        "total_monthly_consumption_value": round(float(sub["monthly_consumption_value"].sum()), 2),
        "by_warehouse": _row_to_records(by_warehouse) if warehouse_id is None else None,
        "method_note": ("Estimated as trailing-90-day average daily demand x unit cost x 30.44 days per "
                         "SKU x Warehouse row, then summed — not a literal sum of one calendar month's "
                         "actual transactions, since only the aggregated 90-day figure is available to "
                         "this app (no raw daily demand table is shipped for deployment size reasons)."),
    }


# --------------------------------------------------------------------------
# 4. Stockout risk
# --------------------------------------------------------------------------

def get_stockout_risks(min_probability: float = 0.5, limit: int = 10) -> dict:
    """Highest stockout-risk SKU x Warehouse rows (30-day horizon)."""
    r = _df("replenishment")
    sub = r[r["stockout_prob_30d"] >= min_probability].sort_values("stockout_prob_30d", ascending=False)
    return {
        "matching_count": int(len(sub)),
        "results": _row_to_records(
            sub[["sku", "warehouse_id", "category", "abc_class", "stockout_prob_30d",
                 "on_hand_qty", "dynamic_reorder_point", "recommended_action"]],
            limit=limit,
        ),
    }


# --------------------------------------------------------------------------
# 4b. Excess inventory items (network-wide, with $ value — unlike
#     get_replenishment_recommendations' "Reduce / hold" filter, which
#     returns the same SKUs but without excess_value)
# --------------------------------------------------------------------------

def get_excess_items(warehouse_id: Optional[str] = None, limit: int = 20) -> dict:
    """SKU x Warehouse rows currently flagged as excess stock (is_excess),
    sorted by excess $ value descending — network-wide, or filtered to one
    warehouse. Use this whenever the user asks for an excess-inventory
    report/list with dollar values, not just a warehouse's top 5."""
    r = _df("replenishment")
    sub = r[r["is_excess"]]
    if warehouse_id:
        sub = sub[sub["warehouse_id"] == warehouse_id]
    if sub.empty:
        return {"matching_count": 0, "total_excess_value": 0.0, "results": []}
    sub = sub.sort_values("excess_value", ascending=False)
    return {
        "matching_count": int(len(sub)),
        "total_excess_value": round(float(sub["excess_value"].sum()), 2),
        "results": _row_to_records(
            sub[["sku", "warehouse_id", "category", "abc_class", "on_hand_qty",
                 "unit_cost", "excess_value", "days_of_supply"]],
            limit=limit,
        ),
    }


# --------------------------------------------------------------------------
# 5. Single-SKU detail
# --------------------------------------------------------------------------

def get_sku_detail(sku: str, warehouse_id: Optional[str] = None) -> dict:
    """Full current state of one SKU — across all warehouses, or one
    specific warehouse if given."""
    r = _df("replenishment")
    sub = r[r["sku"].astype(str) == str(sku)]
    if warehouse_id:
        sub = sub[sub["warehouse_id"] == warehouse_id]
    if sub.empty:
        return {"error": f"No data for sku='{sku}'" + (f" at warehouse_id='{warehouse_id}'" if warehouse_id else "")}
    return {"sku": sku, "rows": _row_to_records(sub)}


# --------------------------------------------------------------------------
# 6. Replenishment recommendations
# --------------------------------------------------------------------------

def get_replenishment_recommendations(warehouse_id: Optional[str] = None,
                                       action: Optional[str] = None, limit: int = 20) -> dict:
    """Filterable list of Phase 5 reorder recommendations. `action` matches
    one of: 'Order now — high stockout risk', 'Order now',
    'Reduce / hold — excess stock', 'Monitor'.

    When `action` is omitted (an open 'what needs doing here' question),
    every non-Monitor row — any 'Order now...' or 'Reduce / hold...' action
    — is always included in full, however low its stockout_prob_30d is
    relative to other rows. Sorting everything by stockout_prob_30d and
    then cutting at `limit` could otherwise bump a genuinely urgent action
    (e.g. a large 'Order now' with a merely middling stockout probability)
    outside the returned window while lower-priority 'Monitor' rows with a
    higher probability fill it — silently hiding the one thing the caller
    most needs to see. Only the routine 'Monitor' rows are subject to
    `limit` here. When `action` IS given, `limit` applies to that filtered
    set as before, sorted by stockout_prob_30d."""
    r = _df("replenishment")
    sub = r
    if warehouse_id:
        sub = sub[sub["warehouse_id"] == warehouse_id]

    cols = ["sku", "warehouse_id", "category", "recommended_action",
            "recommended_order_qty", "days_until_reorder", "stockout_prob_30d"]

    if action:
        sub = sub[sub["recommended_action"] == action].sort_values("stockout_prob_30d", ascending=False)
        return {
            "matching_count": int(len(sub)),
            "results": _row_to_records(sub[cols], limit=limit),
        }

    urgent = sub[sub["recommended_action"] != "Monitor"].sort_values("stockout_prob_30d", ascending=False)
    monitor = sub[sub["recommended_action"] == "Monitor"].sort_values("stockout_prob_30d", ascending=False)
    monitor_shown = max(limit - len(urgent), 0)
    combined = pd.concat([urgent, monitor.head(monitor_shown)])

    return {
        "matching_count": int(len(sub)),
        "urgent_action_count": int(len(urgent)),
        "note": ("All non-Monitor (urgent) rows are included above regardless of limit; only "
                 f"{monitor_shown} of {len(monitor)} routine 'Monitor' rows are shown, sorted by "
                 "stockout_prob_30d, to keep the response short — ask for a specific warehouse or "
                 "action to see more."),
        "results": _row_to_records(combined[cols]),
    }


# --------------------------------------------------------------------------
# 7. Transfer vs. fresh PO (Phase 5 optimization result)
# --------------------------------------------------------------------------

def get_transfer_recommendations() -> dict:
    """Phase 5's inter-warehouse transfer plan (excess at one warehouse
    covering a shortage of the same SKU elsewhere) and what's left over
    needing a fresh purchase order."""
    t = _df("transfers")
    u = _df("unmet_shortage")
    return {
        "transfer_moves": _row_to_records(t),
        "total_transfer_qty": int(t["transfer_qty"].sum()) if len(t) else 0,
        "total_transfer_cost": round(float(t["total_cost"].sum()), 2) if len(t) else 0.0,
        "fresh_po_needed_units": round(float(u["unmet_via_fresh_po"].sum()), 0),
        "fresh_po_needed_value": round(float((u["unmet_via_fresh_po"] * u["unit_cost"]).sum()), 2),
        "note": ("Most genuine shortages have no matching excess of the same SKU elsewhere in the "
                 "network, so they route to a fresh PO rather than an invented transfer — see "
                 "docs/SYNTHETIC_DATA_METHODOLOGY.md and notebooks/05_prescriptive_optimization.ipynb."),
    }


# --------------------------------------------------------------------------
# 8. Forecast accuracy
# --------------------------------------------------------------------------

def get_forecast_accuracy(sku: Optional[str] = None, warehouse_id: Optional[str] = None) -> dict:
    """WAPE (Weighted Absolute Percentage Error) for the Phase 4 demand
    forecast, overall or filtered to one SKU/warehouse."""
    f = _df("forecast_test")
    sub = f
    if sku:
        sub = sub[sub["sku"].astype(str) == str(sku)]
    if warehouse_id:
        sub = sub[sub["warehouse_id"] == warehouse_id]
    if sub.empty:
        return {"error": "No matching forecast test rows for that filter."}
    denom = sub["demand_qty"].abs().sum()
    wape = float((sub["demand_qty"] - sub["prediction"]).abs().sum() / denom) if denom > 0 else None
    return {
        "rows_evaluated": int(len(sub)),
        "wape": round(wape, 4) if wape is not None else None,
        "total_actual_demand": round(float(sub["demand_qty"].sum()), 1),
        "total_forecasted_demand": round(float(sub["prediction"].sum()), 1),
    }


# --------------------------------------------------------------------------
# 9. Health score explanation (rule-based decomposition — see module docstring)
# --------------------------------------------------------------------------

def explain_health_score(sku: str, warehouse_id: str) -> dict:
    """Breaks the 0-100 Inventory Health Score down into its 6 weighted
    components for one SKU x Warehouse row, so 'why is this Critical/Watch'
    has a concrete answer instead of just the number."""
    r = _df("replenishment")
    sub = r[(r["sku"].astype(str) == str(sku)) & (r["warehouse_id"] == warehouse_id)]
    if sub.empty:
        return {"error": f"No data for sku='{sku}' at warehouse_id='{warehouse_id}'."}
    row = sub.iloc[0]

    dos_ratio = min(max((row["days_of_supply"] if np.isfinite(row["days_of_supply"]) else 9999) / row["target_dos"], 0), 5)
    dos_score = max(100 - abs(dos_ratio - 1) * 40, 0)
    turnover_score = min(max(row["inventory_turnover"], 0), 20) / 20 * 100
    variability_score = 100 - min(max(row["cv"], 0), 3) / 3 * 100
    stockout_score = 100 - row["stockout_prob_30d"] * 100
    excess_score = 20 if row["is_excess"] else 100
    service_score = row["service_level"] * 100

    components = {
        "days_of_supply_score": {"value": round(dos_score, 1), "weight": HEALTH_SCORE_WEIGHTS["dos"],
                                  "raw_days_of_supply": round(float(row["days_of_supply"]), 1) if np.isfinite(row["days_of_supply"]) else "inf",
                                  "target_days_of_supply": row["target_dos"]},
        "turnover_score": {"value": round(turnover_score, 1), "weight": HEALTH_SCORE_WEIGHTS["turnover"],
                            "raw_turnover": round(float(row["inventory_turnover"]), 2)},
        "variability_score": {"value": round(variability_score, 1), "weight": HEALTH_SCORE_WEIGHTS["variability"],
                               "raw_demand_cv": round(float(row["cv"]), 2)},
        "stockout_score": {"value": round(stockout_score, 1), "weight": HEALTH_SCORE_WEIGHTS["stockout"],
                            "raw_stockout_prob_30d": round(float(row["stockout_prob_30d"]), 3)},
        "excess_score": {"value": excess_score, "weight": HEALTH_SCORE_WEIGHTS["excess"],
                          "is_excess": bool(row["is_excess"])},
        "service_score": {"value": round(service_score, 1), "weight": HEALTH_SCORE_WEIGHTS["service"],
                           "raw_service_level": round(float(row["service_level"]), 3)},
    }
    return {
        "sku": sku, "warehouse_id": warehouse_id,
        "inventory_health_score": round(float(row["inventory_health_score"]), 1),
        "health_tier": row["health_tier"],
        "components": components,
        "method_note": ("Rule-based decomposition of the weighted composite score (recomputed from "
                        "this row's own KPI fields, same weights as kpi_engine.compute_health_score) "
                        "— not a live SHAP explanation, since the Phase 4 models were trained in a "
                        "notebook and never serialized to disk."),
    }


# --------------------------------------------------------------------------
# OpenAI function-calling schemas
# --------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "get_network_summary": get_network_summary,
    "get_warehouse_detail": get_warehouse_detail,
    "get_inventory_trend": get_inventory_trend,
    "get_consumption_value": get_consumption_value,
    "why_is_inventory_high": why_is_inventory_high,
    "get_stockout_risks": get_stockout_risks,
    "get_excess_items": get_excess_items,
    "get_sku_detail": get_sku_detail,
    "get_replenishment_recommendations": get_replenishment_recommendations,
    "get_transfer_recommendations": get_transfer_recommendations,
    "get_forecast_accuracy": get_forecast_accuracy,
    "explain_health_score": explain_health_score,
}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_network_summary",
        "description": "Overall health of the whole network: total inventory value, total excess value, health tier counts, average service level and turnover.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_warehouse_detail",
        "description": "Inventory value, excess value, health tier breakdown, and top 5 excess SKUs for one warehouse (e.g. 'PLANT03').",
        "parameters": {"type": "object", "properties": {
            "warehouse_id": {"type": "string", "description": "Warehouse ID, e.g. 'PLANT03'."},
        }, "required": ["warehouse_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_inventory_trend",
        "description": "Month-by-month total inventory value trend, network-wide (omit warehouse_id) or for one warehouse, plus the single biggest increase and decrease in the window. Use this FIRST for any general 'why did inventory go up/down' question, at any scope.",
        "parameters": {"type": "object", "properties": {
            "warehouse_id": {"type": "string", "description": "Omit for the whole network."},
            "months_back": {"type": "integer", "description": "How many recent months to include (default 6)."},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_consumption_value",
        "description": "Estimated monthly consumption ($) value, network-wide (with a per-warehouse breakdown) or for one warehouse. Use for any 'how much is being consumed/used per month, in dollars' question — different from inventory VALUE (get_inventory_trend), which is stock sitting on hand, not stock being used up.",
        "parameters": {"type": "object", "properties": {
            "warehouse_id": {"type": "string", "description": "Omit for the whole network (returns a per-warehouse breakdown)."},
        }},
    }},
    {"type": "function", "function": {
        "name": "why_is_inventory_high",
        "description": "Deep-dive on ONE warehouse's excess drivers: recent month-over-month value change by ABC class plus the top 5 SKUs currently sitting in excess there. Use only when the user wants excess-SKU-level detail for a specific warehouse; for general trend questions use get_inventory_trend instead.",
        "parameters": {"type": "object", "properties": {
            "warehouse_id": {"type": "string"},
            "months_back": {"type": "integer", "description": "How many recent months of trend to include (default 3)."},
        }, "required": ["warehouse_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_stockout_risks",
        "description": "Highest stockout-risk SKU x Warehouse rows over the next 30 days, sorted descending by risk.",
        "parameters": {"type": "object", "properties": {
            "min_probability": {"type": "number", "description": "Minimum 30-day stockout probability, 0-1 (default 0.5)."},
            "limit": {"type": "integer", "description": "Max rows to return (default 10)."},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_excess_items",
        "description": "Network-wide (or one-warehouse) report of SKUs currently flagged as excess stock, sorted by excess dollar value descending, with a total excess value for the filter. Use this for any 'excess inventory report/list' question — get_replenishment_recommendations' 'Reduce / hold' filter returns the same SKUs but without dollar values.",
        "parameters": {"type": "object", "properties": {
            "warehouse_id": {"type": "string", "description": "Omit for the whole network."},
            "limit": {"type": "integer", "description": "Max rows to return (default 20)."},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_sku_detail",
        "description": "Full current KPI/health/risk state of one SKU, across all warehouses or one specific warehouse.",
        "parameters": {"type": "object", "properties": {
            "sku": {"type": "string"},
            "warehouse_id": {"type": "string", "description": "Optional — omit to see the SKU at every warehouse it's stocked in."},
        }, "required": ["sku"]},
    }},
    {"type": "function", "function": {
        "name": "get_replenishment_recommendations",
        "description": "Filterable list of Phase 5 reorder recommendations, optionally by warehouse and/or action type.",
        "parameters": {"type": "object", "properties": {
            "warehouse_id": {"type": "string"},
            "action": {"type": "string", "enum": ["Order now — high stockout risk", "Order now",
                                                   "Reduce / hold — excess stock", "Monitor"]},
            "limit": {"type": "integer", "description": "Max rows to return (default 20)."},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_transfer_recommendations",
        "description": "Phase 5's inter-warehouse transfer plan and what's left needing a fresh purchase order instead.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_forecast_accuracy",
        "description": "WAPE for the Phase 4 demand forecast, overall or filtered to one SKU/warehouse.",
        "parameters": {"type": "object", "properties": {
            "sku": {"type": "string"},
            "warehouse_id": {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "explain_health_score",
        "description": "Breaks one SKU x Warehouse's Inventory Health Score down into its 6 weighted components, to answer 'why is this Critical/Watch/Healthy'.",
        "parameters": {"type": "object", "properties": {
            "sku": {"type": "string"},
            "warehouse_id": {"type": "string"},
        }, "required": ["sku", "warehouse_id"]},
    }},
]
