"""
optimization_engine.py
=======================
Phase 5 — prescriptive optimization. Turns Phase 3's diagnostics and Phase
4's predictive signals into concrete, costed actions:

1. A dynamic reorder recommendation per SKU x Warehouse (order now / don't,
   how much, how urgent) — reorder point recomputed from the latest observed
   90-day demand rather than the static generator-time policy table, so it
   reflects what's actually been happening lately.
2. Inter-warehouse transfer recommendations: SKUs that are excess in one
   warehouse and short in another (of the SAME SKU) are matched via a small
   linear program per SKU (`scipy.optimize.linprog`) that minimizes total
   network cost — internal transfer vs. a fresh, rush purchase order — for
   every unit of shortage across the network.

A note on the transfer-cost model (read this before trusting the dollar
figures): the real Brunel `FreightRates` table looks, at first glance, like
exactly what a transfer-cost matrix needs — but every single lane in it
terminates at one port (PORT09). It is an *inbound* freight-rate card (cost
to ship components INTO the plants' receiving hub), not a plant-to-plant
distribution network, so it cannot legitimately be repurposed as an
any-to-any transfer-cost matrix — there is no real lane data for, say,
PLANT01 -> PLANT05. Rather than force-fit data that doesn't fit, transfer
cost here uses the real per-warehouse `handling_cost_per_unit` (from Brunel
WhCosts) at both ends, plus an explicit, illustrative same-region /
cross-region surcharge (`REGION_SURCHARGE` below) standing in for the real
freight-lane data this public dataset doesn't include. This is a documented
planning assumption, not a fitted or scraped number — same spirit as
`TARGET_DOS_BY_ABC` in `kpi_engine.py`.
"""
from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Illustrative planning assumptions (documented, not fitted/scraped) — see
# module docstring for why real freight-rate data can't be used here.
# --------------------------------------------------------------------------
REGION_SURCHARGE = {"same": 0.50, "cross": 2.00}   # $ per unit, on top of real handling costs
PO_RUSH_PREMIUM_FRACTION = 0.08   # extra cost/risk of a fresh expedited PO vs. an internal transfer, as a fraction of unit cost
EXCESS_MULTIPLIER = 2.0           # keep in sync with kpi_engine.EXCESS_MULTIPLIER


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_inputs(root: Path) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(root / "data" / "inventory_performance.db")
    dim_warehouse = pd.read_sql("SELECT * FROM dim_warehouse", con)
    dim_policy = pd.read_sql("SELECT * FROM dim_replenishment_policy", con)
    con.close()

    health = pd.read_parquet(root / "data" / "processed" / "kpi_tables" / "sku_warehouse_health_scored.parquet")
    dim_policy = dim_policy.rename(columns={"warehouse": "warehouse_id"})
    return {"health": health, "policy": dim_policy, "warehouse": dim_warehouse}


# --------------------------------------------------------------------------
# 1. Dynamic replenishment recommendation
# --------------------------------------------------------------------------

def compute_dynamic_replenishment(health: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    df = health.merge(
        policy[["sku", "warehouse_id", "lead_time_days", "safety_stock", "eoq"]],
        on=["sku", "warehouse_id"], how="left",
    )
    # Reorder point recomputed from the latest observed 90-day demand (Phase
    # 3/4 signal), not the static generator-time value in dim_replenishment_policy.
    df["dynamic_reorder_point"] = df["avg_daily_demand_90d"] * df["lead_time_days"] + df["safety_stock"]
    df["needs_replenishment"] = df["on_hand_qty"] <= df["dynamic_reorder_point"]
    df["recommended_order_qty"] = np.where(df["needs_replenishment"], df["eoq"].round(0), 0)

    # Days until on-hand crosses the reorder point, for items not yet due —
    # a simple urgency signal (0 for items already due).
    days_until = (df["on_hand_qty"] - df["dynamic_reorder_point"]) / df["avg_daily_demand_90d"].replace(0, np.nan)
    df["days_until_reorder"] = np.where(df["needs_replenishment"], 0, days_until.clip(lower=0)).round(1)
    df["days_until_reorder"] = df["days_until_reorder"].fillna(9999)  # zero recent demand => not reorder-driven

    df["recommended_action"] = np.select(
        [df["needs_replenishment"] & (df["stockout_prob_30d"] >= 0.5),
         df["needs_replenishment"],
         df["is_excess"]],
        ["Order now — high stockout risk", "Order now", "Reduce / hold — excess stock"],
        default="Monitor",
    )
    return df


# --------------------------------------------------------------------------
# 2. Excess / shortage quantities feeding the transfer LP
# --------------------------------------------------------------------------

def compute_excess_and_shortage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    target_units = out["target_dos"] * out["avg_daily_demand_90d"] * EXCESS_MULTIPLIER
    out["excess_qty"] = np.where(out["is_excess"], (out["on_hand_qty"] - target_units).clip(lower=0), 0.0)
    out["shortage_qty"] = np.where(out["needs_replenishment"],
                                    (out["dynamic_reorder_point"] - out["on_hand_qty"]).clip(lower=0), 0.0)
    return out


# --------------------------------------------------------------------------
# Transfer cost matrix (see module docstring for methodology)
# --------------------------------------------------------------------------

def build_transfer_cost_matrix(dim_warehouse: pd.DataFrame) -> pd.DataFrame:
    wh = dim_warehouse.set_index("warehouse_id")
    ids = wh.index.tolist()
    rows = []
    for o in ids:
        for d in ids:
            if o == d:
                continue
            surcharge = REGION_SURCHARGE["same"] if wh.loc[o, "region"] == wh.loc[d, "region"] else REGION_SURCHARGE["cross"]
            cost = wh.loc[o, "handling_cost_per_unit"] + wh.loc[d, "handling_cost_per_unit"] + surcharge
            rows.append({"from_warehouse": o, "to_warehouse": d, "cost_per_unit": cost})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Per-SKU transfer optimization (small LP, scipy.optimize.linprog)
# --------------------------------------------------------------------------

def _optimize_one_sku(sku: str, supply: pd.Series, demand: pd.Series, unit_cost: float,
                       cost_matrix: pd.DataFrame) -> list[dict]:
    """supply: warehouse_id -> excess_qty (>0). demand: warehouse_id -> shortage_qty (>0).
    Decision vars: x_ij (transfer i->j) for every (i in supply, j in demand) pair,
    plus u_j (unmet shortage at j, covered by a fresh rush PO instead)."""
    sources = list(supply.index)
    sinks = list(demand.index)
    n_x = len(sources) * len(sinks)
    po_cost = unit_cost * (1 + PO_RUSH_PREMIUM_FRACTION)

    cm = cost_matrix.set_index(["from_warehouse", "to_warehouse"])["cost_per_unit"]
    c = []
    for i in sources:
        for j in sinks:
            c.append(cm.get((i, j), REGION_SURCHARGE["cross"] * 3))  # fallback if pair missing
    c += [po_cost] * len(sinks)  # u_j costs

    n_vars = n_x + len(sinks)
    A_ub, b_ub = [], []
    # supply constraints: sum_j x_ij <= excess_i
    for si, i in enumerate(sources):
        row = [0.0] * n_vars
        for sj in range(len(sinks)):
            row[si * len(sinks) + sj] = 1.0
        A_ub.append(row)
        b_ub.append(supply[i])

    A_eq, b_eq = [], []
    # demand constraints: sum_i x_ij + u_j == shortage_j
    for sj, j in enumerate(sinks):
        row = [0.0] * n_vars
        for si in range(len(sources)):
            row[si * len(sinks) + sj] = 1.0
        row[n_x + sj] = 1.0
        A_eq.append(row)
        b_eq.append(demand[j])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return []

    out = []
    x = res.x[:n_x].reshape(len(sources), len(sinks))
    for si, i in enumerate(sources):
        for sj, j in enumerate(sinks):
            qty = x[si, sj]
            if qty > 0.5:
                out.append({"sku": sku, "from_warehouse": i, "to_warehouse": j,
                            "transfer_qty": round(qty), "cost_per_unit": round(cm.get((i, j), np.nan), 2),
                            "total_cost": round(qty * cm.get((i, j), 0), 2)})
    return out


def optimize_transfers(df: pd.DataFrame, cost_matrix: pd.DataFrame) -> pd.DataFrame:
    results = []
    multi_wh_skus = df.groupby("sku")["warehouse_id"].nunique()
    candidate_skus = multi_wh_skus[multi_wh_skus > 1].index
    for sku in candidate_skus:
        sub = df[df["sku"] == sku]
        supply = sub.loc[sub["excess_qty"] > 0].set_index("warehouse_id")["excess_qty"]
        demand = sub.loc[sub["shortage_qty"] > 0].set_index("warehouse_id")["shortage_qty"]
        if supply.empty or demand.empty:
            continue
        unit_cost = sub["unit_cost"].iloc[0]
        results.extend(_optimize_one_sku(sku, supply, demand, unit_cost, cost_matrix))
    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["sku", "from_warehouse", "to_warehouse", "transfer_qty", "cost_per_unit", "total_cost"])


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_all_recommendations(root: Path) -> dict[str, pd.DataFrame]:
    out_dir = root / "data" / "processed" / "recommendations"
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(root)
    df = compute_dynamic_replenishment(inputs["health"], inputs["policy"])
    df = compute_excess_and_shortage(df)
    cost_matrix = build_transfer_cost_matrix(inputs["warehouse"])
    transfers = optimize_transfers(df, cost_matrix)

    # Fresh-PO fallback for any shortage the LP left unmatched (no excess
    # anywhere in the network for that SKU) — still a valid recommendation,
    # just not a transfer.
    matched_shortage = transfers.groupby(["sku", "to_warehouse"])["transfer_qty"].sum() if len(transfers) else pd.Series(dtype=float)
    shortage_rows = df.loc[df["shortage_qty"] > 0, ["sku", "warehouse_id", "shortage_qty", "unit_cost"]].copy()
    shortage_rows["matched_by_transfer"] = shortage_rows.apply(
        lambda r: matched_shortage.get((r["sku"], r["warehouse_id"]), 0.0), axis=1)
    shortage_rows["unmet_via_fresh_po"] = (shortage_rows["shortage_qty"] - shortage_rows["matched_by_transfer"]).clip(lower=0).round(0)

    tables = {
        "replenishment_recommendations": df,
        "transfer_recommendations": transfers,
        "unmet_shortage_fresh_po": shortage_rows,
        "transfer_cost_matrix": cost_matrix,
    }
    for name, tdf in tables.items():
        tdf.to_parquet(out_dir / f"{name}.parquet", index=False)
    return tables


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    tables = build_all_recommendations(root)
    df = tables["replenishment_recommendations"]
    transfers = tables["transfer_recommendations"]
    unmet = tables["unmet_shortage_fresh_po"]

    print(f"SKU x Warehouse rows evaluated: {len(df)}")
    print(df["recommended_action"].value_counts().to_string())
    print()
    print(f"Transfer recommendations: {len(transfers)} moves, "
          f"{transfers['transfer_qty'].sum() if len(transfers) else 0:.0f} units, "
          f"${transfers['total_cost'].sum() if len(transfers) else 0:,.0f} total transfer cost")
    print(f"Shortage still requiring a fresh PO: {unmet['unmet_via_fresh_po'].sum():.0f} units "
          f"across {(unmet['unmet_via_fresh_po'] > 0).sum()} SKU x Warehouse rows")
    if len(transfers):
        est_po_cost = (transfers.merge(df[["sku", "unit_cost"]].drop_duplicates(), on="sku")
                        .assign(would_be_po_cost=lambda d: d["transfer_qty"] * d["unit_cost"] * (1 + PO_RUSH_PREMIUM_FRACTION))
                        ["would_be_po_cost"].sum())
        print(f"Estimated savings vs. fresh POs for the transferred units: ${est_po_cost - transfers['total_cost'].sum():,.0f}")
    print("Done.")
