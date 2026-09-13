"""
simulate_inventory.py
======================
Generates the synthetic daily inventory & demand layer for this project,
calibrated to the REAL warehouse network in the Brunel Supply Chain Logistics
Problem Dataset (plant capacities, handling costs, and freight lead times).

No public dataset audited for this project publishes real on-hand stock
levels (see docs/DATA_PROVENANCE.md and the Inventory Intelligence
Sourcebook, Section A.3). This module is the documented, reproducible
answer to that gap — every assumption it makes is listed in
docs/SYNTHETIC_DATA_METHODOLOGY.md. License: CC0 (this project's own work).

Two things it does NOT claim to be:
  1. Real historical Metso/Brunel operational data — it is a simulation.
  2. A copy of the Brunel OrderList time axis — it runs on its own
     independent 2-year simulation window; only the *network structure*
     (which SKUs sit at which plants, plant capacity/cost, real transit
     lead times) is taken from the real data.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Reproducibility: every "random" draw below is seeded from a hash of a
# stable business key (SKU, warehouse, ...), never from a bare global seed.
# That means re-running this script always regenerates byte-identical data,
# and adding one more SKU later doesn't reshuffle everyone else's numbers.
# --------------------------------------------------------------------------

def _seed_from(*parts: str) -> int:
    h = hashlib.sha256("::".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


SIM_START = pd.Timestamp("2024-01-01")
SIM_END = pd.Timestamp("2025-12-31")
SIM_DATES = pd.date_range(SIM_START, SIM_END, freq="D")
N_DAYS = len(SIM_DATES)

# Illustrative region labels. The Brunel dataset anonymizes plant/port codes
# (PLANT01, PORT01, ...) with no country field, so these are analyst-assigned
# for narrative/Power-BI-map purposes only — NOT derived from the source
# data. Documented explicitly; never presented as real geography.
REGIONS = [
    "North America", "Europe", "Asia-Pacific", "Latin America",
    "Middle East & Africa", "South Asia",
]
PRODUCT_CATEGORIES = [
    "Wear Parts", "Consumables", "Spare Components", "Filtration & Fluids",
    "Electrical & Sensors", "Fasteners & Hardware",
]

HOLDING_COST_RATE = 0.22          # annual holding cost as a fraction of unit value (industry-typical 18-25%)
ORDERING_COST_RANGE = (35.0, 180.0)  # illustrative fixed cost per replenishment order, by category
SERVICE_LEVEL_BY_ABC = {"A": 0.975, "B": 0.95, "C": 0.90}
Z_BY_SERVICE_LEVEL = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600}


@dataclass
class Policy:
    sku: str
    warehouse: str
    abc_class: str
    unit_cost: float
    mean_daily_demand: float
    demand_cv: float
    lead_time_days: int
    eoq: float
    safety_stock: float
    reorder_point: float


def load_brunel_tables(processed_dir: Path) -> dict[str, pd.DataFrame]:
    names = ["OrderList", "FreightRates", "WhCosts", "WhCapacities", "ProductsPerPlant", "PlantPorts"]
    return {n: pd.read_parquet(processed_dir / f"brunel_{n}.parquet") for n in names}


def build_dim_warehouse(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    wh = tables["WhCosts"].merge(tables["WhCapacities"], on="Plant_Code")
    ports = tables["PlantPorts"].groupby("Plant_Code")["Ports"].apply(list).reset_index()
    wh = wh.merge(ports, on="Plant_Code", how="left")
    wh["region"] = wh["Plant_Code"].apply(lambda p: REGIONS[_seed_from("region", p) % len(REGIONS)])
    wh = wh.rename(columns={
        "Plant_Code": "warehouse_id",
        "Cost_Per_Unit": "handling_cost_per_unit",
        "Daily_Capacity": "daily_throughput_capacity",
        "Ports": "linked_ports",
    })
    return wh


def build_dim_sku(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sku_wh = tables["ProductsPerPlant"].rename(columns={"Plant_Code": "warehouse_id", "Product_ID": "sku"})
    unique_skus = sku_wh["sku"].drop_duplicates().reset_index(drop=True)

    rows = []
    for sku in unique_skus:
        seed = _seed_from("sku", sku)
        rng = np.random.default_rng(seed)
        unit_cost = float(np.round(rng.lognormal(mean=4.0, sigma=1.15), 2))       # ~ $5 to a few thousand $, spare-parts-like spread
        base_annual_demand = float(rng.lognormal(mean=5.2, sigma=1.3))             # heavy right tail -> a few fast movers, many slow movers
        category = PRODUCT_CATEGORIES[seed % len(PRODUCT_CATEGORIES)]
        rows.append((sku, unit_cost, base_annual_demand, category))

    dim = pd.DataFrame(rows, columns=["sku", "unit_cost", "base_annual_demand", "category"])
    dim["annual_value"] = dim["unit_cost"] * dim["base_annual_demand"]

    dim = dim.sort_values("annual_value", ascending=False).reset_index(drop=True)
    cum_share = dim["annual_value"].cumsum() / dim["annual_value"].sum()
    dim["abc_class"] = np.select(
        [cum_share <= 0.70, cum_share <= 0.90],
        ["A", "B"],
        default="C",
    )
    return dim, sku_wh


def _lead_time_by_warehouse(tables: dict[str, pd.DataFrame], dim_wh: pd.DataFrame) -> dict[str, int]:
    freight = tables["FreightRates"]
    global_avg = int(round(freight["TPT_Day_Count"].mean()))
    lt = {}
    for _, row in dim_wh.iterrows():
        ports = row["linked_ports"] if isinstance(row["linked_ports"], list) else []
        matched = freight[freight["Orig_Port"].isin(ports)]
        lt[row["warehouse_id"]] = int(round(matched["TPT_Day_Count"].mean())) if len(matched) else global_avg
    return lt


def _demand_series(mean_daily: float, cv: float, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Intermittent, over-dispersed daily demand via a Negative Binomial draw
    (captures the lumpy, many-zero-days pattern typical of industrial spare
    parts far better than a Normal/Poisson would)."""
    mean_daily = max(mean_daily, 1e-6)
    variance = max((cv * mean_daily) ** 2, mean_daily * 1.05)
    p = mean_daily / variance
    p = min(max(p, 1e-4), 0.999)
    r = mean_daily * p / (1 - p)
    r = max(r, 1e-3)
    return rng.negative_binomial(r, p, size=n_days)


def simulate_all(processed_dir: Path, synthetic_dir: Path) -> None:
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    tables = load_brunel_tables(processed_dir)

    dim_wh = build_dim_warehouse(tables)
    dim_sku, sku_wh_pairs = build_dim_sku(tables)
    lead_time_by_wh = _lead_time_by_warehouse(tables, dim_wh)

    sku_lookup = dim_sku.set_index("sku")
    wh_lookup = dim_wh.set_index("warehouse_id")

    inventory_rows = []
    demand_rows = []
    order_rows = []
    policy_rows = []
    order_id_counter = 1

    for _, pair in sku_wh_pairs.iterrows():
        sku, wh = pair["sku"], pair["warehouse_id"]
        if sku not in sku_lookup.index or wh not in wh_lookup.index:
            continue
        srow = sku_lookup.loc[sku]
        seed = _seed_from("series", sku, wh)
        rng = np.random.default_rng(seed)

        abc = srow["abc_class"]
        target_service = SERVICE_LEVEL_BY_ABC[abc]
        z = Z_BY_SERVICE_LEVEL[target_service]

        cv = {"A": rng.uniform(0.3, 0.6), "B": rng.uniform(0.6, 1.1), "C": rng.uniform(1.1, 2.3)}[abc]
        # split each SKU's total annual demand across the (usually few) warehouses it's stocked at
        n_wh_for_sku = (sku_wh_pairs["sku"] == sku).sum()
        mean_daily = (srow["base_annual_demand"] / max(n_wh_for_sku, 1)) / 365.0

        unit_cost = float(srow["unit_cost"])
        lead_time = lead_time_by_wh.get(wh, 7)
        annual_demand = mean_daily * 365.0
        ordering_cost = ORDERING_COST_RANGE[0] + (seed % 1000) / 999 * (ORDERING_COST_RANGE[1] - ORDERING_COST_RANGE[0])
        holding_cost = max(HOLDING_COST_RATE * unit_cost, 0.01)

        eoq = float(np.sqrt(max(2 * annual_demand * ordering_cost / holding_cost, 1.0)))
        demand_std_daily = cv * mean_daily
        safety_stock = float(z * demand_std_daily * np.sqrt(lead_time))
        reorder_point = mean_daily * lead_time + safety_stock

        demand = _demand_series(mean_daily, cv, N_DAYS, rng)

        on_hand = float(max(reorder_point + eoq / 2, 1))
        in_transit: list[tuple[int, float]] = []  # (arrival_day_index, qty)
        backorder = 0.0

        for day_idx, date in enumerate(SIM_DATES):
            arrived_today = sum(q for d, q in in_transit if d == day_idx)
            if arrived_today:
                on_hand += arrived_today
                in_transit = [(d, q) for d, q in in_transit if d != day_idx]

            todays_demand = float(demand[day_idx])
            fulfillable = min(on_hand, todays_demand + backorder)
            on_hand -= fulfillable
            unmet = (todays_demand + backorder) - fulfillable
            backorder = max(unmet, 0.0)
            stockout = 1 if (on_hand <= 0 and todays_demand > 0) else 0

            on_order_qty = sum(q for _, q in in_transit)
            if (on_hand + on_order_qty) <= reorder_point:
                order_qty = round(eoq)
                arrival_idx = day_idx + lead_time
                if arrival_idx < N_DAYS:
                    in_transit.append((arrival_idx, order_qty))
                order_rows.append((order_id_counter, sku, wh, date, order_qty, lead_time,
                                    SIM_DATES[arrival_idx] if arrival_idx < N_DAYS else pd.NaT, "Placed"))
                order_id_counter += 1

            inventory_rows.append((date, sku, wh, round(on_hand, 2), round(on_order_qty, 2),
                                    round(backorder, 2), stockout))
            demand_rows.append((date, sku, wh, todays_demand))

        policy_rows.append(Policy(sku, wh, abc, unit_cost, mean_daily, cv, lead_time, eoq, safety_stock, reorder_point))

    fact_inventory = pd.DataFrame(inventory_rows, columns=[
        "date", "sku", "warehouse_id", "on_hand_qty", "in_transit_qty", "backorder_qty", "stockout_flag"])
    fact_demand = pd.DataFrame(demand_rows, columns=["date", "sku", "warehouse_id", "demand_qty"])
    fact_orders = pd.DataFrame(order_rows, columns=[
        "order_id", "sku", "warehouse_id", "order_date", "order_qty", "lead_time_days", "expected_arrival", "status"])
    dim_policy = pd.DataFrame([p.__dict__ for p in policy_rows])

    fact_inventory.to_parquet(synthetic_dir / "fact_inventory.parquet", index=False)
    fact_demand.to_parquet(synthetic_dir / "fact_demand.parquet", index=False)
    fact_orders.to_parquet(synthetic_dir / "fact_orders.parquet", index=False)
    dim_sku.to_parquet(synthetic_dir / "dim_sku.parquet", index=False)
    dim_wh.drop(columns=["linked_ports"]).to_parquet(synthetic_dir / "dim_warehouse.parquet", index=False)
    dim_policy.to_parquet(synthetic_dir / "dim_replenishment_policy.parquet", index=False)

    dim_date = pd.DataFrame({"date": SIM_DATES})
    dim_date["year"] = dim_date["date"].dt.year
    dim_date["month"] = dim_date["date"].dt.month
    dim_date["quarter"] = dim_date["date"].dt.quarter
    dim_date["day_of_week"] = dim_date["date"].dt.day_name()
    dim_date.to_parquet(synthetic_dir / "dim_date.parquet", index=False)

    print(f"SKU x Warehouse series simulated: {len(policy_rows):,}")
    print(f"Fact_Inventory rows: {len(fact_inventory):,}")
    print(f"Fact_Demand rows: {len(fact_demand):,}")
    print(f"Fact_Orders rows: {len(fact_orders):,}")
    print(f"Overall simulated stockout-day rate: {fact_inventory['stockout_flag'].mean():.2%}")
    print(f"ABC mix: {dim_sku['abc_class'].value_counts(normalize=True).round(3).to_dict()}")


def load_into_sqlite(synthetic_dir: Path, processed_dir: Path, db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    for name in ["fact_inventory", "fact_demand", "fact_orders", "dim_sku", "dim_warehouse",
                 "dim_replenishment_policy", "dim_date"]:
        pd.read_parquet(synthetic_dir / f"{name}.parquet").to_sql(name, con, if_exists="replace", index=False)
    for name in ["OrderList", "FreightRates", "WhCosts", "WhCapacities", "ProductsPerPlant", "PlantPorts"]:
        pd.read_parquet(processed_dir / f"brunel_{name}.parquet").to_sql(f"brunel_{name.lower()}", con, if_exists="replace", index=False)
    con.commit()
    con.close()
    print(f"Loaded star schema into {db_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    simulate_all(root / "data" / "processed", root / "data" / "synthetic")
    load_into_sqlite(root / "data" / "synthetic", root / "data" / "processed", root / "data" / "inventory_performance.db")
