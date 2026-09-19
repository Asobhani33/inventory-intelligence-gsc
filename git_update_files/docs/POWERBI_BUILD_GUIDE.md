# Power BI Build Guide — Phase 6

Step-by-step for building `powerbi/inventory_performance.pbix` from the
tables in `data/powerbi/` (produced by `notebooks/06_powerbi_prep.ipynb`).
Do this in Power BI Desktop on your own machine — we'll go through it
together, one step at a time.

## 0. Prerequisites

- Run `notebooks/06_powerbi_prep.ipynb` first (Colab or local) so
  `data/powerbi/*.parquet` (14 files) exists and is current.
- Power BI Desktop installed (you already have this).
- Your Power BI Desktop version should be recent enough to have a native
  **Parquet** connector (`Get Data` search box → type "parquet"). If it's
  missing, update Power BI Desktop from the Microsoft Store/website first —
  it's a free update, not a plugin.

## 1. Import the tables

1. Open Power BI Desktop → `Get Data` → search "**Folder**" → point it at
   your local `data/powerbi/` folder → `Combine & Transform` is NOT what we
   want here (that merges same-shape files into one table); instead choose
   **"Combine"** → **"Load"** only if prompted per-file, OR — simpler and
   more reliable — import each of the 14 files individually:
   `Get Data` → search "**Parquet**" → select one `.parquet` file → repeat
   14 times (tedious but avoids any folder-combine surprises).
2. Confirm all 14 tables loaded: check the **Fields** pane lists
   `dim_sku`, `dim_warehouse`, `dim_date`, `dim_replenishment_policy`,
   `dim_warehouse_coords`, `fact_inventory`, `fact_demand`, `fact_orders`,
   `kpi_monthly_diagnostic`, `predictive_forecast_test`,
   `reco_replenishment`, `reco_transfers`, `reco_unmet_shortage`,
   `reco_transfer_cost_matrix`.

## 2. Mark the date table

`dim_date` is your date dimension: select it in the Fields pane → **Table
tools** ribbon → **Mark as date table** → pick the `date` column.

## 3. Build relationships (Model view)

Switch to **Model view** (left sidebar icon). Power BI auto-detects some of
these from matching column names — verify each one exists and fix the
cardinality/direction if it guessed wrong:

| From | To | Cardinality |
|---|---|---|
| `dim_sku[sku]` | `fact_inventory[sku]` | 1 → many |
| `dim_sku[sku]` | `fact_demand[sku]` | 1 → many |
| `dim_sku[sku]` | `fact_orders[sku]` | 1 → many |
| `dim_sku[sku]` | `dim_replenishment_policy[sku]` | 1 → many |
| `dim_sku[sku]` | `reco_replenishment[sku]` | 1 → many |
| `dim_sku[sku]` | `predictive_forecast_test[sku]` | 1 → many |
| `dim_sku[sku]` | `reco_transfers[sku]` | 1 → many |
| `dim_sku[sku]` | `reco_unmet_shortage[sku]` | 1 → many |
| `dim_warehouse[warehouse_id]` | `fact_inventory[warehouse_id]` | 1 → many |
| `dim_warehouse[warehouse_id]` | `fact_demand[warehouse_id]` | 1 → many |
| `dim_warehouse[warehouse_id]` | `fact_orders[warehouse_id]` | 1 → many |
| `dim_warehouse[warehouse_id]` | `dim_replenishment_policy[warehouse_id]` | 1 → many |
| `dim_warehouse[warehouse_id]` | `reco_replenishment[warehouse_id]` | 1 → many |
| `dim_warehouse[warehouse_id]` | `predictive_forecast_test[warehouse_id]` | 1 → many |
| `dim_warehouse[warehouse_id]` | `kpi_monthly_diagnostic[warehouse_id]` | 1 → many |
| `dim_warehouse[warehouse_id]` | `reco_transfer_cost_matrix[from_warehouse]` | 1 → many |
| `dim_warehouse[warehouse_id]` | `reco_transfers[from_warehouse]` | 1 → many (**active**) |
| `dim_warehouse[warehouse_id]` | `reco_transfers[to_warehouse]` | 1 → many (**inactive** — Power BI only allows one active path between two tables; leave this one inactive, see the DAX reference for `USERELATIONSHIP`) |
| `dim_warehouse[warehouse_id]` | `dim_warehouse_coords[warehouse_id]` | 1 → 1 |
| `dim_date[date]` | `fact_inventory[date]` | 1 → many |
| `dim_date[date]` | `fact_demand[date]` | 1 → many |

Leave `fact_orders[order_date]` and `kpi_monthly_diagnostic[month]`
unrelated to `dim_date` for now (different grain — order-level and
month-level respectively) unless you want to build a separate month
dimension; the pages below don't require it.

## 4. Add the DAX measures

Go through `docs/POWERBI_DAX_REFERENCE.md` top to bottom and paste each
measure into the table it's listed under (right-click the table → **New
measure**). Do this now, before building pages, so every visual below has
what it needs.

## 5. Build the 7 pages

Add a page (`+` at the bottom tab bar) for each. Suggested visuals per page
— you have full creative freedom on layout, this is a starting point:

### Page 1 — Executive Overview
- KPI cards: `[Total Inventory Value]`, `[Total Excess Value]`,
  `[Avg Inventory Turnover]`, `[Avg Service Level]`, `[SKU x Warehouse Count]`
- Bar chart: `Total Inventory Value` by `dim_warehouse[region]`
- Slicers: `dim_sku[abc_class]`, `dim_warehouse[region]`

### Page 2 — Inventory Health
- KPI cards: `[Avg Health Score]`, `[Critical Count]`, `[Watch Count]`, `[Healthy Count]`
- Matrix: ABC x XYZ (see DAX reference)
- Bar chart: `Excess Value` by `dim_warehouse[warehouse_id]` (this is where the $693K in PLANT03 shows up)

### Page 3 — Demand Forecast
- KPI cards: `[WAPE]`, `[Forecast Accuracy]` (live DAX measures, not a static card — see DAX reference)
- Line chart: `predictive_forecast_test[demand_qty]` vs `[prediction]` over `week`, filtered to a slicer-selected SKU
- Table: `predictive_forecast_test` sorted by largest forecast error

### Page 4 — Inventory Risk
- KPI cards: `[Avg Stockout Prob (30d)]`, `[High Risk SKU Count]`
- Scatter plot: `stockout_prob_30d` (x) vs `inventory_health_score` (y), size = `on_hand_qty` × `unit_cost`, color = `health_tier`
- Table: `reco_replenishment` filtered to `recommended_action = "Order now — high stockout risk"`

### Page 5 — Replenishment
- KPI cards: `[Order Now Count]`, `[Total Recommended Order Qty]`, `[Transfer Move Count]`, `[Fresh PO Value]`
- Table: `reco_transfers` (the LP's recommended moves)
- Table: `reco_unmet_shortage` sorted by `unmet_via_fresh_po` descending

### Page 6 — Global Network
- Map visual using `dim_warehouse_coords[approx_lat]`/`[approx_lon]`, bubble size = `[Total Inventory Value]`
- **Add a text box on this page:** *"Warehouse positions are illustrative (region-center + deterministic jitter) — the source datasets record a region per plant, not real coordinates."* This matters for portfolio credibility — don't let the map imply more precision than the data has.
- Bar chart: `[Total Orders Placed]` by `dim_warehouse[region]`

### Page 7 — AI Advisor
- Placeholder for now: a text box "Coming in Phase 7" is fine. Once Phase 7 ships, this page embeds the chat panel (design TBD — see `docs/AI_ADVISOR_DESIGN.md` when it exists).

## 6. Save

`File` → `Save As` → `powerbi/inventory_performance.pbix` inside your
project folder (creates the `powerbi/` folder if it doesn't exist yet).
