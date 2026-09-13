# Power BI DAX Reference

Every measure used across the 7 dashboard pages, grouped by the page that
needs it first (several are reused on more than one page). Copy-paste these
into Power BI as new measures on the table listed — right-click the table in
the **Fields** pane → **New measure**.

All measures assume the table names and relationships in
`POWERBI_BUILD_GUIDE.md`. Where a measure needs a specific relationship path
(e.g. `reco_transfers` relates to `dim_warehouse` twice — from and to — so
`USERELATIONSHIP` is required), that's called out explicitly.

---

## Shared / Executive Overview (measures on `reco_replenishment`)

```dax
Total Inventory Value =
SUMX(reco_replenishment, reco_replenishment[on_hand_qty] * reco_replenishment[unit_cost])

Total Excess Value = SUM(reco_replenishment[excess_value])

Excess Value % of Total = DIVIDE([Total Excess Value], [Total Inventory Value])

SKU x Warehouse Count = COUNTROWS(reco_replenishment)

Excess SKU Count =
CALCULATE(COUNTROWS(reco_replenishment), reco_replenishment[is_excess] = TRUE)

Avg Inventory Turnover = AVERAGE(reco_replenishment[inventory_turnover])

Avg Days of Supply =
AVERAGEX(FILTER(reco_replenishment, reco_replenishment[days_of_supply] <> BLANK()), reco_replenishment[days_of_supply])

Avg Service Level = AVERAGE(reco_replenishment[service_level])

Obsolete SKU Count =
CALCULATE(COUNTROWS(reco_replenishment), reco_replenishment[is_obsolete] = TRUE)
```

---

## Inventory Health page (measures on `reco_replenishment`)

```dax
Avg Health Score = AVERAGE(reco_replenishment[inventory_health_score])

Critical Count =
CALCULATE(COUNTROWS(reco_replenishment), reco_replenishment[health_tier] = "Critical")

Watch Count =
CALCULATE(COUNTROWS(reco_replenishment), reco_replenishment[health_tier] = "Watch")

Healthy Count =
CALCULATE(COUNTROWS(reco_replenishment), reco_replenishment[health_tier] = "Healthy")

Critical % = DIVIDE([Critical Count], [SKU x Warehouse Count])
```

**ABC x XYZ matrix**: use a Matrix visual — Rows = `reco_replenishment[abc_class]`, Columns = `reco_replenishment[xyz_class]`, Values = `[SKU x Warehouse Count]` (or `[Total Inventory Value]` to see dollars, not just counts).

---

## Demand Forecast page (measures on `predictive_forecast_test`)

```dax
WAPE =
DIVIDE(
    SUMX(predictive_forecast_test, ABS(predictive_forecast_test[demand_qty] - predictive_forecast_test[prediction])),
    SUM(predictive_forecast_test[demand_qty])
)

Forecast Accuracy = 1 - [WAPE]

Total Actual Demand (Test Weeks) = SUM(predictive_forecast_test[demand_qty])

Total Forecasted Demand (Test Weeks) = SUM(predictive_forecast_test[prediction])

Forecast Bias =
DIVIDE(
    SUM(predictive_forecast_test[prediction]) - SUM(predictive_forecast_test[demand_qty]),
    SUM(predictive_forecast_test[demand_qty])
)
```

Headline card for this page: **WAPE 0.409 · MASE 0.786** (fixed values from the Phase 4 model run — a static text card or reference these live measures on the test-week slice).

---

## Inventory Risk page (measures on `reco_replenishment` and `fact_inventory`)

```dax
Avg Stockout Prob (30d) = AVERAGE(reco_replenishment[stockout_prob_30d])

High Risk SKU Count =
CALCULATE(COUNTROWS(reco_replenishment), reco_replenishment[stockout_prob_30d] >= 0.5)

Total Stockout Days (Historical) = SUM(fact_inventory[stockout_flag])

Stockout Rate (Historical) =
DIVIDE(SUM(fact_inventory[stockout_flag]), COUNTROWS(fact_inventory))
```

---

## Replenishment page (measures on `reco_replenishment`, `reco_transfers`, `reco_unmet_shortage`)

```dax
Order Now Count =
CALCULATE(COUNTROWS(reco_replenishment), reco_replenishment[needs_replenishment] = TRUE)

High Risk Order Count =
CALCULATE(
    COUNTROWS(reco_replenishment),
    reco_replenishment[recommended_action] = "Order now — high stockout risk"
)

Total Recommended Order Qty = SUM(reco_replenishment[recommended_order_qty])

Transfer Move Count = COUNTROWS(reco_transfers)

Total Transfer Qty = SUM(reco_transfers[transfer_qty])

Total Transfer Cost = SUM(reco_transfers[total_cost])

Fresh PO Units Needed = SUM(reco_unmet_shortage[unmet_via_fresh_po])

Fresh PO Value =
SUMX(reco_unmet_shortage, reco_unmet_shortage[unmet_via_fresh_po] * reco_unmet_shortage[unit_cost])

Estimated Transfer Savings vs. Fresh PO =
SUMX(
    reco_transfers,
    reco_transfers[transfer_qty] * RELATED(dim_sku[unit_cost]) * 1.08 - reco_transfers[total_cost]
)
```

> `reco_transfers` has TWO relationships to `dim_warehouse` (`from_warehouse` and `to_warehouse`). Power BI will only activate one by default (shown solid in the model view; the other is dashed/inactive). To slice a visual by the *destination* warehouse when the active relationship is on `from_warehouse`, wrap the measure: `CALCULATE([Total Transfer Qty], USERELATIONSHIP(reco_transfers[to_warehouse], dim_warehouse[warehouse_id]))`.

---

## Global Network page (measures on `fact_orders`, `dim_warehouse`)

```dax
Total Orders Placed = COUNTROWS(fact_orders)

Avg Lead Time (Days) = AVERAGE(fact_orders[lead_time_days])

Total Order Qty = SUM(fact_orders[order_qty])
```

Map visual: use `dim_warehouse_coords[approx_lat]` / `[approx_lon]` as Latitude/Longitude, `dim_warehouse[warehouse_id]` as legend/tooltip, bubble size = `[Total Inventory Value]` filtered to that warehouse. **Label the visual "illustrative"** — see the callout in `POWERBI_BUILD_GUIDE.md`.

---

## AI Advisor page

No DAX here — this page is a placeholder for Phase 7 (a Streamlit chat panel embedded via the Power BI web-content visual, or a "coming soon" static page for now). See `docs/AI_ADVISOR_DESIGN.md` once Phase 7 is built.
