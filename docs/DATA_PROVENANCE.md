# Data Provenance & License Record

This file is the single source of truth for where every raw data file in this
repository came from, under what license, and what (if anything) was changed.
Keep it in sync with `data/raw/`.

---

## 1. Supply Chain Logistics Problem Dataset (network backbone)

| Field | Value |
|---|---|
| Publisher | Brunel University London |
| Authors | Tatiana Kalganova, Ivars Dzalbs |
| Original host | Figshare — https://brunel.figshare.com/articles/dataset/Supply_Chain_Logistics_Problem_Dataset/7558679 |
| License | Stated as CC BY 4.0 on the Figshare item page (Figshare's institutional default). **Re-confirm the badge on the page yourself before final publication** — it could not be scraped automatically from this build environment. |
| Data type | Real (historical order records) + fixed network parameters |
| Retrieved via | GitHub mirror https://github.com/jaredbach/LogisticsDataset (the *original* Brunel Excel file, re-exported to CSV by the mirror author with no content changes — used here only because this build sandbox's network policy blocks figshare.com directly; the download notebook (`01_data_acquisition.ipynb`) pulls straight from Figshare when run somewhere with open internet access, e.g. Google Colab) |
| Retrieved on | 2026-09-11 |

### Confirmed table sizes (verified in this session, not estimates)

| Table | Rows | Columns | Description |
|---|---:|---:|---|
| `OrderList.csv` | 9,215 | 14 | Historical orders: order id, date, origin port, carrier, transit-day count, service level, ship-ahead/ship-late day counts, customer, product id, plant code, destination port, quantity, weight |
| `FreightRates.csv` | 1,540 | 11 | Freight rate card: carrier, origin/dest port, weight bands, min cost, rate, transport mode, transit-day count, carrier type |
| `WhCosts.csv` | 19 | 2 | Cost per unit, by plant |
| `WhCapacities.csv` | 19 | 2 | Daily capacity, by plant |
| `ProductsPerPlant.csv` | 2,036 | 2 | Product → plant mapping |
| `VmiCustomers.csv` | 14 | 2 | VMI-managed customer → plant mapping |
| `PlantPorts.csv` | 22 | 2 | Plant → port mapping |

**Required attribution when this data is used or shown:**
> Kalganova, T. & Dzalbs, I., "Supply Chain Logistics Problem Dataset," Brunel University London (Figshare). https://brunel.figshare.com/articles/dataset/Supply_Chain_Logistics_Problem_Dataset/7558679

---

## 2. Online Retail (demand-forecasting module)

| Field | Value |
|---|---|
| Publisher | UCI Machine Learning Repository |
| Donor/author | Dr. Daqing Chen |
| License | CC BY 4.0 |
| Data type | Real UK online gift-ware retailer transactions |

**Important — read before publishing:** the file currently sitting in
`data/raw/online_retail/online_retail_dev_placeholder.csv` is a **development
stand-in**, not the final source file. It is a cleaned community re-export
(531,283 rows, invoices starting at 536365, Dec 2010–Dec 2011) of the
**original "Online Retail" dataset** (UCI dataset #352), pulled from a GitHub
mirror because this build sandbox cannot reach `archive.ics.uci.edu`
directly. It let development continue without blocking on network access.

For the published version of this project, swap this file for the authentic
**"Online Retail II"** dataset (UCI dataset #502 — 1,067,371 rows, Dec
2009–Dec 2011, the one profiled in the Inventory Intelligence Sourcebook)
by running `01_data_acquisition.ipynb` in an environment with open internet
access (Google Colab works). It downloads directly from:
https://archive.ics.uci.edu/dataset/502/online+retail+ii — same schema
(InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice,
CustomerID, Country), so no pipeline code changes are needed either way.

**Required attribution:**
> Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

---

## 3. Synthetic inventory & demand layer (`data/synthetic/`)

Authored entirely by this project (see `src/simulate_inventory.py`) —
calibrated to the real plant/warehouse counts, capacities, costs, and
transit lead times in table 1 above. **License: CC0 / public domain** — it
is original work with no third-party rights attached, generated to fill the
one gap every public dataset in the Inventory Intelligence Sourcebook shares
(no dataset publishes real on-hand stock levels). Document the exact
generation methodology in `docs/SYNTHETIC_DATA_METHODOLOGY.md` before
publishing so reviewers can see it is a deliberate, defensible simulation
and not fabricated "real" data.

---

## 4. What NOT to commit

Do not add the DataCo "Smart Supply Chain" dataset to this repository in any
form (raw file, derivative table, or near-complete transformation). Its
original Mendeley license states *"All rights reserved by original author
only. This is only for educational purpose,"* which is incompatible with
public redistribution. See the Inventory Intelligence Sourcebook, Section B,
for the full legal teardown.
