# AI-Powered Inventory Performance & Optimization — Global Industrial Supply Chain

A portfolio project demonstrating descriptive, diagnostic, predictive, and
prescriptive inventory analytics — plus a generative-AI advisor — on a
simulated global industrial aftermarket network (parts/consumables, in the
spirit of a company like Metso's Consumables business).

**Status:** in progress. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the
phase-by-phase build plan and current status.

## Why this project

Most public "supply chain analytics" portfolio pieces reuse the same one
dataset (DataCo) without checking its license, and stop at exploratory
dashboards or a single classifier. This project instead:

1. Starts from a **real, multi-plant / multi-warehouse / multi-port
   logistics network** (Brunel University London's Supply Chain Logistics
   Problem Dataset) rather than an unlicensed mirror.
2. Adds a **transparently synthetic, self-owned inventory & demand layer**
   calibrated to that real network, because no public dataset combines
   real on-hand stock levels with industrial-network structure — see
   `docs/DATA_PROVENANCE.md` for exactly why.
3. Chains together forecasting → stockout risk → inventory health scoring →
   optimization (safety stock / ROP / EOQ / inter-warehouse transfer) →
   Power BI → an AI advisor that explains the results, instead of stopping
   at EDA.

Full dataset audit, license teardown, and scoring of 10 public candidates:
see the companion **Inventory Intelligence Sourcebook**.

## Repository layout

```
data/
  raw/                  third-party source data, as downloaded (see docs/DATA_PROVENANCE.md)
  synthetic/             this project's own generated inventory/demand layer (CC0)
  processed/              cleaned, modeled tables ready for analysis / Power BI
notebooks/               numbered, run-in-order Jupyter/Colab notebooks
src/                      reusable Python modules imported by the notebooks
powerbi/                  .pbix file(s) and DAX reference
docs/                     roadmap, data provenance, methodology, legal checklist
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/01_data_acquisition.ipynb
```

Or open any notebook in `notebooks/` directly in Google Colab — each one
downloads its own source data on first run.

## License

Code in this repository: MIT (see `LICENSE`). Third-party datasets keep
their original licenses (see `docs/DATA_PROVENANCE.md`); this project's own
synthetic data is CC0.
