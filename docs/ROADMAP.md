# Build Roadmap

Full living version (status, checklists) is published here — ask Claude for
the current status any time:
**Inventory Project Roadmap** (Artifact link shared in conversation).

| Phase | Status | Deliverable |
|---|---|---|
| 0. Environment & repo scaffold | ✅ Done | README, LICENSE, requirements.txt, .gitignore |
| 1. Data acquisition & validation | ✅ Done | `notebooks/01_data_acquisition.ipynb`, `data/raw/`, `docs/DATA_PROVENANCE.md` |
| 2. Data engineering — synthetic inventory layer | ⏭ Next | `src/simulate_inventory.py`, star schema |
| 3. Descriptive & diagnostic analytics | ⏳ Waiting | `notebooks/03_descriptive_diagnostic.ipynb` |
| 4. Predictive analytics | ⏳ Waiting | forecasting + stockout-risk notebooks |
| 5. Prescriptive optimization | ⏳ Waiting | safety stock / ROP / EOQ / transfer LP |
| 6. Power BI decision dashboard | ⏳ Waiting | `powerbi/inventory_performance.pbix` |
| 7. AI Inventory Advisor | ⏳ Waiting | `src/ai_advisor/`, `app.py` |
| 8. Packaging & portfolio polish | ⏳ Waiting | GitHub push, write-up, walkthrough |
