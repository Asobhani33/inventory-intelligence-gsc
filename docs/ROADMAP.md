# Build Roadmap

Full living version (status, checklists) is published here — ask Claude for
the current status any time:
**Inventory Project Roadmap** (Artifact link shared in conversation).

| Phase | Status | Deliverable |
|---|---|---|
| 0. Environment & repo scaffold | ✅ Done | README, LICENSE, requirements.txt, .gitignore |
| 1. Data acquisition & validation | ✅ Done | `notebooks/01_data_acquisition.ipynb`, `data/raw/`, `docs/DATA_PROVENANCE.md` |
| 2. Data engineering — synthetic inventory layer | ✅ Done | `src/simulate_inventory.py`, star schema |
| 3. Descriptive & diagnostic analytics | ✅ Done | `src/kpi_engine.py`, `notebooks/03_descriptive_diagnostic.ipynb` |
| 4. Predictive analytics | ✅ Done | `src/predictive_models.py`, `notebooks/04_predictive_analytics.ipynb` |
| 5. Prescriptive optimization | ✅ Done | `src/optimization_engine.py`, `notebooks/05_prescriptive_optimization.ipynb` |
| 6. Power BI decision dashboard | ✅ Done | `powerbi/inventory_performance.pbix`, `src/powerbi_prep.py`, `docs/POWERBI_DAX_REFERENCE.md`, `docs/POWERBI_BUILD_GUIDE.md` |
| 7. AI Inventory Advisor | ✅ Done | `src/ai_advisor/`, `app.py`, `docs/AI_ADVISOR_DESIGN.md` |
| 8. Packaging & portfolio polish | ✅ Done | GitHub push ✅, public Streamlit deployment ✅, Power BI "open advisor" button ✅, README rewrite ✅, `docs/WRITEUP.md` ✅, all 7 Power BI page screenshots ✅ |
