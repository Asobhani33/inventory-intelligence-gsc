# Project Write-Up: Problem, Approach, and What a Real-ERP Version Would Need

## The problem

Industrial aftermarket networks — spare parts, consumables, wear parts for
heavy equipment — carry a specific, expensive tension: stock out on the
wrong SKU at the wrong plant and a customer's production line stops; carry
too much of the wrong SKU and working capital sits idle in a warehouse for
months. Most public portfolio projects in this space either (a) reuse the
same unlicensed retail dataset (DataCo) with no real network structure, or
(b) stop at a dashboard of historical KPIs without ever turning "what
happened" into "what should we do about it." Neither answers the question
an actual inventory planner asks every morning: *which SKUs, at which
warehouses, need action this week — and why?*

This project's goal was to answer that question end-to-end, on a network
with a real logistics backbone, using only public/self-generated data —
and to be explicit, at every step, about where the analysis is grounded in
real data versus a documented assumption.

## The approach

**1. Ground the network in something real.** Rather than starting from a
purely synthetic network, the project uses Brunel University London's
Supply Chain Logistics Problem Dataset as the backbone: real plant count,
warehouse capacities, handling costs, and port/freight relationships for a
multi-region industrial network. No public dataset pairs that kind of
network structure with real on-hand inventory levels, so a synthetic
inventory & demand layer was built on top of it — calibrated to the real
network's scale, documented field-by-field in
`docs/SYNTHETIC_DATA_METHODOLOGY.md` so every assumption is auditable
rather than implied.

**2. Descriptive → diagnostic.** KPI engine computes days-of-supply,
turnover, ABC/XYZ classification, and a six-component Inventory Health
Score per SKU×warehouse, then decomposes *why* a warehouse's inventory
value moved month over month (excess buildup vs. genuine demand vs.
shortage-driven ordering) — not just that it moved.

**3. Predictive.** Two models per SKU×warehouse: a demand forecast
(LightGBM/statsmodels) and a 30-day stockout-risk classifier, both
evaluated honestly (forecast WAPE reported as 33.8%, using a global
LightGBM model with a Tweedie objective — suited to this demand series'
zero-inflation, 22% of weekly rows are exactly zero — and early stopping
against a held-out validation window; the synthetic demand series is
intentionally noisy, and the number isn't smoothed over to look better
than it is).

**4. Prescriptive.** Dynamic reorder points from observed 90-day demand,
an excess/shortage optimizer that matches excess stock at one warehouse
against shortages at another via a per-SKU transportation-problem LP
(`scipy.optimize.linprog`), and a clear fallback to "needs a fresh PO"
when no matching excess exists anywhere in the network — rather than
inventing a transfer that wouldn't actually solve the shortage.

**5. Decision layer, not just analysis layer.** A 7-page Power BI report
turns all of the above into something a planner would actually open every
morning, and a generative-AI advisor (OpenAI tool-calling over 12 typed
Python functions, never free-text generation against raw data) lets
anyone ask the network a question in plain language and get back an exact,
traceable number — deployed publicly so the chatbot isn't a local demo
only its author can run.

**Design choice made throughout:** every limitation encountered was
documented rather than papered over — the Brunel freight-rate data
turning out unusable for transfer costing (single hub port only), SHAP
explanations that couldn't be persisted because the Phase 4 models were
never serialized, a 130MB SQLite file that had to be replaced with small
parquet exports for public deployment, a Power BI platform restriction
that rules out a literally-embedded chatbot in a shared `.pbix` file, and
the default page-navigation strip in Power BI Desktop that can't be fully
hidden for a standalone `.pbix` opened in Desktop (that control only
exists after publishing to Power BI Service or a Power BI App, neither of
which this project uses) — the report's own custom sidebar navigation
sits alongside it rather than replacing it, a known Desktop limitation,
not an oversight. A portfolio project that only shows the parts that worked isn't a credible
demonstration of how someone works through the parts that didn't.

## What a real-ERP version would need

This project deliberately used public/synthetic data so it could be
built and shared without any company's proprietary information. A version
built against a real ERP (SAP, Oracle, Infor, etc.) would need:

- **Real demand and lead-time history** in place of the synthetic layer —
  the forecast and stockout-risk models would very likely perform
  meaningfully better against real transaction history than against this
  project's intentionally-noisy synthetic demand (WAPE 33.8% here should
  not be read as a ceiling on the approach itself).
- **Real freight/transfer cost data** by lane, replacing the
  region-surcharge approximation this project uses — this is the single
  biggest accuracy gap between what's built here and a production-grade
  transfer recommendation.
- **Persisted, versioned models** (not re-trained fresh in a notebook each
  run) with real SHAP values served at inference time, so
  `explain_health_score`-style answers are backed by true per-prediction
  attributions instead of a transparent rule-based approximation.
- **Guardrails well beyond this project's tool-calling design** before
  allowing any kind of open-ended query generation against a live
  production database — this project deliberately avoided that (12 fixed,
  narrow tools instead of "let the model write SQL") precisely because a
  real ERP raises the cost of a wrong query enormously; a production
  version would need row-level access control, query auditing, and
  probably a human-in-the-loop step before any tool that could write
  data, not just read it.
- **Multi-tenant auth and rate limiting** on the AI Advisor itself — this
  deployment intentionally shares one API key across all visitors, which
  is fine for a portfolio demo and not fine for a real internal tool.
- **A real embedding surface for the chatbot** — likely a Power BI Service
  publish (not a shared `.pbix` file) or a genuinely custom Power BI
  visual built and signed for the organization, both of which sidestep
  the Desktop iframe restriction this project hit.
- **A live data feed to trigger real alerts.** This project deliberately did
  not build a push-alert mechanism (email/Slack when a SKU crosses into
  Critical, or stockout risk passes a threshold) — the underlying data here
  is a static snapshot, not a live feed, so a scheduled check would never
  find anything new to alert on, which would make the feature a hollow
  demo rather than a real capability. The health-tier and stockout-risk
  logic already in this project (`kpi_engine.compute_health_score`,
  `get_stockout_risks`) is exactly the rule set a real alert engine would
  run — against a live ERP feed, the same thresholds would drive an actual
  notification pipeline instead of only being visible on request (Power BI
  report or AI Advisor query).

None of these are hidden gaps discovered after the fact — each is a
direct, named consequence of a scope decision made explicitly to keep this
project buildable and shareable as a public portfolio piece.
