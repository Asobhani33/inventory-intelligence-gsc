# AI Inventory Advisor — Design (Phase 7)

## What it is

A small Streamlit chat app (`app.py`) backed by OpenAI's function/tool
calling. It answers questions about the project's own data — warehouse
health, stockout risk, replenishment recommendations, transfer plans,
forecast accuracy — by calling Python functions that read the actual
processed tables from Phases 3-5, never from the model's own "knowledge."

## Why tool-calling, not RAG or a fine-tuned model

The data here is small, structured, and precise (KPIs, probabilities,
dollar values) — exactly wrong for embedding-based retrieval, which is
built for fuzzy matching over unstructured text and would blur numbers
that need to stay exact. Tool-calling instead gives the model typed
Python functions it can call with specific arguments (a SKU, a warehouse
ID, a threshold) and get back exact, current numbers. This also means the
Advisor's answers can never be "stale" the way a fine-tuned model's
baked-in knowledge would be — every answer reflects whatever's currently
in `data/processed/`.

## Architecture

```
app.py (Streamlit UI)
  -> src/ai_advisor/agent.py   (OpenAI tool-calling loop + system prompt)
       -> src/ai_advisor/tools.py        (10 tool functions + their JSON schemas)
            -> src/ai_advisor/data_access.py   (loads/caches the processed parquet tables)
```

- **`data_access.py`** — loads every table the tools need once, cached in
  memory (`functools.lru_cache`). A "Reload data" button in the app's
  sidebar clears the cache, so re-running the pipeline notebooks and
  clicking reload picks up fresh numbers without restarting the app.
  Everything now loads from small parquet files under `data/processed/`
  (including `dim_sku.parquet` / `dim_warehouse.parquet`, exported once
  from the project's SQLite database) — earlier this read `dim_sku` /
  `dim_warehouse` live from `data/inventory_performance.db`, but that file
  is 130+MB, over GitHub's 100MB limit and not something a public
  deployment can regenerate on its own, so those two small dimension
  tables were exported to parquet and the SQLite dependency was dropped
  entirely. `load_all()` never touches the raw fact tables or the
  database file — only the small recommendation, KPI, forecast, and
  dimension tables the tools actually need.
- **`tools.py`** — ten narrow functions, each answering one kind of
  question (network summary, one warehouse's detail, network- or
  warehouse-level inventory trend over time, why inventory moved,
  stockout risk, one SKU's detail, replenishment recommendations, transfer
  plan, forecast accuracy, health-score breakdown). Every function returns
  plain JSON-serializable data — no free text — so there's nothing for the
  model to embellish.
- **`agent.py`** — the OpenAI chat-completions loop with `tools=` set to
  the schemas in `tools.py`. The system prompt is explicit: answer only
  from tool output, say so when a tool returns nothing, and surface any
  `note`/`method_note` field a tool includes (see below) rather than
  smoothing it over. `chat()` returns not just the reply text but also the
  list of tool calls made that turn (name, arguments, result), so the UI
  layer can act on a tool's raw result — e.g. plot it — without re-parsing
  the conversation.
- **`app.py`** — the chat UI. When `get_inventory_trend` is called, the
  app pulls its monthly series straight out of the tool-call result and
  renders it as a bar chart under the reply, so trend questions get a
  picture, not just a paragraph of numbers. The OpenAI key comes from
  Streamlit Secrets when deployed (see "Public deployment" below) so the
  app works for any visitor without them supplying their own key; locally,
  it falls back to the `OPENAI_API_KEY` environment variable or a sidebar
  input. The sidebar's key input only ever renders when no key is already
  configured — once a key is present (from Secrets or the environment) the
  input is hidden, so a visitor typing into it on a public deployment can
  never overwrite the shared key for everyone else.

## A note on "explainability" — what's real SHAP and what isn't

Phase 4 computed SHAP explanations for the forecast and stockout-risk
models *inside a notebook, at training time* (`docs/phase4_shap_*.png`) —
but those trained model objects were never serialized to disk (no
`joblib.dump`/pickle step), so there's no saved model for this Advisor to
re-load and query live.

Rather than fake it, `explain_health_score` does something more modest and
fully honest: it recomputes the Inventory Health Score's six weighted
components (days-of-supply fit, turnover, demand variability, stockout
risk, excess flag, service level) directly from that row's own KPI fields,
using the exact same formula and weights as `kpi_engine.compute_health_score`.
It's a transparent rule-based decomposition, not SHAP — and the tool's
output says so explicitly (`method_note`), so the Advisor's answers do too
when relevant. A natural next step, if this project continues past Phase
7, is to serialize the Phase 4 models (`joblib.dump`) and swap in true
per-prediction SHAP values here without changing the tool's interface.

## Setup & running it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...          # PowerShell: $env:OPENAI_API_KEY="sk-..."
streamlit run app.py
```

Or paste the key into the sidebar text box at runtime instead of setting
an environment variable — either way, the key is never written to disk or
committed to the repo. `AI_ADVISOR_MODEL` (env var, default `gpt-4o-mini`)
picks the OpenAI model if you want to try a different one.

## Example questions it handles well

- "How's the network doing overall?"
- "Why is inventory so high at PLANT03?"
- "Why has network inventory dropped over the last few months?" (network-
  or warehouse-wide, increases and decreases — `get_inventory_trend`)
- "Which SKUs have the highest stockout risk right now?"
- "What's the transfer plan, and what still needs a fresh PO?"
- "How accurate is the demand forecast?"
- "Why is SKU 1667817 at PLANT03 rated Watch instead of Healthy?"

## Public deployment

The app is deployed on Streamlit Community Cloud, deployed straight from
this GitHub repo (`app.py` as the entry point, Python 3.11, `OPENAI_API_KEY`
set as a Streamlit Secret rather than typed in by each visitor). That
deployment is what makes the "shared, always-on" AI Advisor possible: the
Power BI report links out to it with a button (see below), rather than
each person needing their own OpenAI key or a local Python environment.

**Why the Power BI page links out instead of embedding the chat in an
iframe:** Power BI Desktop does not render `<iframe>` content inside any
custom visual, in any report — this is a platform-level sandboxing
restriction, not a gap in any particular visual, and it holds even for a
custom "HTML content" visual side-loaded from a `.pbiviz` file. Since this
project's `.pbix` is shared as a file that recipients open in their own
Power BI Desktop (not published to Power BI Service), a true embedded
chat panel isn't achievable here. The "AI Advisor" report page instead
carries a styled panel with a button (`Action` → `Web URL`) that opens the
public Streamlit app in a new browser tab — functionally the same
outcome (anyone with the report can use the live chatbot with one click),
just not literally inside the report canvas.

## Known limitations (documented on purpose, not hidden)

- No conversation memory beyond the current browser session (Streamlit's
  `session_state` resets on page reload).
- The ten tools cover the questions this project's data can actually
  answer well — they don't attempt open-ended SQL generation, which would
  risk the model writing an incorrect query against a real production
  database. For a portfolio project this trade-off (safety and precision
  over flexibility) is the right one; a production version behind a real
  ERP would need much stronger guardrails before any generated-query
  approach.
- `get_forecast_accuracy` and `why_is_inventory_high` read from the fixed
  Phase 4 test-period snapshot and Phase 3 monthly diagnostic table
  respectively — both reflect whatever was true when those notebooks were
  last run, not a live re-computation.
- The public deployment shares one OpenAI key (and its usage/cost) across
  every visitor — fine for a portfolio demo, but a real multi-tenant
  deployment would need per-user auth and rate limiting.
