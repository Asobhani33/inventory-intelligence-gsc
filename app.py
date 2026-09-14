"""
app.py
=======
Phase 7 — AI Inventory Advisor, a small Streamlit chat app. Run locally:

    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...        # PowerShell: $env:OPENAI_API_KEY="sk-..."
    streamlit run app.py

Never commit a real API key to this repo — set it as an environment
variable (or paste it into the sidebar box below, which keeps it in this
browser session only, never written to disk).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ai_advisor import agent, data_access, tools  # noqa: E402

st.set_page_config(page_title="AI Inventory Advisor", page_icon="📦", layout="wide")

st.title("📦 AI Inventory Advisor")
st.caption(
    "Ask about warehouse health, stockout risk, replenishment recommendations, "
    "transfer opportunities, or forecast accuracy — every answer is grounded in "
    "the project's own processed tables (Phases 3-5), never invented."
)

# On Streamlit Community Cloud, a key set in the app's Secrets is available
# via st.secrets and is shared by every visitor (one server process serves
# everyone), so the public chatbot works without each person needing their
# own OpenAI key. Locally, there's no secrets.toml — st.secrets access is
# wrapped so that doesn't crash the app.
try:
    _cloud_key = st.secrets.get("OPENAI_API_KEY")
except Exception:  # noqa: BLE001 — no secrets.toml present (local run)
    _cloud_key = None
if _cloud_key and not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = _cloud_key

with st.sidebar:
    st.header("Setup")
    if os.environ.get("OPENAI_API_KEY"):
        # Key already available (from Secrets, or an env var on your own
        # machine) — don't show an input that could let a visitor overwrite
        # the shared key for everyone else using this same deployment.
        st.success("OpenAI connected ✅")
    else:
        api_key_input = st.text_input(
            "OpenAI API key", type="password",
            help="Kept only in this browser session — never written to disk or committed to the repo.",
        )
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input

    st.divider()
    st.subheader("Data snapshot")
    try:
        summary = tools.get_network_summary()
        st.metric("Total inventory value", f"${summary['total_inventory_value']:,.0f}")
        st.metric("Total excess value", f"${summary['total_excess_value']:,.0f}")
        st.write("Health tiers:", summary["health_tier_counts"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't load project data: {exc}")

    if st.button("🔄 Reload data (after re-running notebooks)"):
        data_access.reload()
        st.rerun()

    st.divider()
    st.subheader("Try asking")
    for example in [
        "How's the network doing overall?",
        "Why is inventory so high at PLANT03?",
        "Which SKUs have the highest stockout risk right now?",
        "What's the transfer plan, and what still needs a fresh PO?",
        "How accurate is the demand forecast?",
    ]:
        st.code(example, language=None)

def _extract_trend_chart(tool_calls: list[dict]) -> pd.DataFrame | None:
    """If a get_inventory_trend tool was called this turn, build a small
    DataFrame from its monthly series so the UI can plot it — not just
    describe it in text."""
    for call in tool_calls:
        if call["name"] != "get_inventory_trend":
            continue
        series = call["result"].get("monthly_total_value_trend")
        if not series:
            continue
        df = pd.DataFrame(series)
        if "month" not in df.columns:
            continue
        df["month"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
        return df.set_index("month")[["total_inventory_value"]]
    return None


# Tool result keys that hold a list of records worth showing as a table,
# in priority order (first match wins) — covers every list-returning tool
# in tools.py so any of them renders as a real table, not just text.
_TABLE_KEYS = ["results", "rows", "transfer_moves"]


def _extract_table(tool_calls: list[dict]) -> pd.DataFrame | None:
    """If any tool called this turn returned a list of records (SKUs,
    recommendations, transfer moves, ...), build a DataFrame from the last
    such call so the UI can show a real table, not just prose."""
    for call in reversed(tool_calls):
        result = call.get("result") or {}
        for key in _TABLE_KEYS:
            records = result.get(key)
            if records:
                df = pd.DataFrame(records)
                if not df.empty:
                    return df
    return None


if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn_charts" not in st.session_state:
    st.session_state.turn_charts = {}  # message index -> DataFrame
if "turn_tables" not in st.session_state:
    st.session_state.turn_tables = {}  # message index -> DataFrame

for i, m in enumerate(st.session_state.messages):
    if m["role"] in ("user", "assistant") and m.get("content"):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if i in st.session_state.turn_charts:
                st.bar_chart(st.session_state.turn_charts[i])
            if i in st.session_state.turn_tables:
                st.dataframe(st.session_state.turn_tables[i], hide_index=True)

prompt = st.chat_input("Ask the Inventory Advisor...")
if prompt:
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Enter your OpenAI API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Checking the data..."):
                try:
                    reply, st.session_state.messages, tool_calls = agent.chat(st.session_state.messages)
                except Exception as exc:  # noqa: BLE001
                    reply, tool_calls = f"Something went wrong calling the model: {exc}", []
                    st.session_state.messages.append({"role": "assistant", "content": reply})
            st.markdown(reply)

            chart_df = _extract_trend_chart(tool_calls)
            if chart_df is not None:
                st.bar_chart(chart_df)
                st.session_state.turn_charts[len(st.session_state.messages) - 1] = chart_df

            table_df = _extract_table(tool_calls)
            if table_df is not None:
                st.dataframe(table_df, hide_index=True)
                st.session_state.turn_tables[len(st.session_state.messages) - 1] = table_df
