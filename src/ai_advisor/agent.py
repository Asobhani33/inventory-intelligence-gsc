"""
agent.py
=========
Phase 7 — the OpenAI tool-calling loop. The system prompt scopes the model
to answer ONLY from what the tools in tools.py return: it never invents a
number, and if a tool has no matching data it says so instead of guessing.

Requires an OpenAI API key in the OPENAI_API_KEY environment variable —
never hardcode a key in this file or commit one to the repo.
"""
from __future__ import annotations

import json
import os

from openai import OpenAI

from . import tools

MODEL = os.environ.get("AI_ADVISOR_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are the AI Inventory Advisor for a global industrial \
supply chain's inventory performance project (19 real warehouses, ~1,540 \
SKUs, spare parts / consumables / wear parts for industrial equipment). Of \
those 19 warehouses, 18 (PLANT01-PLANT18) have SKUs assigned and active \
inventory/demand data; PLANT19 is a real warehouse (it has handling-cost \
and capacity data) but has zero SKUs assigned to it in the source data, so \
it has no inventory data to report — don't treat that as an error, and \
don't call get_warehouse_detail for PLANT19 just to complete a "check all \
19" loop unless the user specifically asks about PLANT19.

Hard rules:
1. Answer ONLY using data returned by the tools you're given. Never invent \
a number, trend, or SKU that a tool didn't return.
2. If a tool returns an error or empty result, say so plainly rather than \
guessing or filling the gap with plausible-sounding text.
3. Call a tool whenever the question needs current data — don't answer \
inventory/demand/risk questions from general knowledge.
4. Be concise and concrete: lead with the number, then the one-sentence \
"why" if a tool explains it. Write like a supply-chain analyst briefing a \
manager, not like a chatbot.
5. When a tool result includes a `method_note` or `note` field explaining a \
limitation or assumption, mention it if it's relevant to the user's question \
— this project is built to be transparent about what's real data vs. a \
documented assumption, and the Advisor should keep that standard.
6. For forecast accuracy questions (get_forecast_accuracy), the `wape` value \
IS the accuracy metric — state it plainly and do not add your own judgment \
of whether `total_actual_demand` and `total_forecasted_demand` "align" or \
"are close." Those two totals are sums across every evaluated period for \
that filter, and a small WAPE does NOT mean those two totals will look close \
to each other by eye (a few periods can over- and under-forecast in ways \
that partly cancel in the totals while still contributing real per-period \
error to WAPE) — never eyeball-compare them yourself, and never describe \
them as "close"/"aligned" unless you are just repeating a qualitative label \
a tool itself returned. If rows_evaluated is small (e.g. under 5), say so — \
a WAPE from very few periods is less reliable than one from many. The `wape` \
value is a plain ratio, NOT already a percentage: multiply it by 100 and \
write it with a "%" sign (wape=0.41 means "41%"). For very low-volume SKUs \
wape can exceed 1 (e.g. wape=2.94 means "294%" — a very BAD forecast, the \
opposite of accurate) — never call a high wape "relatively low" just because \
the raw number looks small; judge it after converting to a percentage.
7. Never build a table/column for a metric no tool actually returned, even \
to honestly mark it "Not Available" in every row — that reads as a broken \
feature, not an honest gap. If part of what the user asked can't be \
answered from any available tool, answer the part you can from a tool \
result, and say in one sentence what's missing and why, instead of \
including a placeholder column. For monthly consumption/usage value in \
dollars by warehouse, use get_consumption_value — don't say that's \
unavailable, since a tool for it exists.
"""


def _execute_tool_call(name: str, arguments_json: str) -> dict:
    fn = tools.TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        args = json.loads(arguments_json) if arguments_json else {}
        return fn(**args)
    except Exception as exc:  # noqa: BLE001 — surface the error to the model, don't crash the app
        return {"error": f"Tool '{name}' raised an exception: {exc}"}


def chat(messages: list[dict], client: OpenAI | None = None,
         max_tool_rounds: int = 5) -> tuple[str, list[dict], list[dict]]:
    """messages: the running conversation (list of {role, content} dicts,
    NOT including the system prompt — that's added here). Returns
    (assistant_reply_text, updated_messages, tool_calls_this_turn) — the
    third item is [{"name": ..., "arguments": {...}, "result": {...}}, ...]
    for every tool call made answering THIS message, so the caller
    (Streamlit) can render a chart from a tool's data alongside the text
    reply without re-parsing the conversation."""
    client = client or OpenAI()
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    tool_calls_this_turn: list[dict] = []

    for _ in range(max_tool_rounds):
        response = client.chat.completions.create(
            model=MODEL,
            messages=full_messages,
            tools=tools.TOOL_SPECS,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            return msg.content or "", messages, tool_calls_this_turn

        # Record the assistant's tool-call request, then run each tool and
        # feed its result back before asking the model again.
        full_messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            result = _execute_tool_call(tc.function.name, tc.function.arguments)
            tool_calls_this_turn.append({
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
                "result": result,
            })
            full_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    # Ran out of tool-call rounds — ask once more, without tools, for a final answer.
    response = client.chat.completions.create(model=MODEL, messages=full_messages)
    final_text = response.choices[0].message.content or "(No answer produced.)"
    messages.append({"role": "assistant", "content": final_text})
    return final_text, messages, tool_calls_this_turn
