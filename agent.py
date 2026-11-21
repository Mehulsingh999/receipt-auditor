"""Receipt Auditor agent: extract (LLM) -> audit (code) -> explain (LLM only when needed).

Built as a small LangGraph so each step is visible, testable and easy to swap.
"""
from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from audit import audit_receipt
from schema import AuditReport, Receipt

load_dotenv()

EXTRACT_PROMPT = """You extract data from a store or restaurant receipt.
Rules:
- Copy every number exactly as printed. Do NOT fix the math, even if it looks wrong.
- One entry per printed line item, in order. Keep repeated lines as separate entries.
- Discounts and coupons go in `discounts` as one positive total, not as items.
- Auto-gratuity or service charge goes in `service_charge`; a tip line goes in `tip`.
- If a field is not printed, leave it empty. Never invent values."""

EXPLAIN_PROMPT = """You are a receipt auditor talking to a shopper.
Given the receipt and the audit findings (computed by exact math, trust them), write a verdict:
- First sentence: the bottom line, with the dollar amount at stake.
- Then the top issues, most costly first, in plain words.
- End with one concrete action (for example: ask for a refund of $X at the register).
Max 4 sentences. Say "possible" for anything the shopper should double-check."""


def get_llm(provider: str):
    """provider: 'anthropic' (hosted) or 'ollama' (local)."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"), temperature=0, max_tokens=2000)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"), temperature=0)
    raise ValueError(f"Unknown provider: {provider}")


class State(TypedDict, total=False):
    text: Optional[str]
    image_path: Optional[str]
    receipt: Receipt
    report: AuditReport
    llm: Any


def _image_message(path: str) -> dict:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def extract(state: State) -> State:
    if state.get("receipt"):  # already structured (tests, or a JSON input)
        return {}
    structured = state["llm"].with_structured_output(Receipt)
    if state.get("image_path"):
        content = [_image_message(state["image_path"]), {"type": "text", "text": "Extract this receipt."}]
    else:
        content = f"Receipt text:\n\n{state['text']}"
    receipt = structured.invoke([("system", EXTRACT_PROMPT), ("human", content)])
    return {"receipt": receipt}


def audit(state: State) -> State:
    return {"report": audit_receipt(state["receipt"])}


def needs_explaining(state: State) -> str:
    serious = [f for f in state["report"].findings if f.severity != "low"]
    return "explain" if serious and state.get("llm") is not None else "clean"


def explain(state: State) -> State:
    rep = state["report"]
    findings = "\n".join(f"- [{f.severity}] {f.message} (at stake: ${f.amount:.2f})" for f in rep.findings)
    msg = state["llm"].invoke([
        ("system", EXPLAIN_PROMPT),
        ("human", f"Merchant: {rep.receipt.merchant}\nCharged: ${rep.receipt.total:.2f}\n"
                  f"Computed: ${rep.computed_total:.2f}\nMoney at stake: ${rep.overcharge_estimate:.2f}\n"
                  f"Findings:\n{findings}"),
    ])
    return {"report": rep.model_copy(update={"verdict": msg.content if isinstance(msg.content, str) else str(msg.content)})}


def clean(state: State) -> State:
    """No serious findings: skip the second LLM call and save the cost."""
    rep = state["report"]
    serious = [f for f in rep.findings if f.severity != "low"]
    if serious:  # no LLM available: plain summary
        verdict = (f"${rep.overcharge_estimate:.2f} possibly at stake. "
                   + " ".join(f.message for f in serious[:3]))
    else:
        verdict = f"Looks right: everything adds up to the ${rep.receipt.total:.2f} you were charged."
    return {"report": rep.model_copy(update={"verdict": verdict})}


def build_graph():
    g = StateGraph(State)
    g.add_node("extract", extract)
    g.add_node("audit", audit)
    g.add_node("explain", explain)
    g.add_node("clean", clean)
    g.set_entry_point("extract")
    g.add_edge("extract", "audit")
    g.add_conditional_edges("audit", needs_explaining, {"explain": "explain", "clean": "clean"})
    g.add_edge("explain", END)
    g.add_edge("clean", END)
    return g.compile()


def run(text: str | None = None, image_path: str | None = None, provider: str | None = "anthropic",
        receipt: Receipt | None = None, llm=None) -> AuditReport:
    """provider=None with a ready `receipt` runs fully offline (no AI calls)."""
    if llm is None and provider:
        llm = get_llm(provider)
    if llm is None and receipt is None:
        raise ValueError("Need a provider (or llm) to extract from text or an image.")
    state: State = {"text": text, "image_path": image_path, "llm": llm}
    if receipt is not None:
        state["receipt"] = receipt
    return build_graph().invoke(state)["report"]
