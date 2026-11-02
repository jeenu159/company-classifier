import json
import operator
import os
import sys
import time
import uuid
from typing import Annotated, Literal, Optional, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from schema import Classification, Label, TAXONOMY, EVIDENCE_GUIDE
from tools import fda_profile, web_search

load_dotenv()

PRIMARY, BACKUP = "gpt-5-mini", "gpt-4.1-mini"
MAX_ATTEMPTS = 2
HITL = os.getenv("HITL", "0") == "1"
FDA_CACHE, WEB_CACHE = "data/profiles.json", "data/web.json"


# ---------- structured LLMs with retries and a fallback model ----------
def structured(schema):
    primary = ChatOpenAI(model=PRIMARY, max_retries=2).with_structured_output(schema)
    backup = ChatOpenAI(model=BACKUP, max_retries=2).with_structured_output(schema)
    return primary.with_fallbacks([backup])


class WebDecision(BaseModel):
    need_web: bool = Field(description="True if the FDA profile alone cannot settle the class")
    reason: str = Field(description="One short sentence")


class Validation(BaseModel):
    verdict: Literal["agree", "disagree"]
    issue: str = Field(description="If disagree: what the evidence contradicts. If agree: one short sentence why.")
    suggested_label: Optional[Label] = None


classifier = structured(Classification)
validator = structured(Validation)


# ---------- graph state ----------
class State(TypedDict, total=False):
    company: str
    fda: dict
    web: Optional[dict]
    used_web: bool
    classification: dict
    validation: dict
    attempts: int
    status: str
    trace: Annotated[list, operator.add]   # each node appends; LangGraph merges with +


# ---------- helpers: retry with backoff, and on-disk caching ----------
def with_retry(fn, *args, tries=3):
    for i in range(tries):
        try:
            return fn(*args)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def cached(path, key, fn):
    cache = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    if key not in cache:
        cache[key] = fn(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1)
    return cache[key]


# ---------- prompts ----------
CLASSIFY_PROMPT = """Classify this pharmaceutical-industry company into exactly one class.

Classes:
{taxonomy}

{guide}

Company name: {company}

FDA label profile:
{profile}

Web search results:
{web}
{feedback}
Pick the single best class and cite the evidence."""

VALIDATE_PROMPT = """You are a strict reviewer checking a company classification.

Classes:
{taxonomy}

{guide}

Company name: {company}
FDA label profile:
{profile}
Web search results:
{web}

Proposed class: {label}
Classifier's evidence: {evidence}

Before deciding, name the strongest ALTERNATIVE class and the evidence for it.
Disagree if that alternative is at least as well supported as the proposed class.
Watch especially for: generic manufacturer vs repackager vs distributor (FDA labels can look identical;
web evidence about the business model decides), store brand vs consumer health vs personal care.
If you disagree, say what the evidence contradicts and suggest the better class."""


def _web_text(state):
    web = state.get("web")
    return json.dumps(web["results"], indent=1) if web else "none (not searched)"


# ---------- nodes ----------
def needs_web(fda):
    """Rule-based router: skip web search only when the FDA profile has a decisive signal."""
    if not fda.get("found"):
        return True, "no FDA labels found"
    if fda["share_homeopathic"] >= 0.8:
        return False, f"decisive: homeopathic share {fda['share_homeopathic']}"
    if fda["share_sunscreen"] >= 0.8 or fda["share_hand_sanitizer"] >= 0.5:
        return False, "decisive: sunscreen / sanitizer products"
    rx = fda["product_types"].get("HUMAN PRESCRIPTION DRUG", 0) / max(fda["labels_sampled"], 1)
    bg = fda.get("rx_share_brand_equals_generic")
    if rx >= 0.8 and bg is not None and bg <= 0.3:
        return False, f"decisive: branded Rx (brand=generic {bg})"
    return True, "ambiguous profile: generic-Rx-like or mixed OTC, FDA labels can't show the business model"


def research(state: State):
    company = state["company"]
    fda = cached(FDA_CACHE, company, lambda c: with_retry(fda_profile, c))
    need, reason = needs_web(fda)
    web = None
    if need:
        try:
            web = cached(WEB_CACHE, company, lambda c: with_retry(web_search, c))
        except Exception as e:                       # fallback: carry on with FDA evidence only
            reason += f" | web search failed: {type(e).__name__}"
    return {"fda": fda, "web": web, "used_web": web is not None, "attempts": 0,
            "trace": [f"research: web={'yes' if web else 'no'} ({reason})"]}

def classify(state: State):
    feedback = ""
    v = state.get("validation")
    if v and v["verdict"] == "disagree":
        feedback = (f"\nA reviewer disagreed with your previous answer "
                    f"({state['classification']['label']}): {v['issue']}\nReconsider using the evidence.\n")
    out = classifier.invoke(CLASSIFY_PROMPT.format(
        taxonomy=TAXONOMY, guide=EVIDENCE_GUIDE, company=state["company"],
        profile=json.dumps(state["fda"], indent=1), web=_web_text(state), feedback=feedback))
    attempts = state.get("attempts", 0) + 1
    return {"classification": out.model_dump(), "attempts": attempts,
            "trace": [f"classify #{attempts}: {out.label} (conf {out.confidence:.2f})"]}


def validate(state: State):
    c = state["classification"]
    out = validator.invoke(VALIDATE_PROMPT.format(
        taxonomy=TAXONOMY, guide=EVIDENCE_GUIDE, company=state["company"],
        profile=json.dumps(state["fda"], indent=1), web=_web_text(state),
        label=c["label"], evidence=c["evidence"]))
    return {"validation": out.model_dump(),
            "trace": [f"validate: {out.verdict} - {out.issue[:120]}"]}


def route(state: State) -> str:
    if state["validation"]["verdict"] == "agree":
        return "finalize"
    if state["attempts"] < MAX_ATTEMPTS:
        return "classify"
    return "human_review"


def human_review(state: State):
    c, v = state["classification"], state["validation"]
    if HITL:
        answer = interrupt({"company": state["company"], "proposed": c["label"],
                            "validator_issue": v["issue"], "validator_suggests": v.get("suggested_label")})
        return {"classification": {**c, "label": answer, "confidence": 1.0},
                "status": "human_reviewed", "trace": [f"human_review: set to {answer}"]}
    return {"status": "needs_review", "trace": ["human_review: queued for a human (auto mode)"]}


def finalize(state: State):
    return {"status": "auto", "trace": ["finalize: accepted"]}


# ---------- wiring ----------
def build_graph():
    g = StateGraph(State)
    g.add_node("research", research)
    g.add_node("classify", classify)
    g.add_node("validate", validate)
    g.add_node("human_review", human_review)
    g.add_node("finalize", finalize)
    g.add_edge(START, "research")
    g.add_edge("research", "classify")
    g.add_edge("classify", "validate")
    g.add_conditional_edges("validate", route, ["finalize", "classify", "human_review"])
    g.add_edge("finalize", END)
    g.add_edge("human_review", END)
    return g.compile(checkpointer=InMemorySaver())   # checkpointer is what lets interrupt() pause and resume


if __name__ == "__main__":
    graph = build_graph()
    company = " ".join(sys.argv[1:])
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke({"company": company}, config)
    if "__interrupt__" in result:                    # paused for a human
        print(json.dumps(result["__interrupt__"][0].value, indent=1))
        label = input("Your label: ").strip()
        result = graph.invoke(Command(resume=label), config)
    for line in result["trace"]:
        print(" ", line)
    print(f"\nFINAL: {result['classification']['label']}  [{result['status']}]")