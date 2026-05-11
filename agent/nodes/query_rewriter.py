"""Query rewriting node for improving retrieval recall and precision."""

from __future__ import annotations

import json
import re
from typing import Any

from agent.prompts import QUERY_REWRITER_SYSTEM_PROMPT, build_query_rewriter_prompt
from agent.state import AgentState, PlanStep
from utils.deepseek_llm import invoke_deepseek
from utils.logger import get_logger


logger = get_logger(__name__)

MAX_QUERY_VARIANTS = 4


def query_rewriter_node(state: AgentState) -> dict[str, list[PlanStep]]:
    rewritten_plan: list[PlanStep] = []
    for step in state.get("plan", []):
        if step.get("query_variants"):
            rewritten_plan.append(step)
            continue
        try:
            rewritten_plan.append(rewrite_step(state, step))
        except Exception:
            logger.exception("Query rewrite failed for step %s; using original query.", step["id"])
            rewritten_plan.append(with_fallback_rewrite(step))

    logger.info("Query rewriter prepared %s retrieval steps", len(rewritten_plan))
    return {"plan": rewritten_plan}


def rewrite_step(state: AgentState, step: PlanStep) -> PlanStep:
    prompt = build_query_rewriter_prompt(
        state["question"],
        step["task"],
        step["query"],
        conversation_context=state.get("conversation_context", ""),
        long_term_context=state.get("long_term_context", ""),
    )
    raw = invoke_deepseek(prompt, system_prompt=QUERY_REWRITER_SYSTEM_PROMPT)
    payload = json.loads(extract_json(raw))
    rewritten_query = str(payload.get("rewritten_query", "")).strip()
    raw_variants = payload.get("query_variants", [])
    variants = clean_query_variants([step["query"], rewritten_query, *coerce_list(raw_variants)])
    if not variants:
        variants = [step["query"]]
    return {
        **step,
        "rewritten_query": rewritten_query or variants[0],
        "query_variants": variants,
    }


def with_fallback_rewrite(step: PlanStep) -> PlanStep:
    query = normalize_query(step["query"])
    variants = clean_query_variants([step["query"], query])
    return {
        **step,
        "rewritten_query": query,
        "query_variants": variants,
    }


def clean_query_variants(items: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        query = " ".join(str(item).split())
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(query)
        if len(cleaned) >= MAX_QUERY_VARIANTS:
            break
    return cleaned


def coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def extract_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in query rewriter output.")
    return match.group(0)


def normalize_query(query: str) -> str:
    replacements: dict[str, str] = {
        "DSE": "DSE design space exploration",
        "MOO": "MOO multi-objective optimization",
        "BO": "BO Bayesian optimization",
        "RL": "RL reinforcement learning",
        "LLM": "LLM large language model",
    }
    normalized = query
    for term, expanded in replacements.items():
        normalized = re.sub(rf"\b{re.escape(term)}\b", expanded, normalized)
    return " ".join(normalized.split())
