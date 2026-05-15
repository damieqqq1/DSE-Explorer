"""Reasoning and synthesis nodes for paper-grounded answers."""

from __future__ import annotations

import json
import re
from typing import Any

from agent.prompts import (
    REASONER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    build_reasoner_prompt,
    build_synthesis_prompt,
)
from agent.state import AgentState, PlanStep, SourceItem
from utils.deepseek_llm import invoke_deepseek
from utils.logger import get_logger


logger = get_logger(__name__)


def reasoner_node(state: AgentState) -> dict[str, Any]:
    compressed_context = compress_evidence(state)
    iteration = int(state.get("iteration", 0)) + 1
    max_iterations = int(state.get("max_iterations", 2))

    if iteration >= max_iterations:
        logger.info("Reasoner reached max iterations: %s", max_iterations)
        return {
            "iteration": iteration,
            "compressed_context": compressed_context,
            "needs_more_evidence": False,
            "reasoner_note": "Reached max iterations; synthesizing with current evidence.",
        }

    try:
        assessment = assess_evidence(
            state["question"],
            compressed_context,
            conversation_context=state.get("conversation_context", ""),
            long_term_context=state.get("long_term_context", ""),
        )
    except Exception:
        logger.exception("Reasoner assessment failed; synthesizing with current evidence.")
        return {
            "iteration": iteration,
            "compressed_context": compressed_context,
            "needs_more_evidence": False,
            "reasoner_note": "Reasoner assessment failed; using current evidence.",
        }

    if assessment.get("sufficient", True):
        return {
            "iteration": iteration,
            "compressed_context": compressed_context,
            "needs_more_evidence": False,
            "reasoner_note": str(assessment.get("reason", "")),
        }

    follow_up_steps = build_follow_up_steps(state, assessment.get("follow_up_steps", []))
    if not follow_up_steps:
        return {
            "iteration": iteration,
            "compressed_context": compressed_context,
            "needs_more_evidence": False,
            "reasoner_note": "Evidence marked insufficient but no usable follow-up steps were produced.",
        }

    logger.info("Reasoner added %s follow-up steps", len(follow_up_steps))
    return {
        "iteration": iteration,
        "plan": list(state.get("plan", [])) + follow_up_steps,
        "compressed_context": compressed_context,
        "needs_more_evidence": True,
        "reasoner_note": str(assessment.get("reason", "")),
    }


def synthesizer_node(state: AgentState) -> dict[str, Any]:
    evidence_text = state.get("compressed_context") or compress_evidence(state)
    prompt = build_synthesis_prompt(
        state["question"],
        evidence_text,
        conversation_context=state.get("conversation_context", ""),
        long_term_context=state.get("long_term_context", ""),
    )
    answer = invoke_deepseek(prompt, system_prompt=SYNTHESIZER_SYSTEM_PROMPT)
    sources = collect_sources(state)
    logger.info("Synthesized answer with %s sources", len(sources))
    return {"final_answer": answer, "sources": sources}


def assess_evidence(
    question: str,
    compressed_context: str,
    conversation_context: str = "",
    long_term_context: str = "",
) -> dict[str, Any]:
    raw = invoke_deepseek(
        build_reasoner_prompt(
            question,
            compressed_context,
            conversation_context=conversation_context,
            long_term_context=long_term_context,
        ),
        system_prompt=REASONER_SYSTEM_PROMPT,
    )
    return json.loads(extract_json(raw))


def extract_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in reasoner output.")
    return match.group(0)


def build_follow_up_steps(state: AgentState, raw_steps: Any) -> list[PlanStep]:
    if not isinstance(raw_steps, list):
        return []

    existing_queries = {step["query"].lower() for step in state.get("plan", [])}
    next_id = max((step["id"] for step in state.get("plan", [])), default=0) + 1
    steps: list[PlanStep] = []
    for item in raw_steps[:2]:
        if not isinstance(item, dict):
            continue
        task = str(item.get("task", "")).strip()
        query = str(item.get("query", "")).strip()
        if not task or not query or query.lower() in existing_queries:
            continue
        steps.append({"id": next_id, "task": task, "query": query})
        next_id += 1
    return steps


# Tools whose output is a block of structured content (table, template, plot), not chunk-text.
_STRUCTURED_TOOLS = {
    "generate_dse_method_matrix",
    "compute_dse_metrics",
    "generate_pareto_front",
    "design_experiment_template",
    "get_rag_corpus_stats",
    "get_memory_stats",
    "get_recent_conversations",
}

# Metadata keys that don't carry answer content — skip them for brevity.
_SKIP_KEYS = {"success", "query", "url", "paper_path", "file_name", "question"}
# Keys whose value is a content block to include at full length.
_CONTENT_KEYS = {"matrix", "content", "template", "stdout", "message", "answer"}


def compress_evidence(state: AgentState, max_results: int = 10, max_chars: int = 900) -> str:
    blocks: list[str] = []

    # 1. Collect structured evidence first — show it in full.
    for item in state.get("evidence", []):
        tool = str(item.get("tool", ""))
        if tool not in _STRUCTURED_TOOLS:
            continue
        for result in item.get("results", []):
            if not isinstance(result, dict):
                continue
            parts: list[str] = [f"- Structured output from {tool} (step {item['step_id']}):"]
            for key, value in sorted(result.items()):
                if key in _SKIP_KEYS:
                    continue
                if isinstance(value, str) and value.strip():
                    cap = 5000 if key in _CONTENT_KEYS else 500
                    parts.append(f"  [{key}]:\n{value[:cap]}")
                elif isinstance(value, dict):
                    parts.append(f"  [{key}]:\n{json.dumps(value, indent=2, ensure_ascii=False)[:3000]}")
                elif isinstance(value, list):
                    parts.append(f"  [{key}]:\n{json.dumps(value, indent=2, ensure_ascii=False)[:3000]}")
                elif value is not None:
                    parts.append(f"  [{key}]: {value}")
            if len(parts) > 1:
                blocks.append("\n".join(parts))

    # 2. Collect chunk-based evidence (search_dse_papers, ask_paper, fetch_web_page, ...).
    rows: list[tuple[float, str, int | str, str, str]] = []
    seen: set[tuple[str, int | str, str]] = set()
    for item in state.get("evidence", []):
        tool = str(item.get("tool", ""))
        if tool in _STRUCTURED_TOOLS:
            continue
        for result in item.get("results", []):
            metadata = result.get("metadata", {})
            source = str(metadata.get("relative_path") or metadata.get("file_name") or "unknown")
            page = metadata.get("page", "?")
            text = " ".join(str(result.get("text") or result.get("content", "")).split())
            key = (source, page, text[:160])
            if key in seen:
                continue
            seen.add(key)
            rows.append((float(result.get("score", 0.0)), source, page, item["task"], text))

    rows.sort(key=lambda row: row[0], reverse=True)
    for score, source, page, task, text in rows[:max_results]:
        blocks.append(
            f"- Evidence label: [{source}, page {page}]\n"
            f"  Task: {task}\n"
            f"  Score: {score:.4f}\n"
            f"  Text: {text[:max_chars]}"
        )
    return "\n".join(blocks)


def format_evidence(state: AgentState) -> str:
    blocks: list[str] = []
    for item in state.get("evidence", []):
        blocks.append(f"Step {item['step_id']}: {item['task']}\nQuery: {item['query']}")
        for result in item.get("results", []):
            metadata = result.get("metadata", {})
            source = metadata.get("relative_path") or metadata.get("file_name") or "unknown"
            page = metadata.get("page", "?")
            score = result.get("score", 0.0)
            text = result.get("text", "")
            blocks.append(
                f"- Evidence label: [{source}, page {page}]\n"
                f"  Score: {score:.4f}\n"
                f"  Text: {text}"
            )
    return "\n".join(blocks)


def collect_sources(state: AgentState) -> list[SourceItem]:
    seen: set[tuple[str, int | str]] = set()
    sources: list[SourceItem] = []
    for item in state.get("evidence", []):
        for result in item.get("results", []):
            metadata = result.get("metadata", {})
            source = str(metadata.get("relative_path") or metadata.get("file_name") or "unknown")
            page = metadata.get("page", "?")
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "source": source,
                    "page": page,
                    "score": float(result.get("score", 0.0)),
                }
            )
    return sources[:10]
