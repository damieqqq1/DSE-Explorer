"""Planning node for decomposing user goals into executable subtasks."""

from __future__ import annotations

import json
import re
from typing import Any

from agent.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from agent.state import AgentState, PlanStep
from utils.deepseek_llm import invoke_deepseek
from utils.logger import get_logger
from utils.tool_registry import get_relevant_tool_context


logger = get_logger(__name__)


def planner_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    tool_context = get_relevant_tool_context(question)
    prompt = build_planner_prompt(
        question,
        conversation_context=state.get("conversation_context", ""),
        long_term_context=state.get("long_term_context", ""),
        node_context=state.get("planner_context", ""),
        tool_context=tool_context,
    )

    try:
        raw = invoke_deepseek(prompt, system_prompt=PLANNER_SYSTEM_PROMPT)
        mode, plan = parse_plan(raw)
    except Exception:
        logger.exception("Planner failed; falling back to a single retrieval plan.")
        mode = "researcher"
        plan = fallback_plan(question)

    logger.info("Planner mode=%s, %s steps", mode, len(plan))
    return {"mode": mode, "plan": plan}


def parse_plan(raw: str) -> tuple[str, list[PlanStep]]:
    data = json.loads(extract_json(raw))
    mode = str(data.get("mode", "researcher")).lower()
    if mode not in ("researcher", "developer"):
        mode = "researcher"

    steps = data.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("Planner JSON must contain a list field named steps.")

    plan: list[PlanStep] = []
    for index, item in enumerate(steps[:4], start=1):
        if not isinstance(item, dict):
            continue
        task = str(item.get("task", "")).strip()
        query = str(item.get("query", "")).strip()
        if not task or not query:
            continue
        step: PlanStep = {"id": int(item.get("id") or index), "task": task, "query": query}
        tool = str(item.get("tool", "")).strip()
        if tool:
            step["tool"] = tool
        tool_args = item.get("tool_args")
        if isinstance(tool_args, dict):
            step["tool_args"] = tool_args
        plan.append(step)

    if not plan:
        raise ValueError("Planner produced no usable steps.")
    return mode, plan


def extract_json(raw: str) -> str:
    text = raw.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in planner output.")
    return match.group(0)


def fallback_plan(question: str) -> list[PlanStep]:
    query = normalize_query(question)
    return [
        {
            "id": 1,
            "task": "Retrieve the most relevant local paper chunks for the user question.",
            "query": query,
            "tool": "search_dse_papers",
        }
    ]


def normalize_query(question: str) -> str:
    replacements: dict[str, str] = {
        "设计空间探索": "design space exploration",
        "微架构": "microarchitecture",
        "贝叶斯优化": "Bayesian optimization",
        "多目标优化": "multi-objective optimization",
        "强化学习": "reinforcement learning",
        "图表示学习": "graph representation learning",
    }
    query = question
    for zh, en in replacements.items():
        query = query.replace(zh, en)
    return query
