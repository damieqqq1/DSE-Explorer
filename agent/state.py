"""Shared agent state definitions."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class PlanStep(TypedDict):
    id: int
    task: str
    query: str
    tool: NotRequired[str]
    tool_args: NotRequired[dict[str, Any]]
    rewritten_query: NotRequired[str]
    query_variants: NotRequired[list[str]]


class EvidenceItem(TypedDict):
    step_id: int
    task: str
    query: str
    tool: NotRequired[str]
    results: list[dict[str, Any]]


class SourceItem(TypedDict):
    source: str
    page: int | str
    score: float


class AgentState(TypedDict, total=False):
    question: str
    conversation_context: str
    long_term_context: str
    planner_context: str
    rewriter_context: str
    reasoner_context: str
    synthesis_context: str
    context_trace: list[dict[str, Any]]
    mode: str
    plan: list[PlanStep]
    executed_step_ids: list[int]
    evidence: list[EvidenceItem]
    iteration: int
    max_iterations: int
    needs_more_evidence: bool
    reasoner_note: str
    compressed_context: str
    final_answer: str
    sources: list[SourceItem]
