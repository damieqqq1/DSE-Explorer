"""Experiment design assistant — generates DSE experiment templates via LLM."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from utils.config import get_settings
from utils.deepseek_llm import invoke_deepseek
from utils.logger import get_logger

logger = get_logger(__name__)

VariableType = Literal["continuous", "discrete", "mixed"]

EXPERIMENT_DESIGNER_SYSTEM_PROMPT = """\
You are an expert in Design Space Exploration (DSE) and Multi-Objective Optimization (MOO) \
experiment design. Based on the user's problem description, recommend a concrete experiment \
pipeline. Output strictly valid JSON with this schema:

{
  "problem_summary": "1-sentence summary of the problem",
  "design_space": {
    "variables": [{"name": "string", "type": "continuous|discrete|integer|categorical", "bounds": [min, max] or ["cat1","cat2"]}],
    "constraints": ["constraint description"],
    "estimated_size": "small|medium|large|huge"
  },
  "pipeline": {
    "sampling_strategy": "lhs|random|sobol|full_factorial|grid|custom",
    "sampling_rationale": "1-2 sentences why",
    "surrogate_model": "gp|random_forest|mlp|none|other",
    "surrogate_rationale": "1-2 sentences why",
    "acquisition_function": "ei|ucb|lcb|ehvi|parego|ts|random|other",
    "acquisition_rationale": "1-2 sentences why",
    "evaluation_budget": 100,
    "budget_rationale": "1-2 sentences why this budget"
  },
  "evaluation": {
    "metrics": ["hv", "igd", "spread"],
    "baselines": ["random_search", "bayesian_optimization"],
    "repetitions": 10,
    "reporting": "brief advice on how to report results"
  },
  "pitfalls": ["pitfall 1", "pitfall 2"],
  "references": ["method or paper name to consult (if relevant)"]
}

Rules:
- Recommend GP-based BO for expensive evaluations; random forest for discrete/mixed spaces.
- For 2-3 objectives, recommend EHVI or ParEGO acquisition; for >3, consider random scalarisation.
- Default to LHS for initial sampling; Sobol for high-dimensional (10+) continuous spaces.
- Default budget 50-100 for continuous, 100-200 for discrete.
- Keep advice concise, grounded in standard DSE/MOO practice.
- Use empty strings "" or empty lists [] for unknown fields.\
"""


def design_experiment_template(
    problem_description: str,
    num_design_variables: int | None = None,
    num_objectives: int | None = None,
    variable_type: VariableType = "continuous",
    constraints: list[str] | None = None,
    evaluation_budget: int | None = None,
    evaluation_cost: str = "",
) -> dict[str, Any]:
    """Generate a DSE/MOO experiment design template from a problem description.

    Args:
        problem_description: Natural-language description of the design problem.
        num_design_variables: Number of design variables.
        num_objectives: Number of objectives to optimise.
        variable_type: continuous, discrete, or mixed.
        constraints: Optional list of constraint descriptions.
        evaluation_budget: Optional budget (number of evaluations available).
        evaluation_cost: Description of evaluation cost (e.g. "cheap simulator", "expensive CFD").

    Returns:
        A dictionary with the recommended experiment pipeline, evaluation setup,
        and known pitfalls.
    """
    desc = problem_description.strip()
    if not desc:
        raise ValueError("problem_description must not be empty.")

    settings = get_settings()

    extra: list[str] = []
    if num_design_variables is not None:
        extra.append(f"Number of design variables: {num_design_variables}")
    if num_objectives is not None:
        extra.append(f"Number of objectives: {num_objectives}")
    extra.append(f"Variable type: {variable_type}")
    if constraints:
        extra.append("Constraints: " + "; ".join(constraints))
    if evaluation_budget is not None:
        extra.append(f"Evaluation budget: {evaluation_budget} evaluations")
    if evaluation_cost:
        extra.append(f"Evaluation cost: {evaluation_cost}")

    context = "\n".join(extra)

    prompt = f"""Problem description:
{desc}

{context}

Recommend a concrete DSE experiment pipeline for this problem."""

    try:
        raw = invoke_deepseek(
            prompt,
            system_prompt=EXPERIMENT_DESIGNER_SYSTEM_PROMPT,
            settings=settings,
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        template = json.loads(match.group()) if match else {"raw_response": raw}
    except Exception:
        logger.exception("LLM call failed for experiment design")
        return {
            "success": False,
            "error": "LLM call failed during experiment design.",
            "problem_description": desc,
        }

    return {
        "success": True,
        "problem_description": desc,
        "inputs": {
            "num_design_variables": num_design_variables,
            "num_objectives": num_objectives,
            "variable_type": variable_type,
            "constraints": constraints or [],
            "evaluation_budget": evaluation_budget,
            "evaluation_cost": evaluation_cost,
        },
        "template": template,
    }


def register_experiment_designer_tools(mcp: Any) -> None:
    """Register experiment design tools on a FastMCP server instance."""

    mcp.tool(
        name="design_experiment_template",
        description=(
            "Generate a DSE/MOO experiment design template from a problem description. "
            "Recommends a complete pipeline: design space characterisation, sampling "
            "strategy, surrogate model, acquisition function, evaluation budget, "
            "metrics, baselines, and known pitfalls. Use this when the user asks "
            "'how should I design my DSE experiment' or 'what pipeline should I use "
            "for this optimisation problem'."
        ),
    )(design_experiment_template)
