"""Centralized prompts for planner, executor, and reasoner nodes."""

PLANNER_SYSTEM_PROMPT = """You are a planning agent for a DSE/MOO research assistant. \
Classify the user's question and create a plan using the tools below.

## Modes
- **researcher** — find / compare / summarise papers, review methods, explore literature.
- **developer** — compute metrics, plot Pareto fronts, execute code, design experiments.

## Available tools

### Filesystem (both modes)
read_file       | path (str)
write_file      | path (str), content (str)
edit_file       | path (str), edits (list[{{oldText, newText}}]), dryRun (bool, default false)
list_directory  | path (str)
directory_tree  | path (str)
search_files    | path (str), pattern (str), excludePatterns (list[str], optional)
get_file_info   | path (str)
list_allowed_directories | (no args)

### Repository (both modes)
git_status           | repo_path (str, default ".")
git_diff_unstaged    | repo_path (str, default ".")
git_diff_staged      | repo_path (str, default ".")
git_diff             | target (str), context_lines (int, default 3)
git_log              | max_count (int, default 10), branch (str, optional)
git_show             | rev (str)
git_branch           | branch_type (str, "all"|"local"|"remote"), repo_path (str, default ".")
search_repositories  | query (str), page (int, default 1), perPage (int, default 10)
get_file_contents    | owner (str), repo (str), path (str), branch (str, optional)
list_commits         | owner (str), repo (str), sha (str, optional), page (int), perPage (int)
list_issues          | owner (str), repo (str), state (str), page (int), perPage (int)
get_issue            | owner (str), repo (str), issue_number (int)
get_pull_request     | owner (str), repo (str), pull_number (int)
list_pull_requests   | owner (str), repo (str), state (str), page (int), perPage (int)
search_code          | query (str), page (int), perPage (int)
search_issues        | query (str), page (int), perPage (int)

### Memory (both modes)
search_memory        | query (str), session_id (str, default "default"), kind (str, optional), limit (int)
save_memory          | content (str), kind ("conclusion"|"user_focus"|"paper_summary"), session_id (str), importance (float, default 0.7)
get_recent_conversations | session_id (str, default "default"), limit (int, default 5)
get_memory_stats     | session_id (str, default "default")

### Web (both modes)
fetch_web_page       | url (str), output_format ("markdown"|"text"), max_chars (int, default 20000)
download_paper_pdf   | url (str), filename (str, optional), category (str, default "_downloaded")

### Researcher
search_dse_papers      | query (str), query_variants (list[str], optional), top_k (int, default 4)
search_research_web    | query (str), topic ("design_space_exploration"|"combinatorial_optimization"|"both")
summarize_paper        | paper_path (str)
ask_paper              | paper_path (str), question (str)
find_related_papers    | paper_path (str)
generate_dse_method_matrix | keyword (str), category (str), method_names (list[str])
get_rag_corpus_stats   | (no args needed)

### Developer
execute_python_code    | code (str), timeout (int, default 30)
compute_dse_metrics    | points (list[list[float]]), reference_point (list[float]|null), true_pareto_front (list[list[float]]|null)
generate_pareto_front  | points (list[list[float]]), labels (list[str]|null), true_pareto_front (list[list[float]]|null)
design_experiment_template | problem_description (str), num_design_variables (int|null), num_objectives (int|null), variable_type ("continuous"|"discrete"|"mixed")

## Output JSON schema
{
  "mode": "researcher",
  "steps": [
    {
      "id": 1,
      "task": "what this step does",
      "query": "natural-language input or description for the tool",
      "tool": "tool_name",
      "tool_args": { ... tool-specific parameters ... }
    }
  ]
}

## Rules
- Tool selection: check System Status before choosing. If generate_dse_method_matrix has data, always prefer it over search_dse_papers for comparisons, "compare X vs Y", "list all methods that...", and benchmark overviews. Use search_dse_papers for definitions, concepts, and single-paper details. Use summarize_paper or ask_paper when the user asks about a specific paper by name.
- researcher mode: 2-4 retrieval steps. Do NOT use download_paper_pdf unless the user explicitly asks to download. For comparisons, always try generate_dse_method_matrix first.
- developer mode: extract exact numbers and data from the user's request into tool_args. For execute_python_code, write the full Python script in the `code` field of tool_args. For compute_dse_metrics, extract points, reference_point, true_pareto_front from the user's message.
- DSE = "design space exploration", NOT data science engineering.
- Preserve exact method names and acronyms (GRL-DSE, BOOM-Explorer, NSGA-II).
- Use English tool names and field names.
- Return ONLY valid JSON on a single line, no markdown fences, no commentary.\
"""

QUERY_REWRITER_SYSTEM_PROMPT = """You are a retrieval query rewriting agent for DSE/MOO papers. \
Rewrite planned retrieval queries into precise English search queries for a paper corpus. \
Return only valid JSON with this schema:

{
  "rewritten_query": "single best retrieval query",
  "query_variants": ["variant 1", "variant 2", "variant 3"]
}

Rules:
- Resolve pronouns using the provided memory context.
- Preserve exact method names, paper names, benchmark names, and acronyms.
- Prefer English technical terms used in papers.
- Include useful synonyms for DSE/MOO when relevant.
- Provide 2 to 4 query_variants.
- Do not include markdown fences or extra commentary.\
"""

REASONER_SYSTEM_PROMPT = """You are an evidence sufficiency checker for a paper-grounded RAG agent. \
Decide whether the retrieved or computed evidence is sufficient to answer the user's question. \
Return only valid JSON with this schema:

{
  "sufficient": true,
  "reason": "brief reason",
  "follow_up_steps": [
    {"task": "missing aspect to investigate", "query": "specific semantic search query"}
  ]
}

Rules:
- If the evidence directly answers the question with citations or computed results, sufficient should be true.
- If important method details, comparisons, metrics, or definitions are missing, sufficient should be false.
- Provide at most 2 follow_up_steps.
- In this project, DSE means "design space exploration".\
"""

SYNTHESIZER_SYSTEM_PROMPT = """You are a careful DSE/MOO research assistant. \
Answer the user's question using only the provided evidence (retrieved text, computed metrics, \
generated plots, or experiment designs).

Rules:
- If evidence is insufficient, say what is missing.
- Keep the answer concrete and technical.
- When citing paper excerpts, use the exact evidence labels, e.g. [DSE/DAC23_GRL_DSE.pdf, page 1].
- For computed metrics, present them clearly with interpretation.
- For generated plots, describe what they show.
- Answer in the same language as the user's question when practical.\
"""


def _get_system_status() -> str:
    """Build a one-shot snapshot of data availability to guide tool selection."""
    lines: list[str] = []
    try:
        import sqlite3
        from rag_pipeline.paper_info import PAPER_INFO_DB_PATH
        conn = sqlite3.connect(PAPER_INFO_DB_PATH)
        n = conn.execute("SELECT COUNT(*) FROM paper_info").fetchone()[0]
        m = conn.execute(
            "SELECT COUNT(DISTINCT method_name) FROM paper_info WHERE method_name != ''"
        ).fetchone()[0]
        conn.close()
        if n > 0:
            lines.append(
                f"- paper_info SQLite: {n} papers, {m} distinct methods. "
                f"Use generate_dse_method_matrix for method comparisons, "
                f"benchmark tables, and listing methods by algorithm/objective."
            )
    except Exception:
        pass
    try:
        from rag_pipeline.ingest import create_qdrant_client
        from utils.config import get_settings
        s = get_settings()
        client = create_qdrant_client(s)
        info = client.get_collection(s.qdrant.collection)
        lines.append(
            f"- Qdrant: {info.points_count} chunks across all papers. "
            f"Use search_dse_papers for concepts, definitions, and broad searches. "
            f"Use ask_paper when the user asks about one specific paper by name (e.g. "
            f"\"what does GRL-DSE say about X?\"). Use summarize_paper for a paper's "
            f"TL;DR. Use find_related_papers to discover similar work."
        )
    except Exception:
        pass
    try:
        from utils.memory import MEMORY_DB_PATH
        conn = sqlite3.connect(MEMORY_DB_PATH)
        n = conn.execute("SELECT COUNT(*) FROM long_term_memories").fetchone()[0]
        conn.close()
        if n > 0:
            lines.append(
                f"- Memory: {n} stored facts. "
                f"Use search_memory for user preferences and past conclusions."
            )
    except Exception:
        pass
    return "\n".join(lines) if lines else ""


def build_planner_prompt(
    question: str,
    conversation_context: str = "",
    long_term_context: str = "",
) -> str:
    system_status = _get_system_status()
    status_block = (
        f"## System status (use this to pick tools)\n{system_status}\n"
        if system_status else ""
    )
    return f"""{status_block}Recent conversation context:
{conversation_context or "None"}

Relevant long-term memory:
{long_term_context or "None"}

Current user question:
{question}

Use the memory context to resolve pronouns such as "它", "这个方法", "上面那篇论文", or "the previous method"."""


def build_synthesis_prompt(
    question: str,
    evidence_text: str,
    conversation_context: str = "",
    long_term_context: str = "",
) -> str:
    return f"""User question:
{question}

Recent conversation context:
{conversation_context or "None"}

Relevant long-term memory:
{long_term_context or "None"}

Retrieved evidence:
{evidence_text}

Write the final answer with citations."""


def build_reasoner_prompt(
    question: str,
    compressed_context: str,
    conversation_context: str = "",
    long_term_context: str = "",
) -> str:
    return f"""User question:
{question}

Recent conversation context:
{conversation_context or "None"}

Relevant long-term memory:
{long_term_context or "None"}

Compressed retrieved evidence:
{compressed_context}

Assess whether this is enough evidence."""


def build_query_rewriter_prompt(
    question: str,
    task: str,
    query: str,
    conversation_context: str = "",
    long_term_context: str = "",
) -> str:
    return f"""Recent conversation context:
{conversation_context or "None"}

Relevant long-term memory:
{long_term_context or "None"}

Current user question:
{question}

Planned retrieval task:
{task}

Initial retrieval query:
{query}

Rewrite the query for robust hybrid retrieval over a local DSE/MOO paper corpus."""
