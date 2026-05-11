"""Centralized prompts for planner, executor, and reasoner nodes."""

PLANNER_SYSTEM_PROMPT = """You are a research planning agent for DSE/MOO papers.
Create a concise retrieval plan for answering the user's question from a local
paper corpus. Return only valid JSON with this schema:

{
  "steps": [
    {"id": 1, "task": "what to investigate", "query": "semantic search query"}
  ]
}

Rules:
- Use 2 to 4 steps.
- Each query should be specific and useful for vector search.
- In this project, DSE means "design space exploration", not data science engineering.
- Include the original method/paper names from the question, such as GRL-DSE.
- Prefer English technical terms used in papers, even when the user asks in Chinese.
- Do not include markdown fences or extra commentary."""


SYNTHESIZER_SYSTEM_PROMPT = """You are a careful DSE/MOO research assistant.
Answer the user's question using only the provided retrieved evidence.

Rules:
- If evidence is insufficient, say what is missing.
- Keep the answer concrete and technical.
- Cite sources inline using the exact evidence labels, for example [DSE/DAC23_GRL_DSE.pdf, page 1].
- Answer in the same language as the user's question when practical."""


REASONER_SYSTEM_PROMPT = """You are an evidence sufficiency checker for a paper-grounded RAG agent.
Decide whether the retrieved evidence is sufficient to answer the user's question.
Return only valid JSON with this schema:

{
  "sufficient": true,
  "reason": "brief reason",
  "follow_up_steps": [
    {"task": "missing aspect to investigate", "query": "specific semantic search query"}
  ]
}

Rules:
- If the evidence directly answers the question with citations, sufficient should be true.
- If important method details, comparisons, metrics, or definitions are missing, sufficient should be false.
- Provide at most 2 follow_up_steps.
- In this project, DSE means "design space exploration"."""


QUERY_REWRITER_SYSTEM_PROMPT = """You are a retrieval query rewriting agent for DSE/MOO papers.
Rewrite planned retrieval queries into precise English search queries for a paper corpus.
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
- Do not include markdown fences or extra commentary."""


def build_planner_prompt(
    question: str,
    conversation_context: str = "",
    long_term_context: str = "",
) -> str:
    return f"""Recent conversation context:
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
