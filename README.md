# DSE-Explorer

面向“组合优化 / 设计空间探索 / 多目标优化”研究的 Agent 系统。

DSE/MOO research agent based on LangGraph, MCP tools, Qdrant-backed RAG, hybrid retrieval, paper information extraction, and structured method comparison.

The project targets research workflows around design space exploration (DSE), multi-objective optimization (MOO), and combinatorial optimization papers.

## Structure

- `agent/`: LangGraph workflow and agent nodes.
- `mcp_server/`: MCP server and tool implementations.
- `rag_pipeline/`: PDF ingestion, hybrid retrieval, paper information extraction, and method matrix generation.
- `utils/`: configuration, LLM wrapper, embedding client, memory, and logging helpers.
- `memory/`: local SQLite conversation and long-term memory.
- `reports/`: optional generated method matrices and reports.

## Configuration Check

```powershell
conda run --no-capture-output -n dse_explorer python main.py --check
conda run --no-capture-output -n dse_explorer python main.py --show-config
```

## PDF Ingestion

Small smoke test:

```powershell
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.ingest --limit 3
```

Full ingestion:

```powershell
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.ingest --all --batch-size 10
```

Dry run without embedding or Qdrant writes:

```powershell
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.ingest --all --dry-run
```

## Hybrid Retrieval

The retriever now uses hybrid search by default:

- Dense retrieval: Qdrant vector search.
- Sparse retrieval: in-memory BM25 keyword search over Qdrant payload text.
- Fusion: Reciprocal Rank Fusion (RRF).
- Reranking: lightweight reranker using fused score, token overlap, exact method-name hits, source hits, and phrase hits.

```powershell
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.retriever "GRL-DSE graph representation learning design space exploration" --top-k 3 --candidate-k 30
```

Disable hybrid search or reranking:

```powershell
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.retriever "LLM design space exploration" --top-k 5 --dense-only
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.retriever "LLM design space exploration" --top-k 5 --no-rerank
```

## Paper Information Extraction

Extract structured paper metadata and method information into local SQLite:

```powershell
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.paper_info --limit 3
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.paper_info --all
```

Dry run and list extracted records:

```powershell
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.paper_info --limit 1 --dry-run
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.paper_info --list --limit 10
```

The extracted data is stored at:

```text
rag_pipeline/data/paper_info.sqlite3
```

Extracted metadata includes title, authors, year, venue, abstract, and keywords.
Extracted method information includes method name, problem, core idea, optimizer, model/agent, design variables, objectives, benchmarks, baselines, metrics, results, and limitations.

The same capability is also exposed through `main.py`:

```powershell
conda run --no-capture-output -n dse_explorer python main.py --extract-paper-info --paper-info-limit 3
conda run --no-capture-output -n dse_explorer python main.py --paper-info-list
```

## Method Matrix

Generate a DSE method comparison matrix from extracted paper information:

```powershell
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.method_matrix --limit 20 --format markdown
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.method_matrix --keyword "Bayesian optimization" --format csv
```

Save a matrix under the project directory:

```powershell
conda run --no-capture-output -n dse_explorer python -m rag_pipeline.method_matrix --limit 20 --save-path reports/dse_method_matrix.md
```

The same matrix generation logic is exposed as an MCP tool named `generate_dse_method_matrix`.

## MCP Server

```powershell
conda run --no-capture-output -n dse_explorer python -m mcp_server.server --transport stdio
```

HTTP mode:

```powershell
conda run --no-capture-output -n dse_explorer python -m mcp_server.server --transport http --host 127.0.0.1 --port 8765
```

Registered tools:

- `search_dse_papers`: search local DSE/MOO paper chunks using hybrid retrieval and reranking.
- `get_rag_corpus_stats`: inspect the current Qdrant collection.
- `generate_dse_method_matrix`: generate a structured DSE method comparison matrix from `paper_info.sqlite3`.
- `search_research_web`: search the internet for DSE or combinatorial optimization research.

The agent executor calls `search_dse_papers` through an MCP stdio client, which starts `mcp_server.server` as a local child process and invokes the tool over MCP.

## Paper-Grounded Agent

```powershell
conda run --no-capture-output -n dse_explorer python main.py --ask "GRL-DSE 的核心思想是什么？"
```

Agent workflow:

```text
Planner -> Query Rewriter -> MCP Hybrid RAG Executor -> Reasoner -> Synthesizer
```

Useful flags:

```powershell
conda run --no-capture-output -n dse_explorer python main.py --ask "GRL-DSE 的核心思想是什么？" --show-trace
conda run --no-capture-output -n dse_explorer python main.py --ask "它用了什么优化方法？" --session demo --show-memory
conda run --no-capture-output -n dse_explorer python main.py --ask "LLM-DSE 和传统 DSE 有什么区别？" --max-iterations 3
```

Memory flags:

- `--session`: conversation memory session id.
- `--memory-turns`: number of recent turns injected into context.
- `--no-memory`: disable conversation and long-term memory.
- `--show-memory`: print injected memory context.

## Direct LLM Prompt

```powershell
conda run --no-capture-output -n dse_explorer python main.py --prompt "请用一句话解释设计空间探索。"
```
