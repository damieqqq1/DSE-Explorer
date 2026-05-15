# dse-explorer 使用说明

`dse-explorer` 是一个面向设计空间探索（DSE）、多目标优化（MOO）研究的论文 Agent 项目。支持研究者模式（检索/总结/对比论文）和开发者模式（计算指标/绘图/设计实验），集成 5 个 MCP 服务器共 50+ 工具。

## 1. 环境准备

进入项目目录：

```powershell
cd C:\Users\Administrator\Desktop\Code\dse-explorer
```

使用 Conda 环境运行：

```powershell
conda activate dse_explorer
```

如果 PowerShell 没有初始化 Conda：

```powershell
conda run --no-capture-output -n dse_explorer python main.py --check
```

外部 MCP 依赖（Node.js）：

```powershell
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-github
pip install mcp-server-git
```

## 2. 配置检查

项目配置在 `.env` 中，包括：

- `LLM_API_KEY` — DeepSeek LLM API Key
- `EMBED_API_KEY` / `DASHSCOPE_API_KEY` — Embedding API Key
- `QDRANT_URL` / `QDRANT_API_KEY` — Qdrant 向量数据库
- `TAVILY_API_KEY` / `SERPAPI_API_KEY` — 网络检索（可选）
- `GITHUB_PERSONAL_ACCESS_TOKEN` — GitHub MCP（可选）

检查配置：

```powershell
python main.py --check
python main.py --show-config
```

## 3. PDF 论文入库

本地论文目录：`Paper_ljh/`（179 篇 PDF）

```powershell
# 小规模测试
python -m rag_pipeline.ingest --limit 3

# 全量入库
python -m rag_pipeline.ingest --all --batch-size 10

# 只解析不写入
python -m rag_pipeline.ingest --all --dry-run
```

入库流程：`扫描 PDF → pypdf 逐页解析 → 文本清洗 → chunk 切分 → embedding → Qdrant upsert`

## 4. Hybrid Retrieval 检索

```text
Query → Dense (Qdrant) + Sparse (BM25) → RRF 融合 → Lightweight Reranker
```

```powershell
python -m rag_pipeline.retriever "GRL-DSE graph reinforcement learning DSE" --top-k 3
python -m rag_pipeline.retriever "LLM DSE" --top-k 5 --dense-only
python -m rag_pipeline.retriever "LLM DSE" --top-k 5 --no-rerank
```

## 5. 论文元数据抽取和方法矩阵

从 PDF 前几页自动抽取结构化论文信息，保存到 SQLite：

```powershell
python -m rag_pipeline.paper_info --limit 3
python -m rag_pipeline.paper_info --all
python -m rag_pipeline.paper_info --list --limit 10
```

通过 main.py：

```powershell
python main.py --extract-paper-info --paper-info-limit 3
python main.py --paper-info-list
```

生成 DSE 方法对比矩阵：

```powershell
python -m rag_pipeline.method_matrix --limit 20 --format markdown
python -m rag_pipeline.method_matrix --keyword "Bayesian optimization" --format csv
python -m rag_pipeline.method_matrix --limit 20 --save-path reports/matrix.md
```

## 6. MCP 服务器架构

项目运行 5 个 MCP 服务器，通过 stdio 持久连接（后台事件循环保活，首次懒初始化）：

```
Agent Executor
  ├─ python -m mcp_server.server          dse-explorer MCP (16 tools)
  ├─ npx @modelcontextprotocol/server-filesystem   Filesystem MCP (8 tools)
  ├─ python -m mcp_server_git             Git MCP (10 tools)
  └─ npx @modelcontextprotocol/server-github      GitHub MCP (15+ tools)
```

启动 DSE MCP Server（仅 dse-explorer 自身）：

```powershell
python -m mcp_server.server --transport stdio
python -m mcp_server.server --transport http --host 127.0.0.1 --port 8765
```

### DSE MCP 工具一览（16 个）

| 分类 | 工具 | 用途 |
|------|------|------|
| 检索 | `search_dse_papers` | Hybrid 搜索本地论文库 |
| 检索 | `search_research_web` | 网络搜索 DSE 研究 |
| 检索 | `get_rag_corpus_stats` | Qdrant 语料统计 |
| 论文 | `summarize_paper` | 论文结构化摘要 |
| 论文 | `ask_paper` | 单篇论文问答（带页码引用） |
| 论文 | `find_related_papers` | 论文关系图谱 |
| 论文 | `generate_dse_method_matrix` | 方法对比矩阵 |
| 开发 | `execute_python_code` | 代码沙箱执行+绘图 |
| 开发 | `compute_dse_metrics` | 计算 HV/IGD/GD/Spread |
| 开发 | `generate_pareto_front` | Pareto 前沿可视化 |
| 开发 | `design_experiment_template` | 实验设计模板生成 |
| 网络 | `fetch_web_page` | 抓取网页转 Markdown |
| 网络 | `download_paper_pdf` | 下载论文 PDF 到本地 |
| 记忆 | `search_memory` | 搜索对话历史和长期记忆 |
| 记忆 | `save_memory` | 写入长期记忆 |
| 记忆 | `get_recent_conversations` / `get_memory_stats` | 查看记忆 |

### 外部 MCP 工具

**Filesystem**：`read_file` `write_file` `edit_file` `list_directory` `directory_tree` `search_files` `get_file_info` `list_allowed_directories`

**Git**：`git_status` `git_diff` `git_diff_staged` `git_diff_unstaged` `git_log` `git_show` `git_branch` `git_commit` `git_add` `git_checkout` `git_create_branch`

**GitHub**：`search_repositories` `search_code` `get_file_contents` `list_commits` `list_issues` `get_issue` `list_pull_requests` `get_pull_request` `create_issue` `create_pull_request` 等 15+

## 7. Agent 问答（研究者 + 开发者模式）

Agent 入口：

```powershell
python main.py --ask "你的问题"
```

### 模式切换

Planner 自动判断问题意图：
- **研究者模式**：检索论文、对比方法、总结、找相关
- **开发者模式**：计算指标、画图、执行代码、设计实验

```powershell
# 研究者模式
python main.py --ask "GRL-DSE 的核心思想是什么？"

# 开发者模式 — Agent 会调用 compute_dse_metrics
python main.py --ask "帮我计算这组解的 HV：points=[[1,7],[2,5],[3,3],[5,1]], reference_point=[10,10]"
```

### Agent 工作流

```text
Planner（模式判定 + 步骤分解）
  → QueryRewriter（检索步骤改写；非检索步骤直通）
  → Executor（按工具路由：search_dse_papers → batch，其他 → 单次调用）
  → Reasoner（证据评估，可循环）
  → Synthesizer（生成最终答案）
```

### 常用参数

```powershell
# 查看 trace
python main.py --ask "..." --show-trace

# 控制最大迭代轮数
python main.py --ask "..." --max-iterations 3

# 指定 session（记忆隔离）
python main.py --ask "..." --session my-session

# 显示注入的记忆上下文
python main.py --ask "它用了什么优化方法？" --session my-session --show-memory
```

## 8. 多轮对话与长期记忆

双层记忆系统，SQLite 存储：

```text
memory/dse_explorer_memory.sqlite3
  ├─ conversation_turns    — 每轮问答 + 摘要
  └─ long_term_memories    — conclusion / user_focus / paper_summary
```

`main.py` 在启动时自动注入近期对话和长期记忆到 Agent，回答后自动保存。

Agent 也可以**主动调用**记忆工具：

```powershell
# 查历史
python main.py --ask "我们之前讨论过 GRL-DSE 和 BOOM-Explorer 的区别，再总结一下"

# 查记忆统计
python main.py --ask "查看 test session 的记忆统计" --session test
```

## 9. 开发者模式工具

### 计算指标

```python
# Agent 会调用 compute_dse_metrics
"计算 [[1,7],[2,5],[3,3],[5,1]] 的 HV，参考点 [10,10]"
"计算这组解的 IGD, GD, Spread，真实前沿是 [[1,7],[2,5],[3,3],[5,1]]"
```

### Pareto 前沿绘制

```python
"画出这组解的 Pareto 前沿：points=[[1,7],[2,5],[3,3],[5,1],[1.5,6]]"
"对比我的解和真实前沿：found=[[1,7],[2,5],[3,3]], true=[[1,7],[2,5],[3,3],[5,1]]"
```

### 代码执行

```python
"写一个 NSGA-II 的 Python 实现并运行"
"用 matplotlib 画一个 ZDT1 问题的 Pareto 前沿"
```

### 实验设计

```python
"我有 10 个连续设计变量，3 个目标，评估一次要 30 分钟，帮我设计实验流程"
```

### 网页抓取

```python
"抓取 https://arxiv.org/abs/2301.00000 的内容"
"下载 https://arxiv.org/pdf/2301.00000.pdf 到论文库"
```

## 10. 直接调用 LLM

```powershell
python main.py --prompt "请用一句话解释设计空间探索。"
```

## 11. 推荐使用流程

```text
1. main.py --check
2. python -m rag_pipeline.ingest --limit 3
3. python -m rag_pipeline.retriever "test query"
4. python -m rag_pipeline.paper_info --limit 3
5. python -m rag_pipeline.method_matrix --limit 10
6. main.py --ask "GRL-DSE 的核心思想是什么？" --show-trace
7. main.py --ask "帮我算这组解的 DSE 指标..." --session demo
```

## 12. 当前能力

已支持：

- **检索**：Hybrid Search (Dense + BM25 + RRF + Rerank)、Query Rewriter、网络搜索
- **论文**：PDF 入库、元数据抽取、结构化摘要、单篇问答、关系图谱、方法对比矩阵
- **Agent**：研究者/开发者双模式、自动工具路由、迭代检索+推理、带引用答案
- **记忆**：双层记忆系统（对话+长期）、Agent 主动读写
- **开发**：代码沙箱、DSE 指标计算(HV/IGD/GD/Spread)、Pareto 前沿可视化、实验设计模板
- **文件**：文件读写/搜索/编辑（Filesystem MCP，项目目录+论文目录白名单）
- **版本**：Git 操作（Git MCP）
- **远程**：GitHub 仓库/Issue/PR/代码搜索（GitHub MCP）
- **网络**：网页抓取转 Markdown、PDF 下载（SSRF 防护）
- **架构**：5 MCP 服务器、持久会话保活、按需懒初始化
- **安全**：代码沙箱超时隔离、URL 内网地址过滤

暂未支持：

- Web UI
- PDF 图表/公式/表格的结构化解析
- 自动将网络搜索结果写入长期知识库
- RAG 自动评估集

## 13. 常见问题

### PowerShell 中 conda activate 失败

```powershell
conda run --no-capture-output -n dse_explorer python main.py --check
```

### Windows 输出乱码

```powershell
conda run --no-capture-output -n dse_explorer python ...
```

### PDF 解析失败

少量 PDF 可能损坏或非文本型 PDF。入库流程会跳过失败文件继续处理。扫描版 PDF 需要额外 OCR。

### 方法矩阵为空

先运行论文信息抽取：

```powershell
python -m rag_pipeline.paper_info --limit 10
python -m rag_pipeline.method_matrix --limit 10
```

### GitHub MCP 报错

在 `.env` 中设置 `GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxx`，否则 GitHub 工具会返回错误但不影响其他功能。

### Node.js 未安装导致 Filesystem/GitHub MCP 不可用

安装 Node.js 后运行：

```powershell
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-github
```
