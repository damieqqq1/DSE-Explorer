"""Project entry point for the DSE explorer agent."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r"The default value of `allowed_objects`.*")

from utils.config import get_settings
from utils.context_engine import build_context_bundle
from utils.deepseek_llm import invoke_deepseek
from utils.logger import configure_logging, get_logger
from utils.memory import MemoryStore


logger = get_logger(__name__)


def count_pdfs(paper_root: Path) -> int:
    if not paper_root.exists():
        return 0
    return sum(1 for _ in paper_root.rglob("*.pdf"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DSE explorer project entry point.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate local configuration without calling external APIs.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print a masked configuration summary.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Send a one-shot prompt to the configured DeepSeek-compatible LLM.",
    )
    parser.add_argument(
        "--ask",
        type=str,
        default="",
        help="Ask the Planner + RAG Executor + Synthesizer agent a paper-grounded question.",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Show plan and retrieval trace when used with --ask.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum retrieval/reasoning iterations when used with --ask.",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="default",
        help="Conversation memory session id when used with --ask.",
    )
    parser.add_argument(
        "--memory-turns",
        type=int,
        default=3,
        help="Number of recent conversation turns to inject as context.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable conversation and long-term memory for this run.",
    )
    parser.add_argument(
        "--show-memory",
        action="store_true",
        help="Show injected conversation and long-term memory context.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show selected context candidates, scores, and node-routed context.",
    )
    parser.add_argument(
        "--no-context-embeddings",
        action="store_true",
        help="Use lexical context relevance only instead of embedding similarity.",
    )
    parser.add_argument(
        "--extract-paper-info",
        action="store_true",
        help="Extract structured metadata and method information from local PDFs.",
    )
    parser.add_argument(
        "--paper-info-limit",
        type=int,
        default=3,
        help="Number of PDFs to process with --extract-paper-info.",
    )
    parser.add_argument(
        "--paper-info-all",
        action="store_true",
        help="Process all PDFs with --extract-paper-info.",
    )
    parser.add_argument(
        "--paper-info-force",
        action="store_true",
        help="Re-extract paper info records that already exist.",
    )
    parser.add_argument(
        "--paper-info-dry-run",
        action="store_true",
        help="Print extracted paper info without writing SQLite.",
    )
    parser.add_argument(
        "--paper-info-list",
        action="store_true",
        help="List recently extracted paper info records.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    if args.show_config:
        print(json.dumps(settings.masked_summary(), indent=2, ensure_ascii=False))

    if args.check:
        settings.validate_for_papers()
        logger.info("Configuration loaded from %s", settings.env_path)
        logger.info("Paper root: %s", settings.paper_root)
        logger.info("PDF files found: %s", count_pdfs(settings.paper_root))
        logger.info("LLM model: %s", settings.llm.model_id)
        logger.info("Qdrant collection: %s", settings.qdrant.collection)
        return

    if args.prompt:
        logger.info("Invoking LLM model: %s", settings.llm.model_id)
        answer = invoke_deepseek(args.prompt, settings=settings)
        print(answer)
        return

    if args.extract_paper_info or args.paper_info_list:
        from rag_pipeline.paper_info import PaperInfoStore, extract_papers

        if args.paper_info_list:
            store = PaperInfoStore()
            print(json.dumps(store.list_recent(limit=args.paper_info_limit), indent=2, ensure_ascii=False))
            return

        limit = None if args.paper_info_all else args.paper_info_limit
        extract_papers(
            settings=settings,
            limit=limit,
            max_pages=4,
            max_chars=16000,
            dry_run=args.paper_info_dry_run,
            force=args.paper_info_force,
        )
        return

    if args.ask:
        logger.info("Running paper-grounded agent.")
        memory_store = None if args.no_memory else MemoryStore()
        context_bundle = None
        if memory_store is not None:
            context_bundle = build_context_bundle(
                memory_store,
                session_id=args.session,
                query=args.ask,
                recent_turns=args.memory_turns,
                use_embeddings=not args.no_context_embeddings,
            )
            if args.show_memory:
                print("Conversation Memory:")
                print(context_bundle.conversation_context or "(empty)")
                print("\nLong-Term Memory:")
                print(context_bundle.long_term_context or "(empty)")
                print()
            if args.show_context:
                print("Context Trace:")
                print(json.dumps(context_bundle.context_trace, indent=2, ensure_ascii=False))
                print("\nPlanner Context:")
                print(context_bundle.planner_context or "(empty)")
                print("\nQuery Rewriter Context:")
                print(context_bundle.rewriter_context or "(empty)")
                print("\nReasoner Context:")
                print(context_bundle.reasoner_context or "(empty)")
                print("\nSynthesis Context:")
                print(context_bundle.synthesis_context or "(empty)")
                print()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*allowed_objects.*")
            from agent.graph import run_agent

        result = run_agent(
            args.ask,
            max_iterations=args.max_iterations,
            conversation_context=context_bundle.conversation_context if context_bundle else "",
            long_term_context=context_bundle.long_term_context if context_bundle else "",
            planner_context=context_bundle.planner_context if context_bundle else "",
            rewriter_context=context_bundle.rewriter_context if context_bundle else "",
            reasoner_context=context_bundle.reasoner_context if context_bundle else "",
            synthesis_context=context_bundle.synthesis_context if context_bundle else "",
            context_trace=context_bundle.context_trace if context_bundle else [],
        )
        print(result.get("final_answer", ""))

        sources = result.get("sources", [])
        if sources:
            print("\nSources:")
            for source in sources:
                print(
                    f"- {source['source']} page {source['page']} "
                    f"(score={source['score']:.4f})"
                )

        if memory_store is not None:
            memory_store.save_turn(
                session_id=args.session,
                question=args.ask,
                answer=result.get("final_answer", ""),
                sources=sources,
            )
            logger.info("Saved conversation and long-term memories for session: %s", args.session)

        if args.show_trace:
            print("\nPlan:")
            print(json.dumps(result.get("plan", []), indent=2, ensure_ascii=False))
            print("\nInjected Conversation Memory:")
            print(result.get("conversation_context", ""))
            print("\nInjected Long-Term Memory:")
            print(result.get("long_term_context", ""))
            print("\nContext Trace:")
            print(json.dumps(result.get("context_trace", []), indent=2, ensure_ascii=False))
            print("\nReasoner:")
            print(result.get("reasoner_note", ""))
            print("\nCompressed Context:")
            print(result.get("compressed_context", ""))
            print("\nEvidence:")
            print(json.dumps(result.get("evidence", []), indent=2, ensure_ascii=False))
        return

    logger.info("dse-explorer is ready. Run with --check, --prompt, or --ask.")


if __name__ == "__main__":
    main()
