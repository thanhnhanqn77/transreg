"""
CLI Runner for TransReg (Formal Translation) and HybLogic (Hybrid Neuro-Symbolic Reasoning).
Anonymous submission artifact for double-blind review.

Usage:
    # 1. Run HybLogic on 5 samples of FOLIO v2:
    python run_inference.py --data_file data/folio_v2_validation.jsonl --mode hybrid --limit 5

    # 2. Run on ProofWriter Depth-5:
    python run_inference.py --data_file data/proofwriter_test_depth5_400.jsonl --mode hybrid --limit 10

    # 3. Run Clover Z3 formal SMT solver only (using pre-translated or newly translated FOL):
    python run_inference.py --data_file data/folio_v2_validation.jsonl --mode clover_only --limit 5

    # 4. Run SymbCoT fallback LLM reasoning only:
    python run_inference.py --data_file data/folio_v2_validation.jsonl --mode symbcot_only --limit 5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add package root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.common import LABELS, digest, load_dataset, read_json, write_json
from src.hybrid_pipeline import OursHybridPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TransReg & HybLogic Dual-Solver Reasoning Pipeline (Z3 Fast-Path + SymbCoT Fallback)"
    )
    parser.add_argument(
        "--data_file",
        default="data/folio_v2_validation.jsonl",
        help="Path to JSONL dataset (e.g., data/folio_v2_validation.jsonl or data/proofwriter_test_depth5_400.jsonl)",
    )
    parser.add_argument(
        "--model_name",
        default="gemma4:31b-cloud",
        help="LLM model identifier for Ollama endpoint",
    )
    parser.add_argument(
        "--host",
        default="https://ollama.com/api/chat",
        help="Ollama API base endpoint",
    )
    parser.add_argument(
        "--api_key",
        default=os.getenv("OLLAMA_API_KEY", ""),
        help="Ollama API token string (recommended; or set via OLLAMA_API_KEY env var)",
    )
    parser.add_argument(
        "--key_file",
        default=None,
        help="Optional path to API keys file (default: None, falls back to keys.txt if present)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=12288,
        help="Max completion tokens per request",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "dual", "clover_only", "symbcot_only"),
        default="hybrid",
        help="Execution mode: hybrid (Z3 fast-path with SymbCoT fallback), dual (both), or single solver",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Output directory for checkpoints and detailed traces (default: runs/<dataset>_<model>_<mode>)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of samples to process (0 = all)",
    )
    parser.add_argument(
        "--retry_errors",
        action="store_true",
        help="Retry previously failed samples in existing checkpoint",
    )
    return parser.parse_args()


def compute_summary(results):
    total = len(results)
    completed = [r for r in results if r.get("status") == "complete"]
    evaluated = [r for r in completed if r.get("correct") is not None]
    correct = sum(1 for r in evaluated if r.get("correct"))

    clover_z3_count = sum(1 for r in completed if r.get("solver_path") == "clover_z3")
    symbcot_fallback_count = sum(1 for r in completed if r.get("solver_path") == "symbcot_fallback")

    acc = (correct / len(evaluated) * 100) if evaluated else 0.0

    matrix = {g: {p: 0 for p in LABELS} for g in LABELS}
    for r in evaluated:
        g = r.get("gold_label")
        p = r.get("predicted_label")
        if g in matrix and p in matrix[g]:
            matrix[g][p] += 1

    return {
        "total_samples": total,
        "completed": len(completed),
        "errors": total - len(completed),
        "evaluated": len(evaluated),
        "correct": correct,
        "accuracy_percent": round(acc, 2),
        "solver_breakdown": {
            "clover_z3_decisive": clover_z3_count,
            "symbcot_fallback": symbcot_fallback_count,
            "z3_resolution_rate_percent": round((clover_z3_count / len(completed) * 100), 2) if completed else 0.0,
        },
        "confusion_matrix": matrix,
    }


def main():
    args = parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"Error: dataset file not found: {data_path}")
        sys.exit(1)

    print(f"Loading dataset from: {data_path}")
    rows = load_dataset(data_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} samples for evaluation.")

    # Output directory
    # Output directory (relative path, avoiding local machine roots)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        dataset_stem = data_path.stem
        model_slug = args.model_name.replace(":", "_").replace("-", "_")
        run_dir = Path("runs") / f"{dataset_stem}_{model_slug}_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints and logs directory: {run_dir}")

    config = {
        "model_name": args.model_name,
        "host": args.host,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "mode": args.mode,
    }
    if args.api_key:
        config["api_key"] = args.api_key
    elif args.key_file:
        config["key_file"] = args.key_file

    # Save sanitized config (no secrets, no local machine paths)
    saved_config = dict(config)
    if "api_key" in saved_config:
        del saved_config["api_key"]
    if "key_file" in saved_config:
        del saved_config["key_file"]
    saved_config["auth"] = "environment_variable" if (args.api_key or os.getenv("OLLAMA_API_KEY")) else "keys_file"
    write_json(run_dir / "config.json", saved_config)

    # Initialize Pipeline
    print(f"\nInitializing OursHybridPipeline (Mode: {args.mode.upper()})...")
    try:
        pipeline = OursHybridPipeline(config)
    except Exception as e:
        print(f"\n[Note] Initialization exception: {e}")
        print("Make sure you provide an Ollama API key via --api_key, OLLAMA_API_KEY environment variable, or keys.txt.")
        sys.exit(1)

    # Resume checkpoint if existing
    results_path = run_dir / "results.json"
    results_map = {}
    if results_path.exists():
        try:
            old_data = read_json(results_path)
            results_map = {r["index"]: r for r in old_data.get("results", [])}
        except Exception:
            pass

    print("\n" + "=" * 75)
    print(f"STARTING EVALUATION | MODE: {args.mode.upper()} | TOTAL: {len(rows)}")
    print("=" * 75)

    for i, row in enumerate(rows):
        idx = row["index"]
        if idx in results_map and results_map[idx].get("status") == "complete" and not args.retry_errors:
            print(f"[{i+1}/{len(rows)}] Sample #{idx}: already cached as completed. Skipping.")
            continue

        started = time.monotonic()
        try:
            res = pipeline.process_sample(row, run_dir=run_dir, mode=args.mode, retry_errors=args.retry_errors)
            results_map[idx] = res
            elapsed = time.monotonic() - started

            path_str = res.get("solver_path", "unknown")
            pred = res.get("predicted_label", "N/A")
            gold = res.get("gold_label", "N/A")
            is_corr = "CORRECT" if res.get("correct") else ("WRONG" if res.get("status") == "complete" else "ERROR")

            print(f"[{i+1}/{len(rows)}] Sample #{idx} | Path: {path_str:<16} | Pred: {pred:<9} | Gold: {gold:<9} | [{is_corr}] | {elapsed:.2f}s")
        except Exception as exc:
            print(f"[{i+1}/{len(rows)}] Sample #{idx} | ERROR: {exc}")
            results_map[idx] = {
                "index": idx,
                "id": row["example_id"],
                "gold_label": row.get("label"),
                "status": "error",
                "error": str(exc),
            }

        # Checkpoint every 5 samples or at the end
        if (i + 1) % 5 == 0 or (i + 1) == len(rows):
            ordered_results = [results_map[r["index"]] for r in rows if r["index"] in results_map]
            summary_data = compute_summary(ordered_results)
            write_json(results_path, {"config": config, "results": ordered_results, "summary": summary_data})
            write_json(run_dir / "summary.json", summary_data)

    ordered_results = [results_map[r["index"]] for r in rows if r["index"] in results_map]
    final_summary = compute_summary(ordered_results)
    write_json(results_path, {"config": config, "results": ordered_results, "summary": final_summary})
    write_json(run_dir / "summary.json", final_summary)

    print("\n" + "=" * 75)
    print("FINAL SUMMARY REPORT")
    print("=" * 75)
    print(f"Total Evaluated: {final_summary['evaluated']} / {final_summary['total_samples']}")
    print(f"Accuracy:        {final_summary['accuracy_percent']}% ({final_summary['correct']}/{final_summary['evaluated']})")
    print(f"Z3 Decisive:     {final_summary['solver_breakdown']['clover_z3_decisive']} ({final_summary['solver_breakdown']['z3_resolution_rate_percent']}%)")
    print(f"SymbCoT Fallback:{final_summary['solver_breakdown']['symbcot_fallback']}")
    print(f"Detailed logs saved to: {results_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
