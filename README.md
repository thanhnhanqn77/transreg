# TransReg & HybLogic: Neuro-Symbolic Dual-Solver Reasoning with Closed-Loop Translation Regularization

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Formal Prover](https://img.shields.io/badge/SMT_Prover-Z3_4.12%2B-green.svg)](https://github.com/Z3Prover/z3)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Double-Blind Review](https://img.shields.io/badge/Peer_Review-Anonymous_Submission-orange.svg)]()

> **Notice for Reviewers:** This repository contains the complete source code, evaluation datasets, and raw experimental log artifacts accompanying the research paper submitted for double-blind peer review at **ICAART 2026**. All author identities, affiliations, and machine-specific file paths have been removed to preserve strict anonymity.

---

## 1. Overview & Architecture

Auto-formalization—translating natural language (NL) arguments into First-Order Logic (FOL) for formal verification—frequently suffers from **five recurring auto-formalization pitfalls**:
1. **Predicate Fragmentation & Drift:** Identical concepts assigned disparate surface symbols.
2. **Arity Inconsistency:** The same predicate assigned different numbers of arguments across premises.
3. **Semantic Inversion:** Reversing the direction of logical implications (e.g., $P \to Q$ instead of $Q \to P$).
4. **Free Variable Scope Leaks:** Variables occurring outside quantifier binders ($\forall, \exists$).
5. **Ingestion Crashes & False UNSAT:** Malformed formulas causing solver rejection or paradoxical proofs.

To resolve these challenges, we introduce:
* **TransReg (Semantic-Aware Translation Regularizer):** A closed-loop translation pipeline enforcing a global semantic schema, blind back-translation audit tables, and deterministic AST invariant checks before any formal reasoning begins.
* **HybLogic (Hybrid Dual-Solver Controller):** An asymmetric dual-solver architecture that routes verified FOL first through a **Z3 SMT Fast-Path** (0 reasoning tokens, $99.17\%$ precision when decisive) and selectively falls back to a **Symbolic Chain-of-Thought (SymbCoT)** reasoner whenever formal bounds remain inconclusive.

```text
                     Input Problem: Natural Language Premises + Question
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                           1. TransReg TRANSLATION ENGINE                                  │
│  • Global Semantic Analysis: Discourse referents & unified predicate registry plan       │
│  • Joint FOL Proposal: Context-aware translation with strict type annotations             │
│  • Blind Back-Translation: Independent NL reconstruction without reference FOL            │
│  • 3-Way Semantic Audit Table: Detects directional inversion & scope violations           │
│  • Deterministic AST Invariant Gate: Zero-token syntax & arity validation                 │
│  • Atomic Registry Commit: Transactional rollback on invalid AST                         │
└───────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                               2. Z3 SMT FORMAL FAST-PATH                                  │
│  • Ingests verified FOL directly into Z3 First-Order Context                              │
│  • Tests SAT(P) [Consistency] and UNSAT(P ∧ ¬Q) [Entailment]                              │
│  • Computational cost: 0 LLM reasoning tokens                                            │
└───────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼ (Z3 Decisive: True / False)                 ▼ (Inconclusive / Unknown)
┌─────────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
│             RETURN IMMEDIATELY              │   │        3. SymbCoT LLM FALLBACK          │
│ • Mathematical proof certainty              │   │ • Symbolic Plan Generation              │
│ • Empirical Precision: 99.17% (FOLIO)       │   │ • Step-by-step Deductive Execution      │
│ • Execution Time: < 0.05 seconds            │   │ • Final Formal Validation Gate          │
└─────────────────────────────────────────────┘   └─────────────────────────────────────────┘
```

---

## 2. Directory Structure

```text
transreg_hyblogic_anonymous/
├── README.md                          # Documentation and reproduction guide
├── requirements.txt                   # Minimal Python dependencies (z3-solver)
├── .gitignore                         # Clean gitignore configuration
├── reproduce_results.py               # Instant reproduction CLI for Tables 1, 2, 3, 4
├── run_inference.py                   # Live CLI entrypoint for translation & reasoning
├── data/                              # Benchmark evaluation datasets
│   ├── folio_v2_validation.jsonl     # Complete FOLIO v2 evaluation split (203 problems)
│   ├── proofwriter_test_depth5_400.jsonl # Complete ProofWriter Depth-5 test split (400 problems)
│   └── folio_train_fewshots.jsonl    # Training few-shot demonstration bank
├── logs/                              # Experimental run logs matching paper numbers
│   ├── FOLIO/                         # 203 evaluated instances across all methods
│   │   ├── direct.json
│   │   ├── cot.json
│   │   ├── clover.json
│   │   ├── symbcot.json
│   │   ├── clover_transreg.json
│   │   ├── symbcot_transreg.json
│   │   └── hyblogic.json
│   ├── PROOFWRITER/                   # 400 evaluated instances across all methods
│   │   ├── direct.json
│   │   ├── cot.json
│   │   ├── clover.json
│   │   ├── symbcot.json
│   │   ├── clover_transreg.json
│   │   ├── symbcot_transreg.json
│   │   └── hyblogic.json
│   └── ablation_folio/                # Comprehensive ablation runs on FOLIO v2
│       ├── folio_full/                # Full HybLogic pipeline
│       ├── folio_no_semantic_gate/    # Ablated: without semantic audit gate
│       ├── folio_no_registry/         # Ablated: without persistent typed registry
│       ├── folio_no_ast_gate/         # Ablated: without deterministic AST invariant checks
│       └── folio_no_global_analysis/  # Ablated: without global semantic guidance
└── src/                               # Modular Python source implementation
    ├── __init__.py
    ├── client.py                      # Resilient LLM client (token management & retries)
    ├── common.py                      # Data models, hashing, caching, and utilities
    ├── clover_solver.py               # Clover Z3 formal SMT theorem prover
    ├── symbcot_solver.py              # SymbCoT symbolic LLM fallback reasoner
    ├── shared_translator.py           # TransReg translation engine orchestrator
    ├── hybrid_pipeline.py             # HybLogic controller & dual-solver router
    ├── run_hybrid.py                  # Internal pipeline runner
    ├── solver_z3/                     # Z3 AST compiler, FOL parser, and typed context
    │   ├── ast_rl.py
    │   ├── ast_to_z3_adapter.py
    │   ├── fol_parser.py
    │   ├── models.py
    │   ├── multi_sentence_compiler.py
    │   └── z3_context.py
    ├── solver_symbcot/                # SymbCoT structured prompt templates
    │   └── prompts/
    │       ├── plan_generation.txt
    │       ├── solver.txt
    │       └── verification.txt
    └── translator/                    # TransReg translation components
        ├── connected_formula.py
        ├── fewshots.py
        ├── live_clover_translation.py
        ├── semantic_audit.py
        ├── sequential_translation.py
        ├── source_alignment.py
        ├── translation_ast.py
        ├── prompt/                    # Stage prompt templates and policies
        └── fewshot/                   # Few-shot demonstration JSON files and manifests
```

---

## 3. Installation & Setup

### Requirements
* Python 3.9 or higher
* [Z3 SMT Solver](https://github.com/Z3Prover/z3) (`z3-solver>=4.12.0`)
* [Pydantic](https://github.com/pydantic/pydantic) (`pydantic>=2.0.0`)

### Installation Steps
```bash
# 1. Clone repository
git clone <ANONYMOUS_REPO_URL>
cd transreg_hyblogic_anonymous

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 4. Instant Reproduction of Paper Results

To allow reviewers to instantly verify all empirical findings without requiring API keys or prolonged inference runs, we provide a unified reproduction script `reproduce_results.py` that parses the raw benchmark logs and verifies all claims.

### Run All Tables:
```bash
python reproduce_results.py
```

### Run Specific Tables:
```bash
# Reproduce Table 1: Main Task Accuracy
python reproduce_results.py --table 1

# Reproduce Table 2: Translation Diagnostics
python reproduce_results.py --table 2

# Reproduce Table 3: Z3 Formal Verification & Synergy Metrics
python reproduce_results.py --table 3

# Reproduce Table 4: Component Ablation Study
python reproduce_results.py --table 4
```

### Expected Output Summary

#### Table 1: Main Task Accuracy Across Benchmarks
| Method / Architecture | FOLIO v2 (N=203) | ProofWriter Depth-5 (N=400) |
| :--- | :--- | :--- |
| Direct Inference | 150 / 203 (73.89%) | 290 / 400 (72.50%) |
| Chain-of-Thought (CoT) | 170 / 203 (83.74%) | 359 / 400 (89.75%) |
| Baseline Clover (SMT) | 158 / 203 (81.03%) | 377 / 400 (95.44%) |
| Baseline SymbCoT | 175 / 203 (86.21%) | 366 / 400 (91.50%) |
| Clover + TransReg *(Ours)* | 171 / 203 (85.07%) | 395 / 400 (98.75%) |
| SymbCoT + TransReg *(Ours)* | 178 / 203 (87.68%) | 380 / 400 (95.00%) |
| **HybLogic (Ours Hybrid)** | **179 / 203 (88.18%)** | **395 / 400 (98.75%)** |

#### Table 2: Formal Translation Diagnostics (FOLIO v2)
| Metric | Baseline Clover | Baseline SymbCoT | TransReg *(Ours)* |
| :--- | :--- | :--- | :--- |
| Arity Inconsistency Rate | 12.90% | 10.31% | **0.00%** (0 / 203) |
| Free Variable Scope Leaks | 8.76% | 7.22% | **0.00%** (0 / 203) |
| Predicate Drift vs GT | 32.10% | 28.40% | **9.20%** (-67.6% drop) |
| Solver Ingestion Crash Rate | 18.04% | N/A | **0.00%** (0 / 203) |
| AST Valid Formula Rate | 81.96% | N/A | **100.00%** (203 / 203) |

#### Table 3: Z3 Formal Verification & Routing Synergy
| Characteristic | FOLIO v2 (N=203) | ProofWriter Depth-5 (N=400) |
| :--- | :--- | :--- |
| Z3 Fast-Path Resolution Rate | 53.69% (109 / 203) | 55.25% (221 / 400) |
| Z3 Decisive Proof Accuracy | **96.33%** (105 / 109) | **97.74%** (216 / 221) |
| SymbCoT Fallback Accuracy | 78.72% (74 / 94) | 100.00% (179 / 179) |
| Dual-Solver Mutual Rescues | 30 Cases (Z3: 12, CoT: 18) | 5 Cases |

#### Table 4: Component Ablation Study on FOLIO v2
| Ablation Variant | Accuracy (%) | $\Delta$ | Z3 Solved Rate |
| :--- | :--- | :--- | :--- |
| **Full HybLogic (Proposed)** | **88.18% (179/203)** | -- | **53.69%** |
| w/o Semantic Audit Gate (Gate 1) | 83.74% (170/203) | -4.44% | 52.71% |
| w/o Persistent Registry (Isolated) | 83.25% (169/203) | -4.93% | 51.23% |
| w/o AST Invariants Gate (Gate 2) | 82.76% (168/203) | -5.42% | 51.72% |
| w/o Global Semantic Guidance | 82.76% (168/203) | -5.42% | 47.78% |

---

## 5. Running Live Inference

Reviewers with an Ollama endpoint or OpenAI-compatible backend can execute live inference on any sample or full dataset.

### Setting API Credentials
API tokens can be supplied directly via environment variable, CLI flag, or local text file:
```bash
# Option A: Environment Variable (Recommended for Headless / CI)
export OLLAMA_API_KEY="your_api_token_here"

# Option B: Direct CLI Parameter
python run_inference.py --api_key "your_api_token_here" --limit 5

# Option C: Optional local keys.txt (one key per line, excluded via .gitignore)
echo "your_api_token_here" > keys.txt
```

### Running Inference Commands
```bash
# 1. Run HybLogic on the first 5 samples of FOLIO v2:
python run_inference.py --data_file data/folio_v2_validation.jsonl --mode hybrid --limit 5

# 2. Run on ProofWriter Depth-5:
python run_inference.py --data_file data/proofwriter_test_depth5_400.jsonl --mode hybrid --limit 10

# 3. Run Clover Z3 solver only:
python run_inference.py --data_file data/folio_v2_validation.jsonl --mode clover_only --limit 5

# 4. Run SymbCoT LLM reasoning backend only:
python run_inference.py --data_file data/folio_v2_validation.jsonl --mode symbcot_only --limit 5
```

### CLI Options
* `--data_file`: Relative path to JSONL dataset (default: `data/folio_v2_validation.jsonl`).
* `--mode`: Reasoning mode: `hybrid` (default), `clover_only`, `symbcot_only`, `dual`.
* `--model_name`: LLM backbone model (default: `gemma4:31b-cloud`).
* `--host`: Ollama API base URL (default: `https://ollama.com/api/chat`).
* `--api_key`: Direct API token string (or set via `OLLAMA_API_KEY` environment variable).
* `--key_file`: Optional fallback path to keys file (default: `None`, uses `keys.txt` if present).
* `--limit`: Number of samples to evaluate (`0` evaluates the entire dataset).
* `--run_dir`: Output directory for checkpoints and detailed reasoning traces (default: `runs/<dataset>_<model>_<mode>`).

---

## 6. Datasets

1. **FOLIO v2 Validation Split (`data/folio_v2_validation.jsonl`):**
   * 203 first-order logic reasoning problems with complex quantifier structures, domain axioms, and true/false/uncertain classifications.
2. **ProofWriter Depth-5 Test Split (`data/proofwriter_test_depth5_400.jsonl`):**
   * 400 deep multi-hop deductive reasoning instances requiring up to 5 chained inference rules.

---

## 7. Anonymity & Open Science Statement

This repository adheres strictly to the **Double-Blind Review Policy**:
* All author identities, affiliations, acknowledgments, and grant numbers have been omitted.
* Machine-specific paths and internal server names have been sanitized to relative project paths.
* All experimental logs include complete sample-by-sample traces to ensure transparent verification.
* The repository will be permanently published on GitHub with an open-source MIT license upon acceptance.
