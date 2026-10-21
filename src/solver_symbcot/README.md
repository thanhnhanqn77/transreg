# SymbCoT Reasoning Module (Symbolic Chain-of-Thought)

Unlike deterministic SMT solvers like Z3 (which require formal AST parsers, compiler contexts, and theory solvers as implemented in `src/solver_z3/`), **SymbCoT** is an **LLM-based neuro-symbolic reasoning framework**.

Instead of executing theorem proving in C/C++, SymbCoT regularizes the LLM's internal reasoning by constraining it to execute three sequential formal deductive phases:

```text
TransReg Verified FOL
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Plan Generation (prompts/plan_generation.txt)            │
│    • LLM outlines deductive path from premises to question  │
└─────────────────────────────────────────────────────────────┘
          │  [[PLAN]]
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Deduction Execution (prompts/solver.txt)                 │
│    • LLM executes forward & backward inference steps        │
│    • Applies Modus Ponens, Universal Instantiation, etc.    │
└─────────────────────────────────────────────────────────────┘
          │  [[EXECUTION]]
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Formal Verification (prompts/verification.txt)           │
│    • LLM audits deductive validity & logical soundness      │
│    • Emits final answer: {True} / {False} / {Uncertain}     │
└─────────────────────────────────────────────────────────────┘
```

## Module Organization
* **`prompts/`:** Houses the canonical prompt contracts for the 3 stages:
  * `plan_generation.txt`: Prompt template for Step 1.
  * `solver.txt`: Prompt template for Step 2.
  * `verification.txt`: Prompt template for Step 3.
* **`src/symbcot_solver.py`:** The Python controller (`OursSymbCoTSolver`) that manages multi-turn communication with the LLM, injects intermediate outputs (`[[PLAN]]`, `[[EXECUTION]]`), parses final predictions, and tracks latency/telemetry.
