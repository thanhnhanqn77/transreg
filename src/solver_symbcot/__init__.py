"""
Symbolic Chain-of-Thought (SymbCoT) Reasoning Package.

This package houses the 3-stage symbolic reasoning prompt templates:
1. Plan Generation (plan_generation.txt)
2. Symbolic Deduction Execution (solver.txt)
3. Formal Verification & Label Emitting (verification.txt)

The Python orchestration engine is implemented in `src/symbcot_solver.py`.
"""
from src.symbcot_solver import (
    OursSymbCoTSolver,
    load_templates,
    format_symbcot_prompt,
)

__all__ = ["OursSymbCoTSolver", "load_templates", "format_symbcot_prompt"]
