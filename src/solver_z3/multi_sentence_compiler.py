import z3
from typing import List, Tuple, Dict, Any
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Add parent directory to sys.path to allow imports from parser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .fol_parser import parse_fol
    from .ast_to_z3_adapter import convert_to_relational_logic
    from .ast_rl import SymbolTable, RelationalLogic
    from .z3_context import Z3Context
except (ImportError, ValueError):
    try:
        from fol_parser import parse_fol
        from ast_to_z3_adapter import convert_to_relational_logic
        from ast_rl import SymbolTable, RelationalLogic
        from z3_context import Z3Context
    except (ImportError, ValueError):
        from parser.fol_parser import parse_fol
        from parser.ast_to_z3_adapter import convert_to_relational_logic
        from parser.ast_rl import SymbolTable, RelationalLogic
        from solver.z3_context import Z3Context

# ---------------------------------------------------------------------------
# Parse cache
# ---------------------------------------------------------------------------
# parse_fol + convert_to_relational_logic are pure functions of the input string
# and the resulting RelationalLogic objects are never mutated by compilation
# (to_z3_expr / collect_symbols only read the AST). Caching the parsed result
# avoids re-parsing the same premise once per MCQ option / per question, which is
# the dominant repeated cost when a single premise set is checked against many
# conclusions.
_PARSE_CACHE: Dict[str, RelationalLogic] = {}


def clear_parse_cache():
    """Clears the global FOL parse cache."""
    _PARSE_CACHE.clear()


def _parse_cached(fol_string: str) -> RelationalLogic:
    """Returns a cached RelationalLogic for `fol_string`, parsing on first use."""
    cached = _PARSE_CACHE.get(fol_string)
    if cached is not None:
        return cached
    ast = parse_fol(fol_string)
    if not ast:
        raise ValueError(f"Parser returned empty AST for: {fol_string}")
    rl = convert_to_relational_logic(ast, fol_string)
    _PARSE_CACHE[fol_string] = rl
    return rl


class MultiSentenceCompiler:
    """
    Parses multiple FOL sentences, merges symbol tables, and instantiates shared Z3 objects.
    Ensures symbol consistency and type safety across premises and questions.
    """
    def __init__(self, ctx: Z3Context, timeout_ms: int = 10000, strict: bool = False):
        self.ctx = ctx
        self.global_st = SymbolTable()
        self.sentences: List[RelationalLogic] = []
        self.timeout_ms = timeout_ms
        self.strict = strict
        self._compiled_exprs: List[z3.BoolRef] | None = None
        self._compiled_groups: List[List[z3.BoolRef]] | None = None
        self._base_solver: z3.Solver | None = None

    def _invalidate_compilation(self) -> None:
        self._compiled_exprs = None
        self._compiled_groups = None
        self._base_solver = None

    def add_sentence(self, fol_string: str):
        """Parses a single FOL sentence and merges its symbols into the global symbol table."""
        try:
            rl = _parse_cached(fol_string)

            # Merge symbol table signatures
            local_st = rl.get_symbol_table()
            for name, sig in local_st.signatures.items():
                if name in self.global_st.signatures:
                    self._resolve_conflict(name, sig)
                else:
                    self.global_st.signatures[name] = sig

            self.sentences.append(rl)
            self._invalidate_compilation()
        except Exception as e:
            if self.strict:
                raise
            logger.debug("Skipping malformed sentence '%s': %s", fol_string, e)
            # Do not re-raise, allow other sentences to proceed

    def _resolve_conflict(self, name: str, new_sig: tuple):
        """Resolves symbol conflicts (arity or sort mismatch) between different sentences."""
        existing_sig = self.global_st.signatures[name]
        if len(existing_sig) != len(new_sig):
            # Arity conflict: objective English description
            raise ValueError(
                f"Arity conflict for symbol '{name}': "
                f"previously used with {len(existing_sig)-1} args, "
                f"now used with {len(new_sig)-1} args."
            )
        
        # Unify sorts using global precedence (promoting sorts if necessary)
        unified = []
        for s1, s2 in zip(existing_sig, new_sig):
            # If one is Entity and the other is specific (like IntSort()), promote to specific.
            if s1 == "Entity" and s2 != "Entity":
                unified.append(s2)
            elif s2 == "Entity" and s1 != "Entity":
                unified.append(s1)
            elif self.global_st.get_prec(s1) >= self.global_st.get_prec(s2):
                unified.append(s1)
            else:
                unified.append(s2)
        self.global_st.signatures[name] = tuple(unified)

    def compile(self) -> List[z3.BoolRef]:
        """Compile premises once and cache both flat and per-premise expressions."""
        if self._compiled_exprs is not None:
            return self._compiled_exprs

        for name, sig in self.global_st.signatures.items():
            if len(sig) == 1:
                self.ctx.get_or_create_const(name, sig[0])
            else:
                self.ctx.get_or_create_func(name, list(sig[:-1]), sig[-1])

        groups: List[List[z3.BoolRef]] = []
        flattened: List[z3.BoolRef] = []
        for rl in self.sentences:
            exprs = list(rl.to_z3_expr(self.ctx))
            if self.strict and not exprs:
                raise ValueError("A premise compiled to no Z3 expressions")
            groups.append(exprs)
            flattened.extend(exprs)

        self._compiled_groups = groups
        self._compiled_exprs = flattened
        return flattened

    def _get_base_solver(self) -> z3.Solver:
        if self._base_solver is None:
            solver = z3.Solver()
            solver.set("timeout", self.timeout_ms)
            solver.add(self.compile())
            self._base_solver = solver
        return self._base_solver

    # -----------------------------------------------------------------------
    # ReasoningEngine Compatibility (v2.5+)
    # -----------------------------------------------------------------------

    def add_premise(self, fol: str, premise_id: Any):
        """Parses and adds a premise sentence."""
        self.add_sentence(fol)
        from .models import PremiseResult
        return PremiseResult(
            index=int(premise_id)-1 if isinstance(premise_id, int) else 0,
            original_fol=fol,
            status="SAT"
        )

    def compile_question(self, fol: str, label: str):
        """Parses and prepares a question/option for entailment check."""
        rl = _parse_cached(fol)
        
        # Merge symbols to ensure consistency
        local_st = rl.get_symbol_table()
        for name, sig in local_st.signatures.items():
            if name in self.global_st.signatures:
                self._resolve_conflict(name, sig)
            else:
                self.global_st.signatures[name] = sig
                
            # Register in Z3 context immediately
            if len(sig) == 1:
                self.ctx.get_or_create_const(name, sig[0])
            else:
                self.ctx.get_or_create_func(name, list(sig[:-1]), sig[-1])
        
        self._invalidate_compilation()
        exprs = rl.to_z3_expr(self.ctx)
        from .models import QuestionResult
        return QuestionResult(
            label=label,
            fol=fol,
            z3_expr=exprs[0] if exprs else None
        )

    def check_consistency(self) -> bool:
        return self._get_base_solver().check() != z3.unsat

    def check_entailment(self, question_expr: z3.BoolRef) -> bool:
        if question_expr is None:
            return False
        solver = self._get_base_solver()
        solver.push()
        try:
            solver.add(z3.Not(question_expr))
            return solver.check() == z3.unsat
        finally:
            solver.pop()

    def entailment_core(self, question_expr: z3.BoolRef) -> List[int]:
        """Return supporting premise indices using one tracked UNSAT check."""
        if question_expr is None:
            return []
        self.compile()
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        trackers: list[z3.BoolRef] = []
        for index, group in enumerate(self._compiled_groups or []):
            if not group:
                continue
            tracker = z3.Bool(f"premise_track_{index}")
            trackers.append(tracker)
            solver.assert_and_track(z3.And(*group), tracker)
        solver.add(z3.Not(question_expr))
        if solver.check() != z3.unsat:
            return []
        names = {str(item) for item in solver.unsat_core()}
        return [
            index for index in range(len(self._compiled_groups or []))
            if f"premise_track_{index}" in names
        ]

    def get_unsat_core(self) -> List[int]:
        """Returns the indices of premises contributing to the last UNSAT result."""
        # Note: Minimal implementation. Real MUS finding requires trackers.
        return []

    def get_z3_code(self) -> str:
        """Return SMT-LIB without compiling the premise bundle again."""
        try:
            return self._get_base_solver().to_smt2()
        except Exception:
            return "; Error generating Z3 code"
