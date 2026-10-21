import z3
import copy
from typing import Dict, List, Optional, Tuple

class Z3Context:
    """
    Shared Z3 environment for maintaining symbol consistency across multiple FOL sentences.

    Improvements over v1:
      - Raises ValueError on genuine arity or argument-sort conflict (no silent corruption)
      - snapshot() / restore() for safe compiler reuse across MCQ options
      - sort_of() helper for external inspection
    """

    _SORT_PRECEDENCE = {
        "Entity":      0,
        "BoolSort()":  1,
        "IntSort()":   2,
        "RealSort()":  3,
        "StringSort()": 4,
        "DateSort":    2, # Same as Int
        "TimeSort":    2, # Same as Int
    }

    def __init__(self):
        self.consts: Dict[str, z3.ExprRef] = {}
        self.funcs:  Dict[str, z3.FuncDeclRef] = {}
        self.sorts:  Dict[str, z3.SortRef] = {
            "Entity":   z3.DeclareSort("Entity"),
            "TimeSort": z3.IntSort(),
            "DateSort": z3.IntSort(),
        }

    # ------------------------------------------------------------------
    # Snapshot / restore  (for safe per-option compilation)
    # ------------------------------------------------------------------

    def snapshot(self) -> Tuple[dict, dict, dict]:
        """Returns a shallow-copy snapshot of (consts, funcs, sorts)."""
        return (
            dict(self.consts),
            dict(self.funcs),
            dict(self.sorts),
        )

    def restore(self, snap: Tuple[dict, dict, dict]):
        """Restores context to a previous snapshot."""
        self.consts, self.funcs, self.sorts = snap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create_const(self, name: str, sort_str: str) -> z3.ExprRef:
        """Retrieves an existing constant or creates a new one."""
        if name not in self.consts:
            sort = self._resolve_sort(sort_str)
            if sort_str == "BoolSort()":
                self.consts[name] = z3.Bool(name)
            elif sort_str == "IntSort()":
                self.consts[name] = z3.Int(name)
            elif sort_str == "RealSort()":
                self.consts[name] = z3.Real(name)
            elif sort_str == "StringSort()":
                self.consts[name] = z3.String(name)
            else:
                self.consts[name] = z3.Const(name, sort)
        return self.consts[name]

    def get_or_create_func(
        self, name: str, arg_sorts: List[str], ret_sort: str
    ) -> z3.FuncDeclRef:
        """
        Retrieves or creates a Z3 function declaration.

        Upgrade policy (ret_sort only):
          - If new ret_sort has strictly higher precedence → redeclare
          - If arg sorts differ → raise ValueError (prevents silent corruption)
        """
        if name in self.funcs:
            existing = self.funcs[name]

            # --- Arity check ---
            if existing.arity() != len(arg_sorts):
                raise ValueError(
                    f"Arity conflict for '{name}': "
                    f"registered with {existing.arity()} args, "
                    f"now requested with {len(arg_sorts)} args."
                )

            # --- Arg-sort conflict check ---
            for i, (reg_sort, new_sort_str) in enumerate(
                zip([str(existing.domain(j)) for j in range(existing.arity())], arg_sorts)
            ):
                new_sort_obj = self._resolve_sort(new_sort_str)
                if str(new_sort_obj) != reg_sort:
                    # Allow IntSort→RealSort promotion only
                    if not (reg_sort == "Int" and str(new_sort_obj) == "Real"):
                        raise ValueError(
                            f"Arg-sort conflict for '{name}' at position {i}: "
                            f"registered='{reg_sort}', requested='{new_sort_str}'."
                        )

            # --- Return-sort upgrade ---
            existing_ret = str(existing.range())
            p_new = self._SORT_PRECEDENCE.get(ret_sort, 0)
            p_old = self._SORT_PRECEDENCE.get(existing_ret, 0)
            if p_new > p_old:
                z3_arg_sorts = [self._resolve_sort(s) for s in arg_sorts]
                z3_ret_sort  = self._resolve_sort(ret_sort)
                self.funcs[name] = z3.Function(name, *z3_arg_sorts, z3_ret_sort)

            return self.funcs[name]

        z3_arg_sorts = [self._resolve_sort(s) for s in arg_sorts]
        z3_ret_sort  = self._resolve_sort(ret_sort)
        self.funcs[name] = z3.Function(name, *z3_arg_sorts, z3_ret_sort)
        return self.funcs[name]

    def sort_of(self, name: str) -> Optional[str]:
        """Returns the sort string of a registered constant, or None."""
        if name in self.consts:
            return str(self.consts[name].sort())
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_sort(self, sort_str: str) -> z3.SortRef:
        if sort_str == "IntSort()":    return z3.IntSort()
        if sort_str == "RealSort()":   return z3.RealSort()
        if sort_str == "BoolSort()":   return z3.BoolSort()
        if sort_str == "StringSort()": return z3.StringSort()
        if sort_str not in self.sorts:
            self.sorts[sort_str] = z3.DeclareSort(sort_str)
        return self.sorts[sort_str]
