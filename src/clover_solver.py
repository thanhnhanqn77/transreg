"""Ours Clover SMT Solver backend (Z3 Fast-Path) - Completely Self-Contained."""
from typing import List, Dict, Any
import z3

from .solver_z3 import Z3Context, MultiSentenceCompiler


class OursCloverSolver:
    """
    Formal Theorem Prover backend leveraging Z3 SMT solver.
    Operates strictly on verified FOL formulas produced by SharedTranslator.
    Zero LLM tokens during inference.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.timeout_ms = int(self.config.get('z3_timeout_ms', 5000))

    def verify_and_solve(self, fol_premises: List[str], conclusion_fol: str) -> Dict[str, Any]:
        """
        Executes Z3 SMT solver on selected FOL formulas.
        Determines True, False, or Unknown directly from Z3.
        """
        ctx = Z3Context()
        compiler = MultiSentenceCompiler(ctx, timeout_ms=self.timeout_ms)

        compile_errors = []
        for i, fol_p in enumerate(fol_premises):
            try:
                compiler.add_sentence(fol_p)
            except Exception as e:
                compile_errors.append(f"Premise {i+1}: {e}")

        z3_decision = "Unknown"
        decision_type = "z3_smt"
        explanation = ""
        premises_used = []

        if not compile_errors and conclusion_fol:
            try:
                is_consistent = compiler.check_consistency()
                if not is_consistent:
                    z3_decision = "Unknown"
                    explanation = "Premises conjunction is inconsistent / UNSAT in Z3."
                else:
                    q_res = compiler.compile_question(conclusion_fol, label="Target")
                    q_expr = q_res.z3_expr

                    if q_expr is not None:
                        # Check entailment: Is Premises ∧ ¬Conclusion UNSAT?
                        is_entailed = compiler.check_entailment(q_expr)
                        if is_entailed:
                            z3_decision = "True"
                            core = compiler.entailment_core(q_expr)
                            premises_used = [idx + 1 for idx in core]
                            explanation = f"Z3 SMT formally proved True (Premises ⊨ Conclusion). Supporting premises: {premises_used or 'all'}."
                        else:
                            # Check refutation: Is Premises ∧ Conclusion UNSAT?
                            is_refuted = compiler.check_entailment(z3.Not(q_expr))
                            if is_refuted:
                                z3_decision = "False"
                                core = compiler.entailment_core(z3.Not(q_expr))
                                premises_used = [idx + 1 for idx in core]
                                explanation = f"Z3 SMT formally proved False (Premises ⊨ ¬Conclusion). Supporting premises: {premises_used or 'all'}."
                            else:
                                z3_decision = "Unknown"
                                explanation = "Z3 SMT solver concluded Unknown (neither True nor False entailed)."
                    else:
                        compile_errors.append("Failed to compile conclusion expression to Z3.")
            except Exception as e:
                compile_errors.append(f"Z3 execution error: {e}")
                z3_decision = "Unknown"

        return {
            "predicted_answer": z3_decision,
            "decision": z3_decision,
            "decision_type": decision_type,
            "explanation": explanation,
            "compiler_errors": compile_errors,
            "premises_used": premises_used
        }

    def solve(self, translation_source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes formal Z3 SMT entailment & refutation checks.
        Args:
            translation_source: Dict containing 'premises_fol' (list) and 'conclusion_fol' (str)
        Returns:
            dict: {
                'predicted_label': 'True' | 'False' | 'Uncertain',
                'is_decisive': bool (True if proven True or False, False if Unknown),
                'z3_decision': raw decision ('True', 'False', 'Unknown'),
                'explanation': str,
                'premises_used': list,
                'compiler_errors': list
            }
        """
        premises_fol = translation_source.get('premises_fol', [])
        conclusion_fol = translation_source.get('conclusion_fol', '')
        
        if not premises_fol or not conclusion_fol:
            return {
                'predicted_label': 'Uncertain',
                'is_decisive': False,
                'z3_decision': 'Unknown',
                'explanation': 'Missing premises or conclusion FOL in translation source',
                'premises_used': [],
                'compiler_errors': ['Empty FOL']
            }

        result = self.verify_and_solve(premises_fol, conclusion_fol)
        
        raw_decision = result.get('predicted_answer') or result.get('decision') or 'Unknown'
        compiler_errors = result.get('compiler_errors', [])
        explanation = result.get('explanation', '')
        premises_used = result.get('premises_used', [])

        norm_dec = str(raw_decision).strip().capitalize()
        if norm_dec in ('True', 'False'):
            is_decisive = True
            predicted_label = norm_dec
        else:
            is_decisive = False
            predicted_label = 'Uncertain'

        return {
            'predicted_label': predicted_label,
            'is_decisive': is_decisive,
            'z3_decision': raw_decision,
            'explanation': explanation,
            'premises_used': premises_used,
            'compiler_errors': compiler_errors,
            'details': result
        }
