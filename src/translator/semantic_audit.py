"""Source-grounded warnings for the existing semantic review, not answer feedback."""
import re

from .translation_ast import parse_strict, walk
try:
    from ..solver_z3 import MultiSentenceCompiler, Z3Context
except (ImportError, ValueError):
    from solver_z3 import MultiSentenceCompiler, Z3Context
import z3


def audit(sentence, fol, declarations, registry, is_question=False, symbol_plan=None, alignment_plan=None):
    warnings = []
    try:
        tree = parse_strict(fol)
    except Exception as exc:
        return [{'kind': 'unparsed_candidate', 'detail': str(exc), 'action': 'Inspect meaning now; AST repair remains a separate later stage.'}]
    atoms = {n['name'] for n in walk(tree) if n.get('type') == 'Atom' and n.get('meaning') == 'predicate'}
    for rule in (alignment_plan or {}).get('generic_rules', []):
        if rule['source'] == sentence and tree.get('type') == 'Quantifier' and tree.get('meaning') == 'exists':
            warnings.append({'kind':'generic_quantifier_drift', 'source':sentence,
                'detail':'This source is a generic class rule under the shared FOLIO convention, not an existence assertion. Inspect forall versus exists without using the target answer.'})
    known = registry.get('symbols', {})
    plan=symbol_plan or {}
    declared=(declarations or {}).get('symbols',{}) if isinstance(declarations,dict) else {}
    for name in atoms:
        root=re.sub(r'(?:InYear|InSeason|AtTime|DuringPeriod|InEvent)$','',name)
        if root!=name and root in known and root not in atoms:
            warnings.append({'kind':'temporal_relation_disconnect','predicate':name,'core':root,
                'detail':'A temporal variant replaced a known core relation without retaining its positive projection. Prefer the planned event argument or source-licensed core + temporal qualification; negative dated facts must NOT imply negative timeless facts.'})
        if name in plan and name in declared:
            changed=[key for key in ['kind','argument_roles','arg_sorts','return_sort'] if declared[name].get(key)!=plan[name].get(key)]
            if changed:
                warnings.append({'kind':'planned_signature_drift','predicate':name,'fields':changed,
                    'detail':'Check the evidence-grounded story schema. Keep this relation and its event/role slots consistent; source semantics takes precedence if the plan is mistaken.'})
    for name in atoms:
        if name.startswith('Not') and len(name) > 3:
            positive = name[3:]
            if positive in atoms or positive in known or re.search(r'\b(?:not|never|neither)\b', sentence, re.I):
                warnings.append({'kind': 'lexicalized_negation', 'predicate': name,
                    'detail': 'A Not-prefixed predicate may erase logical negation. Check the source span and reuse Not(AST) for ordinary grammatical negation; genuine lexical properties need separate evidence.'})
    if is_question:
        new = sorted(atoms - set(known))
        if new:
            warnings.append({'kind': 'query_only_predicates', 'predicates': new,
                'detail': 'Check canonical reuse, compositional subsumption and temporal/event alignment. A genuinely new property can correctly remain Unknown; do not add premises to prove this query.'})
    if re.search(r'\bonly\b', sentence, re.I):
        warnings.append({'kind': 'only_scope', 'detail': 'Only restricts values of its source relation and semantic type. A music-style restriction must not exclude instruments or other unrelated registry properties.'})
    for node in walk(tree):
        if node.get('type') == 'Quantifier' and node.get('meaning') == 'exists':
            warnings.append({'kind': 'existential_reference', 'detail': 'Preserve existential scope. For definite/anaphoric objects, check whether the source designates an existing office, event or person rather than an arbitrary witness.'})
            break
    # KB consistency is a semantic warning only; legitimate inconsistent inputs
    # are retained. The conclusion is never asserted into the premise KB.
    if not is_question:
        try:
            compiler = MultiSentenceCompiler(Z3Context(), timeout_ms=1000, strict=True)
            for source in registry.get('sources', []):
                compiler.add_sentence(source['fol'])
            compiler.add_sentence(fol)
            solver = z3.Solver()
            solver.set(timeout=1000)
            solver.add(compiler.compile())
            status = solver.check()
            if status == z3.unsat:
                warnings.append({'kind': 'premise_kb_inconsistent', 'detail': 'This candidate makes the current premise KB inconsistent. Inspect source spans, scope and semantic types. Do not remove a genuine source contradiction to obtain an answer.'})
            elif status == z3.unknown:
                warnings.append({'kind': 'consistency_unresolved', 'detail': solver.reason_unknown()})
        except Exception as exc:
            warnings.append({'kind': 'consistency_check_unavailable', 'detail': str(exc)})
    return warnings
