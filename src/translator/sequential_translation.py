"""Sequential translation with separate bounded semantic and AST repair stages."""
import copy
import json
import logging
import re
from pathlib import Path

from .translation_ast import validate, walk, fingerprint, parse_strict, normalize_declarations
from .fewshots import append_fewshots
from .semantic_audit import audit
from .connected_formula import canonicalize, grounded_symbol_plan, grounded_domain_plan, retain_positive_temporal_core, infer_subject_domain
from .source_alignment import source_alignment_plan


class AtomicRegistry:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = {'symbols': {}, 'entities': {}, 'atomics': [], 'chunks': [],
                     'aliases': [], 'links': [], 'sources': []}

    @property
    def predicates(self):
        return {name: dict(spec, name=name, arity=len(spec['argument_roles']),
                           description=spec['meaning'])
                for name, spec in self.data['symbols'].items() if spec['kind'] == 'predicate'}

    @property
    def entities(self):
        return set(self.data['entities'])

    def snapshot(self):
        return copy.deepcopy(self.data)

    def _core_draft(self, declarations, ast, sentence, label, fol):
        """Minimal accepted state derives from checked declarations and AST, not extraction."""
        draft = self.snapshot()
        for category in ('symbols', 'entities'):
            for name, record in declarations.get(category, {}).items():
                fields = ('kind', 'meaning', 'argument_roles', 'arg_sorts', 'return_sort') if category == 'symbols' else ('meaning', 'sort')
                clean = {k: copy.deepcopy(record[k]) for k in fields if k in record}
                existing = draft[category].setdefault(name, dict(clean, provenance=[]))
                if label not in existing['provenance']:
                    existing['provenance'].append(label)
        for node in walk(ast):
            if (node['type'] == 'Atom' and node['meaning'] == 'predicate') or node['type'] == 'Comparison':
                self._add_formula(draft, 'atomics', node, '', label)
        self._add_formula(draft, 'chunks', ast, '', label)
        draft['sources'].append({'label': label, 'sentence': sentence, 'fol': fol})
        return draft

    @staticmethod
    def _add_formula(draft, category, tree, meaning, label):
        signature = json.dumps(fingerprint(tree), sort_keys=True)
        existing = next((r for r in draft[category] if r['ast_key'] == signature), None)
        if existing is None:
            existing = {'fol': tree['formula'], 'meaning': meaning, 'ast_key': signature, 'provenance': []}
            draft[category].append(existing)
        elif not existing['meaning'] and meaning:
            existing['meaning'] = meaning
        if label not in existing['provenance']:
            existing['provenance'].append(label)

    def commit_core(self, declarations, ast, sentence, label, fol):
        self.data = self._core_draft(declarations, ast, sentence, label, fol)

    def commit(self, extraction, declarations, ast, sentence, label, fol):
        """Commit checked core state plus valid auxiliary items in one transaction."""
        warnings = []
        draft = self._core_draft(declarations, ast, sentence, label, fol)
        if not isinstance(extraction, dict):
            warnings.append('Ignored malformed extraction metadata')
            extraction = {}
        nodes = list(walk(ast))
        key = lambda n: json.dumps(fingerprint(n), sort_keys=True)
        subtree_keys = {key(n) for n in nodes}
        atomic_keys = {key(n) for n in nodes if
                       (n['type'] == 'Atom' and n['meaning'] == 'predicate') or n['type'] == 'Comparison'}
        def items(category):
            value = extraction.get(category, [])
            if not isinstance(value, list):
                warnings.append('Ignored malformed optional ' + category)
                return []
            return value
        for category, allowed in (('atomics', atomic_keys), ('chunks', subtree_keys)):
            seen = set()
            for item in items(category):
                try:
                    tree = parse_strict(item['fol'])
                    if key(tree) not in allowed:
                        raise ValueError('formula is not an accepted AST subtree')
                    seen.add(key(tree))
                    meaning = item.get('meaning', '')
                    if not isinstance(meaning, str):
                        warnings.append('Ignored malformed optional ' + category + ' description')
                        meaning = ''
                    self._add_formula(draft, category, tree, meaning, label)
                except Exception as exc:
                    warnings.append(f'Skipped invalid optional {category} entry: {exc}')
            required = atomic_keys if category == 'atomics' else {key(ast)}
            if not required <= seen:
                warnings.append('Incomplete optional ' + category + '; retained AST-derived records')
        kinds = {'predicate': 'symbol', 'function': 'symbol', 'symbol': 'symbol',
                 'constant': 'entity', 'entity': 'entity'}
        def metadata_record(item):
            if not isinstance(item, dict):
                raise ValueError('record must be an object')
            kind = kinds.get(str(item.get('kind', '')).strip().lower())
            if kind is None:
                raise ValueError('unsupported metadata kind')
            return dict(item, kind=kind), {'symbol': 'symbols', 'entity': 'entities'}[kind]
        def words(text):
            irregular = {'people': 'person', 'persons': 'person', 'children': 'child', 'men': 'man', 'women': 'woman'}
            return [irregular.get(w, w) for w in re.findall(r'\w+', text.casefold())]
        for raw in items('aliases'):
            try:
                alias, category = metadata_record(raw)
                target, surface = alias.get('target'), alias.get('alias')
                if not isinstance(target, str) or target not in declarations.get(category, {}):
                    raise ValueError('target is not used in accepted FOL')
                if not isinstance(surface, str) or not surface.strip():
                    raise ValueError('missing alias string')
                source_words, alias_words = words(sentence), words(surface)
                grounded = bool(alias_words) and any(source_words[i:i+len(alias_words)] == alias_words
                                                     for i in range(len(source_words)))
                if not grounded:
                    raise ValueError('ungrounded alias ' + surface)
                if any(old['kind'] == alias['kind'] and words(old['alias']) == alias_words and old['target'] != target
                       for old in draft['aliases']):
                    raise ValueError('ambiguous alias target')
                clean = {k: alias[k] for k in ('kind', 'target', 'alias')}
                if clean not in draft['aliases']:
                    draft['aliases'].append(clean)
            except Exception as exc:
                warnings.append('Ignored optional alias: ' + str(exc))
        for raw in items('links'):
            try:
                link, category = metadata_record(raw)
                if not isinstance(link.get('target'), str) or link['target'] not in draft[category]:
                    raise ValueError('unresolved registry link target ' + str(link.get('target')))
                if (not isinstance(link.get('source'), str) or link['source'] not in declarations.get(category, {}) or
                        link.get('relation') != 'related' or not isinstance(link.get('evidence'), str) or not link['evidence'].strip()):
                    raise ValueError('invalid advisory registry link')
                clean = {k: link[k] for k in ('kind', 'source', 'target', 'relation', 'evidence')}
                if clean not in draft['links']:
                    draft['links'].append(clean)
            except Exception as exc:
                warnings.append('Skipped ' + str(exc))
        self.data = draft
        return warnings


class SequentialTranslator:
    def __init__(self, llm, model_name=None, logger=None):
        self.llm, self.model_name = llm, model_name
        self.logger = logger or logging.getLogger(__name__)
        self.registry = AtomicRegistry()
        self.global_analysis = {}
        self.analysis_error = None
        self.progress_context = ''
        self.problem_context = {}
        self.symbol_plan = {}
        self.domain_plan = {}
        self.alignment_plan = {}
        self.joint_proposal_diagnostic = {}
        prompt_folder = Path(__file__).parent / 'prompt'
        if not prompt_folder.exists():
            prompt_folder = Path(__file__).parent / 'prompts' / 'sequential'
        fewshot_folder = Path(__file__).parent / 'fewshot'
        if not fewshot_folder.exists():
            fewshot_folder = Path(__file__).parent / 'fewshots'
            if not fewshot_folder.exists():
                fewshot_folder = prompt_folder / 'fewshots'
        self.prompts = {name: (prompt_folder / (name + '.txt')).read_text(encoding='utf-8')
                        for name in ('global_analysis', 'translate', 'backtranslate', 'verify', 'extract')}
        policy = (prompt_folder / 'interpretation_policy.txt').read_text(encoding='utf-8')
        for stage in ('global_analysis', 'translate', 'backtranslate', 'verify'):
            self.prompts[stage] += '\n\n' + policy
        joint_prompt = (prompt_folder / 'translate_joint.txt').read_text(encoding='utf-8')
        joint_guidance, joint_contract = joint_prompt.split('JOINT OUTPUT CONTRACT', 1)
        self.prompts['translate_joint'] = joint_guidance+'\n\n'+policy
        self.fewshot_config = append_fewshots(self.prompts, fewshot_folder)
        alignment = (prompt_folder / 'alignment_policy.txt').read_text(encoding='utf-8')
        for stage in ('global_analysis', 'translate', 'verify', 'extract', 'translate_joint'):
            self.prompts[stage] += '\n\n' + alignment
        symbcot_reference = (prompt_folder / 'symbcot_reference.txt').read_text(encoding='utf-8')
        for stage in ('global_analysis', 'translate_joint', 'translate', 'verify'):
            self.prompts[stage] += '\n\n' + symbcot_reference
        # End joint instructions with its complete JSON example/contract, after
        # all FOL demonstrations. No hidden format-repair generation is added.
        self.prompts['translate_joint'] += '\n\nJOINT OUTPUT CONTRACT' + joint_contract
        self.logger.info('FEW_SHOT_CONFIG %s', json.dumps(self.fewshot_config))

    def _json(self, stage, payload):
        names = {'global_analysis': 'Phan tich ngu nghia toan cuc',
                 'translate': 'Dich NL -> FOL', 'backtranslate': 'Dich nguoc FOL -> NL',
                 'verify': 'Kiem tra ngu nghia', 'extract': 'Trich xuat atomics/chunks',
                 'translate_joint':'Tao proposal FOL lien ket cho toan bai'}
        self.progress(names[stage])
        raw = self.llm.chat([{'role': 'system', 'content': self.prompts[stage]},
                             {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                            model=self.model_name, temperature=0.0,
                            max_tokens=16384 if stage=='translate_joint' else 8192)
        # No hidden format-repair generations. Stage budgets are enforced by the caller.
        if stage == 'translate_joint':
            self.joint_proposal_diagnostic = {'raw_response': raw, 'parse_status': 'pending'}
        raw = raw.strip()
        if raw.startswith('```') and raw.endswith('```'):
            raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            if stage == 'translate_joint':
                self.joint_proposal_diagnostic.update(parse_status='failed', error=str(exc),
                    error_line=exc.lineno, error_column=exc.colno, error_position=exc.pos)
            raise
        if not isinstance(value, dict):
            raise ValueError('Expected JSON object for ' + stage)
        if stage == 'translate_joint':
            self.joint_proposal_diagnostic['parse_status'] = 'passed'
        return value

    def progress(self, stage):
        self.logger.info('%s%s', self.progress_context, stage, extra={'progress': True})

    def begin_problem(self, premises, question):
        self.progress_context = ''
        self.registry.reset()
        self.problem_context = {'premises': list(premises), 'question': question}
        self.global_analysis, self.analysis_error, self.symbol_plan = {}, None, {}
        self.joint_proposal_diagnostic = {}
        self.alignment_plan = source_alignment_plan(self.problem_context)
        self.domain_plan = self.alignment_plan.get('carrier') or infer_subject_domain(self.problem_context) or {}
        try:
            result = self._json('global_analysis', {'premises': premises, 'question': question,
                'alignment_plan': self.alignment_plan})
            if not isinstance(result.get('global_analysis'), dict):
                raise ValueError('Missing structured global_analysis')
            self.global_analysis = result['global_analysis']
            self.symbol_plan = grounded_symbol_plan(self.global_analysis,self.problem_context)
            self.domain_plan = self.domain_plan or grounded_domain_plan(self.global_analysis,self.problem_context)
        except Exception as exc:
            # Analysis is optional guidance; a failed analysis is explicitly recorded.
            self.analysis_error = str(exc)
        self.logger.info('GLOBAL_ANALYSIS %s', json.dumps(
            {'global_analysis': self.global_analysis, 'error': self.analysis_error}, ensure_ascii=False))

    def propose_problem(self):
        """One joint initial proposal; subsequent review/repair stays per sentence."""
        error=None
        candidates={}
        try:
            value=self._json('translate_joint',{'problem_context':self.problem_context,
                'global_analysis':self.global_analysis,'symbol_plan':self.symbol_plan,'domain_plan':self.domain_plan,
                'alignment_plan':self.alignment_plan})
            premises=value.get('premises')
            conclusion=value.get('conclusion')
            if not isinstance(premises,list) or len(premises)!=len(self.problem_context['premises']) or not isinstance(conclusion,dict):
                raise ValueError('Joint translation must contain exactly one entry per source premise and one conclusion.')
            for i,candidate in enumerate(premises,1):
                if not isinstance(candidate,dict):raise ValueError('Malformed joint premise entry')
                candidates[f'Premise {i}']=candidate
            candidates['Conclusion']=conclusion
            for label,candidate in candidates.items():
                if not isinstance(candidate.get('fol'),str) or not candidate['fol'].strip():
                    raise ValueError('Missing nonempty joint FOL for '+label)
                declarations=candidate.get('declarations')
                if not isinstance(declarations,dict) or any(not isinstance(declarations.get(k),dict) for k in ('symbols','entities')):
                    raise ValueError('Missing joint declaration maps for '+label)
            self.joint_proposal_diagnostic['schema_status'] = 'passed'
        except Exception as exc:
            error=str(exc)
            candidates={}
            self.joint_proposal_diagnostic.update(schema_status='failed', error=error)
        return candidates,error

    def process_sentence(self, sentence, sentence_label='Premise', initial_candidate=None, initial_error=None):
        before = self.registry.snapshot()
        is_first_premise = (sentence_label == 'Premise 1' or not before.get('sources'))
        is_question = sentence_label.lower().startswith(('conclusion', 'question'))
        result = {'sentence': sentence, 'label': sentence_label, 'registry_before': before,
                  'attempts': [], 'ast_checks': [], 'selected_fol': '', 'status': 'forwarded',
                  'metadata_warnings': [], 'validation_warnings': [],
                  'semantic_status': 'failed', 'ast_status': 'failed',
                  'semantic_verify_calls': 0, 'semantic_repair_calls': 0, 'ast_repair_calls': 0,
                  'structure': '', 'accumulation': [], 'translation_steps': [], 'equivalence_groups': []}
        fol, declarations, ast = '', {}, None
        feedback = None
        reviewed_fol, semantic_passed = None, False
        semantic_invalidated = False

        def propose(phase, number, prior_feedback):
            nonlocal fol, declarations, semantic_invalidated
            self.progress_context = f'{sentence_label} | {phase} {number}/1 | '
            trace = {'attempt': len(result['attempts']) + 1, 'phase': phase,
                     'fol': fol, 'back_translation': None, 'verifier_feedback': None,
                     'ast_validation': {'status': 'not_run'}, 'extracted': None,
                     'accepted': False, 'retained': False, 'metadata_warnings': []}
            result['attempts'].append(trace)
            try:
                payload = {'sentence': sentence, 'label': sentence_label,
                    'problem_context': self.problem_context,
                    'symbol_plan': self.symbol_plan,
                    'domain_plan': self.domain_plan,
                    'alignment_plan': self.alignment_plan,
                    'global_analysis': self.global_analysis, 'registry': before,
                    'feedback': prior_feedback, 'repair_stage': phase,
                    'repair_instruction': (
                        'Repair only the reported AST, binding, declaration or signature errors. '
                        'Preserve the intended meaning and all clauses of the latest FOL; return fol and declarations.'
                        if phase == 'ast_repair' else 'Translate or repair the FOL using the semantic feedback.')}
                if phase=='semantic' and initial_error:
                    raise ValueError('Joint initial proposal unavailable: '+initial_error)
                if phase=='semantic' and initial_candidate is not None:
                    candidate=copy.deepcopy(initial_candidate)
                    trace['from_joint_proposal']=True
                else:
                    candidate = self._json('translate', payload)
                trace['candidate'] = candidate
                proposed_fol = candidate.get('fol')
                if not isinstance(proposed_fol, str) or not proposed_fol.strip():
                    raise ValueError('Missing FOL string; keeping the last available formula')
                # Even an unverified or structurally invalid formula is the latest
                # candidate. A failed/empty API response cannot erase that candidate.
                proposed_declarations = candidate.get('declarations')
                proposed_fol,proposed_declarations,canonical_changes = canonicalize(
                    proposed_fol,proposed_declarations,is_question,self.domain_plan,
                    generic_rule=not is_question and any(g['source'] == sentence
                        for g in self.alignment_plan.get('generic_rules', [])))
                if phase=='semantic_repair' and not is_question:
                    proposed_fol,projection_changes=retain_positive_temporal_core(fol,proposed_fol,prior_feedback,before)
                    canonical_changes+=projection_changes
                trace['canonicalization'] = canonical_changes
                if phase == 'ast_repair' and (proposed_fol != fol or proposed_declarations != declarations):
                    semantic_invalidated = True
                fol, declarations = proposed_fol, proposed_declarations
            except Exception as exc:
                trace['error'] = str(exc)
            trace['fol'] = fol
            return trace

        # One semantic review; a rejection permits one rewrite without re-review.
        # AST diagnostics never restart this stage.
        for number in (1,):
            trace = propose('semantic', number, feedback)
            if fol:
                try:
                    back = self._json('backtranslate', {'fol': fol, 'declarations': declarations,
                        'registry': {k: before[k] for k in ('symbols', 'entities', 'aliases')}})
                    if not isinstance(back.get('back_translation'), str) or not back['back_translation'].strip():
                        raise ValueError('Missing back_translation')
                    trace['back_translation'] = back['back_translation']
                except Exception as exc:
                    trace['metadata_warnings'].append('Back-translation unavailable: ' + str(exc))
            # A back-translation outage does not prevent the verifier inspecting
            # the original source and actual FOL, or consume extra verifier calls.
            result['semantic_verify_calls'] += 1
            reviewed_fol = fol
            result['semantic_audit'] = audit(sentence, fol, declarations, before, is_question, self.symbol_plan, self.alignment_plan)
            try:
                review = self._json('verify', {'original_sentence': sentence, 'fol': fol,
                    'problem_context': self.problem_context, 'semantic_audit': result['semantic_audit'],
                    'symbol_plan': self.symbol_plan, 'canonicalization': trace.get('canonicalization',[]),
                    'domain_plan': self.domain_plan,
                    'alignment_plan': self.alignment_plan,
                    'back_translation': trace['back_translation'], 'declarations': declarations,
                    'global_analysis': self.global_analysis, 'registry': before})
                trace['verifier_feedback'] = review
                if (type(review.get('equivalent')) is not bool or
                        not isinstance(review.get('issues'), list) or
                        not isinstance(review.get('suggested_corrections'), list)):
                    raise ValueError('Invalid semantic verifier JSON schema')
                semantic_passed = bool(fol.strip()) and review['equivalent'] and not review['issues']
                if not semantic_passed:
                    raise ValueError('Semantic verification rejected translation')
            except Exception as exc:
                semantic_passed = False
                trace['semantic_error'] = str(exc)
            if semantic_passed:
                break
            feedback = {'previous_fol': fol, 'back_translation': trace['back_translation'],
                        'semantic_audit': result['semantic_audit'],
                        'verifier_feedback': trace['verifier_feedback'],
                        'error': trace.get('semantic_error', trace.get('error'))}
        if not semantic_passed:
            result['semantic_repair_calls'] = 1
            trace = propose('semantic_repair', 1, feedback)
            result['validation_warnings'].append('Semantic review failed; retained the single rewrite without re-review')

        # One deterministic check before repair, then up to ONE LLM repair,
        # each followed by a deterministic AST/compile check (no semantic loop).
        ast_reg = None if is_first_premise else before
        ast_fols = () if is_first_premise else [s['fol'] for s in before['sources']]
        for repair_count in range(2):
            self.progress_context = f'{sentence_label} | AST check {repair_count + 1}/2 | '
            try:
                self.progress('Kiem tra AST')
                checked_declarations = normalize_declarations(
                    fol, declarations, ast_reg, trace['metadata_warnings'])
                ast = validate(fol, checked_declarations, ast_reg, ast_fols)
                declarations = checked_declarations
                check = {'status': 'passed', 'issues': [], 'ast': ast, 'fol': fol}
            except Exception as exc:
                ast = None
                check = {'status': 'failed', 'issues': [str(exc)], 'fol': fol}
            result['ast_checks'].append(copy.deepcopy(check))
            trace['ast_validation'] = check
            if ast is not None or repair_count == 1:
                break
            result['ast_repair_calls'] += 1
            trace = propose('ast_repair', repair_count + 1, {
                'previous_fol': fol, 'declarations': declarations,
                'ast_validation': check, 'error': '; '.join(check['issues'])})

        result['ast_status'] = 'passed' if ast is not None else 'failed'
        result['semantic_status'] = ('unverified_after_ast_repair' if semantic_invalidated else
                                     'unverified_after_semantic_repair' if result['semantic_repair_calls'] and fol != reviewed_fol else
                                     'passed' if semantic_passed and fol == reviewed_fol else 'failed')
        if result['semantic_status'] == 'unverified_after_ast_repair':
            result['validation_warnings'].append('AST repair changed FOL after the semantic review stage')
        if ast is None:
            result['validation_warnings'].append('AST repair exhausted; retained the latest FOL for Z3')
        fully_checked = result['semantic_status'] == 'passed' and ast is not None
        result.update(selected_fol=fol, status='accepted' if fully_checked else 'forwarded',
                      translation_steps=[sentence + ' ::: ' + fol])
        trace['accepted'], trace['retained'] = fully_checked, True
        trace['fol'] = fol

        if is_question:
            trace['extraction_status'] = 'skipped_question'
        elif ast is None:
            # Never insert an invalid AST into the structured registry. The
            # formula still goes to the final solver via selected_fol.
            trace['extraction_status'] = 'skipped_invalid_ast'
        else:
            extraction = {}
            try:
                extraction = self._json('extract', {'source_nl': sentence, 'accepted_fol': fol,
                    'ast': ast, 'declarations': declarations,
                    'global_analysis': self.global_analysis, 'registry': before,
                    'semantic_status': result['semantic_status']})
            except Exception as exc:
                trace['metadata_warnings'].append('Atomic/chunk extraction failed: ' + str(exc))
            trace['extracted'] = extraction
            try:
                self.progress('Cap nhat registry')
                trace['metadata_warnings'].extend(self.registry.commit(
                    extraction, declarations, ast, sentence, sentence_label, fol))
            except Exception as exc:
                trace['metadata_warnings'].append('Registry metadata update failed; retained core only: ' + str(exc))
                try:
                    self.registry.commit_core(declarations, ast, sentence, sentence_label, fol)
                except Exception as core_exc:
                    trace['metadata_warnings'].append('Registry core update failed; FOL retained: ' + str(core_exc))
            if self.registry.data['sources'] and self.registry.data['sources'][-1]['label'] == sentence_label:
                self.registry.data['sources'][-1]['semantic_status'] = result['semantic_status']
        result['metadata_warnings'] = [w for attempt in result['attempts'] for w in attempt['metadata_warnings']]
        result['registry_after'] = self.registry.snapshot()
        result['final_semantic_audit'] = audit(sentence, fol, declarations, before, is_question, self.symbol_plan, self.alignment_plan)
        self.logger.info('SENTENCE_TRANSLATION %s', json.dumps(result, ensure_ascii=False))
        return result
