"""Unified Hybrid Reasoning Pipeline combining Shared Translation, Z3 Fast-Path, and SymbCoT Fallback."""
from pathlib import Path
import time

from .common import (cache_path, digest, input_hash, public_label, read_json,
                     sanitize, write_json)
from .clover_solver import OursCloverSolver
from .shared_translator import SharedTranslator
from .symbcot_solver import OursSymbCoTSolver


class OursHybridPipeline:
    """
    Unified Neuro-Symbolic Pipeline:
    1. Shared SAA Translation Engine (T_ours)
    2. Clover Z3 Formal SMT Fast-Path (Checks SAT(P) & UNSAT(P & ¬Q))
    3. SymbCoT Symbolic Fallback (Plan -> Execute -> Verify when Z3 is inconclusive)
    """
    def __init__(self, config):
        self.config = config
        self.translator = SharedTranslator(config)
        self.clover = OursCloverSolver(config)
        self.symbcot = OursSymbCoTSolver(config)

    def process_sample(self, row, run_dir=None, mode='hybrid', retry_errors=False):
        """
        Executes pipeline on a single problem sample.
        Args:
            row: dict containing 'index', 'example_id', 'premises', 'conclusion', 'label'
            run_dir: Path/str where intermediate translation caches are stored
            mode: 'hybrid' (Z3-first with SymbCoT fallback) | 'dual' (both) | 'clover_only' | 'symbcot_only'
            retry_errors: bool
        Returns:
            dict containing full trace, solver path, and prediction
        """
        fingerprint = digest(self.config)
        
        # 1. Shared Translation Stage
        if run_dir:
            cached = self.translator.translate_and_cache(row, run_dir, fingerprint, retry_error=retry_errors)
            if cached.get('status') != 'complete':
                return {
                    'index': row['index'],
                    'id': row['example_id'],
                    'gold_label': row.get('label'),
                    'status': 'error',
                    'error': cached.get('error', 'Translation failed'),
                    'solver_path': 'failed_translation'
                }
            translation_text = cached['translation']
            translation_source = cached['translation_source']
            translation_trace = cached.get('translation_trace')
        else:
            translation_text, translation_source, translation_trace = self.translator.translate(row)

        record = {
            'index': row['index'],
            'id': row['example_id'],
            'gold_label': row.get('label'),
            'translation': translation_text,
            'translation_source': translation_source,
            'mode': mode,
            'status': 'complete'
        }

        # 2. Execution by Mode
        if mode == 'clover_only':
            clover_res = self.clover.solve(translation_source)
            record.update({
                'predicted_label': clover_res['predicted_label'],
                'solver_path': 'clover_z3',
                'clover_details': clover_res
            })

        elif mode == 'symbcot_only':
            symbcot_res = self.symbcot.solve(row, translation_text)
            record.update({
                'predicted_label': symbcot_res['predicted_label'],
                'solver_path': 'symbcot_cot',
                'symbcot_details': symbcot_res
            })

        elif mode == 'dual':
            clover_res = self.clover.solve(translation_source)
            symbcot_res = self.symbcot.solve(row, translation_text)
            record.update({
                'clover_predicted': clover_res['predicted_label'],
                'clover_decisive': clover_res['is_decisive'],
                'clover_details': clover_res,
                'symbcot_predicted': symbcot_res['predicted_label'],
                'symbcot_details': symbcot_res,
                'predicted_label': clover_res['predicted_label'] if clover_res['is_decisive'] else symbcot_res['predicted_label'],
                'solver_path': 'clover_z3' if clover_res['is_decisive'] else 'symbcot_fallback'
            })

        else:  # Default 'hybrid': Z3 Fast-Path with SymbCoT Fallback
            started_z3 = time.monotonic()
            clover_res = self.clover.solve(translation_source)
            z3_elapsed = time.monotonic() - started_z3

            if clover_res['is_decisive']:
                # Decisive mathematical proof from Z3!
                record.update({
                    'predicted_label': clover_res['predicted_label'],
                    'solver_path': 'clover_z3',
                    'is_fallback': False,
                    'clover_details': clover_res,
                    'z3_seconds': round(z3_elapsed, 3)
                })
            else:
                # Inconclusive (Unknown / timeout) -> Fallback to SymbCoT
                started_cot = time.monotonic()
                symbcot_res = self.symbcot.solve(row, translation_text)
                cot_elapsed = time.monotonic() - started_cot
                
                record.update({
                    'predicted_label': symbcot_res['predicted_label'],
                    'solver_path': 'symbcot_fallback',
                    'is_fallback': True,
                    'clover_details': clover_res,
                    'symbcot_details': symbcot_res,
                    'z3_seconds': round(z3_elapsed, 3),
                    'cot_seconds': round(cot_elapsed, 3)
                })

        if 'label' in row and record.get('predicted_label'):
            record['correct'] = (record['predicted_label'] == row['label'])

        return record
