"""Shared SAA Translation Engine (T_ours) producing verified FOL bundles (Self-Contained)."""
from pathlib import Path
import sys

from .client import OllamaClient, read_keys
from .common import (WORK, cache_path, digest, inference_row, input_hash,
                     observe_client, read_json, sanitize, write_json)
from .translator.live_clover_translation import LiveCloverTranslations


class SharedTranslator:
    """
    Closed-Loop SAA Translation Engine:
    - Global Semantic Analysis
    - Joint Initial FOL Proposal
    - Sequential Blind Back-Translation & 3-Way Semantic Verification
    - Deterministic AST Invariant Checking (0 LLM tokens)
    - Atomic Predicate Registry Updates
    """
    def __init__(self, config):
        self.config = config
        
        key_input = config.get('api_key') or config.get('key') or config.get('key_file')
        keys = read_keys(key_input)
        
        self.client = OllamaClient(
            keys,
            config.get('model_name', 'gemma4:31b-cloud'),
            config.get('host', 'https://ollama.com/api/chat'),
            timeout=int(config.get('timeout', 600)),
            max_tokens=int(config.get('max_tokens', 12288))
        )
        self.translator = LiveCloverTranslations(self.client)
        observe_client(self)
        self.error = None

        chat = self.translator.translator.llm.chat
        def checked_chat(*args, **kwargs):
            if self.error:
                raise RuntimeError(self.error)
            try:
                return chat(*args, **kwargs)
            except RuntimeError as exc:
                self.error = str(exc)
                raise
        self.translator.translator.llm.chat = checked_chat

    def translate(self, row):
        """
        Translates a single dataset sample (premises + conclusion).
        Returns:
            text: Human-readable FOL string formatted for SymbCoT
            source: Dict containing raw premises_fol and conclusion_fol for Clover/Z3
            trace: Full diagnostic audit trace
        """
        self.generation = []
        inf_row = inference_row(row)
        text, source, trace = self.translator.translation(inf_row, row['index'])
        if not text:
            raise ValueError('Shared translator returned empty FOL after bounded repairs')
        return text, source, trace

    def translate_and_cache(self, row, run_dir, fingerprint, retry_error=False):
        """
        Translates row and writes to disk cache at run_dir/translations/{index:05d}.json.
        Reuses existing complete cache if valid.
        """
        path = cache_path(run_dir, row)
        if path.exists():
            try:
                cached = read_json(path)
                if cached.get('status') == 'complete' and cached.get('input_sha256') == input_hash(row):
                    return cached
                if not retry_error and cached.get('status') == 'error':
                    return cached
            except Exception:
                pass

        cache_entry = {
            'fingerprint': fingerprint,
            'index': row['index'],
            'id': row['example_id'],
            'input_sha256': input_hash(row)
        }
        try:
            text, source, trace = self.translate(row)
            cache_entry.update({
                'status': 'complete',
                'translation': text,
                'translation_source': source,
                'translation_trace': trace,
                'generation': list(self.generation)
            })
        except Exception as exc:
            cache_entry.update({
                'status': 'error',
                'error': sanitize(exc),
                'generation': list(self.generation)
            })
        write_json(path, cache_entry)
        return cache_entry
