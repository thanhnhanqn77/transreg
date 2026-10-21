"""Ours SymbCoT Reasoning Backend (Self-Contained) consuming shared FOL."""
from pathlib import Path

from .client import OllamaClient, read_keys
from .common import (WORK, final_label, inference_row, label_templates,
                     observe_client, public_label)


def load_templates(prompt_dir=None):
    """Load SymbCoT reasoning prompt templates (plan, execution, verification)."""
    if prompt_dir and Path(prompt_dir).is_dir():
        p_dir = Path(prompt_dir)
        return {stage: label_templates((p_dir / filename).read_text(encoding='utf-8')) for stage, filename in {
            'plan': 'plan_generation.txt',
            'execution': 'solver.txt',
            'verification': 'verification.txt'
        }.items()}
    default_dir = Path(__file__).parent / 'solver_symbcot' / 'prompts'
    return {stage: label_templates((default_dir / filename).read_text(encoding='utf-8')) for stage, filename in {
        'plan': 'plan_generation.txt',
        'execution': 'solver.txt',
        'verification': 'verification.txt'
    }.items()}


def format_symbcot_prompt(stage, row, record, templates):
    """
    Format prompt for SymbCoT stage using verified FOL and previous reasoning steps.
    """
    text = templates[stage].replace("[[CONTEXT]]", record["translation"])
    if stage == "execution":
        return text.replace("[[PLAN]]", record["plan"])
    if stage == "verification":
        return text.replace("[[EXECUTION]]", record["execution"])
    return text


class OursSymbCoTSolver:
    """
    Symbolic LLM CoT Reasoning backend:
    - Ingests shared FOL from SharedTranslator.
    - Executes 3 sequential structured stages:
        1. Plan Generation: Outlines proof steps.
        2. Execution: Derives intermediate deductions.
        3. Verification: Validates deductive validity and emits final label.
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
        self.templates = load_templates(config.get('prompt_dir'))
        observe_client(self)

    def solve(self, row, translation_text):
        """
        Executes Plan -> Execution -> Verification on top of shared FOL.
        Args:
            row: Dataset sample dict with 'premises', 'conclusion'
            translation_text: Shared formatted FOL string
        Returns:
            dict: {
                'predicted_label': 'True' | 'False' | 'Uncertain',
                'draft_label': 'True' | 'False' | 'Uncertain',
                'plan': str,
                'execution': str,
                'verification': str,
                'generation': list of telemetry metadata
            }
        """
        self.generation = []
        inf_row = inference_row(row)
        record = {'translation': translation_text}
        
        stages = ('plan', 'execution', 'verification')
        for stage in stages:
            prompt = format_symbcot_prompt(stage, inf_row, record, self.templates)
            text = self.client.generate(prompt)
            record[stage] = text

        draft_label = final_label(record['execution'])
        predicted_label = final_label(record['verification'])

        return {
            'predicted_label': predicted_label,
            'draft_label': draft_label,
            'plan': record['plan'],
            'execution': record['execution'],
            'verification': record['verification'],
            'generation': list(self.generation)
        }
