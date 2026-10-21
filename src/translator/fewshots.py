"""Append fixed, stage-specific demonstrations without changing inference logic."""
import json
import hashlib
from pathlib import Path


def append_fewshots(prompts, folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    sources = json.loads((folder / 'source_samples.json').read_text(encoding='utf-8'))
    if manifest.get('version', 0) >= 4:
        checksum = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        if checksum != manifest['source_nl_sha256']:
            raise ValueError('Few-shot source checksum mismatch')
        if len(sources) != manifest['source_count'] or [s['dataset_index'] for s in sources] != manifest['source_indices']:
            raise ValueError('Few-shot source manifest mismatch')
    counts = {}
    for stage in prompts:
        examples = json.loads((folder / f'{stage}.json').read_text(encoding='utf-8'))
        if len(examples) != manifest['examples_per_stage'].get(stage):
            raise ValueError(f'{stage}: demonstration count differs from manifest')
        for example in examples:
            source = example['source']
            if source.get('kind') == 'synthetic':
                if source.get('synthetic_id') not in manifest.get('synthetic_ids', []):
                    raise ValueError(f'{stage}: undeclared synthetic demonstration')
            elif source.get('dataset_index') not in manifest['source_indices']:
                raise ValueError(f'{stage}: demonstration source not in manifest')
            if stage == 'global_analysis':
                required = {'symbol_plan', 'domain_plan', 'discourse_referents', 'quantifier_plan'}
                if not required <= set(example['output']['global_analysis']):
                    raise ValueError('Global demonstration is missing connected-schema guidance')
        blocks = ['\n\nFEW-SHOT EXAMPLES (separate demonstration problems).\n'
                  'Follow the FOLIO representation conventions demonstrated by the few-shot examples:\n'
                  '- Do not automatically introduce a type predicate for generic domain nouns such as "people", "customers", "persons", etc. if the sentence is naturally quantified over that discourse class.\n'
                  '- Prefer canonical reusable semantic predicates over surface-form predicates.\n'
                  '- When a meaning can be compositionally represented using existing predicates, reuse/decompose into those predicates instead of introducing a new monolithic predicate.\n'
                  '- Preserve the representation style used consistently across the current problem and the few-shot examples.\n'
                  '- Do not add facts that are not stated or logically licensed.\n'
                  '- Bound variables must be single lowercase letters ([a-z]).\n'
                  'Use these to learn the output format and reasoning conventions only. '
                  'Do not copy their entities, predicates or facts into the current problem. '
                  'Demonstration registries do NOT initialize the actual registry. '
                  'The final user message is the only current input.\n']
        for index, example in enumerate(examples, 1):
            blocks.append(f'Example {index}\nInput: ' + json.dumps(example['input'], ensure_ascii=False, separators=(',', ':'))
                          + '\nOutput: ' + json.dumps(example['output'], ensure_ascii=False, separators=(',', ':')))
        prompts[stage] += '\n\n'.join(blocks)
        counts[stage] = len(examples)
    return {'version': manifest['version'], 'source_indices': manifest['source_indices'],
            'examples_per_stage': counts, 'source_nl_sha256': manifest['source_nl_sha256']}
