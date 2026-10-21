"""Core dataset boundary, labels, hashing, caching, and utilities for the Ours Hybrid Pipeline."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
DATA_DIR = PACKAGE_ROOT / 'data'
LOGS_DIR = PACKAGE_ROOT / 'logs'
WORK = PACKAGE_ROOT

LABELS = ('True', 'False', 'Uncertain')
SECRET = re.compile(r'[a-f0-9]{32}\.[A-Za-z0-9_\-]+')


def public_label(value):
    """Normalize label strings to True, False, or Uncertain."""
    labels = {'true': 'True', 'false': 'False', 'unknown': 'Uncertain', 'uncertain': 'Uncertain'}
    if not isinstance(value, str) or value.strip().lower() not in labels:
        raise ValueError(f'Invalid classification label: {value!r}')
    return labels[value.strip().lower()]


def load_dataset(path):
    """Load JSONL dataset (FOLIO or ProofWriter format)."""
    rows = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        original = json.loads(line)
        premises = original.get('premises')
        if isinstance(premises, str):
            premises = [line.strip() for line in premises.splitlines() if line.strip()]
        elif isinstance(premises, list):
            premises = [p.strip() for p in premises if isinstance(p, str) and p.strip()]
        if not isinstance(premises, list) or not premises:
            raise ValueError(f'Invalid premises at dataset index {len(rows)}')
        conclusion = original.get('conclusion')
        if not isinstance(conclusion, str) or not conclusion.strip():
            raise ValueError(f'Invalid conclusion at dataset index {len(rows)}')
        rows.append({
            'index': len(rows),
            'story_id': original.get('story_id'),
            'example_id': original.get('example_id', len(rows)),
            'premises': list(premises),
            'conclusion': conclusion,
            'label': public_label(original.get('label'))
        })
    if not rows or len({str(r['example_id']) for r in rows}) != len(rows):
        raise ValueError('Empty dataset or duplicate example IDs')
    return rows


def inference_row(row):
    """Extract raw inference input without gold labels."""
    return {key: row[key] for key in ('story_id', 'example_id', 'premises', 'conclusion')}


def digest(value):
    """Compute deterministic SHA256 hex digest for JSON-serializable object."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def input_hash(row):
    """Hash premises and conclusion."""
    return digest({'premises': row['premises'], 'conclusion': row['conclusion']})


def read_json(path):
    """Read JSON file."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """Atomic write to JSON file with fallback."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.2 * (attempt + 1))


def cache_path(run_dir, row):
    """Return path to cached translation JSON file."""
    return Path(run_dir) / 'translations' / f"{row['index']:05d}.json"


def sanitize(text):
    """Redact sensitive API keys from strings."""
    return re.sub(r'[a-f0-9]{8}\.\.\.[A-Za-z0-9_\-]{4}', '[masked credential]', SECRET.sub('[REDACTED]', str(text)))


def label_templates(text):
    """Normalize unknown -> Uncertain in prompt templates."""
    return re.sub(r'\bunknown\b', 'Uncertain', text, flags=re.I)


def final_label(text):
    """Extract {true}/{false}/{uncertain} from LLM response text."""
    found = re.findall(r'\{\s*(true|false|unknown|uncertain)\s*\}', text, re.I)
    if not found:
        raise ValueError('Missing valid {true}/{false}/{uncertain} final answer')
    return public_label(found[-1])


def response_metadata(data, elapsed=None):
    """Extract telemetry metadata from LLM JSON response."""
    value = {key: data[key] for key in ('done_reason', 'eval_count', 'prompt_eval_count',
             'total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration') if key in data}
    message = data.get('message') or {}
    value.update(thinking_requested=False, thinking_returned=bool(message.get('thinking')),
                 content_characters=len(message.get('content') or ''))
    if elapsed is not None:
        value['request_wall_seconds'] = round(elapsed, 3)
    return value


def require_no_thinking(data):
    """Enforce strict thinking=false policy."""
    message = data.get('message') or {}
    if (str(message.get('thinking') or '').strip() or
            re.search(r'<\s*/?\s*think\s*>', str(message.get('content') or ''), re.I)):
        raise RuntimeError('Thinking policy violated: server returned thinking despite think=false')


def observe_client(owner):
    """Wrap client urllib opener to enforce think=false and record telemetry."""
    owner.generation = []
    opener = owner.client.opener

    def capture(request, *args, **kwargs):
        payload = json.loads(request.data)
        payload['think'] = False
        options = payload.setdefault('options', {})
        options['num_predict'] = min(int(options.get('num_predict', owner.client.max_tokens)), owner.client.max_tokens)
        request.data = json.dumps(payload).encode()
        started = time.monotonic()
        with opener(request, *args, **kwargs) as response:
            raw = response.read()
        data = json.loads(raw)
        owner.generation.append(response_metadata(data, time.monotonic() - started))
        require_no_thinking(data)
        return io.BytesIO(raw)
    owner.client.opener = capture
