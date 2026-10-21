"""CLOVER translation adapter. Downstream SymbCoT receives only FOL + source NL."""
import hashlib
import json
import logging
import socket
from pathlib import Path
from urllib import error, request

from .sequential_translation import SequentialTranslator


class TranslationClient:
    def __init__(self, client):
        self.client = client

    def chat(self, messages, model=None, temperature=0, max_tokens=8192):
        c = self.client
        payload = json.dumps({'model': model or c.model, 'stream': False,
            'messages': messages, 'options': {'temperature': temperature,
            'num_predict': max_tokens, 'num_ctx': max(max_tokens*2,32768)}}).encode()
        attempts = max(len(c.keys) * max(c.retries, 15), 60)
        for attempt in range(1,attempts+1):
            req = request.Request(c.url,data=payload,headers={
                'Authorization': 'Bearer '+c.keys[c.index], 'Content-Type':'application/json'})
            try:
                with c.opener(req,timeout=c.timeout) as response:
                    data = json.load(response)
                if data.get('error'):
                    c.rotate('translation API error',attempt)
                    continue
                content = data.get('message',{}).get('content','').strip()
                if not content:
                    raise RuntimeError('Translation API returned empty final content.')
                if data.get('done_reason')=='length':
                    raise RuntimeError('Translation output truncated; final FOL retained if available.')
                return content
            except error.HTTPError as exc:
                status=exc.code
                exc.close()
                if status not in (401,402,403,408,429,500,502,503,504):
                    raise RuntimeError(f'Translation Ollama HTTP {status}.') from None
                c.rotate(f'translation HTTP {status}',attempt)
            except (error.URLError,TimeoutError,socket.timeout):
                if attempt==attempts:
                    raise RuntimeError('Translation network/timeout after retries.') from None
                c.sleep(1)
        raise RuntimeError('Translation keys unavailable after retries.')


class LiveCloverTranslations:
    def __init__(self, client):
        logger=logging.getLogger('symbcot.clover_translation')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler=logging.StreamHandler()
            handler.addFilter(lambda record: bool(getattr(record,'progress',False)))
            logger.addHandler(handler)
        logger.propagate=False
        self.translator=SequentialTranslator(TranslationClient(client),client.model,logger)

    def translation(self,row,index):
        # Explicit allowlist; never pass label, reference FOL or cached answers.
        premises,question=row['premises'],row['conclusion']
        t=self.translator
        t.begin_problem(premises,question)
        candidates,joint_error=t.propose_problem()
        details=[t.process_sentence(nl,f'Premise {i}',candidates.get(f'Premise {i}'),joint_error)
                 for i,nl in enumerate(premises,1)]
        conclusion=t.process_sentence(question,'Conclusion',candidates.get('Conclusion'),joint_error)
        fols=[r['selected_fol'] for r in details]
        query=conclusion['selected_fol']
        provenance={'kind':'clover_live','sample_index':index,
            'policy':'clover_joint_proposal_verify1_semantic_repair1_ast_repair1_keep_last_connected_v5_4_connected_fewshots',
            'premises_fol':fols,'conclusion_fol':query,
            'input_sha256':hashlib.sha256(json.dumps({'premises':premises,'conclusion':question},sort_keys=True).encode()).hexdigest()}
        trace={'global_analysis':t.global_analysis,'analysis_error':t.analysis_error,
            'grounded_symbol_plan':t.symbol_plan,
            'domain_plan':t.domain_plan,'alignment_plan':t.alignment_plan,
            'joint_proposal':candidates,'joint_proposal_error':joint_error,
            'joint_proposal_diagnostic':t.joint_proposal_diagnostic,
            'premises':details,'conclusion':conclusion,'registry':t.registry.snapshot()}
        if any(not f.strip() for f in [*fols,query]):
            # Still return the diagnostic trace to the runner for checkpointing.
            return None,provenance,trace
        symbols=t.registry.snapshot()['symbols'].copy()
        for attempt in reversed(conclusion['attempts']):
            declarations=attempt.get('candidate',{}).get('declarations') or {}
            for name,spec in (declarations.get('symbols') or {}).items():
                symbols.setdefault(name,spec)
        glossary=[]
        for name,spec in symbols.items():
            if spec.get('kind') not in {'predicate','function'}:
                continue
            args=', '.join('x'+str(i) for i in range(len(spec.get('argument_roles',[]))))
            signature=f"{name}({args})"
            if spec.get('kind')=='function':signature+=' -> '+spec.get('return_sort','Entity')
            glossary.append(f"{signature} ::: {spec.get('meaning','')}")
        text='\n'.join(['Predicates:',*glossary,'Premises:',
            *[f'{fol} ::: {nl}' for fol,nl in zip(fols,premises)],
            'Conclusion:',f'{query} ::: {question}'])
        return text,provenance,trace


def module_fingerprint():
    root=Path(__file__).resolve().parent
    paths=[Path(__file__),*sorted((root/'clover_translation').rglob('*.py')),
        *sorted((root/'prompt').rglob('*.txt')),
        *sorted((root/'fewshot').rglob('*.json')),
        *sorted((root/'clover_translation/prompts').rglob('*.txt')),
        *sorted((root/'clover_translation/prompts').rglob('*.json'))]
    h=hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()
