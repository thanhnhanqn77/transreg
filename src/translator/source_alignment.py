"""Source-only guidance for generic scope; no labels, witnesses or inferred facts."""
import re


def source_alignment_plan(context):
    premises = context['premises']
    # Episodic anaphora can designate an individual rather than a generic class.
    anaphoric = any(re.match(r'\s*(?:he|she|this person|that person|the same person)\b', s, re.I)
                    for s in premises)
    generic = []
    if not anaphoric:
        for i, text in enumerate(premises, 1):
            match = re.match(r'\s*(?:either\s+)?(?:a|an)\s+(person|animal|app)\s+'
                             r'(?:is|either|likes|eats|studies|has|does|can|will)\b', text, re.I)
            if match and not re.search(r'\b(?:named|called|yesterday|once|there was|there is)\b', text, re.I):
                generic.append({'label':f'Premise {i}', 'source':text,
                    'subject_class':match.group(1).lower(), 'interpretation':'generic_class_rule',
                    'quantifier':'forall',
                    'basis':'Shared FOLIO generic-subject convention, not an existential claim about a new witness.'})
    carriers = {g['subject_class'] for g in generic}
    carrier = None
    if len(carriers) == 1:
        category = next(iter(carriers))
        incompatible = any(re.search(r'\b(?:all|some|every|each|no)\s+'+other+r's?\b', text, re.I)
                           for text in premises for other in {'person','animal','app'}-{category})
        incompatible = incompatible or any(re.search(r'\b(?:is|was|are|were)\s+not\s+(?:a|an)\s+'
                                                     +category+r'\b', text, re.I) for text in premises)
        if not incompatible:
            carrier = {'mode':'homogeneous', 'category':category,
                'evidence':[g['source'] for g in generic], 'source':'explicit_generic_subject',
                'reason':'The generic subject names the discourse carrier under the FOLIO convention. Preserve subclass restrictions; only this carrier guard is redundant.'}
    return {'generic_rules':generic, 'carrier':carrier,
        'constraints':[
            'Generic class statements are universal restrictions, not existential witnesses.',
            'Equivalent active/passive or inverse phrasings reuse one relation and ordered roles.',
            'A query-only predicate needs a source-based distinction; absence alone does not justify deletion.',
            'Compose event locations and containment from source phrases; do not add a world-knowledge rule.',
            'Do not infer a major, organization subtype, novelty or any missing fact merely to produce a determinate answer.']}
