"""Answer-independent canonicalization of classically equivalent formulae."""
import copy
import re

from .translation_ast import parse_strict,walk


def normalize_prefix_connectives(text):
    """Convert reserved Boolean constructors to infix without touching strings."""
    changes=[];output=[];i=0
    while i<len(text):
        if text[i] in {'"',"'"}:
            quote=text[i];j=i+1
            while j<len(text):
                if text[j]=='\\':j+=2;continue
                if text[j]==quote:j+=1;break
                j+=1
            output.append(text[i:j]);i=j;continue
        match=re.match(r'(xor|and|or|not|implies|iff)\s*\(',text[i:],re.I) if i==0 or not (text[i-1].isalnum() or text[i-1]=='_') else None
        if not match:
            output.append(text[i]);i+=1;continue
        start=i+match.end();depth=1;quote=None;j=start;parts=[];part_start=start
        while j<len(text):
            char=text[j]
            if quote:
                if char=='\\':j+=2;continue
                if char==quote:quote=None
            elif char in {'"',"'"}:quote=char
            elif char=='(':depth+=1
            elif char==')':
                depth-=1
                if depth==0:
                    parts.append(text[part_start:j].strip());break
            elif char==',' and depth==1:
                parts.append(text[part_start:j].strip());part_start=j+1
            j+=1
        op=match.group(1).casefold()
        supported=depth==0 and all(parts) and ((op=='not' and len(parts)==1) or (op in {'and','or'} and len(parts)>=2) or (op in {'xor','implies','iff'} and len(parts)==2))
        if not supported:
            output.append(text[i]);i+=1;continue
        args=[]
        for part in parts:
            converted,subchanges=normalize_prefix_connectives(part)
            args.append(converted);changes+=subchanges
        replacement='not ('+args[0]+')' if op=='not' else '('+(' '+{'implies':'->','iff':'<->'}.get(op,op)+' ').join(args)+')'
        if replacement!=text[i:j+1]:changes.append('reserved_prefix_connective_to_infix')
        output.append(replacement);i=j+1
    return ''.join(output),changes


def canonicalize(fol, declarations, is_question=False, domain_plan=None, generic_rule=False):
    fol,changes=normalize_prefix_connectives(fol)
    try:
        ast=parse_strict(fol)
    except Exception:
        return fol,declarations,changes
    if generic_rule and ast.get('type') == 'Quantifier' and ast.get('meaning') == 'exists':
        ast = copy.deepcopy(ast)
        ast['meaning'] = 'forall'
        changes.append('source_policy_generic_subject_as_universal')
    declarations=copy.deepcopy(declarations)
    symbols=(declarations or {}).get('symbols',{})
    used={n['name'] for n in walk(ast) if n.get('type')=='Var' and re.fullmatch(r'[a-z]',n.get('name',''))}
    names={}
    plan=domain_plan or {}
    category=str(plan.get('category','')).strip().casefold()
    domain_names={category,'is'+category} if plan.get('mode')=='homogeneous' and re.fullmatch(r'[a-z]+',category) else set()

    def domain_guard(node):
        return node.get('type')=='Atom' and node.get('meaning')=='predicate' and node.get('name','').casefold() in domain_names and len(node['args'])==1 and node['args'][0].get('type')=='Var' and node['args'][0]['name'] in names

    def emit(n,expected=None):
        kind=n['type']
        if kind=='Quantifier':
            old=n['variable'];prior=names.get(old)
            variable=old
            if not re.fullmatch(r'[a-z]',old):
                variable=next((letter for letter in 'abcdefghijklmnopqrstuvwxyz' if letter not in used),None)
                if variable is None:raise ValueError('No capture-free single-letter variable available')
                used.add(variable)
                changes.append('capture_avoiding_alpha_renaming')
            names[old]=variable
            body=emit(n['body'])
            if prior is None:names.pop(old,None)
            else:names[old]=prior
            return f"{n['meaning']} {variable} ({body})"
        if kind=='Atom':
            spec=symbols.get(n['name'],{})
            sorts=spec.get('arg_sorts',[])
            args=[emit(a,sorts[i] if i<len(sorts) else None) for i,a in enumerate(n['args'])]
            return n['name']+'('+', '.join(args)+')'
        if kind=='UnaryOp':
            child=n['operand']
            if child.get('type')=='UnaryOp' and child.get('meaning')=='not':
                changes.append('double_negation_elimination')
                return emit(child['operand'])
            return 'not '+emit(child) if child.get('type')=='Atom' else 'not ('+emit(child)+')'
        if kind=='BinaryOp' and n.get('meaning')=='implies':
            left,right=n['left'],n['right']
            if domain_guard(left):
                changes.append('homogeneous_subject_domain_guard_elision')
                return emit(right)
            if all(a.get('type')=='UnaryOp' and a.get('meaning')=='not' for a in [left,right]):
                changes.append('negative_implication_contraposition')
                return '('+emit(right['operand'])+' -> '+emit(left['operand'])+')'
            if is_question and left.get('type')=='UnaryOp' and left.get('meaning')=='not':
                changes.append('negative_antecedent_query_as_disjunction')
                return '('+emit(left['operand'])+' or '+emit(right)+')'
            return '('+emit(left)+' -> '+emit(right)+')'
        if kind=='BinaryOp' and n.get('meaning')!='arithmetic':
            if n.get('meaning')=='and':
                if domain_guard(n['left']):
                    changes.append('homogeneous_subject_domain_guard_elision')
                    return emit(n['right'])
                if domain_guard(n['right']):
                    changes.append('homogeneous_subject_domain_guard_elision')
                    return emit(n['left'])
            op={'and':'and','or':'or','xor':'xor','iff':'<->','only_if':'->'}.get(n['meaning'])
            if op is None:raise ValueError('Unsupported logical operator')
            return '('+emit(n['left'])+' '+op+' '+emit(n['right'])+')'
        if kind in {'Comparison','BinaryOp'}:
            operator=n['operator']
            # A real-valued function forces real literal serialization; numerical
            # values remain unchanged. Do not change quantified integer domains.
            left_spec=symbols.get(n['left'].get('name'),{})
            right_spec=symbols.get(n['right'].get('name'),{})
            hint='RealSort()' if any(s.get('return_sort')=='RealSort()' for s in [left_spec,right_spec]) else None
            op={'≠':'!=','≥':'>=','≤':'<='}.get(operator,operator)
            return '('+emit(n['left'],hint)+' '+op+' '+emit(n['right'],hint)+')'
        if kind=='NumConstant' and expected=='RealSort()':
            literal=n['formula']
            if re.fullmatch(r'-?\d+',literal):
                changes.append('real_literal_serialization')
                return literal+'.0'
        if kind=='Var':return names.get(n['name'],n['formula'])
        return n['formula']

    try:
        result=emit(ast)
        parse_strict(result)
        return result,declarations,sorted(set(changes))
    except Exception:
        return fol,declarations,[]


def grounded_symbol_plan(analysis,context):
    """Only schemas with verbatim evidence from source NL guide the translator."""
    accepted={}
    texts=list(context['premises'])+[context['question']]
    for spec in analysis.get('symbol_plan',[]) if isinstance(analysis.get('symbol_plan'),list) else []:
        if not isinstance(spec,dict):continue
        name=spec.get('name','')
        roles=spec.get('argument_roles');sorts=spec.get('arg_sorts');evidence=spec.get('evidence')
        if not re.fullmatch(r'[A-Z][A-Za-z0-9_]*',name):continue
        if spec.get('kind') not in {'predicate','function'} or not isinstance(roles,list) or not isinstance(sorts,list) or len(roles)!=len(sorts):continue
        if not isinstance(evidence,list) or not evidence or not all(isinstance(span,str) and span.strip() and any(span in text for text in texts) for span in evidence):continue
        if not isinstance(spec.get('meaning'),str) or not spec['meaning'].strip():continue
        allowed={'Entity','IntSort()','RealSort()','StringSort()','BoolSort()'}
        # The backend exposes Entity for semantic categories such as Season or
        # Office; retain that category as metadata, not an unsupported Z3 sort.
        semantic_types=list(sorts)
        sorts=[s if s in allowed else 'Entity' if isinstance(s,str) and re.fullmatch(r'[A-Z][A-Za-z0-9_]*',s) else s for s in sorts]
        if any(s not in allowed-{'BoolSort()'} for s in sorts) or spec.get('return_sort') not in allowed:continue
        if any(not isinstance(role,str) or not role.strip() for role in roles):continue
        accepted[name]=dict(copy.deepcopy(spec),arg_sorts=sorts,semantic_arg_types=semantic_types)
    return accepted


def retain_positive_temporal_core(previous, repaired, feedback, registry):
    """Keep a grounded positive core when its sole defect was missing time.

    This is compositional translation of a qualified assertion, not an added
    inference. Never project a negative, quantified, disjunctive or conditional
    assertion, and never identify a new existential witness.
    """
    review=(feedback or {}).get('verifier_feedback') or {}
    issues=review.get('issues',[])
    if not issues or not all(re.search(r'year|date|temporal|time|season',str(i),re.I) for i in issues):
        return repaired,[]
    try:
        old=parse_strict(previous);new=parse_strict(repaired)
    except Exception:
        return repaired,[]
    def positives(node):
        if node.get('type')=='Atom' and node.get('meaning')=='predicate':
            if all(a.get('type') in {'Var','NumConstant','StringConstant'} and
                   not (a.get('type')=='Var' and re.fullmatch(r'[a-z]',a['name'])) for a in node['args']):
                return [node]
        if node.get('type')=='BinaryOp' and node.get('meaning')=='and':
            return positives(node['left'])+positives(node['right'])
        return []
    old_atoms=positives(old);new_atoms=positives(new)
    additions=[]
    for a in old_atoms:
        if a['name'] not in registry.get('symbols',{}):continue
        names=[x['name'] for x in new_atoms]
        if a['name'] in names:continue
        for b in new_atoms:
            root=re.sub(r'(?:InYear|InSeason|AtTime|DuringPeriod|InEvent)$','',b['name'])
            if root==a['name'] and len(b['args'])>len(a['args']) and all(x['formula']==y['formula'] for x,y in zip(a['args'],b['args'])):
                additions.append(a['formula']);break
    if not additions:return repaired,[]
    return '('+' and '.join([*additions,repaired])+')',['retained_source_positive_temporal_core']


def grounded_domain_plan(analysis,context):
    plan=analysis.get('domain_plan',{})
    if not isinstance(plan,dict) or plan.get('mode') not in {'homogeneous','mixed'}:return {}
    evidence=plan.get('evidence')
    texts=[*context['premises'],context['question']]
    if not isinstance(evidence,list) or not evidence or not all(isinstance(s,str) and s.strip() and any(s in text for text in texts) for s in evidence):return {}
    return copy.deepcopy(plan)


def infer_subject_domain(context):
    """Conservative source-based FOLIO subject-carrier convention, no new facts."""
    evidence=[];classes=set()
    def singular(word):
        return word[:-3]+'y' if word.endswith('ies') else word[:-1] if word.endswith('s') and not word.endswith('ss') else word
    for text in context['premises']:
        for match in re.finditer(r'\b(?:all|some|every|each|no)\s+([a-z]+)\b',text,re.I):
            noun=match.group(1).casefold()
            if noun in {'one','people','person','persons','thing','things','of','other'}:return {}
            tail=text[match.end():].lstrip()
            if not re.match(r'(?:who|that|which|is|are|was|were|has|have|in|with|produced|do|does|can|will)\b',tail,re.I):return {}
            classes.add(singular(noun));evidence.append(text)
        for match in re.finditer(r'\b(?:is|are|was|were)\s+(?:a|an)\s+([a-z]+)\b',text,re.I):
            classes.add(singular(match.group(1).casefold()))
        if re.search(r'\b(?:is|are|was|were)\s+not\s+(?:a|an)\s+[a-z]+\b',text,re.I):return {}
    if len(classes)!=1 or len(set(evidence))<2:return {}
    category=next(iter(classes))
    return {'mode':'homogeneous','category':category,'evidence':list(dict.fromkeys(evidence)),
            'source':'repeated_quantified_subject_classes',
            'reason':'All explicitly quantified subject classes agree; no other explicit positive/negative class membership is stated. Apply the FOLIO subject-domain convention consistently, without asserting named taxonomy facts.'}
