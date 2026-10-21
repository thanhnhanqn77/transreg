"""Structural gate adapted from task1-tmp's FOL AST and scope validators.

The solver's parser is byte-identical to task1-tmp/logic/parser/fol_parser.py.
Keep it unchanged; tighten its permissive leaf/argument parsing here instead.
"""
import re
import copy

try:
    from ..solver_z3.fol_parser import FOLParser, Lexer, TokenType
    from ..solver_z3 import MultiSentenceCompiler, Z3Context
except (ImportError, ValueError):
    from solver_z3.fol_parser import FOLParser, Lexer, TokenType
    from solver_z3 import MultiSentenceCompiler, Z3Context


class StrictParser(FOLParser):
    def parse(self, tokens=None, is_term=False, expected_sort=None):
        tokens = self.tokens if tokens is None else tokens
        if not tokens:
            raise ValueError("Missing operand or quantifier body")
        node = super().parse(tokens, is_term, expected_sort)
        if node['type'] in {'StringConstant', 'NumConstant'} and len(tokens) != 1:
            raise ValueError("Unconsumed tokens after literal")
        return node

    def _parse_args(self, tokens):
        if tokens and tokens[-1].type == TokenType.COMMA:
            raise ValueError("Trailing comma in arguments")
        return super()._parse_args(tokens)


def walk(node):
    yield node
    for key in ('body', 'left', 'right', 'operand'):
        if isinstance(node.get(key), dict):
            yield from walk(node[key])
    for arg in node.get('args', []):
        yield from walk(arg)


def fingerprint(node):
    """Ignore parser ids and formatting, preserving argument order and scope."""
    return {k: ([fingerprint(x) for x in v] if isinstance(v, list) else
                fingerprint(v) if isinstance(v, dict) else v)
            for k, v in node.items() if k not in {'id', 'formula'}}


def parse_strict(fol):
    return StrictParser(Lexer(fol).tokenize()).parse_strict()


def normalize_declarations(fol, declarations, registry=None, warnings=None):
    """Recover optional annotations from the formula; never repair the formula itself."""
    if warnings is None:
        warnings = []
    registry = registry or {'symbols': {}, 'entities': {}, 'aliases': [], 'sources': []}
    ast = parse_strict(fol)
    if not isinstance(declarations, dict):
        warnings.append('Missing/malformed optional declarations; deriving from AST')
        declarations = {}
    normalized = {'symbols': {}, 'entities': {}}
    for category in normalized:
        if category not in declarations:
            warnings.append(f'Missing optional {category} field; defaulted to {{}}')
        elif not isinstance(declarations[category], dict):
            warnings.append(f'Ignored malformed optional {category} metadata')
    symbols = declarations.get('symbols') if isinstance(declarations.get('symbols'), dict) else {}
    entities = declarations.get('entities') if isinstance(declarations.get('entities'), dict) else {}

    def visit(node, bound):
        if node['type'] == 'Quantifier':
            visit(node['body'], bound | {node['variable']})
            return
        if node['type'] == 'Atom':
            name = node['name']
            record = symbols.get(name, registry['symbols'].get(name))
            if not isinstance(record, dict):
                warnings.append('Derived missing symbol metadata from AST: ' + name)
                record = {'kind': node['meaning'], 'argument_roles': [f'arg{i+1}' for i in range(len(node['args']))],
                          'arg_sorts': [a.get('sort', 'Entity') for a in node['args']],
                          'return_sort': node.get('return_sort') or 'Entity'}
            record = copy.deepcopy(record)
            if record.get('kind') == 'symbol':
                record['kind'] = node['meaning']
            if not isinstance(record.get('meaning'), str) or not record['meaning'].strip():
                record['meaning'] = registry['symbols'].get(name, {}).get('meaning', name)
                warnings.append('Defaulted optional symbol description: ' + name)
            normalized['symbols'][name] = record
        if node['type'] == 'Var' and node['name'] not in bound:
            name = node['name']
            # Preserve free-variable detection in validate().
            if not re.fullmatch(r'[a-z](?:_\d+)?', name):
                record = entities.get(name, registry['entities'].get(name))
                if not isinstance(record, dict):
                    warnings.append('Derived optional entity metadata from AST: ' + name)
                    record = {'sort': node.get('sort', 'Entity')}
                record = copy.deepcopy(record)
                record.setdefault('sort', node.get('sort', 'Entity'))
                if not isinstance(record.get('meaning'), str):
                    record['meaning'] = name
                normalized['entities'][name] = record
        for key in ('left', 'right', 'operand'):
            if isinstance(node.get(key), dict):
                visit(node[key], bound)
        for arg in node.get('args', []):
            visit(arg, bound)
    visit(ast, set())
    for category, records in (('symbols', symbols), ('entities', entities)):
        for name in records.keys() - normalized[category].keys():
            warnings.append(f'Ignored unused optional {category} entry: {name}')
    return normalized


def validate(fol, declarations, registry=None, accepted_fols=()):
    """No entailment or consistency checks: only structure, signatures and typing."""
    registry = registry or {'symbols': {}, 'entities': {}, 'aliases': [], 'sources': []}
    declarations = normalize_declarations(fol, declarations, registry, [])
    ast = parse_strict(fol)
    if not isinstance(declarations, dict):
        raise ValueError('declarations must be an object')
    symbols = declarations.get('symbols', {})
    entities = declarations.get('entities', {})
    if not isinstance(symbols, dict) or not isinstance(entities, dict):
        raise ValueError('symbols and entities must be objects')
    used_symbols, used_entities = set(), set()
    numeric = {'IntSort()', 'RealSort()'}
    valid_sorts = numeric | {'Entity', 'StringSort()', 'BoolSort()'}
    for name, spec in symbols.items():
        if not re.fullmatch(r'[A-Z][A-Za-z0-9_]*', name) or not isinstance(spec, dict):
            raise ValueError('Invalid symbol declaration')
        if spec.get('return_sort') not in valid_sorts or any(s not in valid_sorts - {'BoolSort()'} for s in spec.get('arg_sorts', [])):
            raise ValueError('Unsupported sort declaration: ' + name)
    for name, spec in entities.items():
        if (not re.fullmatch(r'[A-Z][A-Za-z0-9_]*', name) or not isinstance(spec, dict) or
                spec.get('sort') not in valid_sorts - {'BoolSort()'} or not isinstance(spec.get('meaning'), str)):
            raise ValueError('Invalid entity declaration: ' + name)

    def compatible(a, b):
        return a == b or {a, b} <= numeric

    def visit(node, bound, term=False, expected=None):
        kind = node['type']
        if kind == 'Quantifier':
            if term or node['meaning'] not in {'forall', 'exists'} or not re.fullmatch(r'[a-z]', node['variable']):
                raise ValueError('Invalid quantifier structure; use single lowercase variables')
            visit(node['body'], dict(bound, **{node['variable']: [None]}))
            return 'BoolSort()'
        if kind == 'Var':
            name = node['name']
            if not term:
                raise ValueError('Term in formula position')
            if name in bound:
                prior = bound[name][0]
                if prior and expected and not compatible(prior, expected):
                    raise ValueError('Incompatible bound variable sorts: ' + name)
                bound[name][0] = prior or expected or node.get('sort', 'Entity')
                return bound[name][0]
            if re.fullmatch(r'[a-z](?:_\d+)?', name):
                raise ValueError('Free variable: ' + name)
            if name not in entities:
                raise ValueError('Undeclared constant: ' + name)
            used_entities.add(name)
            return entities[name]['sort']
        if kind in {'NumConstant', 'StringConstant'}:
            if not term:
                raise ValueError('Literal in formula position')
            return 'StringSort()' if kind == 'StringConstant' else node['sort']
        if kind == 'Atom':
            name = node['name']
            if name not in symbols:
                raise ValueError('Undeclared predicate/function: ' + name)
            spec = symbols[name]
            expected_kind = 'function' if term else 'predicate'
            roles, sorts = spec.get('argument_roles'), spec.get('arg_sorts')
            if (spec.get('kind') != expected_kind or not isinstance(roles, list) or
                    not isinstance(sorts, list) or len(roles) != len(node['args']) or len(sorts) != len(roles)):
                raise ValueError('Kind/arity/ordered role signature mismatch: ' + name)
            if not isinstance(spec.get('meaning'), str) or not spec['meaning'].strip():
                raise ValueError('Missing symbol meaning: ' + name)
            if any(not isinstance(r, str) or not r.strip() for r in roles):
                raise ValueError('Invalid argument roles: ' + name)
            result = spec.get('return_sort')
            if (not term and result != 'BoolSort()') or (term and result == 'BoolSort()'):
                raise ValueError('Invalid return sort: ' + name)
            for arg, sort in zip(node['args'], sorts):
                actual = visit(arg, bound, True, sort)
                if not compatible(actual, sort):
                    raise ValueError(f'Argument sort mismatch: {name}: {actual} vs {sort}')
            old = registry['symbols'].get(name)
            if old and any(spec.get(k) != old.get(k) for k in ('kind', 'argument_roles', 'arg_sorts', 'return_sort')):
                raise ValueError('Accepted signature/argument order changed: ' + name)
            used_symbols.add(name)
            return result
        if kind == 'UnaryOp':
            if term or node['meaning'] != 'not':
                raise ValueError('Invalid unary operator')
            visit(node['operand'], bound)
            return 'BoolSort()'
        if kind == 'BinaryOp' and node.get('meaning') != 'arithmetic':
            if term or node['meaning'] not in {'and', 'or', 'xor', 'implies', 'only_if', 'iff'}:
                raise ValueError('Invalid logical operator')
            visit(node['left'], bound)
            visit(node['right'], bound)
            return 'BoolSort()'
        if kind in {'Comparison', 'BinaryOp'}:
            arithmetic = kind == 'BinaryOp'
            if arithmetic != term:
                raise ValueError('Formula/term structure mismatch')
            hint = node['left'].get('sort') if arithmetic or node['operator'] not in {'=', '≠'} else None
            left = visit(node['left'], bound, True, hint)
            right = visit(node['right'], bound, True, left)
            if not compatible(left, right):
                raise ValueError('Incompatible comparison/arithmetic sorts')
            if (arithmetic or node['operator'] not in {'=', '≠'}) and not {left, right} <= numeric:
                raise ValueError('Numeric operator on nonnumeric terms')
            return left if arithmetic else 'BoolSort()'
        raise ValueError('Unsupported AST structure: ' + kind)

    # Validate the unrefined AST first: solver sort inference must not hide mismatches.
    visit(ast, {})
    if set(symbols) != used_symbols or set(entities) != used_entities:
        raise ValueError('Declarations must describe exactly the symbols/constants in this sentence')
    for name in used_entities:
        if name in registry['entities'] and entities[name]['sort'] != registry['entities'][name]['sort']:
            raise ValueError('Accepted entity sort changed: ' + name)
    # Compile through the actual unchanged adapter, but never ask Z3 about truth.
    compiler = MultiSentenceCompiler(Z3Context(), strict=True)
    for previous in accepted_fols:
        compiler.add_sentence(previous)
    compiler.add_sentence(fol)
    compiled = compiler.compile_question(fol, label='SyntaxCheck')
    if compiled.z3_expr is None:
        raise ValueError('Solver adapter could not compile formula')
    # Check all preceding premises too; compiling only the current question can
    # otherwise conceal a cross-sentence numeric sort conflict.
    compiler.compile()
    # Declarations are guidance, not executable solver declarations. Reject types
    # the unchanged parser cannot infer rather than silently changing their meaning.
    for name, spec in symbols.items():
        actual = compiler.global_st.signatures[name]
        wanted = [*spec['arg_sorts'], spec['return_sort']]
        if len(actual) != len(wanted) or any(not compatible(a, b) for a, b in zip(actual, wanted)):
            raise ValueError('Declared signature differs from solver inference: ' + name)
    return ast
