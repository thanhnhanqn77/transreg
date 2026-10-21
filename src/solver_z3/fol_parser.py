import enum
import json
import re

class TokenType(enum.Enum):
    FORALL = '∀'
    EXISTS = '∃'
    NOT = '¬'
    AND = '∧'
    OR = '∨'
    IMPLIES = '→'
    IFF = '↔'
    XOR = '⊕'
    ONLY_IF = '←'
    EQ = '='
    NEQ = '≠'
    GEQ = '≥'
    LEQ = '≤'
    GT = '>'
    LT = '<'
    MUL = '*'
    ADD = '+'
    SUB = '-'
    DIV = '/'
    LPAREN = '('
    RPAREN = ')'
    COMMA = ','
    IDENT = 'IDENT'
    STRING = 'STRING'

class Token:
    def __init__(self, type, value=None):
        self.type = type
        self.value = value if value is not None else type.value
    
    def __repr__(self):
        if self.type == TokenType.IDENT:
            return f"IDENT({self.value})"
        if self.type == TokenType.STRING:
            return f"STRING({self.value})"
        return self.type.name

    def to_dict(self):
        res = {"type": self.type.name}
        if self.value:
            res["value"] = self.value
        return res

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.tokens = []
        self.char_to_token = {
            '∀': TokenType.FORALL, '∃': TokenType.EXISTS, '¬': TokenType.NOT, '∧': TokenType.AND,
            '∨': TokenType.OR, '→': TokenType.IMPLIES, '↔': TokenType.IFF, '⊕': TokenType.XOR,
            '←': TokenType.ONLY_IF, '=': TokenType.EQ, '≠': TokenType.NEQ, '≥': TokenType.GEQ,
            '≤': TokenType.LEQ, '>': TokenType.GT, '<': TokenType.LT, '*': TokenType.MUL,
            '+': TokenType.ADD, '-': TokenType.SUB, '/': TokenType.DIV, '(': TokenType.LPAREN,
            ')': TokenType.RPAREN, ',': TokenType.COMMA,
        }

    def tokenize(self):
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char.isspace():
                self.pos += 1
                continue
            
            # 0. Text-based operators
            word_operator = re.match(r'(forall|exists|not|and|or|xor|iff)\b', self.text[self.pos:], re.IGNORECASE)
            if word_operator:
                keyword = word_operator.group(1).lower()
                token_type = {'forall': TokenType.FORALL, 'exists': TokenType.EXISTS,
                              'not': TokenType.NOT, 'and': TokenType.AND, 'or': TokenType.OR,
                              'xor': TokenType.XOR, 'iff': TokenType.IFF}[keyword]
                self.tokens.append(Token(token_type))
                self.pos += len(keyword)
                continue
            
            # 1. Multi-character operators
            if self.text[self.pos:self.pos+3] == '<->':
                self.tokens.append(Token(TokenType.IFF)); self.pos += 3; continue
            elif self.text[self.pos:self.pos+2] == '->':
                self.tokens.append(Token(TokenType.IMPLIES)); self.pos += 2; continue
            elif self.text[self.pos:self.pos+2] == '<-':
                self.tokens.append(Token(TokenType.ONLY_IF)); self.pos += 2; continue
            elif self.text[self.pos:self.pos+2] == '==':
                self.tokens.append(Token(TokenType.EQ)); self.pos += 2; continue
            elif self.text[self.pos:self.pos+2] == '!=':
                self.tokens.append(Token(TokenType.NEQ)); self.pos += 2; continue
            elif self.text[self.pos:self.pos+2] == '>=':
                self.tokens.append(Token(TokenType.GEQ)); self.pos += 2; continue
            elif self.text[self.pos:self.pos+2] == '<=':
                self.tokens.append(Token(TokenType.LEQ)); self.pos += 2; continue

            # 2. Strings, Identifiers, Numbers...
            if char in ["'", '"']:
                quote = char; start = self.pos; self.pos += 1
                while self.pos < len(self.text) and self.text[self.pos] != quote: self.pos += 1
                if self.pos >= len(self.text): raise Exception("Unterminated string")
                self.tokens.append(Token(TokenType.STRING, self.text[start+1:self.pos])); self.pos += 1; continue
            
            if char.isalnum() or char == '_':
                start = self.pos
                while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] in '_.'):
                    if self.text[self.pos] == '.' and not (self.pos > 0 and self.text[self.pos-1].isdigit()): break
                    self.pos += 1
                self.tokens.append(Token(TokenType.IDENT, self.text[start:self.pos])); continue

            if char in self.char_to_token:
                self.tokens.append(Token(self.char_to_token[char])); self.pos += 1; continue
            
            if char == '.': self.tokens.append(Token(TokenType.COMMA)); self.pos += 1; continue
            raise Exception(f"Unknown char: {char}")
        return self.tokens

class FOLParser:
    BUILTIN_CONSTRUCTORS = {"Time": {"arity": 5, "return_sort": "TimeSort"}, "Date": {"arity": 3, "return_sort": "DateSort"}}

    def __init__(self, tokens):
        self.tokens = tokens
        self.node_id_counter = 0
        self.precedence = {
            TokenType.IFF: 1, TokenType.ONLY_IF: 1, TokenType.IMPLIES: 2, TokenType.OR: 3, TokenType.XOR: 3,
            TokenType.AND: 4, TokenType.EQ: 5, TokenType.NEQ: 5, TokenType.GEQ: 5, TokenType.LEQ: 5, TokenType.GT: 5, TokenType.LT: 5,
            TokenType.ADD: 6, TokenType.SUB: 6, TokenType.MUL: 7, TokenType.DIV: 7
        }
        self.meanings = {
            '∀': 'forall', '∃': 'exists', '¬': 'not', '∧': 'and', '∨': 'or', '→': 'implies',
            '↔': 'iff', '⊕': 'xor', '←': 'only_if', '=': '=', '≠': '!=', '≥': '>=', '≤': '<=', '>': '>', '<': '<',
            '*': '*', '+': '+', '-': '-', '/': '/'
        }

    def _next_id(self): self.node_id_counter += 1; return self.node_id_counter

    def _get_formula_str(self, tokens):
        res = []
        for i, t in enumerate(tokens):
            val = f"'{t.value}'" if t.type == TokenType.STRING else t.value
            if i > 0:
                prev = tokens[i-1]
                if not (t.type in [TokenType.LPAREN, TokenType.RPAREN, TokenType.COMMA] or prev.type in [TokenType.LPAREN, TokenType.FORALL, TokenType.EXISTS, TokenType.NOT]):
                    res.append(" ")
            res.append(val)
        return "".join(res)

    def parse_strict(self):
        full = self._get_formula_str(self.tokens)
        depth = 0
        for t in self.tokens:
            if t.type == TokenType.LPAREN: depth += 1
            elif t.type == TokenType.RPAREN: depth -= 1
            if depth < 0: raise Exception(f"Unbalanced parens: {full}")
        if depth > 0: raise Exception(f"Missing {depth} closing parens: {full}")
        ast = self.parse(self.tokens)
        if not ast: raise Exception("Empty input")
        return ast

    def _infer_sort(self, tokens):
        if any(t.type == TokenType.STRING for t in tokens): return "StringSort()"
        has_num, has_dot = False, False
        for t in tokens:
            if t.type == TokenType.IDENT:
                if t.value.isdigit(): has_num = True
                elif '.' in t.value and t.value.replace('.','',1).isdigit(): has_num, has_dot = True, True
        if has_num: return "RealSort()" if has_dot else "IntSort()"
        if any(t.type in [TokenType.ADD, TokenType.SUB, TokenType.MUL, TokenType.DIV] for t in tokens): return "RealSort()"
        return "Entity"

    def parse(self, tokens=None, is_term=False, expected_sort=None):
        if tokens is None: tokens = self.tokens
        if not tokens: return None
        f_str = self._get_formula_str(tokens)

        # 1. Quantifiers
        if tokens[0].type in [TokenType.FORALL, TokenType.EXISTS]:
            if len(tokens) > 1 and tokens[1].type == TokenType.IDENT:
                start = 3 if (len(tokens) > 2 and tokens[2].type == TokenType.COMMA) else 2
                return {"id": self._next_id(), "type": "Quantifier", "meaning": self.meanings[tokens[0].type.value],
                        "quantifier": tokens[0].type.value, "variable": tokens[1].value, "formula": f_str,
                        "body": self.parse(tokens[start:], is_term=False)}
            raise Exception("Quantifier needs variable")

        # 2. Binary Operators
        op_idx = self._find_main_op(tokens)
        if op_idx != -1:
            t_op, left_t, right_t = tokens[op_idx], tokens[:op_idx], tokens[op_idx+1:]
            if t_op.type in [TokenType.EQ, TokenType.NEQ, TokenType.GEQ, TokenType.LEQ, TokenType.GT, TokenType.LT]:
                s_l, s_r = self._infer_sort(left_t), self._infer_sort(right_t)
                prio = {"RealSort()": 4, "IntSort()": 3, "StringSort()": 2, "Entity": 1}
                inf_s = s_l if prio.get(s_l,0) >= prio.get(s_r,0) else s_r
                if inf_s == "Entity" and t_op.type not in [TokenType.EQ, TokenType.NEQ]: inf_s = "RealSort()"
                return {"id": self._next_id(), "type": "Comparison", "operator": t_op.type.value, "formula": f_str,
                        "left": self.parse(left_t, is_term=True, expected_sort=inf_s),
                        "right": self.parse(right_t, is_term=True, expected_sort=inf_s)}
            if t_op.type in [TokenType.ADD, TokenType.SUB, TokenType.MUL, TokenType.DIV]:
                res_s = self._infer_sort(tokens); res_s = "RealSort()" if res_s == "Entity" else res_s
                return {"id": self._next_id(), "type": "BinaryOp", "meaning": "arithmetic", "operator": t_op.type.value,
                        "result_sort": res_s, "formula": f_str, "left": self.parse(left_t, is_term=True, expected_sort=res_s),
                        "right": self.parse(right_t, is_term=True, expected_sort=res_s)}
            return {"id": self._next_id(), "type": "BinaryOp", "meaning": self.meanings[t_op.type.value], "operator": t_op.type.value,
                    "formula": f_str, "left": self.parse(left_t, is_term=False), "right": self.parse(right_t, is_term=False)}

        # 3. Unary
        if tokens[0].type == TokenType.NOT:
            return {"id": self._next_id(), "type": "UnaryOp", "meaning": "not", "operator": "¬", "formula": f_str,
                    "operand": self.parse(tokens[1:], is_term=False)}
        if tokens[0].type == TokenType.SUB:
            return {"id": self._next_id(), "type": "BinaryOp", "meaning": "arithmetic", "operator": "-", "formula": f_str,
                    "left": {"id": self._next_id(), "type": "NumConstant", "name": "0", "sort": "IntSort()"},
                    "right": self.parse(tokens[1:], is_term=True, expected_sort="RealSort()")}

        # 4. Parens
        if tokens[0].type == TokenType.LPAREN and tokens[-1].type == TokenType.RPAREN and self._is_matching_parens(tokens):
            return self.parse(tokens[1:-1], is_term=is_term, expected_sort=expected_sort)

        # 5. String/Atom/Var
        if tokens[0].type == TokenType.STRING:
            return {"id": self._next_id(), "type": "StringConstant", "name": tokens[0].value, "formula": f_str}
        
        if tokens[0].type == TokenType.IDENT:
            if len(tokens) > 1 and tokens[1].type == TokenType.LPAREN:
                depth, r_idx = 0, -1
                for i in range(1, len(tokens)):
                    if tokens[i].type == TokenType.LPAREN: depth += 1
                    elif tokens[i].type == TokenType.RPAREN:
                        depth -= 1
                        if depth == 0: r_idx = i; break
                if r_idx == len(tokens) - 1:
                    name, args = tokens[0].value, self._parse_args(tokens[2:-1])
                    if name in self.BUILTIN_CONSTRUCTORS:
                        spec = self.BUILTIN_CONSTRUCTORS[name]
                        return {"id": self._next_id(), "type": "BuiltinConstructor", "name": name, "args": args,
                                "return_sort": spec["return_sort"], "formula": f_str}
                    node = {"id": self._next_id(), "type": "Atom", "meaning": "function" if is_term else "predicate",
                            "name": name, "formula": f_str, "args": args}
                    node["return_sort"] = expected_sort if is_term else "BoolSort()"
                    return node
            
            val = tokens[0].value
            if val.isdigit() or ('.' in val and val.replace('.','',1).isdigit()):
                return {"id": self._next_id(), "type": "NumConstant", "name": val, "formula": f_str, "sort": "RealSort()" if "." in val else "IntSort()"}
            
            if not is_term and len(tokens) == 1:
                return {"id": self._next_id(), "type": "Atom", "meaning": "predicate", "name": val, "formula": f_str, "args": [], "return_sort": "BoolSort()"}
            
            if is_term and len(tokens) == 1:
                return {"id": self._next_id(), "type": "Var", "name": val, "formula": f_str, "sort": expected_sort or "Entity"}
        
        raise Exception(f"Unexpected tokens in {f_str}: {tokens}")

    def _find_main_op(self, tokens):
        idx, min_p, d = -1, float('inf'), 0
        for i, t in enumerate(tokens):
            if t.type == TokenType.LPAREN: d += 1
            elif t.type == TokenType.RPAREN: d -= 1
            elif d == 0 and t.type in self.precedence:
                if t.type == TokenType.SUB and (i==0 or tokens[i-1].type in [TokenType.LPAREN, TokenType.COMMA] or tokens[i-1].type in self.precedence): continue
                p = self.precedence[t.type]
                if t.type == TokenType.IMPLIES:
                    if p < min_p: min_p, idx = p, i
                elif p <= min_p: min_p, idx = p, i
        return idx

    def _is_matching_parens(self, tokens):
        d = 0
        for i, t in enumerate(tokens):
            if t.type == TokenType.LPAREN: d += 1
            elif t.type == TokenType.RPAREN: d -= 1
            if d == 0 and i < len(tokens)-1: return False
        return d == 0

    def _parse_args(self, tokens):
        if not tokens: return []
        args, curr, d = [], [], 0
        for t in tokens:
            if t.type == TokenType.COMMA and d == 0: args.append(self.parse(curr, True)); curr = []
            else:
                if t.type == TokenType.LPAREN: d += 1
                elif t.type == TokenType.RPAREN: d -= 1
                curr.append(t)
        if curr: args.append(self.parse(curr, True))
        return args

class UnionFind:
    def __init__(self):
        self.parent, self.sorts = {}, {}
        self.prio = {"Entity": 0, "BoolSort()": 1, "IntSort()": 2, "RealSort()": 3, "StringSort()": 4}

    def find(self, i):
        if i not in self.parent: self.parent[i] = i; return i
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i]); return self.parent[i]

    def unify(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj
            self.sorts[rj] = self._stronger(self.sorts.get(ri, "Entity"), self.sorts.get(rj, "Entity"))

    def set_sort(self, i, s):
        r = self.find(i); self.sorts[r] = self._stronger(self.sorts.get(r, "Entity"), s)

    def get_sort(self, i): return self.sorts.get(self.find(i), "Entity")

    def _stronger(self, s1, s2):
        if s1 == "BoolSort()": return s2
        if s2 == "BoolSort()": return s1
        return s1 if self.prio.get(s1, 0) >= self.prio.get(s2, 0) else s2

class SortRefiner:
    def __init__(self):
        self.sort_uf, self.var_uf = UnionFind(), UnionFind()
        self.scopes, self.id_cnt = [{}], 0
        self.pred_sigs, self.orig_names, self.unique_names, self.used = {}, {}, {}, set()

    def _new_id(self): self.id_cnt -= 1; return self.id_cnt

    def refine(self, ast):
        if not ast: return ast
        self._collect(ast); self._apply(ast); return ast

    def _collect(self, node):
        if not node: return
        nid, ntype = node['id'], node.get("type")
        if ntype == "Var":
            name = node["name"]
            for s in reversed(self.scopes):
                if name in s: vid = s[name]; break
            else: vid = self._new_id(); self.scopes[0][name] = vid
            self.var_uf.unify(nid, vid); root = self.var_uf.find(nid); self.orig_names[root] = name
            if "sort" in node: self.sort_uf.set_sort(nid, node["sort"])
            self.sort_uf.unify(nid, vid)
        elif ntype == "Quantifier":
            self.scopes.append({}); name = node["variable"]; qid = self._new_id(); self.scopes[-1][name] = qid
            self.var_uf.unify(nid, qid); root = self.var_uf.find(nid); self.orig_names[root] = name
            if "sort" in node: self.sort_uf.set_sort(nid, node["sort"])
            self.sort_uf.unify(nid, qid); self._collect(node["body"]); self.scopes.pop()
        elif ntype == "Atom":
            name = node["name"]
            if name not in self.pred_sigs: self.pred_sigs[name] = [self._new_id() for _ in node.get("args", [])]
            sig = self.pred_sigs[name]
            for i, arg in enumerate(node.get("args", [])):
                self._collect(arg)
                if arg and i < len(sig):
                    self.sort_uf.unify(arg['id'], sig[i])
                    if arg.get("type") == "NumConstant": self.sort_uf.set_sort(sig[i], arg.get("sort", "IntSort()"))
            self.sort_uf.set_sort(nid, node.get("return_sort", "BoolSort()"))
        elif ntype == "NumConstant": self.sort_uf.set_sort(nid, node.get("sort", "IntSort()"))
        elif ntype == "Comparison":
            self._collect(node["left"]); self._collect(node["right"])
            if node["left"] and node["right"]: self.sort_uf.unify(node["left"]['id'], node["right"]['id'])
            self.sort_uf.set_sort(nid, "BoolSort()")
        elif ntype == "BinaryOp":
            self._collect(node["left"]); self._collect(node["right"])
            if node.get("meaning") == "arithmetic" and node["left"] and node["right"]:
                self.sort_uf.unify(node["left"]['id'], node["right"]['id']); self.sort_uf.unify(nid, node["left"]['id'])
            else: self.sort_uf.set_sort(nid, "BoolSort()")
        elif ntype == "UnaryOp": self._collect(node["operand"]); self.sort_uf.set_sort(nid, "BoolSort()")
        for k in ["body", "left", "right", "operand"]:
            if k in node: self._collect(node[k])
        if "args" in node:
            for a in node["args"]: self._collect(a)

    def _apply(self, node):
        if not node: return
        nid, ntype = node['id'], node.get("type")
        res_s = self.sort_uf.get_sort(nid)
        v_root = self.var_uf.find(nid)
        if v_root in self.orig_names and v_root not in self.unique_names:
            o = self.orig_names[v_root]; n, c = o, 1
            while n in self.used: n = f"{o}_{c}"; c += 1
            self.unique_names[v_root] = n; self.used.add(n)
        if ntype == "Var":
            if v_root in self.unique_names: node["name"] = self.unique_names[v_root]
            node["sort"] = res_s
        elif ntype == "Quantifier":
            if v_root in self.unique_names: node["variable"] = self.unique_names[v_root]
            node["sort"] = res_s; self._apply(node["body"]); return
        elif ntype == "Atom": node["return_sort"] = res_s
        elif ntype in ["NumConstant", "StringConstant", "BinaryOp", "Comparison", "UnaryOp"]:
            for k in ["sort", "result_sort", "return_sort"]:
                if k in node: node[k] = res_s
        for k in ["body", "left", "right", "operand"]:
            if k in node: self._apply(node[k])
        if "args" in node:
            for a in node["args"]: self._apply(a)

def parse_fol(text):
    lexer = Lexer(text); tokens = lexer.tokenize()
    if not tokens: return None
    parser = FOLParser(tokens); ast = parser.parse_strict()
    refiner = SortRefiner(); return refiner.refine(ast)

if __name__ == "__main__":
    tc = "¬JobMarketWeak"
    try: print(json.dumps(parse_fol(tc), indent=2, ensure_ascii=False))
    except Exception as e: print(f"Error: {e}")
