import z3
from typing import List, Union, Literal, Dict, Set, TYPE_CHECKING
from pydantic import BaseModel, Field, PrivateAttr
import re

if TYPE_CHECKING:
    from solver.z3_context import Z3Context

def sanitize_name(name: str) -> str:
    """Sanitizes a name to be a valid Python/Z3 variable name."""
    if not name: return "v"
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if safe[0].isdigit():
        safe = "_" + safe
    return safe

class SymbolTable:
    def __init__(self):
        self.decls: Dict[str, str] = {}
        self.signatures: Dict[str, tuple] = {}
        self.builtin_sorts: Set[str] = set()
        self.precedence = {
            "Entity": 0, 
            "BoolSort()": 1, 
            "IntSort()": 2, 
            "RealSort()": 3, 
            "StringSort()": 4,
            "DateSort": 2, # Map to Int
            "TimeSort": 2  # Map to Int
        }

    def get_prec(self, s):
        return self.precedence.get(s, 0)

    def register_builtin_sort(self, sort_name: str):
        """Registers a sort like TimeSort, DateSort if not already present."""
        self.builtin_sorts.add(sort_name)

    def add_const(self, name: str, sort: str):
        safe = sanitize_name(name)
        
        if sort in ["TimeSort", "DateSort"]:
            sort = "IntSort()" # Treat as Int for declaration

        if safe in self.decls:
            existing_sort = self.signatures[safe][0]
            if self.get_prec(sort) <= self.get_prec(existing_sort):
                return # Keep existing if it's stronger or same
        
        # Idiomatic Z3 shorthands
        if sort == "IntSort()":
            self.decls[safe] = f"{safe} = Int('{name}')\n"
        elif sort == "RealSort()":
            self.decls[safe] = f"{safe} = Real('{name}')\n"
        elif sort == "StringSort()":
            self.decls[safe] = f"{safe} = String('{name}')\n"
        elif sort == "BoolSort()":
            self.decls[safe] = f"{safe} = Bool('{name}')\n"
        else:
            self.decls[safe] = f"{safe} = Const('{name}', {sort})\n"
            
        self.signatures[safe] = (sort,)

    def add_function(self, name: str, arg_sorts: List[str], return_sort: str):
        safe = sanitize_name(name)
        
        # Built-in constructor specs
        BUILTIN_CONSTRUCTOR_SPECS = {
            "MakeTime": (["IntSort()"]*5, "IntSort()"),
            "MakeDate": (["IntSort()"]*3, "IntSort()"),
        }
        
        if name in BUILTIN_CONSTRUCTOR_SPECS:
            # We handle these via arithmetic, no need for Z3 function decl
            return

        sig = tuple(arg_sorts + [return_sort])
        
        if safe in self.decls:
            existing_sig = self.signatures[safe]
            existing_arg_sorts = list(existing_sig[:-1])
            existing_return_sort = existing_sig[-1]
            
            # Check if we should upgrade return sort
            upgrade_return = False
            if self.get_prec(return_sort) > self.get_prec(existing_return_sort):
                # Don't let BoolSort override numeric sorts or vice versa unless it was Entity
                if existing_return_sort == "Entity" or (return_sort != "BoolSort()" and existing_return_sort != "BoolSort()"):
                    upgrade_return = True
            
            # Check if we should upgrade any argument sorts
            upgrade_args = False
            if len(arg_sorts) == len(existing_arg_sorts):
                for i in range(len(arg_sorts)):
                    if self.get_prec(arg_sorts[i]) > self.get_prec(existing_arg_sorts[i]):
                        existing_arg_sorts[i] = arg_sorts[i]
                        upgrade_args = True
            
            if not upgrade_return and not upgrade_args:
                return
            
            return_sort = return_sort if upgrade_return else existing_return_sort
            arg_sorts = existing_arg_sorts
            sig = tuple(arg_sorts + [return_sort])
            
        # Update or Add declaration
        self.signatures[safe] = sig
        if not arg_sorts:
            # Proposition / Constant
            if return_sort == "BoolSort()":
                self.decls[safe] = f"{safe} = Bool('{name}')\n"
            else:
                self.decls[safe] = f"{safe} = Const('{name}', {return_sort})\n"
            return

        arg_sorts_str = ", ".join(arg_sorts)
        self.decls[safe] = f"{safe} = Function('{name}', {arg_sorts_str}, {return_sort})\n"

class NumericLiteral(BaseModel):
    value: str
    def __str__(self): return self.value
    def to_dict(self): return {"node_type": "NumericLiteral", "value": self.value}
    def collect_symbols(self, st: SymbolTable): pass
    def z3expression_pass(self): return self.value
    def to_z3_expr(self, ctx: 'Z3Context'):
        if "." in self.value:
            return z3.RealVal(self.value)
        return z3.IntVal(self.value)
    def get_sort(self):
        return "RealSort()" if "." in self.value else "IntSort()"

class StringLiteral(BaseModel):
    value: str
    def __str__(self): return f'"{self.value}"'
    def to_dict(self): return {"node_type": "StringLiteral", "value": self.value}
    def collect_symbols(self, st: SymbolTable): pass
    def z3expression_pass(self): 
        return f'StringVal("{self.value}")'
    def to_z3_expr(self, ctx: 'Z3Context'):
        return z3.StringVal(self.value)
    def get_sort(self):
        return "StringSort()"

class Constant(BaseModel):
    name : str
    sort : str = "Entity"
    def __str__(self): return self.name
    def to_dict(self): return {"node_type": "Constant", "name": self.name, "sort": self.sort}
    def collect_symbols(self, st: SymbolTable):
        st.add_const(self.name, self.sort)
    def z3expression_pass(self):
        return sanitize_name(self.name)
    def to_z3_expr(self, ctx: 'Z3Context'):
        return ctx.get_or_create_const(self.name, self.sort)
    def get_sort(self):
        return self.sort

class Variable(BaseModel):
    name : str
    sort : str = "Entity"
    def __str__(self): return self.name
    def to_dict(self): return {"node_type": "Variable", "name": self.name, "sort": self.sort}
    def collect_symbols(self, st: SymbolTable):
        st.add_const(self.name, self.sort)
    def z3expression_pass(self):
        return sanitize_name(self.name)
    def to_z3_expr(self, ctx: 'Z3Context'):
        return ctx.get_or_create_const(self.name, self.sort)
    def get_sort(self):
        return self.sort

class ArithmeticOperation(BaseModel):
    left: "Term"
    right: "Term"
    operator: str # +, -, *, /
    def __str__(self): return f"({self.left} {self.operator} {self.right})"
    def to_dict(self):
        return {"node_type": "ArithmeticOperation", "operator": self.operator, "left": self.left.to_dict(), "right": self.right.to_dict()}
    def collect_symbols(self, st: SymbolTable):
        self.left.collect_symbols(st)
        self.right.collect_symbols(st)
    def z3expression_pass(self):
        return f"({self.left.z3expression_pass()} {self.operator} {self.right.z3expression_pass()})"
    def to_z3_expr(self, ctx: 'Z3Context'):
        l = self.left.to_z3_expr(ctx)
        r = self.right.to_z3_expr(ctx)
        if self.operator == "+": return l + r
        if self.operator == "-": return l - r
        if self.operator == "*": return l * r
        if self.operator == "/": return l / r
        return None
    def get_sort(self):
        l_sort = self.left.get_sort()
        r_sort = self.right.get_sort()
        if l_sort == "RealSort()" or r_sort == "RealSort()":
            return "RealSort()"
        return "IntSort()"

class FunctionApplication(BaseModel):
    name: str
    args: List["Term"]
    return_sort: str = "RealSort()" # Default to Real for arithmetic safety
    
    def __str__(self):
        args_str = ",".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"
    def to_dict(self):
        return {"node_type": "FunctionApplication", "name": self.name, "args": [a.to_dict() for a in self.args]}
    def collect_symbols(self, st: SymbolTable):
        for a in self.args: a.collect_symbols(st)
        arg_sort_list = [a.get_sort() for a in self.args]
        st.add_function(self.name, arg_sort_list, self.return_sort)

    def z3expression_pass(self):
        safe = sanitize_name(self.name)
        if not self.args:
            return safe
        args_str = ", ".join(a.z3expression_pass() for a in self.args)
        return f"{safe}({args_str})"
    
    def to_z3_expr(self, ctx: 'Z3Context'):
        # Handle Built-in Date/Time Constructors with integer packing logic
        if self.name == "MakeDate":
            # args: year, month, day
            args = [a.to_z3_expr(ctx) for a in self.args]
            y, m, d = args[0], args[1], args[2]
            return y * 10000 + m * 100 + d
        
        if self.name == "MakeTime":
            # args: h, m, s, ms, ns
            args = [a.to_z3_expr(ctx) for a in self.args]
            h, mi = args[0], args[1]
            s = args[2] if len(args) > 2 else 0
            ms = args[3] if len(args) > 3 else 0
            ns = args[4] if len(args) > 4 else 0
            # Pack as HHMMSSmmmnnn
            return h * 10000000000 + mi * 100000000 + s * 1000000 + ms * 1000 + ns

        # Normal function
        f = ctx.get_or_create_func(self.name, [a.get_sort() for a in self.args], self.return_sort)
        args = [a.to_z3_expr(ctx) for a in self.args]
        return f(*args)

    def get_sort(self):
        if self.name in ["MakeDate", "MakeTime"]:
            return "IntSort()"
        return self.return_sort

Term = Union[Constant, Variable, NumericLiteral, StringLiteral, FunctionApplication, ArithmeticOperation]

class PredicateApplication(BaseModel):
    name: str
    args: List[Term]
    def __str__(self):
        args_str = ",".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"
    def to_dict(self):
        return {"node_type": "PredicateApplication", "name": self.name, "args": [a.to_dict() for a in self.args]}
    def collect_symbols(self, st: SymbolTable):
        for a in self.args: a.collect_symbols(st)
        arg_sort_list = [a.get_sort() for a in self.args]
        st.add_function(self.name, arg_sort_list, "BoolSort()")

    def z3expression_pass(self):
        safe = sanitize_name(self.name)
        if not self.args:
            return safe
        args_str = ", ".join(a.z3expression_pass() for a in self.args)
        return f"{safe}({args_str})"

    def to_z3_expr(self, ctx: 'Z3Context'):
        f = ctx.get_or_create_func(self.name, [a.get_sort() for a in self.args], "BoolSort()")
        args = [a.to_z3_expr(ctx) for a in self.args]
        return f(*args)

class Comparison(BaseModel):
    left: Term
    right: Term
    operator: str # ==, >=, <=, >, <, !=
    
    def __str__(self): return f"{self.left} {self.operator} {self.right}"
    def to_dict(self):
        return {"node_type": "Comparison", "operator": self.operator, "left": self.left.to_dict(), "right": self.right.to_dict()}
    def collect_symbols(self, st: SymbolTable):
        self.left.collect_symbols(st)
        self.right.collect_symbols(st)
    def z3expression_pass(self):
        l_expr = self.left.z3expression_pass()
        r_expr = self.right.z3expression_pass()
        
        # Map Unicode and common math operators to Python/Z3 operators
        op_map = {
            "=": "==",
            "==": "==",
            "≥": ">=",
            ">=": ">=",
            "≤": "<=",
            "<=": "<=",
            "≠": "!=",
            "!=": "!=",
            ">": ">",
            "<": "<"
        }
        
        z3_op = op_map.get(self.operator, self.operator)
        
        if z3_op == "!=":
            return f"Not({l_expr} == {r_expr})"
            
        return f"{l_expr} {z3_op} {r_expr}"

    def to_z3_expr(self, ctx: 'Z3Context'):
        l = self.left.to_z3_expr(ctx)
        r = self.right.to_z3_expr(ctx)
        op = self.operator
        if op in ["=", "=="]: return l == r
        if op == "≠": return l != r
        if op == "≥": return l >= r
        if op == "≤": return l <= r
        if op == ">": return l > r
        if op == "<": return l < r
        return None

# Logical Sentences
class BinaryOperator(BaseModel):
    left : "Sentence"
    right : "Sentence"
    operator : Literal["And", "Or", "If", "OnlyIf", "IfAndOnlyIf", "Xor"]
    def __str__(self):
        op_map = {"And": "∧", "Or": "∨", "If": "→", "OnlyIf": "←", "IfAndOnlyIf": "↔", "Xor": "⊕"}
        return f"({self.left}) {op_map[self.operator]} ({self.right})"
    def to_dict(self):
        return {"node_type": "BinaryOperator", "operator": self.operator, "left": self.left.to_dict(), "right": self.right.to_dict()}
    def collect_symbols(self, st: SymbolTable):
        self.left.collect_symbols(st)
        self.right.collect_symbols(st)
    def z3expression_pass(self):
        l_expr = self.left.z3expression_pass()
        r_expr = self.right.z3expression_pass()
        if self.operator == "And": return f"And({l_expr}, {r_expr})"
        if self.operator == "Or": return f"Or({l_expr}, {r_expr})"
        if self.operator == "Xor": return f"Xor({l_expr}, {r_expr})"
        if self.operator == "If": return f"Implies({l_expr}, {r_expr})"
        if self.operator == "OnlyIf": return f"Implies({r_expr}, {l_expr})"
        if self.operator == "IfAndOnlyIf": return f"{l_expr} == {r_expr}"
        return ""

    def to_z3_expr(self, ctx: 'Z3Context'):
        l = self.left.to_z3_expr(ctx)
        r = self.right.to_z3_expr(ctx)
        if self.operator == "And": return z3.And(l, r)
        if self.operator == "Or": return z3.Or(l, r)
        if self.operator == "Xor": return z3.Xor(l, r)
        if self.operator == "If": return z3.Implies(l, r)
        if self.operator == "OnlyIf": return z3.Implies(r, l)
        if self.operator == "IfAndOnlyIf": return l == r
        return None

class UnaryOperator(BaseModel):
    sentence : "Sentence"
    operator : Literal["Not"]
    def __str__(self): return f"¬({self.sentence})"
    def to_dict(self):
        return {"node_type": "UnaryOperator", "operator": self.operator, "sentence": self.sentence.to_dict()}
    def collect_symbols(self, st: SymbolTable):
        self.sentence.collect_symbols(st)
    def z3expression_pass(self):
        if self.operator == "Not": return f"Not({self.sentence.z3expression_pass()})"
        return ""

    def to_z3_expr(self, ctx: 'Z3Context'):
        s = self.sentence.to_z3_expr(ctx)
        if self.operator == "Not": return z3.Not(s)
        return None

class QuantifiedSentence(BaseModel):
    quantifier : Literal["ForAll", "ThereExists"]
    variable : Variable
    sentence : "Sentence"
    def __str__(self):
        c = "∀" if self.quantifier == "ForAll" else "∃"
        return f"{c}{self.variable}. ({self.sentence})"
    def to_dict(self):
        return {"node_type": "QuantifiedSentence", "quantifier": self.quantifier, "variable": self.variable.to_dict(), "sentence": self.sentence.to_dict()}
    def collect_symbols(self, st: SymbolTable):
        self.variable.collect_symbols(st)
        self.sentence.collect_symbols(st)
    def z3expression_pass(self):
        if self.quantifier == "ForAll": return f"ForAll([{self.variable.z3expression_pass()}], {self.sentence.z3expression_pass()})"
        if self.quantifier == "ThereExists": return f"Exists([{self.variable.z3expression_pass()}], {self.sentence.z3expression_pass()})"
        return ""

    def to_z3_expr(self, ctx: 'Z3Context'):
        v = self.variable.to_z3_expr(ctx)
        s = self.sentence.to_z3_expr(ctx)
        if self.quantifier == "ForAll": return z3.ForAll([v], s)
        if self.quantifier == "ThereExists": return z3.Exists([v], s)
        return None

Sentence = Union[PredicateApplication, Comparison, ArithmeticOperation, BinaryOperator, UnaryOperator, QuantifiedSentence]

# Model Rebuilds for circular references
for m in (BinaryOperator, UnaryOperator, QuantifiedSentence, FunctionApplication, ArithmeticOperation):
    m.model_rebuild()

class RelationalLogic(BaseModel):
    original_sentence : str
    sentences : List[Sentence]
    def __str__(self):
        return "\n".join(str(s) for s in self.sentences)
    def to_dict(self):
        return {"node_type": "RelationalLogic", "original_sentence" : self.original_sentence, "sentences": [s.to_dict() for s in self.sentences]}
    
    def get_symbol_table(self) -> SymbolTable:
        st = SymbolTable()
        for s in self.sentences:
            s.collect_symbols(st)
        return st

    def z3declaration_pass(self):
        st = self.get_symbol_table()
        output = ""
        # 1. Emit builtin sorts
        for sort in sorted(st.builtin_sorts):
            if sort == "DateSort":
                output += "DateSort = IntSort()\n"
            elif sort == "TimeSort":
                output += "TimeSort = IntSort()\n"
            else:
                output += f"{sort} = DeclareSort('{sort}')\n"
        # 2. Emit variable/constant/function decls
        # Filter out built-in constructors from Z3 Function declarations
        filtered_decls = {k: v for k, v in st.decls.items() if k not in ["MakeTime", "MakeDate"]}
        output += "".join(sorted(filtered_decls.values()))
        return output

    def z3expression_pass(self):
        return "".join(f"solver.add({s.z3expression_pass()})\n" for s in self.sentences)

    def to_z3_expr(self, ctx: 'Z3Context') -> List[z3.BoolRef]:
        return [s.to_z3_expr(ctx) for s in self.sentences]

    def convert_to_z3(self):
        z3_code = "# Auto-generated Z3 code\nfrom z3 import *\n\n"
        z3_code += "Entity = DeclareSort('Entity')\nsolver = Solver()\n\n"
        z3_code += "# --- Helpers ---\n"
        z3_code += "def MakeDate(y, m, d): return y * 10000 + m * 100 + d\n"
        z3_code += "def MakeTime(h, m, s=0, ms=0, ns=0): return h * 10000000000 + m * 100000000 + s * 1000000 + ms * 1000 + ns\n\n"
        z3_code += "# --- Declarations ---\n"
        z3_code += self.z3declaration_pass() + "\n\n"
        z3_code += "# --- Expressions ---\n"       
        z3_code += self.z3expression_pass()
        z3_code += "\nprint(f'Checking satisfiability...')\nresult = solver.check()\nprint(f'Result: {result}')\n"
        return z3_code
