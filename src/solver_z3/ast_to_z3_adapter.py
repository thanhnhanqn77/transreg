import json
import sys
import os

try:
    from .ast_rl import (
        Variable, Constant, NumericLiteral, StringLiteral,
        PredicateApplication, FunctionApplication, Comparison,
        ArithmeticOperation, UnaryOperator, BinaryOperator, QuantifiedSentence,
        RelationalLogic
    )
except (ImportError, ValueError):
    try:
        from ast_rl import (
            Variable, Constant, NumericLiteral, StringLiteral,
            PredicateApplication, FunctionApplication, Comparison,
            ArithmeticOperation, UnaryOperator, BinaryOperator, QuantifiedSentence,
            RelationalLogic
        )
    except (ImportError, ValueError):
        from parser.ast_rl import (
            Variable, Constant, NumericLiteral, StringLiteral,
            PredicateApplication, FunctionApplication, Comparison,
            ArithmeticOperation, UnaryOperator, BinaryOperator, QuantifiedSentence,
            RelationalLogic
        )

def map_ast_to_z3_objects(node):
    """
    Recursively maps our parser's JSON AST to nl2logic's ast_rl objects.
    """
    node_type = node.get("type")
    
    if node_type == "Quantifier":
        quantifier_map = {
            "forall": "ForAll",
            "exists": "ThereExists"
        }
        var = Variable(name=node["variable"], sort=node.get("sort", "Entity"))
        body = map_ast_to_z3_objects(node["body"])
        return QuantifiedSentence(
            quantifier=quantifier_map.get(node["meaning"], "ForAll"),
            variable=var,
            sentence=body
        )
    
    elif node_type == "Comparison":
        left = map_term(node["left"])
        right = map_term(node["right"])
        return Comparison(
            left=left, 
            right=right, 
            operator=node["operator"]
        )
    
    elif node_type == "BinaryOp":
        if node.get("meaning") == "arithmetic":
            return ArithmeticOperation(
                left=map_term(node["left"]),
                right=map_term(node["right"]),
                operator=node["operator"]
            )

        operator_map = {
            "and": "And",
            "or": "Or",
            "implies": "If",
            "only_if": "OnlyIf",
            "iff": "IfAndOnlyIf",
            "xor": "Xor"
        }
        left = map_ast_to_z3_objects(node["left"])
        right = map_ast_to_z3_objects(node["right"])
        return BinaryOperator(
            operator=operator_map.get(node["meaning"], "And"),
            left=left,
            right=right
        )
    
    elif node_type == "UnaryOp":
        operand = map_ast_to_z3_objects(node["operand"])
        return UnaryOperator(
            operator="Not",
            sentence=operand
        )
    
    elif node_type == "BuiltinConstructor":
        name = node["name"]   # "Time" or "Date"
        args = [map_term(arg) for arg in node.get("args", [])]
        return FunctionApplication(
            name=f"Make{name}",   # MakeTime or MakeDate
            args=args,
            return_sort=node.get("return_sort", "IntSort()")
        )

    elif node_type == "Atom":
        if node.get("meaning") == "function":
             return FunctionApplication(
                 name=node["name"], 
                 args=[map_term(arg) for arg in node.get("args", [])],
                 return_sort=node.get("return_sort", "RealSort()")
             )
        name = node["name"]
        args = [map_term(arg) for arg in node.get("args", [])]
        return PredicateApplication(name=name, args=args)

    elif node_type == "Var":
        return Variable(name=node["name"], sort=node.get("sort", "Entity"))
    
    elif node_type == "NumConstant":
        return NumericLiteral(value=node["name"])
    
    elif node_type == "StringConstant":
        return StringLiteral(value=node["name"])
        
    elif node_type == "Constant":
        return Constant(name=node["name"], sort=node.get("sort", "Entity"))

    return None

def map_term(arg_node):
    node_type = arg_node.get("type")
    if node_type == "Var":
        return Variable(name=arg_node["name"], sort=arg_node.get("sort", "Entity"))
    elif node_type == "NumConstant":
        return NumericLiteral(value=arg_node["name"])
    elif node_type == "StringConstant":
        return StringLiteral(value=arg_node["name"])
    elif node_type == "Constant":
        return Constant(name=arg_node["name"], sort=arg_node.get("sort", "Entity"))
    elif node_type == "Atom":
        name = arg_node["name"]
        args = [map_term(arg) for arg in arg_node.get("args", [])]
        return FunctionApplication(
            name=name, 
            args=args,
            return_sort=arg_node.get("return_sort", "RealSort()")
        )
    elif node_type == "BinaryOp" and arg_node.get("meaning") == "arithmetic":
        return ArithmeticOperation(
            left=map_term(arg_node["left"]),
            right=map_term(arg_node["right"]),
            operator=arg_node["operator"]
        )
    elif node_type == "BuiltinConstructor":
        name = arg_node["name"]
        args = [map_term(arg) for arg in arg_node.get("args", [])]
        return FunctionApplication(
            name=f"Make{name}",
            args=args,
            return_sort=arg_node.get("return_sort", "IntSort()")
        )
    else:
        raise ValueError(
            f"map_term() encountered unexpected node type '{node_type}' "
            f"(name={arg_node.get('name')}, meaning={arg_node.get('meaning')}). "
            f"Node: {arg_node}"
        )

def convert_to_relational_logic(json_ast, original_text=""):
    """Entry point for MultiSentenceCompiler to get AST objects."""
    # 1. Map to Pydantic objects
    z3_ast_node = map_ast_to_z3_objects(json_ast)
    
    # 2. Wrap in RelationalLogic
    return RelationalLogic(
        original_sentence=original_text,
        sentences=[z3_ast_node]
    )

def convert_json_ast_to_z3(json_ast, original_text=""):
    # 1. Map to Pydantic objects
    z3_ast_node = map_ast_to_z3_objects(json_ast)
    
    # 2. Wrap in RelationalLogic
    logic_system = RelationalLogic(
        original_sentence=original_text,
        sentences=[z3_ast_node]
    )
    
    # 3. Generate Z3 code
    return logic_system.convert_to_z3()
