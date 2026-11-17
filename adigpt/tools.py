"""Small, safe local tools the model can call (Ollama tool calling)."""
from __future__ import annotations

import ast
import math
import operator
from datetime import datetime

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
          "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round,
          "floor": math.floor, "ceil": math.ceil, "min": min, "max": max}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or abs(left) > 1e6):
            raise ValueError("exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate arithmetic without eval(): only numbers, operators and math functions."""
    if len(expression) > 300:
        return "error: expression too long"
    try:
        value = _eval(ast.parse(expression.replace("^", "**"), mode="eval").body)
    except ZeroDivisionError:
        return "error: division by zero"
    except Exception as exc:  # noqa: BLE001 - report any failure back to the model
        return f"error: {exc}"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return str(value)


def current_datetime() -> str:
    now = datetime.now().astimezone()
    return now.strftime("%A, %Y-%m-%d %H:%M:%S %Z (UTC%z)")


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly. Use for any calculation.",
        "parameters": {"type": "object", "required": ["expression"], "properties": {
            "expression": {"type": "string",
                           "description": "e.g. (1200 * 1.08) / 12 or sqrt(2) ** 3"}}}}},
    {"type": "function", "function": {
        "name": "current_datetime",
        "description": "Get the current local date, time and time zone.",
        "parameters": {"type": "object", "properties": {}}}},
]


def run_tool(name: str, args: dict) -> str:
    if name == "calculator":
        return calculator(str((args or {}).get("expression", "")))
    if name == "current_datetime":
        return current_datetime()
    return f"error: unknown tool {name}"
