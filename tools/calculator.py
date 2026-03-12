import ast
import operator

from tools.base import Tool

# Safe operators for math evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval(expr: str) -> float:
    """Evaluate a math expression safely without using eval()."""
    tree = ast.parse(expr, mode="eval")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            return SAFE_OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        else:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    return _eval(tree)


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression. Supports +, -, *, /, //, %, **. "
        "Use this for any numeric calculations needed during research."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate, e.g. '(10000 * 0.08) / 12'.",
            },
        },
        "required": ["expression"],
    }

    def run(self, expression: str) -> str:
        try:
            result = safe_eval(expression)
            return f"{expression} = {result}"
        except ZeroDivisionError:
            return "Error: Division by zero."
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"
