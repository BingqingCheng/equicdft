"""Every constructor call in the documentation must still be callable.

A keyword that a class no longer accepts survives in prose indefinitely and
fails only when a reader copies the snippet. Parsing the fenced ``python``
blocks and checking their keywords against the real signatures catches that
in a second, without running any of the examples.
"""

import ast
import inspect
import pathlib
import re
import unittest

import equicdft
from equicdft.objectives import LossTerm, MetricCollection
from equicdft.response.stability import FourierStabilityLoss
from equicdft.training.loss import Loss, TensorLoss
from equicdft.training.metrics import Metrics
from equicdft.training.noise import NoiseOptions
from equicdft.training.trainer import Trainer, TrainingStream


REPOSITORY = pathlib.Path(__file__).resolve().parents[2]

DOCUMENTED_CLASSES = {
    name: value
    for name, value in vars(equicdft).items()
    if isinstance(value, type)
}
DOCUMENTED_CLASSES.update({
    "FourierStabilityLoss": FourierStabilityLoss,
    "Loss": Loss,
    "LossTerm": LossTerm,
    "MetricCollection": MetricCollection,
    "Metrics": Metrics,
    "NoiseOptions": NoiseOptions,
    "TensorLoss": TensorLoss,
    "Trainer": Trainer,
    "TrainingStream": TrainingStream,
})


def _documents():
    for path in sorted(REPOSITORY.glob("*.md")):
        yield path
    for path in sorted(REPOSITORY.glob("docs/*.md")):
        yield path


def _calls(source):
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            yield function.id, node
        elif isinstance(function, ast.Attribute):
            yield function.attr, node


class TestDocumentedConstructorCalls(unittest.TestCase):
    def test_every_documented_keyword_is_a_real_parameter(self):
        checked = 0
        for path in _documents():
            text = path.read_text()
            for block in re.findall(r"```python\n(.*?)```", text, re.S):
                try:
                    tree_source = block
                    ast.parse(tree_source)
                except SyntaxError:
                    # Fragments and pseudo-code are not callable examples.
                    continue
                for name, call in _calls(tree_source):
                    cls = DOCUMENTED_CLASSES.get(name)
                    if cls is None:
                        continue
                    try:
                        parameters = inspect.signature(cls.__init__).parameters
                    except (TypeError, ValueError):
                        continue
                    if any(p.kind == p.VAR_KEYWORD for p in parameters.values()):
                        continue
                    checked += 1
                    for keyword in call.keywords:
                        if keyword.arg is None:
                            continue
                        self.assertIn(
                            keyword.arg,
                            parameters,
                            "{}: {}(... {}=...) is not a parameter of {}".format(
                                path.name, name, keyword.arg, name,
                            ),
                        )
        self.assertGreater(checked, 0, "no documented constructor calls found")


if __name__ == "__main__":
    unittest.main()
