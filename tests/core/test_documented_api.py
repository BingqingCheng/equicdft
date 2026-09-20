"""Every documented call must still be callable.

A keyword that a class or method no longer accepts survives in prose
indefinitely and fails only when a reader copies the snippet. Parsing the
fenced ``python`` blocks and checking their keywords against the real
signatures catches that in a second, without running any of the examples.

Constructors are matched by class name. Methods are matched by name against
an explicit map, because a bare ``.solve(...)`` in a snippet says nothing
about its receiver; the map lists the methods whose keyword sets the
documentation actually describes.
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
from equicdft.solver import GridSolver
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


# method name -> the unbound function whose signature the docs describe.
# ``solve`` lost its two dozen flat keywords, which is exactly the kind of
# change a stale snippet outlives.
DOCUMENTED_METHODS = {
    "solve": GridSolver.solve,
    "evaluate": GridSolver.evaluate,
    "fit": Trainer.fit,
    "fit_streams": Trainer.fit_streams,
}


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


class TestDocumentedCalls(unittest.TestCase):
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
                    method = DOCUMENTED_METHODS.get(name)
                    if cls is None and method is None:
                        continue
                    target = cls.__init__ if method is None else method
                    try:
                        parameters = inspect.signature(target).parameters
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
        self.assertGreater(checked, 0, "no documented calls found")


if __name__ == "__main__":
    unittest.main()
