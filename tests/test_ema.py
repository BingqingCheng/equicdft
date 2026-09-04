import unittest

import torch
from torch import nn

from equicdft._ema import _ParameterEMA


class TestParameterEMA(unittest.TestCase):
    def test_first_update_copies_then_averages_trainable_parameters(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(2.0)
        ema = _ParameterEMA(0.5)

        ema.update(model)
        self.assertEqual(ema.num_updates, 1)
        self.assertTrue(
            torch.equal(ema.shadow_parameters["weight"], model.weight)
        )

        model.weight.data.fill_(4.0)
        ema.update(model)
        self.assertEqual(ema.num_updates, 2)
        self.assertEqual(ema.shadow_parameters["weight"].item(), 3.0)

    def test_average_context_restores_raw_parameters(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(2.0)
        ema = _ParameterEMA(0.5)
        ema.update(model)
        model.weight.data.fill_(6.0)

        with ema.average_parameters(model):
            self.assertEqual(model.weight.item(), 2.0)

        self.assertEqual(model.weight.item(), 6.0)

    def test_state_round_trip_validates_model_parameters(self):
        model = nn.Linear(1, 1)
        ema = _ParameterEMA(0.9)
        ema.update(model)
        restored = _ParameterEMA(0.9)
        evaluation_state = ema.evaluation_state_dict(model)
        state = ema.state_dict()
        # Checkpoints written by the initial EMA implementation carried this
        # redundant entry; the compact loader deliberately ignores it.
        state["shadow_parameters"] = ema.shadow_parameters

        restored.load_state_dict(state, evaluation_state, model)

        self.assertEqual(restored.num_updates, 1)
        self.assertNotIn("shadow_parameters", ema.state_dict())
        for name in ema.shadow_parameters:
            self.assertTrue(
                torch.equal(
                    restored.shadow_parameters[name],
                    ema.shadow_parameters[name],
                )
            )


if __name__ == "__main__":
    unittest.main()
