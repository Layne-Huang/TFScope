import unittest

import torch.nn as nn

from scripts.train import checkpoint_model_state


class FakeLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.lora_A = nn.Parameter(self.linear.weight.new_zeros(2, 4))
        self.lora_B = nn.Parameter(self.linear.weight.new_zeros(4, 2))


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_weights = nn.Parameter(nn.Parameter().new_zeros(2))
        self._esm_model = nn.Module()
        self._esm_model.adapter = FakeLoRA()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.head = nn.Linear(4, 2)


class CheckpointStateTest(unittest.TestCase):
    def test_saves_lora_but_omits_frozen_esm_weights(self):
        state = checkpoint_model_state(FakeModel())

        self.assertIn(
            "backbone._esm_model.adapter.lora_A",
            state,
        )
        self.assertIn(
            "backbone._esm_model.adapter.lora_B",
            state,
        )
        self.assertNotIn(
            "backbone._esm_model.adapter.linear.weight",
            state,
        )
        self.assertIn("backbone.layer_weights", state)
        self.assertIn("head.weight", state)


if __name__ == "__main__":
    unittest.main()
