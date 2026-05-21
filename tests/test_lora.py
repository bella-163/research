import torch
import torch.nn as nn

from ga_lora_sub.models.lora import LoRALinear


def test_lora_linear_initially_matches_base():
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    lora = LoRALinear(linear, rank=2, alpha=4)
    x = torch.randn(7, 5)
    y0 = linear(x)
    y1 = lora(x)
    assert torch.allclose(y0, y1, atol=1e-6)


def test_sub_delta_changes_output():
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    lora = LoRALinear(linear, rank=2, alpha=4)
    x = torch.randn(7, 5)
    delta = torch.ones_like(lora.weight) * 0.01
    lora.set_sub_delta(delta)
    y = lora(x)
    assert y.shape == (7, 3)
    assert not torch.allclose(y, linear(x))
