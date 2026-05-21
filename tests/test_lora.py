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


def test_drs_project_lora_a_weights_keeps_projected_subspace():
    from ga_lora_sub.drs import project_lora_a_weights

    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    lora = LoRALinear(linear, rank=2, alpha=4)
    # Use a basis that only preserves the first two input dimensions.
    p = torch.zeros(5, 2)
    p[0, 0] = 1.0
    p[1, 1] = 1.0
    model = nn.Sequential(lora)
    project_lora_a_weights(model, {"0": p})
    assert torch.allclose(lora.lora_A.weight[:, 2:], torch.zeros_like(lora.lora_A.weight[:, 2:]), atol=1e-6)


def test_compute_projectors_from_covariances():
    from ga_lora_sub.drs import compute_projectors_from_covariances

    cov = torch.eye(5)
    projectors, diag = compute_projectors_from_covariances({"layer": cov}, {"layer": 10}, rank=2)
    assert projectors["layer"].shape == (5, 2)
    assert diag.ranks["layer"] == 2
