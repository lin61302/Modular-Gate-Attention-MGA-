import torch
from src.models.modules import GatingNetwork

def test_gating_network_softmax_sum():
    batch_size = 2
    seq_len = 4
    d_model = 8
    gate_hidden = 4
    num_outputs = 3

    gating = GatingNetwork(d_model, gate_hidden, num_outputs, activation='relu')
    x = torch.randn(batch_size, seq_len, d_model)
    weights = gating(x)
    assert weights.shape == (batch_size, seq_len, num_outputs)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
