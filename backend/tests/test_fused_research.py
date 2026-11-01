"""Small deterministic numerical checks; never load the 3.2 GB backbone."""
import torch
import torch.nn.functional as F

from scripts.fused_research_eval import TensorEvaluator, lattice_image, lattice_tokens


def test_lattice_coordinates_and_roundtrip():
    grid = torch.arange(2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
    tokens = lattice_tokens(grid, 4)
    assert torch.equal(tokens[0, :, 0], grid[0, 0, ::4, ::4].flatten())
    assert torch.equal(tokens[1, :, 2], grid[0, 2, ::4, 1::4].flatten())
    assert torch.equal(lattice_image(tokens, 4, 16), grid)


def test_pool_matches_torch_module_with_packed_weights():
    torch.manual_seed(7)
    mha = torch.nn.MultiheadAttention(32, 8, dropout=0.1, batch_first=True).eval()
    norm = torch.nn.LayerNorm(32)
    out = torch.nn.Linear(32, 32)
    latent = torch.randn(1, 3, 32)
    tensors = {"pool.latents": latent}
    for prefix, module in (("mha", mha), ("norm", norm), ("out_proj", out)):
        tensors.update({f"pool.{prefix}.{key}": value for key, value in module.state_dict().items()})
    x = torch.randn(2, 12, 32)
    expected, _ = mha(latent.expand(2, -1, -1), x, x, need_weights=True)
    expected = out(norm(expected))
    actual = TensorEvaluator(tensors).pool(x, "pool")
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_mixture_matches_explicit_per_token_weighted_experts():
    torch.manual_seed(29)
    tensors = {"moe_ffn.router.weight": torch.randn(8, 4)}
    for expert in range(8):
        prefix = f"moe_ffn.experts.{expert}.ffn"
        tensors.update({prefix + ".fc.weight": torch.randn(6, 4), prefix + ".fc.bias": torch.randn(6),
                        prefix + ".proj.weight": torch.randn(4, 3), prefix + ".proj.bias": torch.randn(4)})
    ops = TensorEvaluator(tensors)
    tokens = torch.randn(1, 5, 4)
    expected = torch.zeros_like(tokens)
    for index, token in enumerate(tokens[0]):
        routing = F.linear(token, tensors["moe_ffn.router.weight"])
        selected = routing.topk(2).indices
        weights = routing.softmax(0)[selected]
        weights = weights / (weights.sum() + 1e-8)
        for expert, weight in zip(selected.tolist(), weights):
            prefix = f"moe_ffn.experts.{expert}.ffn"
            values, gates = F.linear(token, tensors[prefix + ".fc.weight"], tensors[prefix + ".fc.bias"]).chunk(2)
            transformed = F.linear(values * F.silu(gates), tensors[prefix + ".proj.weight"], tensors[prefix + ".proj.bias"])
            expected[0, index] += transformed * weight
    torch.testing.assert_close(ops.mixture(tokens), expected, atol=2e-6, rtol=1e-6)
