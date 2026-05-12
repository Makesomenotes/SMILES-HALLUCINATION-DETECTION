"""
aggregation.py — 3 layers + geometric = 2859 features.
"""
from __future__ import annotations
import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    n_layers = hidden_states.shape[0]
    mask_bool = attention_mask.bool()
    real_positions = mask_bool.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_positions[-1].item())
    n_real = int(mask_bool.sum().item())
    device = hidden_states.device
    features = []
    for li in [n_layers - 1, n_layers - 2, n_layers - 3]:
        features.append(hidden_states[li, last_pos])
    last_token_reps = [hidden_states[li, last_pos] for li in range(n_layers)]
    layer_norms = torch.tensor(
        [rep.norm().item() for rep in last_token_reps],
        dtype=torch.float32, device=device
    )
    features.append(layer_norms)
    cos_sims = []
    for li in range(1, n_layers):
        cos = torch.nn.functional.cosine_similarity(
            last_token_reps[li - 1].unsqueeze(0),
            last_token_reps[li].unsqueeze(0)
        ).item()
        cos_sims.append(cos)
    features.append(torch.tensor(cos_sims, dtype=torch.float32, device=device))
    norm_ratios = []
    for li in range(1, n_layers):
        ratio = layer_norms[li].item() / (layer_norms[li - 1].item() + 1e-8)
        norm_ratios.append(ratio)
    features.append(torch.tensor(norm_ratios, dtype=torch.float32, device=device))
    l2_dists = []
    for li in range(1, n_layers):
        dist = (last_token_reps[li] - last_token_reps[li - 1]).norm().item()
        l2_dists.append(dist)
    features.append(torch.tensor(l2_dists, dtype=torch.float32, device=device))
    skip_cos = []
    for li in range(1, n_layers):
        cos = torch.nn.functional.cosine_similarity(
            last_token_reps[0].unsqueeze(0),
            last_token_reps[li].unsqueeze(0)
        ).item()
        skip_cos.append(cos)
    features.append(torch.tensor(skip_cos, dtype=torch.float32, device=device))
    skip_cos_last = []
    for li in range(0, n_layers - 1):
        cos = torch.nn.functional.cosine_similarity(
            last_token_reps[li].unsqueeze(0),
            last_token_reps[-1].unsqueeze(0)
        ).item()
        skip_cos_last.append(cos)
    features.append(torch.tensor(skip_cos_last, dtype=torch.float32, device=device))
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    cos_lt_mean = []
    for li in key_layers:
        mean_rep = hidden_states[li][mask_bool].mean(dim=0)
        cos = torch.nn.functional.cosine_similarity(
            hidden_states[li, last_pos].unsqueeze(0),
            mean_rep.unsqueeze(0)
        ).item()
        cos_lt_mean.append(cos)
    features.append(torch.tensor(cos_lt_mean, dtype=torch.float32, device=device))
    for li in key_layers:
        token_norms = hidden_states[li][mask_bool].norm(dim=1)
        features.append(torch.tensor([
            token_norms.mean().item(),
            token_norms.std().item() if n_real > 1 else 0.0,
            token_norms.max().item() - token_norms.min().item(),
        ], dtype=torch.float32, device=device))
    features.append(torch.tensor([n_real / 512.0], dtype=torch.float32, device=device))

    cos_t = torch.tensor(cos_sims, dtype=torch.float32, device=device)
    features.append(torch.tensor([
        cos_t.mean().item(),
        cos_t.std().item(),
        cos_t.min().item(),
        cos_t.max().item(),
        layer_norms.std().item(),
    ], dtype=torch.float32, device=device))
    return torch.cat(features, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    return aggregate(hidden_states, attention_mask)