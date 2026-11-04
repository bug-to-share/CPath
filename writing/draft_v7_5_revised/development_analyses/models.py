"""Fixed candidate architectures for the 2026-11-04 algorithm experiment.

The full and matched-attention controls preserve the historical implementation.
Direct/subspace attention are explicitly local, nnMIL-inspired adaptations,
not a reproduction of the complete nnMIL training pipeline.

``build_model(variant, seed)`` is the only RNG-resetting function. Forward
passes consume the caller's RNG stream and return one binary logit per bag.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TRIDENT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TRIDENT_ROOT), str(TRIDENT_ROOT / "scripts")]
import ss_experiment_suite as suite

INPUT_DIM = 1536
HIDDEN_DIM = 256
VARIANTS = (
    "hybrid_control", "attention_control", "hybrid_l2", "attention_l2",
    "attention_mask", "direct_attention", "subspace_attention",
    "subspace_largebatch", "hybrid_twobag", "hybrid_consistency",
)


class MatchedAttention(nn.Module):
    """Exactly the three copied modules used in the previous controlled run."""

    def __init__(self, template: nn.Module):
        super().__init__()
        self.project = copy.deepcopy(template.encoder.project)
        self.attention = copy.deepcopy(template.encoder.attention)
        self.classifier = copy.deepcopy(template.classifier)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.project(batch["features"])
        pooled, _ = self.attention(h)
        return self.classifier(pooled).squeeze(1)


class InputL2(nn.Module):
    """Per-tile directional normalization at a fixed sqrt(D) feature norm."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @staticmethod
    def normalize(features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features, p=2, dim=-1) * math.sqrt(INPUT_DIM)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Do not modify the cached feature tensor or caller's dictionary.
        return self.model({**batch, "features": self.normalize(batch["features"])})


class StochasticTopKMaskedPool(nn.Module):
    """Train-only top-attention masking; evaluation exactly matches its source.

    For the planned 512-instance bags, independently mask six uniformly chosen
    members of each bag's top ten pre-softmax logits. No pseudo patch labels
    are constructed and no instance is masked at evaluation.
    """

    def __init__(self, source: nn.Module, top_k: int = 10, drop_k: int = 6):
        super().__init__()
        self.value = copy.deepcopy(source.value)
        self.attention_a = copy.deepcopy(source.attention_a)
        self.attention_b = copy.deepcopy(source.attention_b)
        self.attention_c = copy.deepcopy(source.attention_c)
        self.top_k = int(top_k)
        self.drop_k = int(drop_k)
        self.last_mask: torch.Tensor | None = None
        self.last_premask_logits: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.value(x)
        logits = self.attention_c(self.attention_a(h) * self.attention_b(h)).squeeze(-1)
        self.last_mask = None
        self.last_premask_logits = None
        if self.training:
            # Preserve at least one finite logit even in a small synthetic bag.
            top_k = min(self.top_k, logits.shape[1])
            drop_k = min(self.drop_k, top_k, logits.shape[1] - 1)
            top_indices = logits.topk(top_k, dim=1).indices
            selected = torch.rand(logits.shape[0], top_k, device=x.device).argsort(dim=1)[:, :drop_k]
            masked_indices = top_indices.gather(1, selected)
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, masked_indices, True)
            self.last_mask = mask.detach()
            self.last_premask_logits = logits.detach()
            logits = logits.masked_fill(mask, -torch.inf)
        weights = logits.softmax(dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        return pooled, weights


class RawFeatureGatedAttention(nn.Module):
    """Attention scoring without a learned value projection."""

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, dropout: float = .25):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.attention_c(self.attention_a(features) * self.attention_b(features)).squeeze(-1)
        return logits.softmax(dim=-1)


class DirectAttention(nn.Module):
    """Gated scoring and weighted pooling in the original 1,536-D space."""

    def __init__(self, shared_classifier: nn.Linear):
        super().__init__()
        self.attention = RawFeatureGatedAttention(INPUT_DIM)
        self.classifier = copy.deepcopy(shared_classifier)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["features"]
        weights = self.attention(features)
        pooled = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        return self.classifier(pooled).squeeze(1)


class SubspaceAttention(nn.Module):
    """Local adaptation: random-coordinate scoring and deterministic averaging.

    Training uses one random set of 256 feature coordinates per minibatch.
    Coordinates are sorted so their order follows the original embedding.
    Attention pools the complete 1,536-D values. Evaluation averages the 21
    bag logits from contiguous 256-D windows with stride 64 (including the
    final window 1280:1536). At most seven windows are materialized together.
    """

    def __init__(self, shared_classifier: nn.Linear, eval_chunk_size: int = 7):
        super().__init__()
        if not 1 <= eval_chunk_size <= 7:
            raise ValueError("eval_chunk_size must be in [1, 7]")
        self.attention = RawFeatureGatedAttention(HIDDEN_DIM)
        self.classifier = copy.deepcopy(shared_classifier)
        windows = torch.arange(0, INPUT_DIM - HIDDEN_DIM + 1, 64)[:, None] + torch.arange(HIDDEN_DIM)[None, :]
        self.register_buffer("eval_feature_indices", windows, persistent=True)
        self.eval_chunk_size = eval_chunk_size
        self.last_training_indices: torch.Tensor | None = None

    def logits_for_subspace(self, features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Single-subspace path, also useful for independent numerical checks."""
        weights = self.attention(features.index_select(-1, indices))
        pooled = torch.bmm(weights.unsqueeze(1), features).squeeze(1)
        return self.classifier(pooled).squeeze(1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch["features"]
        if self.training:
            indices = torch.randperm(INPUT_DIM, device=features.device)[:HIDDEN_DIM].sort().values
            self.last_training_indices = indices.detach()
            return self.logits_for_subspace(features, indices)
        self.last_training_indices = None
        total = features.new_zeros(features.shape[0])
        for indices in self.eval_feature_indices.split(self.eval_chunk_size):
            # B,N,S,H -> B,S,N,H; the final axis always contains a subspace.
            subset = features[..., indices].permute(0, 2, 1, 3)
            weights = self.attention(subset)
            pooled = torch.einsum("bsn,bnd->bsd", weights, features)
            logits = self.classifier(pooled).squeeze(-1)
            total = total + logits.sum(dim=1)
        return total / self.eval_feature_indices.shape[0]


def build_model(variant: str, seed: int) -> nn.Module:
    """Build a fresh, reproducible model without reading any checkpoint/data."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Keep construction order identical to the previous controlled experiment.
    template = suite.SlideClassifier("hybrid_evidence_mil", INPUT_DIM, HIDDEN_DIM, .3)
    if variant in {"hybrid_control", "hybrid_twobag", "hybrid_consistency"}:
        return copy.deepcopy(template)
    if variant == "hybrid_l2":
        return InputL2(copy.deepcopy(template))
    if variant in {"attention_control", "attention_l2", "attention_mask"}:
        model = MatchedAttention(template)
        if variant == "attention_l2":
            return InputL2(model)
        if variant == "attention_mask":
            model.attention = StochasticTopKMaskedPool(model.attention)
        return model
    # Construct the head before variant-specific attention layers so direct
    # and subspace variants have bitwise-identical classifier initialization.
    shared_classifier = nn.Linear(INPUT_DIM, 1)
    if variant == "direct_attention":
        return DirectAttention(shared_classifier)
    return SubspaceAttention(shared_classifier)
