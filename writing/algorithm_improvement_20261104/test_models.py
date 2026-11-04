"""CPU checks against historical execution and independent candidate formulas."""
from __future__ import annotations

import copy
import io
import json
import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from models import INPUT_DIM, VARIANTS, build_model, suite


def assert_same_state(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def main():
    torch.set_num_threads(2)
    torch.manual_seed(901)
    x = torch.randn(3, 48, INPUT_DIM)
    coords = torch.randint(0, 5000, (3, 48, 2))
    batch = {"features": x, "coords": coords}
    labels = torch.tensor([0., 1., 1.])
    report = {"device": "cpu", "checks": {}, "variants": {}}

    # Independently construct the original template in the historical order.
    seed = 17
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    original = suite.SlideClassifier("hybrid_evidence_mil", 1536, 256, .3)
    baseline = build_model("hybrid_control", seed)
    assert_same_state(original.state_dict(), baseline.state_dict())
    for training in [False, True]:
        original.train(training)
        baseline.train(training)
        torch.manual_seed(773)
        reference = original.classifier(original.encoder(batch, device="cpu")).squeeze(1)
        torch.manual_seed(773)
        actual = baseline(batch)
        assert torch.equal(reference, actual)
    report["checks"]["historical_full_init_and_train_eval_logits_exact"] = True

    matched = build_model("attention_control", seed)
    assert_same_state(matched.project.state_dict(), original.encoder.project.state_dict())
    assert_same_state(matched.attention.state_dict(), original.encoder.attention.state_dict())
    assert_same_state(matched.classifier.state_dict(), original.classifier.state_dict())
    for training in [False, True]:
        original.train(training)
        matched.train(training)
        torch.manual_seed(881)
        projected = original.encoder.project(x)
        pooled, _ = original.encoder.attention(projected)
        expected = original.classifier(pooled).squeeze(1)
        torch.manual_seed(881)
        actual = matched(batch)
        assert torch.equal(expected, actual)
    report["checks"]["historical_matched_attention_init_and_logits_exact"] = True

    for variant, parent in [("hybrid_l2", "hybrid_control"), ("attention_l2", "attention_control")]:
        model = build_model(variant, seed).eval()
        control = build_model(parent, seed).eval()
        assert_same_state(model.model.state_dict(), control.state_dict())
        xn = model.normalize(x)
        assert torch.allclose(xn.norm(dim=-1), torch.full(x.shape[:-1], math.sqrt(INPUT_DIM)), atol=5e-6)
        assert torch.allclose(model.normalize(x * 3.7), xn, atol=2e-6)
        zero = model.normalize(torch.zeros_like(x))
        assert torch.isfinite(zero).all() and zero.eq(0).all()
        assert torch.equal(model(batch), control({**batch, "features": xn}))
    report["checks"]["l2_norm_direction_zero_and_wrapped_execution"] = True

    masking = build_model("attention_mask", seed).eval()
    control = build_model("attention_control", seed).eval()
    assert_same_state(masking.state_dict(), control.state_dict())
    assert torch.equal(masking(batch), control(batch))
    masking.train()
    masking(batch)
    mask = masking.attention.last_mask
    logits = masking.attention.last_premask_logits
    assert mask.sum(dim=1).eq(6).all()
    allowed = torch.zeros_like(mask).scatter_(1, logits.topk(10, dim=1).indices, True)
    assert not (mask & ~allowed).any()
    assert torch.isfinite(masking({"features": x[:, :1], "coords": coords[:, :1]})).all()
    report["checks"]["mask_eval_exact_and_six_of_top_ten_training_only"] = True

    direct = build_model("direct_attention", seed).eval()
    subspace = build_model("subspace_attention", seed).eval()
    largebatch = build_model("subspace_largebatch", seed).eval()
    assert_same_state(direct.classifier.state_dict(), subspace.classifier.state_dict())
    assert_same_state(subspace.state_dict(), largebatch.state_dict())
    indices = subspace.eval_feature_indices
    assert indices.shape == (21, 256)
    assert torch.equal(indices[0], torch.arange(256))
    assert torch.equal(indices[-1], torch.arange(1280, 1536))
    assert torch.equal(indices[:, 0], torch.arange(0, 1281, 64))
    # Independent scalar-window implementation, without model helper pooling.
    naive_logits = []
    for window in indices:
        subfeatures = x[:, :, window]
        a = subspace.attention.attention_a(subfeatures)
        b = subspace.attention.attention_b(subfeatures)
        attention_logits = subspace.attention.attention_c(a * b).squeeze(-1)
        attention_weights = attention_logits.softmax(dim=1)
        pooled = (attention_weights[:, :, None] * x).sum(dim=1)
        naive_logits.append(F.linear(pooled, subspace.classifier.weight, subspace.classifier.bias).squeeze(1))
    expected = torch.stack(naive_logits).mean(dim=0)
    actual = subspace(batch)
    error = (expected - actual).abs().max().item()
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6), error
    for chunk in [1, 3, 7]:
        subspace.eval_chunk_size = chunk
        assert torch.allclose(subspace(batch), expected, atol=1e-7, rtol=1e-6)
    subspace.train()
    subspace(batch)
    sampled = subspace.last_training_indices
    assert len(sampled) == 256 and sampled.unique().numel() == 256
    assert torch.equal(sampled, sampled.sort().values)
    report["checks"]["direct_subspace_shared_head_initialization"] = True
    report["checks"]["subspace_windows_and_naive_logit_average"] = True
    report["checks"]["subspace_max_naive_logit_error"] = error
    report["checks"]["subspace_random_sorted_coordinates"] = True

    for variant in VARIANTS:
        model = build_model(variant, seed).train()
        scores = model(batch)
        assert scores.shape == (3,) and torch.isfinite(scores).all()
        F.binary_cross_entropy_with_logits(scores, labels).backward()
        bad = [k for k, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not bad, (variant, bad)
        model.eval()
        with torch.no_grad():
            before = model(batch)
        # State dictionaries alone suffice for a fresh factory model.
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        recreated = build_model(variant, 29).eval()
        recreated.load_state_dict(torch.load(stream, weights_only=True))
        with torch.no_grad():
            after = recreated(batch)
        assert torch.equal(before, after), variant
        report["variants"][variant] = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "finite_all_parameter_gradients": True,
            "state_dict_roundtrip_exact": True,
        }
    report["status"] = "PASS"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
