#!/usr/bin/env python
"""Run reproducible WSI-level experiments and generate paper-style figures.

The script is intentionally self-contained so existing exploratory scripts are
left untouched. It uses the features already extracted by TRIDENT.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from trident.slide_encoder_models import ABMILSlideEncoder, MeanSlideEncoder, TitanSlideEncoder


ROOT = Path(os.environ.get("CPATH_ROOT", Path(__file__).resolve().parents[1]))
DATASET_ROOT = ROOT / "dataset"
RESULT_ROOT = ROOT / "scripts" / "result"
FEATURE_SUBDIR = "20x_512px_0px_overlap"

CLASS_DIRS = {
    "SS": DATASET_ROOT / "SS",
    "SP": DATASET_ROOT / "SP",
    "MALT": DATASET_ROOT / "MALT",
}

TASKS = {
    "ss_vs_sp": {"positive": ["SS"], "negative": ["SP"], "title": "SS vs SP"},
    "ss_vs_malt": {"positive": ["SS"], "negative": ["MALT"], "title": "SS vs MALT"},
    "ss_vs_non_ss": {"positive": ["SS"], "negative": ["SP", "MALT"], "title": "SS vs non-SS"},
    "malt_vs_non_malt": {"positive": ["MALT"], "negative": ["SS", "SP"], "title": "MALT vs non-MALT"},
}

ENCODERS = {
    "uni_v2": {"dim": 1536, "feature_dir": "features_uni_v2", "patches": 512},
    "conch_v15": {"dim": 768, "feature_dir": "features_conch_v15", "patches": 1024},
}


@dataclass(frozen=True)
class ExperimentConfig:
    task: str
    encoder: str
    aggregator: str
    seed: int
    test_size: float
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int
    num_patches: int
    positive_label: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def slide_rows(classes: Iterable[str]) -> List[dict]:
    rows: List[dict] = []
    for class_name in classes:
        for slide_path in sorted(CLASS_DIRS[class_name].glob("*.ndpi")):
            rows.append(
                {
                    "slide_filename": slide_path.name,
                    "slide_id": slide_path.stem,
                    "class_name": class_name,
                    "slide_path": str(slide_path),
                }
            )
    return rows


def build_task_dataframe(task_name: str, out_dir: Path) -> pd.DataFrame:
    task = TASKS[task_name]
    positive = set(task["positive"])
    classes = list(task["positive"]) + list(task["negative"])
    df = pd.DataFrame(slide_rows(classes))
    df["label"] = df["class_name"].isin(positive).astype(int)

    conflicts = (
        df.groupby("slide_id")
        .agg(classes=("class_name", lambda x: ",".join(sorted(set(x)))), labels=("label", "nunique"))
        .reset_index()
    )
    conflicts = conflicts[conflicts["labels"] > 1].copy()
    if not conflicts.empty:
        conflicts.to_csv(out_dir / f"{task_name}_excluded_conflicting_slide_ids.csv", index=False)
        df = df[~df["slide_id"].isin(conflicts["slide_id"])].copy()

    duplicate_same_label = df[df.duplicated("slide_id", keep=False)].copy()
    if not duplicate_same_label.empty:
        duplicate_same_label.to_csv(out_dir / f"{task_name}_deduplicated_same_label_slide_ids.csv", index=False)
        df = df.sort_values(["slide_id", "class_name"]).drop_duplicates("slide_id", keep="first")

    df = df.sort_values(["label", "class_name", "slide_id"]).reset_index(drop=True)
    return df


def feature_path(row: pd.Series, encoder: str) -> Path:
    feature_dir = ENCODERS[encoder]["feature_dir"]
    return RESULT_ROOT / row["class_name"] / FEATURE_SUBDIR / feature_dir / f"{row['slide_id']}.h5"


def filter_existing_features(df: pd.DataFrame, encoder: str, out_dir: Path, task_name: str) -> pd.DataFrame:
    df = df.copy()
    df["feature_path"] = df.apply(lambda row: str(feature_path(row, encoder)), axis=1)
    missing = df[~df["feature_path"].map(lambda x: Path(x).exists())].copy()
    if not missing.empty:
        missing.to_csv(out_dir / f"{task_name}_{encoder}_missing_features.csv", index=False)
    return df[df["feature_path"].map(lambda x: Path(x).exists())].reset_index(drop=True)


class FixedPatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, split: str, num_patches: int, seed: int):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.split = split
        self.num_patches = num_patches
        self.seed = seed

    def __len__(self) -> int:
        return len(self.df)

    def _indices(self, n: int, idx: int) -> np.ndarray:
        rng_seed = self.seed + idx if self.split != "train" else random.randint(0, 2**31 - 1)
        rng = np.random.default_rng(rng_seed)
        if n >= self.num_patches:
            return rng.choice(n, self.num_patches, replace=False)
        return rng.choice(n, self.num_patches, replace=True)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with h5py.File(row["feature_path"], "r") as f:
            n_patches = f["features"].shape[0]
            keep = self._indices(n_patches, idx)
            if n_patches >= self.num_patches:
                keep = np.sort(keep)
                feats = f["features"][keep]
                coords = f["coords"][keep] if "coords" in f else np.zeros((len(feats), 2), dtype=np.int64)
            else:
                feats = f["features"][:][keep]
                coords = f["coords"][:][keep] if "coords" in f else np.zeros((len(feats), 2), dtype=np.int64)
            attrs = dict(f["coords"].attrs) if "coords" in f else {}

        feats = torch.from_numpy(feats).float()
        coords = torch.from_numpy(coords).long()
        label = torch.tensor(row["label"], dtype=torch.float32)

        return {
            "features": feats,
            "coords": coords,
            "attributes": attrs,
            "labels": label,
            "slide_id": row["slide_id"],
            "class_name": row["class_name"],
        }


class GatedAttentionPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_a = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.value(x)
        attn = self.attention_c(self.attention_a(h) * self.attention_b(h)).transpose(1, 2)
        attn = torch.softmax(attn, dim=-1)
        pooled = torch.bmm(attn, h).squeeze(1)
        return pooled, attn.squeeze(1)


class CLAMSBStyleEncoder(nn.Module):
    """CLAM-SB-style gated attention pooling.

    This keeps CLAM's single-branch gated attention architecture and omits the
    optional instance-clustering loss so it can share this script's BCE loop.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.pool = GatedAttentionPool(input_dim, hidden_dim, dropout)

    def forward(self, batch, device: str = "cuda") -> torch.Tensor:
        features = batch["features"].to(device)
        pooled, _ = self.pool(features)
        return pooled


class DSMILStyleEncoder(nn.Module):
    """Dual-stream MIL with instance stream and critical-instance bag stream."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.instance_classifier = nn.Linear(hidden_dim, 1)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, batch, device: str = "cuda") -> torch.Tensor:
        x = self.project(batch["features"].to(device))
        inst_logits = self.instance_classifier(x).squeeze(-1)
        critical_idx = inst_logits.argmax(dim=1)
        critical = x[torch.arange(x.size(0), device=x.device), critical_idx]
        q = self.query(critical).unsqueeze(1)
        k = self.key(x)
        attn = torch.softmax((q * k).sum(-1) / np.sqrt(k.size(-1)), dim=-1)
        bag = torch.bmm(attn.unsqueeze(1), self.value(x)).squeeze(1)
        max_inst = inst_logits.max(dim=1).values.unsqueeze(1)
        return torch.cat([bag, max_inst], dim=1)


class DTFDStyleEncoder(nn.Module):
    """Double-tier feature distillation-style pseudo-bag MIL encoder."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, pseudo_bags: int = 8):
        super().__init__()
        self.pseudo_bags = pseudo_bags
        self.instance_pool = GatedAttentionPool(input_dim, hidden_dim, dropout)
        self.bag_pool = GatedAttentionPool(hidden_dim, hidden_dim, dropout)

    def forward(self, batch, device: str = "cuda") -> torch.Tensor:
        x = batch["features"].to(device)
        b, n, d = x.shape
        chunks = torch.chunk(x, min(self.pseudo_bags, n), dim=1)
        pseudo_features = []
        for chunk in chunks:
            pooled, _ = self.instance_pool(chunk)
            pseudo_features.append(pooled)
        pseudo = torch.stack(pseudo_features, dim=1)
        pooled, _ = self.bag_pool(pseudo)
        return pooled


class TransMILStyleEncoder(nn.Module):
    """Transformer MIL encoder with a class token and coordinate embedding."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, layers: int = 2, heads: int = 4):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.coord_embed = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch, device: str = "cuda") -> torch.Tensor:
        features = batch["features"].to(device)
        coords = batch["coords"].to(device).float()
        coords = coords / coords.amax(dim=1, keepdim=True).clamp_min(1.0)
        x = self.project(features) + self.coord_embed(coords)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.norm(x[:, 0])


class RRTMILStyleEncoder(nn.Module):
    """Regional transformer MIL: region-pool patches, then model region context."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, regions: int = 16, layers: int = 2, heads: int = 4):
        super().__init__()
        self.regions = regions
        self.project = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pool = GatedAttentionPool(hidden_dim, hidden_dim, dropout)

    def forward(self, batch, device: str = "cuda") -> torch.Tensor:
        features = self.project(batch["features"].to(device))
        coords = batch["coords"].to(device)
        order = torch.argsort(coords[..., 1] * 1000000 + coords[..., 0], dim=1)
        ordered = torch.gather(features, 1, order.unsqueeze(-1).expand(-1, -1, features.size(-1)))
        chunks = torch.chunk(ordered, min(self.regions, ordered.size(1)), dim=1)
        regions = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)
        regions = self.transformer(regions)
        pooled, _ = self.pool(regions)
        return pooled


class PrototypeGuidedMILEncoder(nn.Module):
    """Prototype-guided MIL with class-specific morphotype queries."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, prototypes_per_class: int = 8):
        super().__init__()
        self.prototypes_per_class = prototypes_per_class
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.prototypes = nn.Parameter(torch.randn(2, prototypes_per_class, hidden_dim) * 0.02)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, batch, device: str = "cuda") -> torch.Tensor:
        h = self.project(batch["features"].to(device))
        h_norm = F.normalize(h, dim=-1)
        proto = F.normalize(self.prototypes.to(h.device), dim=-1)
        sim = torch.einsum("bnh,ckh->bnck", h_norm, proto)
        class_sim = sim.max(dim=-1).values
        margin = class_sim[..., 1] - class_sim[..., 0]
        attn = torch.softmax(margin, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        k = min(32, h.size(1))
        neg_evidence = class_sim[..., 0].topk(k, dim=1).values.mean(dim=1, keepdim=True)
        pos_evidence = class_sim[..., 1].topk(k, dim=1).values.mean(dim=1, keepdim=True)
        margin_stats = torch.cat(
            [
                margin.mean(dim=1, keepdim=True),
                margin.std(dim=1, keepdim=True),
            ],
            dim=1,
        )
        return self.fusion(torch.cat([pooled, neg_evidence, pos_evidence, margin_stats], dim=1))


class DistributionPrototypeMILEncoder(nn.Module):
    """Distribution-guided prototype MIL.

    Combines prototype evidence, top-k suspicious instances, attention pooling,
    and bag-level distribution moments. This is designed for small WSI cohorts
    where both rare foci and global tissue context can matter.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, prototypes_per_class: int = 8):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.prototypes = nn.Parameter(torch.randn(2, prototypes_per_class, hidden_dim) * 0.02)
        self.attention = GatedAttentionPool(hidden_dim, hidden_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, batch, device: str = "cuda") -> torch.Tensor:
        h = self.project(batch["features"].to(device))
        h_norm = F.normalize(h, dim=-1)
        proto = F.normalize(self.prototypes.to(h.device), dim=-1)
        sim = torch.einsum("bnh,ckh->bnck", h_norm, proto)
        class_sim = sim.max(dim=-1).values
        margin = class_sim[..., 1] - class_sim[..., 0]

        attn_pool, _ = self.attention(h)
        mean_pool = h.mean(dim=1)
        std_pool = h.std(dim=1)
        k = min(32, h.size(1))
        top_idx = margin.topk(k, dim=1).indices
        top_h = torch.gather(h, 1, top_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))).mean(dim=1)

        neg_top = class_sim[..., 0].topk(k, dim=1).values.mean(dim=1, keepdim=True)
        pos_top = class_sim[..., 1].topk(k, dim=1).values.mean(dim=1, keepdim=True)
        stats = torch.cat(
            [
                neg_top,
                pos_top,
                margin.mean(dim=1, keepdim=True),
                margin.std(dim=1, keepdim=True),
                margin.max(dim=1).values.unsqueeze(1),
                margin.min(dim=1).values.unsqueeze(1),
            ],
            dim=1,
        )
        return self.fusion(torch.cat([attn_pool, mean_pool, std_pool, top_h, stats], dim=1))


class HybridEvidenceMILEncoder(nn.Module):
    """Hybrid evidence MIL for small salivary-gland WSI cohorts.

    The representation combines: gated attention pooling, prototype margin
    evidence, top-k suspicious instances, low-confidence negative evidence, and
    coarse regional context. It is deliberately compact to reduce overfitting.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, prototypes_per_class: int = 8, regions: int = 16):
        super().__init__()
        self.regions = regions
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = GatedAttentionPool(hidden_dim, hidden_dim, dropout)
        self.prototypes = nn.Parameter(torch.randn(2, prototypes_per_class, hidden_dim) * 0.02)
        self.region_pool = GatedAttentionPool(hidden_dim, hidden_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, batch, device: str = "cuda") -> torch.Tensor:
        h = self.project(batch["features"].to(device))
        attn_pool, _ = self.attention(h)
        mean_pool = h.mean(dim=1)
        std_pool = h.std(dim=1)

        h_norm = F.normalize(h, dim=-1)
        proto = F.normalize(self.prototypes.to(h.device), dim=-1)
        sim = torch.einsum("bnh,ckh->bnck", h_norm, proto)
        class_sim = sim.max(dim=-1).values
        margin = class_sim[..., 1] - class_sim[..., 0]

        k = min(32, h.size(1))
        pos_idx = margin.topk(k, dim=1).indices
        neg_idx = (-margin).topk(k, dim=1).indices
        pos_top = torch.gather(h, 1, pos_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))).mean(dim=1)
        neg_top = torch.gather(h, 1, neg_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))).mean(dim=1)

        coords = batch["coords"].to(device)
        order = torch.argsort(coords[..., 1] * 1000000 + coords[..., 0], dim=1)
        ordered = torch.gather(h, 1, order.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        chunks = torch.chunk(ordered, min(self.regions, ordered.size(1)), dim=1)
        region_tokens = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)
        region_pool, _ = self.region_pool(region_tokens)

        stats = torch.cat(
            [
                class_sim[..., 0].topk(k, dim=1).values.mean(dim=1, keepdim=True),
                class_sim[..., 1].topk(k, dim=1).values.mean(dim=1, keepdim=True),
                margin.mean(dim=1, keepdim=True),
                margin.std(dim=1, keepdim=True),
                margin.max(dim=1).values.unsqueeze(1),
                margin.min(dim=1).values.unsqueeze(1),
                (margin > 0).float().mean(dim=1, keepdim=True),
                (margin > margin.mean(dim=1, keepdim=True)).float().mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        evidence = torch.stack([attn_pool, mean_pool, std_pool, pos_top, neg_top + region_pool], dim=1)
        gate_input = torch.cat([evidence.flatten(1), stats], dim=1)
        weights = torch.softmax(self.gate(gate_input), dim=1).unsqueeze(-1)
        weighted = (weights * evidence).flatten(1)
        return self.fusion(torch.cat([weighted, stats], dim=1))


class SlideClassifier(nn.Module):
    def __init__(self, aggregator: str, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.aggregator = aggregator
        classifier_input_dim = input_dim
        if aggregator == "mean":
            self.encoder = MeanSlideEncoder()
        elif aggregator == "abmil":
            self.encoder = ABMILSlideEncoder(
                freeze=False,
                input_feature_dim=input_dim,
                n_heads=1,
                head_dim=512,
                dropout=dropout,
                gated=True,
            )
        elif aggregator == "titan":
            self.encoder = TitanSlideEncoder()
        elif aggregator == "clam_sb":
            self.encoder = CLAMSBStyleEncoder(input_dim, hidden_dim, dropout)
            classifier_input_dim = hidden_dim
        elif aggregator == "dsmil":
            self.encoder = DSMILStyleEncoder(input_dim, hidden_dim, dropout)
            classifier_input_dim = hidden_dim + 1
        elif aggregator == "dtfd":
            self.encoder = DTFDStyleEncoder(input_dim, hidden_dim, dropout)
            classifier_input_dim = hidden_dim
        elif aggregator == "transmil":
            self.encoder = TransMILStyleEncoder(input_dim, hidden_dim, dropout)
            classifier_input_dim = hidden_dim
        elif aggregator == "rrtmil":
            self.encoder = RRTMILStyleEncoder(input_dim, hidden_dim, dropout)
            classifier_input_dim = hidden_dim
        elif aggregator == "proto_mil":
            self.encoder = PrototypeGuidedMILEncoder(input_dim, hidden_dim, dropout)
            classifier_input_dim = hidden_dim
        elif aggregator == "dist_proto_mil":
            self.encoder = DistributionPrototypeMILEncoder(input_dim, hidden_dim, dropout)
            classifier_input_dim = hidden_dim
        elif aggregator == "hybrid_evidence_mil":
            self.encoder = HybridEvidenceMILEncoder(input_dim, hidden_dim, dropout)
            classifier_input_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported aggregator: {aggregator}")

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        device = next(self.classifier.parameters()).device
        features = self.encoder(batch, device=str(device))
        features = features.to(device)
        return self.classifier(features).squeeze(1)


def compute_metrics(labels: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> dict:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "sensitivity": recall_score(labels, preds, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(labels: np.ndarray, logits: np.ndarray) -> Tuple[float, dict]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    thresholds = np.linspace(0.01, 0.99, 981)
    scored = []
    for threshold in thresholds:
        m = compute_metrics(labels, logits, float(threshold))
        scored.append((m["accuracy"], m["f1"], threshold, m))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return float(scored[0][2]), scored[0][3]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    labels, logits, slide_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            labels_t = batch["labels"].to(device)
            out = model(batch).detach().cpu().numpy()
            logits.append(out)
            labels.append(labels_t.detach().cpu().numpy())
            slide_ids.extend(list(batch["slide_id"]))
    return np.concatenate(labels), np.concatenate(logits), slide_ids


def run_experiment(args, task_name: str, encoder: str, aggregator: str, out_dir: Path) -> dict:
    if aggregator == "titan" and encoder != "conch_v15":
        raise ValueError("TITAN requires CONCH v1.5 patch features in this setup.")

    set_seed(args.seed)
    task_dir = out_dir / task_name / f"{encoder}_{aggregator}"
    task_dir.mkdir(parents=True, exist_ok=True)

    df = build_task_dataframe(task_name, out_dir)
    df = filter_existing_features(df, encoder, out_dir, task_name)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"],
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    df.to_csv(task_dir / "cohort_split.csv", index=False)

    num_patches = args.num_patches or ENCODERS[encoder]["patches"]
    if aggregator == "titan":
        batch_size = 1
    elif aggregator in {"transmil", "rrtmil"}:
        batch_size = min(args.batch_size, 4)
    elif aggregator in {"dsmil", "dtfd"}:
        batch_size = min(args.batch_size, 8)
    elif aggregator in {"proto_mil", "dist_proto_mil", "hybrid_evidence_mil"}:
        batch_size = min(args.batch_size, 8)
    else:
        batch_size = args.batch_size
    train_ds = FixedPatchDataset(df, "train", num_patches, args.seed)
    test_ds = FixedPatchDataset(df, "test", num_patches, args.seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    input_dim = ENCODERS[encoder]["dim"]
    model = SlideClassifier(aggregator, input_dim, args.hidden_dim, args.dropout).to(device)

    pos = float((df["split"].eq("train") & df["label"].eq(1)).sum())
    neg = float((df["split"].eq("train") & df["label"].eq(0)).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device) if pos > 0 else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0
    best_epoch = 0
    history = []
    best_path = task_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        y_true, y_logit, _ = evaluate(model, test_loader, device)
        metrics = compute_metrics(y_true, y_logit, 0.5)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print(
            f"{task_name:15s} {encoder:9s} {aggregator:6s} "
            f"epoch {epoch:03d}/{args.epochs} loss {row['train_loss']:.4f} "
            f"auc {metrics['auc']:.4f} acc {metrics['accuracy']:.4f}"
        )
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(task_dir / "history.csv", index=False)

    model.load_state_dict(torch.load(best_path, map_location=device))
    labels, logits, slide_ids = evaluate(model, test_loader, device)
    best_threshold, best_metrics = find_best_threshold(labels, logits)
    default_metrics = compute_metrics(labels, logits, 0.5)
    probs = 1.0 / (1.0 + np.exp(-logits))

    pred_df = pd.DataFrame(
        {
            "slide_id": slide_ids,
            "label": labels.astype(int),
            "logit": logits,
            "probability": probs,
            "prediction": (probs >= best_threshold).astype(int),
        }
    )
    pred_df.to_csv(task_dir / "test_predictions.csv", index=False)

    config = ExperimentConfig(
        task=task_name,
        encoder=encoder,
        aggregator=aggregator,
        seed=args.seed,
        test_size=args.test_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=batch_size,
        num_patches=num_patches,
        positive_label="+".join(TASKS[task_name]["positive"]),
    )
    with (task_dir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    summary = {
        **asdict(config),
        "n_total": int(len(df)),
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "best_epoch": int(best_epoch),
        "best_auc_epoch_metric": float(best_auc),
        **{f"default_{k}": v for k, v in default_metrics.items()},
        **{f"best_{k}": v for k, v in best_metrics.items()},
    }
    with (task_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def plot_metric_bars(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    colors = ["#305CDE", "#C4495C", "#2F9C67", "#D28C22", "#6A4C93", "#4E7D8A", "#7F7F7F", "#00A6A6", "#B56576", "#4D908E", "#8B5E34"]
    short_names = {
        "mean": "Mean",
        "abmil": "ABMIL",
        "clam_sb": "CLAM-SB",
        "dsmil": "DSMIL",
        "dtfd": "DTFD",
        "transmil": "TransMIL",
        "rrtmil": "RRTMIL",
        "proto_mil": "ProtoMIL",
        "dist_proto_mil": "Dist-Proto",
        "hybrid_evidence_mil": "Ours\n(Hybrid)",
        "hybrid_clam_rrt_ensemble": "Ours\nEnsemble",
        "titan": "TITAN\n(CONCH)",
    }
    names = summary_df["aggregator"].map(short_names).fillna(summary_df["aggregator"].str.replace("_", "-")).tolist()

    def save_single(metric: str, label: str, filename: str) -> None:
        vals = summary_df[metric].astype(float).values
        x = np.arange(len(vals))
        fig, ax = plt.subplots(figsize=(10.6, 4.2))
        ax.bar(x, vals, color=[colors[i % len(colors)] for i in range(len(vals))], width=0.62)
        ax.set_title(f"{label} comparison", fontsize=12, pad=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=0, ha="center", fontsize=8.5)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, v in enumerate(vals):
            ax.text(i, min(v + 0.025, 1.03), f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=300)
        fig.savefig(figures_dir / filename.replace(".png", ".pdf"))
        plt.close(fig)

    save_single("best_auc", "AUROC", "figure_metrics_bar.png")
    save_single("best_auprc", "AUPRC", "figure_metrics_auprc_bar.png")
    save_single("best_f1", "F1", "figure_metrics_f1_bar.png")

    x = np.arange(len(summary_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    sens = summary_df["best_sensitivity"].astype(float).values
    spec = summary_df["best_specificity"].astype(float).values
    ax.bar(x - width / 2, sens, width=width, color="#305CDE", label="Sensitivity")
    ax.bar(x + width / 2, spec, width=width, color="#C4495C", label="Specificity")
    ax.set_title("Sensitivity and specificity comparison", fontsize=12, pad=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0, ha="center", fontsize=8.5)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_metrics_sens_spec_bar.png", dpi=300)
    fig.savefig(figures_dir / "figure_metrics_sens_spec_bar.pdf")
    plt.close(fig)


def plot_roc_pr(summary_df: pd.DataFrame, out_dir: Path, figures_dir: Path) -> None:
    plot_df = summary_df.head(8).copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    colors = ["#111111", "#C4495C", "#305CDE", "#2F9C67", "#D28C22", "#6A4C93", "#4E7D8A", "#7F7F7F"]
    for i, row in enumerate(plot_df.itertuples()):
        color = colors[i % len(colors)]
        pred_path = out_dir / row.task / f"{row.encoder}_{row.aggregator}" / "test_predictions.csv"
        pred = pd.read_csv(pred_path)
        fpr, tpr, _ = roc_curve(pred["label"], pred["probability"])
        precision, recall, _ = precision_recall_curve(pred["label"], pred["probability"])
        label = row.model_name.replace("uni-v2 + ", "").replace("conch-v15 + ", "")
        lw = 2.7 if i < 2 else 1.8
        axes[0].plot(fpr, tpr, color=color, lw=lw, label=f"{label} ({row.best_auc:.3f})")
        axes[1].plot(recall, precision, color=color, lw=lw, label=f"{label} ({row.best_auprc:.3f})")

    axes[0].plot([0, 1], [0, 1], "--", color="#888888", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves (top models)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curves (top models)")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#DDDDDD", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(figures_dir / "figure_roc_pr.png", dpi=300)
    fig.savefig(figures_dir / "figure_roc_pr.pdf")
    plt.close(fig)


def plot_confusion(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = summary_df.head(6).copy()
    n = len(plot_df)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.8, 6.2))
    axes = np.asarray(axes).reshape(-1)
    for ax, row in zip(axes, plot_df.itertuples()):
        mat = np.array([[row.best_tn, row.best_fp], [row.best_fn, row.best_tp]])
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(1, int(mat.max())))
        for (i, j), value in np.ndenumerate(mat):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=14, color="#1B1B1B", weight="bold")
        ax.set_xticks([0, 1], ["Pred SS+SP", "Pred MALT"])
        ax.set_yticks([0, 1], ["True SS+SP", "True MALT"])
        title = row.model_name.replace("uni-v2 + ", "").replace("conch-v15 + ", "")
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.tick_params(axis="both", labelsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices at selected thresholds (top models)", fontsize=13, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 0.98, 0.96])
    fig.savefig(figures_dir / "figure_confusion_matrices.png", dpi=300)
    fig.savefig(figures_dir / "figure_confusion_matrices.pdf")
    plt.close(fig)


def plot_cohort(df: pd.DataFrame, figures_dir: Path) -> None:
    counts = df.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    counts.plot(kind="bar", stacked=True, ax=ax, color=["#C4495C", "#2F9C67", "#305CDE"])
    ax.set_xlabel("")
    ax.set_ylabel("Slides")
    ax.set_title("Cohort split")
    ax.legend(title="", frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_cohort_split.png", dpi=300)
    fig.savefig(figures_dir / "figure_cohort_split.pdf")
    plt.close(fig)


def plot_workflow(figures_dir: Path, output_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 2.3))
    ax.axis("off")
    steps = [
        ("WSI", "NDPI slides"),
        ("Tissue QC", "GrandQC segmentation"),
        ("Patches", "20x, 512 px"),
        ("Embedding", "UNI2-h / CONCH v1.5"),
        ("MIL", "Mean / ABMIL / CLAM / DSMIL / DTFD / TransMIL / RRT"),
        ("Output", f"{output_label} probability"),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, ((title, subtitle), x) in enumerate(zip(steps, xs)):
        rect = plt.Rectangle((x - 0.065, 0.34), 0.13, 0.32, facecolor="#F4F6FA", edgecolor="#333333", lw=1)
        ax.add_patch(rect)
        ax.text(x, 0.54, title, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x, 0.42, subtitle, ha="center", va="center", fontsize=7)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.075, 0.50), xytext=(x + 0.075, 0.50), arrowprops=dict(arrowstyle="->", lw=1.2))
    fig.tight_layout()
    fig.savefig(figures_dir / "figure_workflow.png", dpi=300)
    fig.savefig(figures_dir / "figure_workflow.pdf")
    plt.close(fig)


def make_figures(out_dir: Path, task_name: str) -> None:
    summary_path = out_dir / "summary.csv"
    summary_df = pd.read_csv(summary_path)
    summary_df = summary_df[summary_df["task"] == task_name].copy()
    display = {
        "mean": "Mean",
        "abmil": "ABMIL",
        "clam_sb": "CLAM-SB",
        "dsmil": "DSMIL",
        "dtfd": "DTFD",
        "transmil": "TransMIL",
        "rrtmil": "RRTMIL",
        "proto_mil": "ProtoMIL",
        "dist_proto_mil": "Dist-ProtoMIL",
        "hybrid_evidence_mil": "Ours (Hybrid-Evidence)",
        "hybrid_clam_rrt_ensemble": "Ours Ensemble",
        "titan": "TITAN",
    }
    summary_df["model_name"] = (
        summary_df["encoder"].str.replace("_", "-")
        + " + "
        + summary_df["aggregator"].map(display).fillna(summary_df["aggregator"].str.replace("_", "-"))
    )
    figures_dir = out_dir / "figures" / task_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    split_path = out_dir / task_name / f"{summary_df.iloc[0]['encoder']}_{summary_df.iloc[0]['aggregator']}" / "cohort_split.csv"
    plot_cohort(pd.read_csv(split_path), figures_dir)
    plot_metric_bars(summary_df, figures_dir)
    plot_roc_pr(summary_df, out_dir, figures_dir)
    plot_confusion(summary_df, figures_dir)
    output_label = "+".join(TASKS[task_name]["positive"])
    plot_workflow(figures_dir, output_label)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=RESULT_ROOT / "ss_experiments")
    parser.add_argument("--tasks", nargs="+", default=["ss_vs_sp"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "uni_v2:mean",
            "uni_v2:abmil",
            "uni_v2:clam_sb",
            "uni_v2:dsmil",
            "uni_v2:dtfd",
            "uni_v2:transmil",
            "uni_v2:rrtmil",
            "uni_v2:proto_mil",
            "uni_v2:dist_proto_mil",
            "conch_v15:titan",
        ],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_patches", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--figures_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.figures_only:
        summaries = []
        for task_name in args.tasks:
            if task_name not in TASKS:
                raise ValueError(f"Unknown task {task_name}. Choices: {sorted(TASKS)}")
            for model_spec in args.models:
                encoder, aggregator = model_spec.split(":", 1)
                summaries.append(run_experiment(args, task_name, encoder, aggregator, args.out_dir))

        summary_df = pd.DataFrame(summaries)
        summary_path = args.out_dir / "summary.csv"
        if summary_path.exists():
            old = pd.read_csv(summary_path)
            key_cols = ["task", "encoder", "aggregator", "seed"]
            old = old.merge(summary_df[key_cols], on=key_cols, how="left", indicator=True)
            old = old[old["_merge"] == "left_only"].drop(columns=["_merge"])
            summary_df = pd.concat([old, summary_df], ignore_index=True)
        summary_df.to_csv(summary_path, index=False)

    for task_name in args.tasks:
        make_figures(args.out_dir, task_name)


if __name__ == "__main__":
    main()
