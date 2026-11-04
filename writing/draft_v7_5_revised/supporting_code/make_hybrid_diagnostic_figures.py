#!/usr/bin/env python
"""Generate Hybrid-Evidence MIL interpretation, stability and ablation figures.

The script uses the real UNI2-h features, coordinates, checkpoints and
predictions already stored in this repository.  It intentionally keeps the
current exploratory split-selection protocol for comparability with the
existing experiment suite; the generated figures are labelled as exploratory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ss_experiment_suite as suite  # noqa: E402


OUT_DIR = ROOT / "scripts" / "result" / "ss_experiments" / "figures" / "malt_vs_non_malt"
EXP_DIR = ROOT / "scripts" / "result" / "ss_experiments" / "malt_vs_non_malt"
HYBRID_DIR = EXP_DIR / "uni_v2_hybrid_evidence_mil"
SEED_ROOT = ROOT / "scripts" / "result"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

COLORS = {
    "navy": "#234E70",
    "teal": "#2D8C8C",
    "orange": "#D17A22",
    "red": "#B94A5C",
    "green": "#2F8F63",
    "purple": "#76579A",
    "grey": "#6D7278",
    "light": "#E9EDF0",
    "ink": "#20252B",
}


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def get_font(size: int = 18):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_df() -> pd.DataFrame:
    return pd.read_csv(HYBRID_DIR / "cohort_split.csv")


def load_model(device: torch.device) -> nn.Module:
    model = suite.SlideClassifier("hybrid_evidence_mil", 1536, 256, 0.3).to(device)
    model.load_state_dict(torch.load(HYBRID_DIR / "best_model.pth", map_location=device))
    model.eval()
    return model


def sampled_slide(row: pd.Series, index: int, seed: int = 42, num_patches: int = 512) -> Tuple[torch.Tensor, np.ndarray, dict]:
    with h5py.File(row["feature_path"], "r") as f:
        feats = f["features"][:]
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs)
    rng = np.random.default_rng(seed + index)
    n = feats.shape[0]
    if n >= num_patches:
        keep = rng.choice(n, size=num_patches, replace=False)
    else:
        keep = rng.choice(n, size=num_patches, replace=True)
    return torch.from_numpy(feats[keep]).float().unsqueeze(0), coords[keep], attrs


def hybrid_details(model: nn.Module, features: torch.Tensor, coords: np.ndarray, attrs: dict, device: torch.device) -> dict:
    """Recompute the named Hybrid branches using the trained encoder modules."""
    enc = model.encoder
    batch_coords = torch.from_numpy(coords).long().unsqueeze(0).to(device)
    h = enc.project(features.to(device))
    attn_pool, attn = enc.attention(h)
    mean_pool = h.mean(dim=1)
    std_pool = h.std(dim=1)
    h_norm = torch.nn.functional.normalize(h, dim=-1)
    proto = torch.nn.functional.normalize(enc.prototypes.to(device), dim=-1)
    sim = torch.einsum("bnh,ckh->bnck", h_norm, proto)
    class_sim = sim.max(dim=-1).values
    margin = class_sim[..., 1] - class_sim[..., 0]
    k = min(32, h.size(1))
    pos_idx = margin.topk(k, dim=1).indices
    neg_idx = (-margin).topk(k, dim=1).indices
    pos_top = torch.gather(h, 1, pos_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))).mean(dim=1)
    neg_top = torch.gather(h, 1, neg_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))).mean(dim=1)
    order = torch.argsort(batch_coords[..., 1] * 1000000 + batch_coords[..., 0], dim=1)
    ordered = torch.gather(h, 1, order.unsqueeze(-1).expand(-1, -1, h.size(-1)))
    chunks = torch.chunk(ordered, min(enc.regions, ordered.size(1)), dim=1)
    region_tokens = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)
    region_pool, region_attn = enc.region_pool(region_tokens)
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
    weights = torch.softmax(enc.gate(gate_input), dim=1)
    fused = enc.fusion(torch.cat([(weights.unsqueeze(-1) * evidence).flatten(1), stats], dim=1))
    with torch.no_grad():
        logits = model.classifier(fused).squeeze(1)
    margin_np = margin.squeeze(0).detach().cpu().numpy()
    attn_np = attn.squeeze(0).detach().cpu().numpy()
    region_np = region_attn.squeeze(0).detach().cpu().numpy()
    region_scores = np.zeros(len(coords), dtype=np.float32)
    ordered_idx = order.squeeze(0).detach().cpu().numpy()
    chunks_idx = np.array_split(ordered_idx, min(enc.regions, len(coords)))
    for score, idxs in zip(region_np, chunks_idx):
        region_scores[idxs] = float(score)
    return {
        "probability": float(torch.sigmoid(logits).item()),
        "margin": margin_np,
        "attention": attn_np,
        "region": region_scores,
        "gate": weights.squeeze(0).detach().cpu().numpy(),
        "stats": stats.squeeze(0).detach().cpu().numpy(),
        "positive_idx": pos_idx.squeeze(0).detach().cpu().numpy(),
        "negative_idx": neg_idx.squeeze(0).detach().cpu().numpy(),
        "attrs": attrs,
    }


def thumbnail_path(row: pd.Series) -> Path:
    for p in [
        ROOT / "scripts" / "result" / row["class_name"] / "thumbnails" / f"{row['slide_id']}.jpg",
        ROOT / "scripts" / "result" / row["class_name"] / "20x_512px_0px_overlap" / "visualization" / f"{row['slide_id']}.jpg",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(f"thumbnail not found for {row['slide_id']}")


def overlay_image(row: pd.Series, coords: np.ndarray, scores: np.ndarray, attrs: dict, cmap_name: str, diverging: bool = False) -> Image.Image:
    base = Image.open(thumbnail_path(row)).convert("RGB")
    w, h = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    lw, lh = float(attrs["level0_width"]), float(attrs["level0_height"])
    ps = float(attrs["patch_size_level0"])
    scores = np.asarray(scores, dtype=float)
    if diverging:
        scale = max(abs(np.percentile(scores, 2)), abs(np.percentile(scores, 98)), 1e-6)
        norm = colors.Normalize(vmin=-scale, vmax=scale)
    else:
        lo, hi = np.percentile(scores, 2), np.percentile(scores, 98)
        norm = colors.Normalize(vmin=float(lo), vmax=float(hi) if hi > lo else float(lo + 1e-6))
    cmap = cm.get_cmap(cmap_name)
    for (x, y), score in zip(coords, scores):
        x0, y0 = int(x / lw * w), int(y / lh * h)
        x1, y1 = int((x + ps) / lw * w), int((y + ps) / lh * h)
        r, g, b, _ = cmap(float(norm(score)))
        alpha = 145 if diverging else 125
        draw.rectangle([x0, y0, max(x1, x0 + 2), max(y1, y0 + 2)], fill=(int(r * 255), int(g * 255), int(b * 255), alpha))
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def extract_patches(row: pd.Series, coords: np.ndarray, indices: np.ndarray, out_dir: Path, prefix: str, top_k: int = 6) -> List[Tuple[Image.Image, int]]:
    import openslide

    slide = openslide.OpenSlide(row["slide_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    patches = []
    for rank, idx in enumerate(indices[:top_k], 1):
        x, y = map(int, coords[int(idx)])
        patch = slide.read_region((x, y), 0, (1024, 1024)).convert("RGB").resize((220, 220), Image.Resampling.LANCZOS)
        patch.save(out_dir / f"{row['slide_id']}_{prefix}{rank:02d}.png")
        patches.append((patch, int(idx)))
    slide.close()
    return patches


def make_case_panel(row: pd.Series, details: dict, coords: np.ndarray, out_dir: Path, category: str, top_k: int = 6) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pos_order = np.argsort(details["margin"])[::-1]
    neg_order = np.argsort(details["margin"])
    pos_patches = extract_patches(row, coords, pos_order, out_dir, "malt_like_", top_k)
    neg_patches = extract_patches(row, coords, neg_order, out_dir, "nonmalt_like_", top_k)
    heat_pos = overlay_image(row, coords, details["margin"], details["attrs"], "coolwarm", diverging=True)
    heat_pos.save(out_dir / f"{row['slide_id']}_malt_margin_overlay.png")

    # Compact panel: one primary Hybrid evidence map and two top-k patch
    # groups. The slide identifier is intentionally omitted from all visible
    # text; it remains only in the filename and manifest.
    canvas = Image.new("RGB", (1800, 760), "white")
    draw = ImageDraw.Draw(canvas)
    font_title, font, font_small = get_font(26), get_font(19), get_font(14)
    threshold = 0.448
    pred = "MALT" if details["probability"] >= threshold else "non-MALT"
    draw.text((30, 22), "Hybrid-Evidence MIL", fill=(25, 25, 25), font=font_title)
    draw.text((30, 60), f"True class: {row['class_name']}   |   MALT probability: {details['probability']:.3f}   |   prediction: {pred}", fill=(75, 75, 75), font=font)

    left_x, map_y = 28, 112
    map_w, map_h = 680, 560
    # Fill the large frame so wide WSI thumbnails do not leave blank bands.
    map_image = ImageOps.fit(heat_pos.convert("RGB"), (map_w, map_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    canvas.paste(map_image, (left_x, map_y))
    draw.rectangle([left_x, map_y, left_x + map_w, map_y + map_h], outline=(50, 50, 50), width=2)
    draw.text((left_x, map_y + map_h + 8), "Hybrid MALT-margin evidence", fill=(35, 35, 35), font=font_small)

    draw.text((780, 112), "MALT-like top-k patches", fill=(35, 35, 35), font=font)
    draw.text((1280, 112), "non-MALT-like top-k patches", fill=(35, 35, 35), font=font)
    for i, (patch, idx) in enumerate(pos_patches):
        x, y = 780 + (i % 2) * 210, 145 + (i // 2) * 176
        canvas.paste(patch.resize((170, 170)), (x, y))
        draw.rectangle([x, y, x + 170, y + 170], outline=(185, 74, 92), width=2)
        draw.text((x, y + 174), f"#{i+1}  margin={details['margin'][idx]:.2f}", fill=(45, 45, 45), font=font_small)
    for i, (patch, idx) in enumerate(neg_patches):
        x, y = 1280 + (i % 2) * 210, 145 + (i // 2) * 176
        canvas.paste(patch.resize((170, 170)), (x, y))
        draw.rectangle([x, y, x + 170, y + 170], outline=(118, 87, 154), width=2)
        draw.text((x, y + 174), f"#{i+1}  margin={details['margin'][idx]:.2f}", fill=(45, 45, 45), font=font_small)
    panel_path = out_dir / f"{row['slide_id']}_hybrid_evidence_panel.png"
    canvas.save(panel_path)
    return {"slide_id": row["slide_id"], "category": category, "class_name": row["class_name"], "probability": details["probability"], "gate": details["gate"].tolist(), "panel": str(panel_path)}


def select_cases(pred: pd.DataFrame, per_category: int = 3) -> List[Tuple[str, str]]:
    groups = [
        ("true_malt", pred[(pred.label == 1) & (pred.prediction == 1)].sort_values("probability", ascending=False)),
        ("true_non_malt", pred[(pred.label == 0) & (pred.prediction == 0)].sort_values("probability")),
        ("false_positive", pred[(pred.label == 0) & (pred.prediction == 1)].sort_values("probability", ascending=False)),
        ("false_negative", pred[(pred.label == 1) & (pred.prediction == 0)].sort_values("probability")),
    ]
    selected = [(name, str(row.slide_id)) for name, frame in groups for row in frame.head(per_category).itertuples()]
    # The best-threshold seed-42 Hybrid run has no false-negative cases. Keep
    # the lowest-scoring true MALT slides as hard-positive candidates so the
    # candidate grid remains useful for pathology review.
    if not any(name == "false_negative" for name, _ in selected):
        hard = pred[pred.label == 1].sort_values("probability").head(per_category)
        selected.extend(("hard_malt_positive", str(row.slide_id)) for row in hard.itertuples())
    return selected


def make_case_grid(manifest: List[dict], out_path: Path) -> None:
    groups = ["true_malt", "true_non_malt", "false_positive", "hard_malt_positive", "false_negative"]
    fig, axes = plt.subplots(len(groups), 3, figsize=(12, 13.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for r, group in enumerate(groups):
        items = [m for m in manifest if m["category"] == group][:3]
        for c, item in enumerate(items):
            image = Image.open(item["panel"]).convert("RGB")
            axes[r, c].imshow(image)
            axes[r, c].set_title(f"{group.replace('_', ' ')} | {item['slide_id']} | p={item['probability']:.3f}", fontsize=8, loc="left")
    fig.suptitle("Hybrid-Evidence MIL: candidate case-level evidence panels\nExploratory seed-42 test split; select representative cases for manuscript", fontsize=13, y=0.995)
    save_figure(fig, out_path)


def make_flat_case_folder(manifest: List[dict], out_dir: Path) -> None:
    """Export one self-contained panel per candidate for manual screening."""
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_manifest = []
    for item in manifest:
        source = Path(item["panel"])
        stem = f"{item['category']}__{item['slide_id']}"
        png_path = out_dir / f"{stem}.png"
        pdf_path = out_dir / f"{stem}.pdf"
        image = Image.open(source).convert("RGB")
        image.save(png_path, dpi=(300, 300))
        image.save(pdf_path, "PDF", resolution=300.0)
        flat_manifest.append({**item, "png": str(png_path), "pdf": str(pdf_path)})
    (out_dir / "manifest.json").write_text(json.dumps(flat_manifest, indent=2))


def load_stability_table() -> pd.DataFrame:
    rows = []
    for seed in [42, 7, 13, 21]:
        if seed == 42:
            path = HYBRID_DIR / "summary.json"
        else:
            path = SEED_ROOT / f"ss_experiments_hybrid_seed_{seed}" / "malt_vs_non_malt" / "uni_v2_hybrid_evidence_mil" / "summary.json"
        with path.open() as f:
            row = json.load(f)
        row["seed"] = seed
        rows.append(row)
    return pd.DataFrame(rows)


def make_stability_figure(out_path: Path) -> pd.DataFrame:
    df = load_stability_table().sort_values("seed")
    metrics = [("best_auc", "AUROC"), ("best_auprc", "AUPRC"), ("best_accuracy", "Accuracy"), ("best_sensitivity", "Sensitivity"), ("best_specificity", "Specificity"), ("best_f1", "F1")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), gridspec_kw={"width_ratios": [1.45, 1]})
    x = np.arange(len(df))
    for metric, label in metrics[:2]:
        axes[0].plot(x, df[metric], marker="o", lw=1.8, ms=5, label=label)
        axes[0].axhline(df[metric].mean(), color=COLORS["grey"], lw=0.8, ls="--")
        axes[0].text(len(df) - 0.05, df[metric].mean() + (0.014 if metric == "best_auc" else -0.022), f"mean={df[metric].mean():.3f} ± {df[metric].std(ddof=1):.3f}", ha="right", fontsize=7, color=COLORS["grey"])
    axes[0].set_xticks(x, [f"seed {s}" for s in df.seed])
    axes[0].set_ylim(0.75, 1.0)
    axes[0].set_ylabel("Metric")
    axes[0].set_title("Ranking metrics vary across split/training seeds", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(axis="y", color="#E5E8EA", lw=0.6)

    vals = df[[m for m, _ in metrics[2:]]].to_numpy().T
    im = axes[1].imshow(vals, aspect="auto", cmap="YlGnBu", vmin=0.55, vmax=1.0)
    axes[1].set_yticks(np.arange(4), [label for _, label in metrics[2:]])
    axes[1].set_xticks(x, [str(s) for s in df.seed])
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            axes[1].text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if vals[i, j] > 0.8 else COLORS["ink"])
    axes[1].set_title("Operating metrics", loc="left", fontweight="bold")
    fig.colorbar(im, ax=axes[1], fraction=0.045, pad=0.04, label="Point estimate")
    fig.suptitle("Hybrid-Evidence MIL stability across repeated exploratory seeds", fontsize=13, y=1.02)
    fig.text(0.01, -0.035, "Each seed changes both the stratified slide split and training randomness; no confidence intervals are implied.", fontsize=7, color=COLORS["grey"])
    save_figure(fig, out_path)
    return df


class MaskedHybridEncoder(suite.HybridEvidenceMILEncoder):
    """Hybrid encoder with masked evidence branches for exploratory ablation."""

    def __init__(self, *args, active: Iterable[int] = (0, 1, 2, 3, 4), equal_gate: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.active = tuple(active)
        self.equal_gate = equal_gate

    def forward(self, batch, device: str = "cuda") -> torch.Tensor:
        h = self.project(batch["features"].to(device))
        attn_pool, _ = self.attention(h)
        mean_pool, std_pool = h.mean(dim=1), h.std(dim=1)
        h_norm = torch.nn.functional.normalize(h, dim=-1)
        proto = torch.nn.functional.normalize(self.prototypes.to(h.device), dim=-1)
        sim = torch.einsum("bnh,ckh->bnck", h_norm, proto)
        class_sim = sim.max(dim=-1).values
        margin = class_sim[..., 1] - class_sim[..., 0]
        k = min(32, h.size(1))
        pos_idx, neg_idx = margin.topk(k, dim=1).indices, (-margin).topk(k, dim=1).indices
        pos_top = torch.gather(h, 1, pos_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))).mean(dim=1)
        neg_top = torch.gather(h, 1, neg_idx.unsqueeze(-1).expand(-1, -1, h.size(-1))).mean(dim=1)
        coords = batch["coords"].to(device)
        order = torch.argsort(coords[..., 1] * 1000000 + coords[..., 0], dim=1)
        ordered = torch.gather(h, 1, order.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        chunks = torch.chunk(ordered, min(self.regions, ordered.size(1)), dim=1)
        region_tokens = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)
        region_pool, _ = self.region_pool(region_tokens)
        stats = torch.cat([class_sim[..., 0].topk(k, dim=1).values.mean(dim=1, keepdim=True), class_sim[..., 1].topk(k, dim=1).values.mean(dim=1, keepdim=True), margin.mean(dim=1, keepdim=True), margin.std(dim=1, keepdim=True), margin.max(dim=1).values.unsqueeze(1), margin.min(dim=1).values.unsqueeze(1), (margin > 0).float().mean(dim=1, keepdim=True), (margin > margin.mean(dim=1, keepdim=True)).float().mean(dim=1, keepdim=True)], dim=1)
        evidence = torch.stack([attn_pool, mean_pool, std_pool, pos_top, neg_top + region_pool], dim=1)
        mask = torch.zeros_like(evidence)
        for idx in self.active:
            mask[:, idx] = 1.0
        evidence = evidence * mask
        stats = stats if len(self.active) >= 4 else stats * 0.0
        if self.equal_gate:
            weights = mask[:, :, :1]
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            weights = weights / denom
        else:
            weights = torch.softmax(self.gate(torch.cat([evidence.flatten(1), stats], dim=1)), dim=1).unsqueeze(-1)
        return self.fusion(torch.cat([(weights * evidence).flatten(1), stats], dim=1))


class CachedPatchDataset(Dataset):
    """Cache H5 arrays once so repeated ablation epochs are I/O-bound only once."""

    def __init__(self, df: pd.DataFrame, split: str, num_patches: int, seed: int):
        self.rows = df[df["split"] == split].reset_index(drop=True)
        self.split = split
        self.num_patches = num_patches
        self.seed = seed
        self.cache = []
        for row in self.rows.itertuples(index=False):
            with h5py.File(row.feature_path, "r") as f:
                self.cache.append((f["features"][:], f["coords"][:]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        features, coords = self.cache[idx]
        if self.split == "train":
            rng = np.random.default_rng()
            keep = rng.choice(features.shape[0], size=self.num_patches, replace=features.shape[0] < self.num_patches)
        else:
            rng = np.random.default_rng(self.seed + idx)
            keep = rng.choice(features.shape[0], size=self.num_patches, replace=features.shape[0] < self.num_patches)
        return {
            "features": torch.from_numpy(features[keep]).float(),
            "coords": torch.from_numpy(coords[keep]).long(),
            "labels": torch.tensor(float(row.label), dtype=torch.float32),
            "slide_id": str(row.slide_id),
        }


def build_masked_model(active: Iterable[int], equal_gate: bool, device: torch.device) -> nn.Module:
    model = suite.SlideClassifier("hybrid_evidence_mil", 1536, 256, 0.3).to(device)
    model.encoder = MaskedHybridEncoder(1536, 256, 0.3, active=active, equal_gate=equal_gate).to(device)
    return model


def run_ablation_training(out_dir: Path, device: torch.device, epochs: int = 30, seed: int = 42, force: bool = False) -> pd.DataFrame:
    specs = {
        "attention_only": ((0,), False),
        "attention_mean_std": ((0, 1, 2), False),
        "attention_mean_std_proto": ((0, 1, 2, 3), False),
        "full_equal_gate": ((0, 1, 2, 3, 4), True),
        "full_learned_gate": ((0, 1, 2, 3, 4), False),
    }
    split = load_split_df()
    rows = []
    for name, (active, equal_gate) in specs.items():
        target = out_dir / name / "summary.json"
        if target.exists() and not force:
            rows.append(json.loads(target.read_text()))
            continue
        set_seed(seed)
        model = build_masked_model(active, equal_gate, device)
        train_df = split[split.split == "train"].reset_index(drop=True)
        test_df = split[split.split == "test"].reset_index(drop=True)
        train_ds = CachedPatchDataset(split, "train", 512, seed)
        test_ds = CachedPatchDataset(split, "test", 512, seed)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
        pos = float((train_df.label == 1).sum())
        neg = float((train_df.label == 0).sum())
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        best_auc, best_epoch, best_state = -1.0, 0, None
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                labels = batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch), labels)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            labels, logits, _ = suite.evaluate(model, test_loader, device)
            metrics = suite.compute_metrics(labels, logits, 0.5)
            metrics["epoch"] = epoch
            metrics["train_loss"] = float(np.mean(losses))
            history.append(metrics)
            if metrics["auc"] > best_auc:
                best_auc, best_epoch = float(metrics["auc"]), epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        labels, logits, slide_ids = suite.evaluate(model, test_loader, device)
        threshold, best_metrics = suite.find_best_threshold(labels, logits)
        summary = {"aggregator": name, "seed": seed, "epochs": epochs, "best_epoch": best_epoch, "n_total": 193, "n_train": 144, "n_test": 49, "active_branches": list(active), "equal_gate": equal_gate, **{f"best_{k}": v for k, v in best_metrics.items()}, "best_auc_epoch_metric": best_auc}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2))
        pd.DataFrame(history).to_csv(target.parent / "history.csv", index=False)
        pd.DataFrame({"slide_id": slide_ids, "label": labels.astype(int), "probability": sigmoid(logits), "prediction": (sigmoid(logits) >= threshold).astype(int)}).to_csv(target.parent / "test_predictions.csv", index=False)
        rows.append(summary)
        print(f"ablation {name}: AUROC={best_auc:.4f}, threshold={threshold:.3f}")
    return pd.DataFrame(rows)


def make_ablation_figure(df: pd.DataFrame, out_path: Path) -> None:
    display = {"attention_only": "Attention only", "attention_mean_std": "+ mean/std", "attention_mean_std_proto": "+ prototype top-k", "full_equal_gate": "Full, equal gate", "full_learned_gate": "Full, learned gate"}
    order = list(display)
    df = df.set_index("aggregator").loc[order].reset_index()
    metrics = [("best_auc", "AUROC"), ("best_auprc", "AUPRC"), ("best_accuracy", "Accuracy"), ("best_sensitivity", "Sensitivity"), ("best_specificity", "Specificity"), ("best_f1", "F1")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.55, 1]})
    x = np.arange(len(df))
    width = 0.12
    for j, (metric, label) in enumerate(metrics):
        axes[0].bar(x + (j - 2.5) * width, df[metric], width=width, label=label, color=plt.cm.Set2(j / 6), edgecolor="white", linewidth=0.4)
    axes[0].set_xticks(x, [display[v] for v in df.aggregator], rotation=25, ha="right")
    axes[0].set_ylim(0.55, 1.0)
    axes[0].set_ylabel("Metric")
    axes[0].set_title("Component-masked Hybrid ablation", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=3, fontsize=7)
    axes[0].grid(axis="y", color="#E5E8EA", lw=0.6)
    vals = df[[m for m, _ in metrics]].to_numpy()
    im = axes[1].imshow(vals, aspect="auto", cmap="YlGnBu", vmin=0.55, vmax=1.0)
    axes[1].set_yticks(np.arange(len(df)), [display[v] for v in df.aggregator])
    axes[1].set_xticks(np.arange(len(metrics)), [label for _, label in metrics], rotation=30, ha="right")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            axes[1].text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if vals[i, j] > 0.8 else COLORS["ink"])
    axes[1].set_title("Metric matrix", loc="left", fontweight="bold")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Point estimate")
    fig.suptitle("Hybrid-Evidence MIL branch ablation on the exploratory seed-42 split", fontsize=13, y=1.02)
    fig.text(0.01, -0.035, "Branches are masked in the same Hybrid architecture and trained for 30 epochs; checkpoint and threshold follow the existing exploratory protocol.", fontsize=7, color=COLORS["grey"])
    save_figure(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--cases-per-category", type=int, default=3)
    parser.add_argument("--ablation-epochs", type=int, default=30)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--force-ablation", action="store_true")
    parser.add_argument("--skip-cases", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_cases:
        print("Generating Hybrid candidate case panels...")
        pred = pd.read_csv(HYBRID_DIR / "test_predictions.csv")
        split = load_split_df()
        split_by_id = {str(row.slide_id): (idx, row) for idx, row in split.reset_index(drop=True).iterrows() if row.split == "test"}
        model = load_model(device)
        case_root = OUT_DIR / "hybrid_evidence_cases"
        manifest = []
        for category, slide_id in select_cases(pred, args.cases_per_category):
            if slide_id not in split_by_id:
                continue
            idx, row = split_by_id[slide_id]
            features, coords, attrs = sampled_slide(row, idx, seed=42, num_patches=512)
            with torch.no_grad():
                details = hybrid_details(model, features, coords, attrs, device)
            manifest.append(make_case_panel(row, details, coords, case_root / category, category))
        (case_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        make_case_grid(manifest, OUT_DIR / "figure_hybrid_evidence_case_grid")
        make_flat_case_folder(manifest, OUT_DIR / "hybrid_evidence_cases_flat")

    stability = make_stability_figure(OUT_DIR / "figure_hybrid_seed_stability")
    stability.to_csv(OUT_DIR / "hybrid_seed_stability.csv", index=False)

    if not args.skip_ablation:
        ablation_root = ROOT / "scripts" / "result" / "hybrid_ablation_seed42"
        ablation = run_ablation_training(ablation_root, device, epochs=args.ablation_epochs, force=args.force_ablation)
        ablation.to_csv(OUT_DIR / "hybrid_ablation_summary.csv", index=False)
        make_ablation_figure(ablation, OUT_DIR / "figure_hybrid_ablation")
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
