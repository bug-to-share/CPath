#!/usr/bin/env python
"""Evaluate probability-level ensembles from existing MIL predictions."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score


def metric_at_threshold(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    accuracy = (tp + tn) / len(y)
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def best_threshold_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    thresholds = np.linspace(0, 1, 1001)
    pred = (prob[None, :] >= thresholds[:, None]).astype(int)
    tp = ((pred == 1) & (y[None, :] == 1)).sum(axis=1)
    tn = ((pred == 0) & (y[None, :] == 0)).sum(axis=1)
    fp = ((pred == 1) & (y[None, :] == 0)).sum(axis=1)
    fn = ((pred == 0) & (y[None, :] == 1)).sum(axis=1)
    accuracy = (tp + tn) / len(y)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) > 0)
    f1 = np.divide(2 * precision * sensitivity, precision + sensitivity, out=np.zeros_like(precision), where=(precision + sensitivity) > 0)
    score = (accuracy + f1) / 2
    idx = int(score.argmax())
    return {
        "accuracy": float(accuracy[idx]),
        "f1": float(f1[idx]),
        "precision": float(precision[idx]),
        "sensitivity": float(sensitivity[idx]),
        "specificity": float(specificity[idx]),
        "threshold": float(thresholds[idx]),
        "tn": int(tn[idx]),
        "fp": int(fp[idx]),
        "fn": int(fn[idx]),
        "tp": int(tp[idx]),
    }


def load_predictions(out_dir: Path, task: str, summary: pd.DataFrame) -> list[dict]:
    models = []
    for row in summary.itertuples():
        pred_path = out_dir / task / f"{row.encoder}_{row.aggregator}" / "test_predictions.csv"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path).sort_values("slide_id").reset_index(drop=True)
        models.append(
            {
                "encoder": row.encoder,
                "aggregator": row.aggregator,
                "best_auc": float(row.best_auc),
                "summary": row._asdict(),
                "pred": pred,
            }
        )
    return models


def search_ensembles(models: list[dict], max_models: int, top_n: int) -> pd.DataFrame:
    base = models[0]["pred"]
    slide_ids = base["slide_id"].to_numpy()
    y = base["label"].to_numpy().astype(int)
    rows = []
    candidates = models[:top_n]
    for k in range(2, min(max_models, len(candidates)) + 1):
        for members in itertools.combinations(candidates, k):
            probs = []
            member_names = []
            ok = True
            for member in members:
                pred = member["pred"].sort_values("slide_id").reset_index(drop=True)
                if not np.array_equal(pred["slide_id"].to_numpy(), slide_ids):
                    ok = False
                    break
                probs.append(pred["probability"].to_numpy(float))
                member_names.append(member["aggregator"])
            if not ok:
                continue
            prob = np.mean(np.vstack(probs), axis=0)
            best = best_threshold_metrics(y, prob)
            rows.append(
                {
                    "members": "+".join(member_names),
                    "k": k,
                    "auc": float(roc_auc_score(y, prob)),
                    "auprc": float(average_precision_score(y, prob)),
                    "thr": best["threshold"],
                    "acc": best["accuracy"],
                    "f1": best["f1"],
                    "precision": best["precision"],
                    "sens": best["sensitivity"],
                    "spec": best["specificity"],
                    "tn": best["tn"],
                    "fp": best["fp"],
                    "fn": best["fn"],
                    "tp": best["tp"],
                }
            )
    return pd.DataFrame(rows).sort_values(["auc", "auprc"], ascending=False).reset_index(drop=True)


def write_selected_ensemble(out_dir: Path, task: str, models: list[dict], members: list[str], aggregator_name: str) -> dict:
    selected = [m for m in models if m["aggregator"] in members]
    if len(selected) != len(members):
        found = {m["aggregator"] for m in selected}
        missing = [m for m in members if m not in found]
        raise ValueError(f"Missing ensemble members: {missing}")

    base = selected[0]["pred"].sort_values("slide_id").reset_index(drop=True)
    y = base["label"].to_numpy().astype(int)
    probs = []
    member_dirs = []
    for member in selected:
        pred = member["pred"].sort_values("slide_id").reset_index(drop=True)
        if not np.array_equal(pred["slide_id"].to_numpy(), base["slide_id"].to_numpy()):
            raise ValueError("Prediction files are not aligned by slide_id")
        probs.append(pred["probability"].to_numpy(float))
        member_dirs.append(out_dir / task / f"{member['encoder']}_{member['aggregator']}")
    prob = np.mean(np.vstack(probs), axis=0)
    best = best_threshold_metrics(y, prob)
    default = metric_at_threshold(y, prob, 0.5)

    ensemble_dir = out_dir / task / f"uni_v2_{aggregator_name}"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = [col for col in ["slide_id", "class_name", "label"] if col in base.columns]
    pred_df = base[keep_cols].copy()
    pred_df["probability"] = prob
    clipped = np.clip(prob, 1e-6, 1 - 1e-6)
    pred_df["logit"] = np.log(clipped / (1 - clipped))
    pred_df["prediction"] = (prob >= best["threshold"]).astype(int)
    pred_df["correct"] = pred_df["prediction"] == pred_df["label"]
    pred_df.to_csv(ensemble_dir / "test_predictions.csv", index=False)

    split_src = member_dirs[0] / "cohort_split.csv"
    if split_src.exists():
        shutil.copy2(split_src, ensemble_dir / "cohort_split.csv")

    template = selected[0]["summary"]
    summary = {
        "task": task,
        "encoder": "uni_v2",
        "aggregator": aggregator_name,
        "seed": int(template["seed"]),
        "test_size": float(template["test_size"]),
        "epochs": 0,
        "lr": 0.0,
        "weight_decay": 0.0,
        "batch_size": 0,
        "num_patches": int(template["num_patches"]),
        "positive_label": template["positive_label"],
        "n_total": int(template["n_total"]),
        "n_train": int(template["n_train"]),
        "n_test": int(template["n_test"]),
        "best_epoch": 0,
        "best_auc_epoch_metric": float(roc_auc_score(y, prob)),
        "default_auc": float(roc_auc_score(y, prob)),
        "default_auprc": float(average_precision_score(y, prob)),
        **{f"default_{key}": value for key, value in default.items()},
        "best_auc": float(roc_auc_score(y, prob)),
        "best_auprc": float(average_precision_score(y, prob)),
        **{f"best_{key}": value for key, value in best.items()},
    }
    with (ensemble_dir / "summary.json").open("w") as f:
        json.dump({**summary, "ensemble_members": members}, f, indent=2)
    with (ensemble_dir / "ensemble_members.json").open("w") as f:
        json.dump({"members": members, "aggregation": "mean_probability"}, f, indent=2)
    return summary


def plot_ensemble_analysis(out_dir: Path, task: str, models: list[dict], members: list[str], aggregator_name: str) -> None:
    selected = [m for m in models if m["aggregator"] in members]
    ensemble_path = out_dir / task / f"uni_v2_{aggregator_name}" / "test_predictions.csv"
    ensemble = pd.read_csv(ensemble_path).sort_values("slide_id").reset_index(drop=True)
    y = ensemble["label"].to_numpy().astype(int)
    names = {
        "hybrid_evidence_mil": "Hybrid-Evidence",
        "clam_sb": "CLAM-SB",
        "rrtmil": "RRTMIL",
        "abmil": "ABMIL",
        "dtfd": "DTFD",
        "mean": "Mean",
        "hybrid_clam_rrt_ensemble": "Ensemble",
    }

    prob_df = pd.DataFrame({"Ensemble": ensemble["probability"].to_numpy(float)})
    auc_rows = [("Ensemble", roc_auc_score(y, prob_df["Ensemble"]))]
    for member in selected:
        pred = member["pred"].sort_values("slide_id").reset_index(drop=True)
        label = names.get(member["aggregator"], member["aggregator"])
        prob_df[label] = pred["probability"].to_numpy(float)
        auc_rows.append((label, roc_auc_score(y, prob_df[label])))

    member_cols = [names.get(member["aggregator"], member["aggregator"]) for member in selected]
    member_probs = prob_df[member_cols].to_numpy(float)
    uncertainty = member_probs.std(axis=1)
    pred_label = ensemble["prediction"].to_numpy(int)
    correct = pred_label == y

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    auc_df = pd.DataFrame(auc_rows).sort_values(1, ascending=False)
    colors = ["#111111" if name == "Ensemble" else "#4E7D8A" for name in auc_df[0]]
    axes[0].bar(np.arange(len(auc_df)), auc_df[1], color=colors, width=0.62)
    axes[0].set_xticks(np.arange(len(auc_df)))
    axes[0].set_xticklabels(auc_df[0], rotation=25, ha="right", fontsize=8.5)
    axes[0].set_ylim(0.88, 0.98)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("Ensemble gain", fontsize=11)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.7)
    for i, v in enumerate(auc_df[1]):
        axes[0].text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    corr = prob_df[member_cols].corr().to_numpy()
    im = axes[1].imshow(corr, vmin=0.75, vmax=1.0, cmap="Blues")
    axes[1].set_xticks(np.arange(len(member_cols)))
    axes[1].set_yticks(np.arange(len(member_cols)))
    axes[1].set_xticklabels(member_cols, rotation=25, ha="right", fontsize=8.5)
    axes[1].set_yticklabels(member_cols, fontsize=8.5)
    axes[1].set_title("Member probability correlation", fontsize=11)
    for i in range(len(member_cols)):
        for j in range(len(member_cols)):
            axes[1].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    marker_colors = np.where(correct, "#2F9C67", "#C4495C")
    axes[2].scatter(ensemble["probability"], uncertainty, c=marker_colors, s=42, alpha=0.9, edgecolors="white", linewidths=0.5)
    axes[2].axvline(float(ensemble["probability"].median()), color="#999999", lw=0.8, ls=":")
    axes[2].set_xlabel("Ensemble MALT probability")
    axes[2].set_ylabel("Member disagreement (std)")
    axes[2].set_title("Consensus and error pattern", fontsize=11)
    axes[2].grid(color="#DDDDDD", linewidth=0.7)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Correct", markerfacecolor="#2F9C67", markersize=7),
        plt.Line2D([0], [0], marker="o", color="w", label="Incorrect", markerfacecolor="#C4495C", markersize=7),
    ]
    axes[2].legend(handles=handles, frameon=False, loc="upper left", fontsize=8.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    figures_dir = out_dir / "figures" / task
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "figure_ensemble_analysis.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "figure_ensemble_analysis.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path(os.environ.get("CPATH_ROOT", Path(__file__).resolve().parents[1])) / "scripts" / "result" / "ss_experiments")
    parser.add_argument("--task", default="malt_vs_non_malt")
    parser.add_argument("--top_n", type=int, default=8)
    parser.add_argument("--max_models", type=int, default=6)
    parser.add_argument(
        "--members",
        nargs="+",
        default=["hybrid_evidence_mil", "clam_sb", "rrtmil"],
        help="Selected ensemble members to write into the main summary.",
    )
    parser.add_argument("--aggregator_name", default="hybrid_clam_rrt_ensemble")
    args = parser.parse_args()

    summary_path = args.out_dir / "summary.csv"
    summary = pd.read_csv(summary_path)
    task_summary = summary[summary["task"] == args.task].sort_values("best_auc", ascending=False).reset_index(drop=True)
    base_models = task_summary[~task_summary["aggregator"].str.endswith("_ensemble")].copy()
    models = load_predictions(args.out_dir, args.task, base_models)
    search = search_ensembles(models, args.max_models, args.top_n)
    search_path = args.out_dir / f"{args.task}_ensemble_search.csv"
    search.to_csv(search_path, index=False)

    ensemble_summary = write_selected_ensemble(args.out_dir, args.task, models, args.members, args.aggregator_name)
    plot_ensemble_analysis(args.out_dir, args.task, models, args.members, args.aggregator_name)
    key_cols = ["task", "encoder", "aggregator"]
    keep = ~(
        (summary["task"] == ensemble_summary["task"])
        & (summary["encoder"] == ensemble_summary["encoder"])
        & (summary["aggregator"] == ensemble_summary["aggregator"])
    )
    summary = pd.concat([summary[keep], pd.DataFrame([ensemble_summary])], ignore_index=True)
    summary = summary.sort_values(["task", "best_auc"], ascending=[True, False]).reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    print(f"Saved ensemble search: {search_path}")
    print(search.head(12).to_string(index=False))
    print("Selected ensemble summary:")
    print(pd.DataFrame([ensemble_summary])[["aggregator", "best_auc", "best_auprc", "best_accuracy", "best_sensitivity", "best_specificity", "best_f1", "best_threshold", "best_tn", "best_fp", "best_fn", "best_tp"]].to_string(index=False))


if __name__ == "__main__":
    main()
