#!/usr/bin/env python3
"""
Generate publication figures (ROC/AUC, confusion matrices, PR curves)
from *actual* test-set predictions of trained checkpoints.

Run from the `project/` directory (after `data/processed/fruit` + `data/thermal`
and checkpoints exist):

  python generate_paper_figures.py \\
    --rgb_checkpoint models/checkpoints/rgb_baseline_resnet18_20250621_150726_best.pth \\
    --fusion_checkpoint models/checkpoints/fusion_rgb_thermal_attention_20250621_151754_best.pth \\
    --output_dir ../Paper/figures_from_code

Outputs (300 DPI PNGs + JSON):
  fig_roc_rgb_baseline.png
  fig_roc_fusion_proposed.png
  fig_confusion_rgb_baseline.png
  fig_confusion_fusion_proposed.png
  fig_pr_rgb_baseline.png  (optional)
  fig_pr_fusion_proposed.png
  metrics_paper_figures.json   (accuracy w/ bootstrap 95% CI, macro AUC, McNemar if both models)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import patheffects as pe
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from models.fusion_model import create_fusion_model
from models.rgb_branch import create_rgb_branch
from scripts.dataloader import create_dataloaders  # type: ignore

try:
    from scipy.stats import chi2
except Exception:  # pragma: no cover
    chi2 = None  # type: ignore

DEFAULT_CLASSES = [
    "Healthy",
    "Anthracnose",
    "Alternaria",
    "Black Mould Rot",
    "Stem and Rot",
]


def _torch_load(path: str, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_rgb(path: str, class_names: List[str], device: torch.device) -> torch.nn.Module:
    ck = _torch_load(path, device)
    back = (ck.get("model_config") or {}).get("backbone", "resnet18")
    feat = (ck.get("model_config") or {}).get("feature_dim", 512)
    model = create_rgb_branch(
        num_classes=len(class_names), backbone=back, feature_dim=feat
    )
    model.load_state_dict(ck["model_state_dict"], strict=False)
    return model


def _load_fusion(path: str, class_names: List[str], device: torch.device) -> torch.nn.Module:
    ck = _torch_load(path, device)
    cfg = ck.get("model_config") or {}
    model = create_fusion_model(
        num_classes=len(class_names),
        feature_dim=cfg.get("feature_dim", 512),
        use_acoustic=cfg.get("use_acoustic", False),
        fusion_type=cfg.get("fusion_type", "attention"),
    )
    model.load_state_dict(ck["model_state_dict"], strict=False)
    return model


@torch.inference_mode()
def collect_eval(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    model_type: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    y_true, y_prob, y_pred = [], [], []

    for batch in tqdm(dataloader, desc=f"eval {model_type}"):
        if model_type == "fusion":
            if len(batch) == 4:
                rgb, th, _ac, lab = batch
            else:
                rgb, th, lab = batch
            rgb = rgb.to(device)
            th = th.to(device)
            lab = lab.to(device)
            out = model(rgb, th)
        else:
            if len(batch) == 3:
                rgb, _, lab = batch
            elif len(batch) == 4:
                rgb, _, _, lab = batch
            else:
                rgb, lab = batch
            rgb = rgb.to(device)
            lab = lab.to(device)
            out = model(rgb)

        prob = F.softmax(out, dim=1)
        pred = out.argmax(1)
        y_true.append(lab.cpu().numpy())
        y_prob.append(prob.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    return (
        np.concatenate(y_true, axis=0),
        np.concatenate(y_prob, axis=0),
        np.concatenate(y_pred, axis=0),
    )


def plot_roc_figure(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
) -> None:
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="No skill (random)")

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        av = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.4, label=f"{name} (AUC = {av:.3f})")

    fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), y_score.ravel())
    ax.plot(fpr_m, tpr_m, "b-", lw=2, label=f"Micro-average (AUC = {auc(fpr_m, tpr_m):.3f})")
    try:
        macro_auc = roc_auc_score(
            y_true, y_score, multi_class="ovr", average="macro"
        )
    except Exception:
        macro_auc = float("nan")
    fig.text(
        0.02,
        0.02,
        f"macro-AUC (OvR) = {macro_auc:.4f}",
        transform=ax.transFigure,
        fontsize=9,
        color="0.2",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    leg = ax.legend(
        loc="lower right", fontsize=7, frameon=True, title="One-vs-rest"
    )
    leg.get_frame().set_edgecolor("0.8")
    for line in leg.get_texts():
        line.set_path_effects(
            [pe.Stroke(linewidth=1.0, foreground="w"), pe.Normal()]
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confusion_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    row = cm.sum(axis=1)[:, np.newaxis]
    row = np.where(row == 0, 1, row)
    cmn = cm.astype("float64") / row
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
    im = ax.imshow(cmn, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            tcol = "white" if cmn[i, j] >= 0.55 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({100 * cmn[i, j]:.1f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color=tcol,
            )
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=40, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True class")
    ax.set_xlabel("Predicted class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Row-normalized (recall share)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pr_figure(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: Path,
) -> None:
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for i, name in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_bin[:, i], y_score[:, i])
        ax.plot(rec, prec, lw=1.5, label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=7, frameon=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        accs.append((y_true[idx] == y_pred[idx]).mean())
    lo, hi = float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975))
    return {
        "mean_accuracy_percent": 100.0 * float((y_true == y_pred).mean()),
        "ci_95_lo_percent": 100.0 * lo,
        "ci_95_hi_percent": 100.0 * hi,
        "n_samples": n,
    }


def mcnemar_test(
    y_true: np.ndarray,
    pred_baseline: np.ndarray,
    pred_proposed: np.ndarray,
) -> Dict[str, float]:
    """Contingency: b = baseline wrong & proposed right; c = opposite."""
    c_b = pred_baseline == y_true
    c_p = pred_proposed == y_true
    b = int(np.sum(~c_b & c_p))
    c_ = int(np.sum(c_b & ~c_p))
    n = b + c_
    if n == 0 or chi2 is None:
        return {
            "b": b,
            "c": c_,
            "n_discordant": n,
            "mcnemar_chi2": 0.0,
            "p_value": 1.0,
        }
    stat = (abs(b - c_) - 1) ** 2 / n
    p = float(1.0 - chi2.cdf(stat, 1))
    return {"b": b, "c": c_, "n_discordant": n, "mcnemar_chi2": float(stat), "p_value": p}


def main() -> None:
    os.chdir(HERE)
    ap = argparse.ArgumentParser(
        description="Export ROC, confusion, PR, and JSON metrics from real checkpoints"
    )
    ap.add_argument("--rgb_checkpoint", type=str, default=None)
    ap.add_argument("--fusion_checkpoint", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="figures_paper")
    ap.add_argument("--rgb_data_path", type=str, default="data/processed/fruit")
    ap.add_argument("--thermal_data_path", type=str, default="data/thermal")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument(
        "--use_acoustic", action="store_true", help="If fusion was trained with acoustic"
    )
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument(
        "--fusion_only",
        action="store_true",
        help="Only eval fusion; skip RGB",
    )
    ap.add_argument("--no_pr", action="store_true")
    args = ap.parse_args()

    if not args.fusion_checkpoint and not args.rgb_checkpoint:
        print("Error: set --rgb_checkpoint and/or --fusion_checkpoint", file=sys.stderr)
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = list(DEFAULT_CLASSES)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device, flush=True)
    print("Test loader...", flush=True)
    dls = create_dataloaders(
        rgb_data_path=args.rgb_data_path,
        thermal_data_path=args.thermal_data_path,
        batch_size=args.batch_size,
        num_workers=0,
        image_size=args.image_size,
        use_acoustic=args.use_acoustic,
    )
    te = dls["test"]
    print("  n_test =", len(te.dataset), flush=True)

    report: Dict[str, Any] = {"n_test": len(te.dataset)}

    y_t_rgb = y_p_rgb = None
    if args.rgb_checkpoint and not args.fusion_only:
        m = _load_rgb(args.rgb_checkpoint, class_names, device)
        y_t, y_s, y_pd = collect_eval(m, te, "rgb", device)
        plot_roc_figure(
            y_t, y_s, class_names, "ROC — RGB baseline (test set)", out_dir / "fig_roc_rgb_baseline.png"
        )
        plot_confusion_figure(
            y_t, y_pd, class_names, "Confusion — RGB baseline (test set)", out_dir / "fig_confusion_rgb_baseline.png"
        )
        if not args.no_pr:
            plot_pr_figure(
                y_t, y_s, class_names, "PR — RGB baseline (test set)", out_dir / "fig_pr_rgb_baseline.png"
            )
        report["rgb"] = {
            "checkpoint": str(args.rgb_checkpoint),
            "macro_auc": float(roc_auc_score(y_t, y_s, multi_class="ovr", average="macro")),
            "bootstrap_95": bootstrap_accuracy(
                y_t, y_pd, n_boot=args.n_bootstrap, seed=42
            ),
        }
        y_t_rgb, y_p_rgb = y_t, y_pd
    else:
        report["rgb"] = None

    if args.fusion_checkpoint:
        m = _load_fusion(args.fusion_checkpoint, class_names, device)
        y_t, y_s, y_pd = collect_eval(m, te, "fusion", device)
        plot_roc_figure(
            y_t,
            y_s,
            class_names,
            "ROC — proposed multi-modal fusion (test set)",
            out_dir / "fig_roc_fusion_proposed.png",
        )
        plot_confusion_figure(
            y_t,
            y_pd,
            class_names,
            "Confusion — proposed (test set)",
            out_dir / "fig_confusion_fusion_proposed.png",
        )
        if not args.no_pr:
            plot_pr_figure(
                y_t, y_s, class_names, "PR — proposed (test set)", out_dir / "fig_pr_fusion_proposed.png"
            )
        report["fusion"] = {
            "checkpoint": str(args.fusion_checkpoint),
            "macro_auc": float(roc_auc_score(y_t, y_s, multi_class="ovr", average="macro")),
            "bootstrap_95": bootstrap_accuracy(
                y_t, y_pd, n_boot=args.n_bootstrap, seed=43
            ),
        }
        if y_t_rgb is not None and y_p_rgb is not None:
            if np.array_equal(y_t, y_t_rgb):
                mcn = mcnemar_test(y_t, y_p_rgb, y_pd)
                report["mcnemar_fusion_vs_rgb"] = mcn
                dacc = 100.0 * ((y_pd == y_t).mean() - (y_p_rgb == y_t).mean())
                report["accuracy_gain_pp"] = {
                    "absolute_points": float(dacc),
                    "from_rgb": float(100.0 * (y_p_rgb == y_t).mean()),
                    "proposed": float(100.0 * (y_pd == y_t).mean()),
                }
    else:
        report["fusion"] = None

    p_json = out_dir / "metrics_paper_figures.json"
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    if chi2 is None:
        print(
            "Note: install scipy for McNemar p-value (or values default to 1.0 if discordant=0).",
            flush=True,
        )
    print("Wrote JSON:", p_json.resolve(), flush=True)
    print("Figures in:", out_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
