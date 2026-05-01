#!/usr/bin/env python3
"""Statistical tests for paired classifiers on the same test set (+ optional seed summaries).

Depends: numpy, scipy, scikit-learn

Which classical test fits your design (see also tests-of-significance overviews):
  https://www.geeksforgeeks.org/maths/tests-of-significance/

+---------------------------+----------------------------------------+----------------------------+
| Design                    | Null idea (informal)                    | Implemented test           |
+===========================+========================================+============================+
| Same test samples, two    | Equal error rates (paired dichotomous) | **McNemar**               |
| models (your NPZ path)    | Asymptotic form uses chi-squared.        | (+ exact binomial)         |
+---------------------------+----------------------------------------+----------------------------+
| Several training seeds,   | Mean metric difference is zero          | **Paired **``t``** **     |
| two models matched        |                                         | (+ **Wilcoxon** optional) |
| per seed (CSV rows)       |                                         |                             |
+---------------------------+----------------------------------------+----------------------------+
| Same seeds, **3+ models** | All methods equivalent                 | **Friedman** (non-par.)    |
+---------------------------+----------------------------------------+----------------------------+

Independent groups (different subjects with no pairing) sometimes use **one-way ANOVA**;
that is **not** the usual deep-learning comparison on a **fixed test set**—use McNemar /
bootstrap instead. Repeated-measures ANOVA exists for matched blocks but Friedman is robust
without normality assumptions.

CLI: no args runs `--demo` (table output). NPZ: `--a`/`--b`. Seed metimage.pngrics CSV: `--paired-csv`.
Full JSON: `--format json`.

Typical workflow
----------------
1. After evaluation, save arrays (e.g. NPZ):
     np.savez("run_baseline.npz", y_true=..., y_pred=..., proba=...)
     np.savez("run_proposed.npz", y_true=..., y_pred=..., proba=...)
2. Run:
     python statistical_tests.py --a run_baseline.npz --b run_proposed.npz

Or import: mcnemar_from_preds, bootstrap_macro_auc_difference, paired_ttest.

McNemar
-------
Tests whether two classifiers disagree symmetrically on the *same* examples.
Use the **exact** variant when b+c is small (default uses exact when b+c < 25).

Bootstrap AUC
---------------
Resamples **test examples** (paired) to get a CI for Δ = AUC(A) − AUC(B).
For multi-class, default is sklearn macro-OvR AUROC.

Seeds / repeated runs
---------------------
Use `summarize_metric_runs` and `paired_ttest_runs` on lists of scalar metrics
from matched seeds (same order = same seed index).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


@dataclass(frozen=True)
class McNemarResult:
    b: int  # only A correct
    c: int  # only B correct
    statistic: float
    pvalue: float
    method: str


def mcnemar_from_preds(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> McNemarResult:
    """Two-sided McNemar test for paired error patterns (same test examples)."""
    y_true = np.asarray(y_true).ravel()
    pred_a = np.asarray(pred_a).ravel()
    pred_b = np.asarray(pred_b).ravel()
    ca = pred_a == y_true
    cb = pred_b == y_true
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    n_disc = b + c
    if n_disc == 0:
        return McNemarResult(b=b, c=c, statistic=0.0, pvalue=1.0, method="no discordant pairs")

    # Exact two-sided binomial test on discordant pairs (standard for small samples)
    if n_disc < 25:
        # B ~ Binomial(b+c, 0.5) under H0; observed count b (or symmetrically c)
        pval = float(stats.binomtest(b, n_disc, 0.5, alternative="two-sided").pvalue)
        chi2_stat = (abs(b - c) - 1) ** 2 / n_disc if n_disc else 0.0
        return McNemarResult(b=b, c=c, statistic=float(chi2_stat), pvalue=pval, method="exact_binomial")

    # Asymptotic with continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / n_disc
    pval = 1.0 - stats.chi2.cdf(chi2_stat, df=1)
    return McNemarResult(b=b, c=c, statistic=float(chi2_stat), pvalue=float(pval), method="chi2_cc")


@dataclass(frozen=True)
class BootstrapAUCResult:
    auc_a_mean: float
    auc_b_mean: float
    delta_mean: float
    ci_low: float
    ci_high: float
    p_two_sided: float  # proportion of bootstrap Δ straddling 0 → approximate


def bootstrap_macro_auc_difference(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    random_state: int | None = 0,
) -> BootstrapAUCResult:
    """Bootstrap CI for macro-OvR AUROC(A) − AUROC(B) over the same test points."""
    y_true = np.asarray(y_true).ravel().astype(np.int64)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    n_classes = proba_a.shape[1]
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))

    def macro_auc(proba: np.ndarray) -> float:
        return float(roc_auc_score(y_bin, proba, multi_class="ovr", average="macro"))

    auc_a_full = macro_auc(proba_a)
    auc_b_full = macro_auc(proba_b)
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    deltas = np.empty(n_bootstrap)
    idx_all = rng.randint(0, n, size=(n_bootstrap, n))
    for i in range(n_bootstrap):
        idx = idx_all[i]
        deltas[i] = macro_auc(proba_a[idx]) - macro_auc(proba_b[idx])

    low, high = np.percentile(deltas, [2.5, 97.5])
    p_approx = 2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    if p_approx > 1.0:
        p_approx = 1.0
    return BootstrapAUCResult(
        auc_a_mean=auc_a_full,
        auc_b_mean=auc_b_full,
        delta_mean=float(np.mean(deltas)),
        ci_low=float(low),
        ci_high=float(high),
        p_two_sided=float(p_approx),
    )


def summarize_metric_runs(values: list[float] | np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        raise ValueError("empty inputs")
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else 0.0
    # 95% normal approx CI around mean when n>=2 (for reporting only)
    half = stats.t.ppf(0.975, df=n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return {"n_runs": n, "mean": mean, "std": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def paired_ttest_runs(
    metric_a: list[float] | np.ndarray,
    metric_b: list[float] | np.ndarray,
) -> dict[str, float]:
    """Two-sided **paired t-test** on matched runs (e.g., same seeds), testing mean(B-A)=0.

    Classic parametric choice when differences are roughly normal; for heavy tails or tiny n use
    `wilcoxon_signed_rank_runs` instead.
    """
    a = np.asarray(metric_a, dtype=np.float64).ravel()
    b = np.asarray(metric_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("metric_a and metric_b must have the same length")
    if a.size < 2:
        raise ValueError("need at least two paired runs for t-test")
    diff = b - a
    n = diff.size
    t_stat, pval = stats.ttest_1samp(diff, 0.0)
    half = stats.t.ppf(0.975, df=n - 1) * float(np.std(diff, ddof=1)) / np.sqrt(n)
    mean_diff = float(np.mean(diff))
    return {
        "mean_delta_B_minus_A": mean_diff,
        "t_statistic": float(t_stat),
        "pvalue_two_sided": float(pval),
        "mean_delta_ci95_low": mean_diff - half,
        "mean_delta_ci95_high": mean_diff + half,
        "n_pairs": int(n),
    }


def wilcoxon_signed_rank_runs(
    metric_a: list[float] | np.ndarray,
    metric_b: list[float] | np.ndarray,
) -> dict[str, float | None]:
    """Two-sided **Wilcoxon signed-rank** on paired differences B-A (non-parametric alternative to paired t-test)."""
    a = np.asarray(metric_a, dtype=np.float64).ravel()
    b = np.asarray(metric_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("metric_a and metric_b must have the same length")
    diff = (b - a).astype(np.float64)
    if diff.size == 0:
        raise ValueError("empty inputs")
    all_zero = bool(np.all(diff == 0))
    out: dict[str, float | None] = {"n_pairs": int(diff.size), "wilcoxon_statistic": None, "pvalue_two_sided": None}
    if all_zero:
        out["pvalue_two_sided"] = 1.0
        return out
    try:
        kwargs = {"zero_method": "wilcox", "alternative": "two-sided"}
        try:
            res = stats.wilcoxon(diff, **kwargs, method="auto")
        except TypeError:
            res = stats.wilcoxon(diff, **kwargs)
        out["wilcoxon_statistic"] = float(res.statistic)
        out["pvalue_two_sided"] = float(res.pvalue)
    except ValueError as e:
        out["error"] = str(e)
    return out


def friedman_test_runs(*metric_columns: list[float] | np.ndarray) -> dict[str, float]:
    """**Friedman test**: same “blocks” (e.g., seeds) across **3 or more** models/treatments.

    Non-parametric analogue of repeated-measures one-way ANOVA when normality is doubtful.
    """
    arrs = [np.asarray(c, dtype=np.float64).ravel() for c in metric_columns]
    if len(arrs) < 3:
        raise ValueError("Friedman needs at least 3 paired columns (methods); use paired t-test / Wilcoxon for two")
    shapes = [a.size for a in arrs]
    if len(set(shapes)) != 1:
        raise ValueError("all metric columns must have the same length (matched seeds)")
    n = shapes[0]
    if n < 3:
        raise ValueError("Friedman is unreliable with very few blocks; collect more seeds or cite small-n limits")
    fr = stats.friedmanchisquare(*arrs)
    return {
        "friedman_chi2_statistic": float(fr.statistic),
        "pvalue_two_sided": float(fr.pvalue),
        "n_blocks": int(n),
        "n_models_k": len(arrs),
    }


def f_oneway_independent_groups(
    *independent_metric_groups: list[float] | np.ndarray,
) -> dict[str, float]:
    """One-way ANOVA (``scipy.stats.f_oneway``) for **unrelated** groups.

    Use **only** when measurements are statistically independent—not the usual case when the
    same seeds are reused across methods (prefer Friedman instead).
    """
    groups = [np.asarray(g, dtype=np.float64).ravel() for g in independent_metric_groups]
    if len(groups) < 2:
        raise ValueError("need at least two groups")
    f = stats.f_oneway(*groups)
    return {
        "f_statistic": float(f.statistic),
        "pvalue": float(f.pvalue),
        "group_sizes": [int(g.size) for g in groups],
    }


def load_paired_metrics_csv(path: str | Path) -> tuple[list[str], dict[str, list[float]]]:
    """CSV: header row; each row one run (usually a seed). Optional column ``seed`` (ignored numeric)."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not rows:
        raise ValueError("empty CSV: %s" % path)
    assert fieldnames is not None
    skip = {"seed", ""}
    columns = []
    raw: dict[str, list[float]] = {}
    for name in fieldnames:
        if name is None or name.strip().lower() in skip:
            continue
        columns.append(name)
        raw[name] = []
        for row in rows:
            v = row.get(name)
            if v is None or v.strip() == "":
                raise ValueError("missing value for column %s" % name)
            raw[name].append(float(v))
    return columns, raw


def analyze_metric_columns(columns: dict[str, list[float]], compare: tuple[str, str] | None = None) -> dict:
    """Run paired ``t``, Wilcoxon, and Friedman (if k>=3) on columns keyed by model name."""
    names = list(columns.keys())
    if len(names) < 2:
        raise ValueError("need at least two numeric columns")
    out: dict = {"columns": names, "friedman": None}
    if compare is None:
        ma, mb = columns[names[0]], columns[names[1]]
        out["paired_between"] = (names[0], names[1])
    else:
        ma = columns[compare[0]]
        mb = columns[compare[1]]
        out["paired_between"] = compare
    out["paired_ttest"] = {"parametric": paired_ttest_runs(ma, mb), "wilcoxon": wilcoxon_signed_rank_runs(ma, mb)}
    if len(names) >= 3:
        arrs = [columns[nm] for nm in names]
        out["friedman"] = friedman_test_runs(*arrs)
    out["means_per_column"] = {n: summarize_metric_runs(columns[n]) for n in names}
    return out


def paired_csv_report_text(summary: dict) -> str:
    """Human-readable block for pasted seed metrics."""
    lines: list[str] = []
    pb = summary.get("paired_between", ("?", "?"))
    lines.append("Matched-run tests (paired t-test / Wilcoxon on %s vs %s)" % pb)
    pt = summary["paired_ttest"]["parametric"]
    lines.append("  Paired t-test  p=%.4f  mean diff (B-A)=%.6f  n=%d" % (pt["pvalue_two_sided"], pt["mean_delta_B_minus_A"], pt["n_pairs"]))
    w = summary["paired_ttest"]["wilcoxon"]
    if w.get("pvalue_two_sided") is not None:
        lines.append("  Wilcoxon       p=%.4f  n=%d" % (w["pvalue_two_sided"], w["n_pairs"]))
    else:
        lines.append("  Wilcoxon       %s" % w.get("error", "(skipped)"))
    fr = summary.get("friedman")
    if fr is not None:
        lines.append("")
        lines.append("Friedman test (all %d models simultaneously, same runs as blocks):" % fr["n_models_k"])
        lines.append("  chi^2=%.4f  p=%.4f  (n_blocks=%d)" % (fr["friedman_chi2_statistic"], fr["pvalue_two_sided"], fr["n_blocks"]))
        lines.append("  (reject only if justified; pairwise post-hoc tests may still be needed).")
    return "\n".join(lines)


def metrics_from_probs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if proba is not None:
        nc = np.asarray(proba).shape[1]
        yb = label_binarize(y_true, classes=np.arange(nc))
        out["macro_ovr_auroc"] = float(roc_auc_score(yb, proba, multi_class="ovr", average="macro"))
    return out


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    d = np.load(path, allow_pickle=False)
    yt = np.asarray(d["y_true"]).ravel()
    yp = np.asarray(d["y_pred"]).ravel()
    proba = np.asarray(d["proba"]) if "proba" in d.files else None
    return yt, yp, proba


def build_report(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    proba_a: np.ndarray | None,
    proba_b: np.ndarray | None,
    *,
    n_bootstrap: int,
    random_state: int,
) -> dict:
    mcn = mcnemar_from_preds(y_true, pred_a, pred_b)
    report: dict = {
        "mcnemar": {
            "b_only_A_correct": mcn.b,
            "c_only_B_correct": mcn.c,
            "statistic": mcn.statistic,
            "pvalue_two_sided": mcn.pvalue,
            "method": mcn.method,
        },
        "metrics_A": metrics_from_probs(y_true, pred_a, proba_a),
        "metrics_B": metrics_from_probs(y_true, pred_b, proba_b),
    }
    if proba_a is not None and proba_b is not None:
        bs = bootstrap_macro_auc_difference(
            y_true, proba_a, proba_b, n_bootstrap=n_bootstrap, random_state=random_state
        )
        report["bootstrap_macro_ovr_auroc"] = {
            "auc_A": bs.auc_a_mean,
            "auc_B": bs.auc_b_mean,
            "delta_mean_bootstrap": bs.delta_mean,
            "ci95_delta": [bs.ci_low, bs.ci_high],
            "p_two_sided_approx": bs.p_two_sided,
        }
    else:
        report["bootstrap_macro_ovr_auroc"] = "skipped (need proba in both NPZs)"
    return report


def report_to_table(
    report: dict,
    *,
    label_a: str = "Model A",
    label_b: str = "Model B",
    decimals: int = 4,
) -> str:
    """Plain-text table: three measures (accuracy, macro-F1, macro-OvR AUROC) for A vs B + diff (B-A)."""
    ma = report["metrics_A"]
    mb = report["metrics_B"]
    measures: list[tuple[str, str]] = [
        ("Accuracy", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("Macro OvR AUROC", "macro_ovr_auroc"),
    ]

    def fmt(x: float | None) -> str:
        return "—" if x is None else f"{float(x):.{decimals}f}"

    lines: list[str] = []
    hdr = f"{'Measure':<22} | {label_a:>{11}} | {label_b:>{11}} | {'Diff (B-A)':>{10}}"
    sep = "-" * len(hdr)
    lines.append(hdr)
    lines.append(sep)
    for name, key in measures:
        va = ma.get(key)
        vb = mb.get(key)
        if va is None or vb is None:
            delta_s = "—"
        else:
            delta_s = f"{float(vb) - float(va):+.{decimals}f}"
        lines.append(
            f"{name:<22} | {fmt(va):>11} | {fmt(vb):>11} | {delta_s:>10}"
        )

    mcn = report.get("mcnemar", {})
    boot = report.get("bootstrap_macro_ovr_auroc")
    lines.append("")
    lines.append("Statistical tests (paired, same test set)")
    if isinstance(mcn, dict) and "pvalue_two_sided" in mcn:
        method = str(mcn.get("method", ""))
        mcn_lab = (
            "McNemar "
            "(exact discordant-binomial)" if method == "exact_binomial"
            else "McNemar (chi-squared approx + cc)"
            if method == "chi2_cc"
            else "McNemar"
        )
        lines.append(f"  {mcn_lab} (two-sided p): {mcn['pvalue_two_sided']:.{max(3, decimals)}f}")
    else:
        lines.append("  McNemar: —")
    if isinstance(boot, dict) and "ci95_delta" in boot:
        lo, hi = boot["ci95_delta"]
        lines.append(
            f"  Bootstrap diff macro-OvR AUROC (95% CI): [{lo:.{decimals}f}, {hi:.{decimals}f}] "
            f"(B minus A on resampled test points)"
        )
    else:
        lines.append("  Bootstrap diff AUROC: — (need proba in both NPZs)")
    return "\n".join(lines)


def demo_synthetic(*, n: int = 300, n_classes: int = 5, n_bootstrap: int = 500, seed: int = 0) -> dict:
    """Same test set; model B slightly more accurate; fake softmax for AUC."""
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, n_classes, size=n)
    pred_a = np.where(rng.rand(n) < 0.82, y_true, rng.randint(0, n_classes, n))
    pred_b = np.where(rng.rand(n) < 0.86, y_true, rng.randint(0, n_classes, n))
    proba_a = rng.dirichlet(np.ones(n_classes), size=n).astype(np.float64)
    proba_b = rng.dirichlet(np.ones(n_classes), size=n).astype(np.float64)
    for i in range(n):
        proba_a[i, pred_a[i]] += 0.35
        proba_b[i, pred_b[i]] += 0.45
        proba_a[i] /= proba_a[i].sum()
        proba_b[i] /= proba_b[i].sum()
    return build_report(
        y_true, pred_a, pred_b, proba_a, proba_b, n_bootstrap=n_bootstrap, random_state=seed + 1
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Paired statistical tests for two saved eval NPZs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python statistical_tests.py --demo
  python statistical_tests.py --a baseline.npz --b proposed.npz
  python statistical_tests.py --paired-csv seeds_accuracy.csv --compare baseline proposed
  python statistical_tests.py --a baseline.npz --b proposed.npz --bootstrap 5000 --seed 42

Each NPZ should contain arrays: y_true (N,), y_pred (N,), proba (N,C) optional but needed for AUC bootstrap.

CSV (--paired-csv): header row + one row per seed; numeric columns named by model (e.g. baseline,concat,avg,proposed).
""",
    )
    p.add_argument("--demo", action="store_true", help="Run on synthetic data (no --a/--b needed)")
    p.add_argument("--a", default=None, help="NPZ with y_true, y_pred, [proba]")
    p.add_argument("--b", default=None, help="NPZ with y_true, y_pred, [proba]")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for AUC delta")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="table = 3 measures + stats lines; json = full machine-readable report",
    )
    p.add_argument("--label-a", default="Model A", metavar="TEXT", dest="label_a")
    p.add_argument("--label-b", default="Model B", metavar="TEXT", dest="label_b")
    p.add_argument(
        "--paired-csv",
        default=None,
        metavar="PATH",
        help=(
            "CSV: first row headers; one row per training seed/run; numeric columns "
            "(optionally omit 'seed' column). Runs paired t-test, Wilcoxon, and Friedman if 3+ models."
        ),
    )
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("COL_A", "COL_B"),
        dest="compare_cols",
        default=None,
        help="Which two CSV columns for paired tests (default: first two numeric columns)",
    )
    args = p.parse_args()

    def emit(report: dict) -> None:
        if args.format == "table":
            if report.get("_note"):
                print(report["_note"], file=sys.stderr)
                report = {k: v for k, v in report.items() if k != "_note"}
            print(report_to_table(report, label_a=args.label_a, label_b=args.label_b))
        else:
            print(json.dumps(report, indent=2))

    if getattr(args, "paired_csv", None):
        _, raw = load_paired_metrics_csv(args.paired_csv)
        cmp_kw: tuple[str, str] | None = None
        if args.compare_cols is not None:
            cmp_kw = (args.compare_cols[0], args.compare_cols[1])
            missing = set(cmp_kw) - set(raw.keys())
            if missing:
                raise SystemExit("missing CSV columns %s (have %s)" % (missing, list(raw.keys())))
        summary = analyze_metric_columns(raw, compare=cmp_kw)
        if args.format == "table":
            print(paired_csv_report_text(summary))
            print("")
            print("Mean +/- summary (matched runs)")
            for name in summary["means_per_column"]:
                m = summary["means_per_column"][name]
                print(
                    "  %s: mean=%.4f std=%.4f [%d runs]"
                    % (name, m["mean"], m["std"], int(m["n_runs"]))
                )
        else:
            print(json.dumps(summary, indent=2))
        return

    if args.demo:
        report = demo_synthetic(n_bootstrap=min(args.bootstrap, 800), seed=args.seed)
        report["_note"] = "synthetic demo; use real NPZs for paper results."
        emit(report)
        return

    if not args.a or not args.b:
        p.error("give --paired-csv, both --a/--b NPZs, or run with --demo")

    y_a, p_a, pr_a = load_npz(args.a)
    y_b, p_b, pr_b = load_npz(args.b)
    if not np.array_equal(y_a, y_b):
        raise SystemExit("y_true must match between A and B (same test order).")

    report = build_report(
        y_a, p_a, p_b, pr_a, pr_b, n_bootstrap=args.bootstrap, random_state=args.seed
    )
    emit(report)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(
            "No args: running synthetic --demo. "
            "For your runs: python statistical_tests.py --a baseline.npz --b proposed.npz "
            "(use --format json for full report)\n",
            file=sys.stderr,
        )
        sys.argv.append("--demo")
    main()
