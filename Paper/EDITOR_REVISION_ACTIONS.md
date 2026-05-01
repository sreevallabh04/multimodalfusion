# Manuscript revisions from editor comments

This document turns the editor’s review into **concrete changes** to aim for in the next revision. Grouped by theme; each item should be reflected in **Methods**, **Results**, **Discussion**, and/or **new experiments** as appropriate.

---

## 1. Validation against real thermal data (highest priority)

**Editor concern:** Lack of validation against real thermal data weakens the claim that synthetic thermal maps approximate true thermal signals.

**Suggested changes:**

- **Collect or obtain** a subset of paired RGB + real thermal images (same leaves/sessions as possible), or a small external thermal dataset with compatible disease classes.
- **Add a dedicated experiment section** (e.g., “Validation on real thermal imagery”) that reports:
  - Correlation / agreement between synthetic thermal predictions and real thermal measurements (per-pixel or region-level, as appropriate).
  - Whether models trained on synthetic thermal still generalize when tested on real thermal (train/val splits described clearly).
- **Update the central claim** so it is bounded: e.g., synthetic thermal as a *proxy* validated to degree *X* on real data, not as a drop-in replacement for a calibrated camera without qualification.
- **Figures/tables:** add at least one quantitative comparison (scatter, Bland–Altman, or RMSE/MAE between synthetic vs real where alignment is possible).

---

## 2. “Physics-informed” thermal synthesis — calibration and justification

**Editor concern:** The approach appears heuristic without proper calibration or justification.

**Suggested changes:**

- **Rename or qualify** “physics-informed” if the method is mainly learned heuristics plus priors (e.g., “physically motivated” or “guided by simple heat-transfer priors”).
- **Methods:** document every physical assumption (heat conduction model, emissivity, ambient terms, etc.) and state what is **fitted from data** vs **fixed by hand**.
- **Add a calibration subsection:** how parameters were set (literature values, grid search, fit to a small labeled real-thermal subset, etc.).
- **Ablation:** one experiment removing or randomizing the “physical” components to show they matter (or honestly report if gains are small).
- **Limitations:** acknowledge where the model departs from rigorous physics.

---

## 3. Dataset size vs. model complexity

**Editor concern:** Dataset is relatively small for the complexity of the architecture.

**Suggested changes:**

- **Report dataset statistics clearly:** total images, per-class counts, train/val/test splits, augmentation.
- **Discuss capacity:** why ResNet/ViT scales are chosen; consider **smaller backbones** or **stronger regularization** with comparable results to show the large model is warranted.
- Optionally add **cross-validation** or **multiple random seeds** (ties to significance below).
- **Discussion/limitations:** state explicitly that results may not transfer to much larger or wilder field datasets without retraining.

---

## 4. Modest gains — relative vs. absolute improvement

**Editor concern:** Performance gains are modest and somewhat overstated.

**Suggested changes:**

- **Results tables:** present **both** absolute metrics (e.g., +X accuracy points) **and** relative change (e.g., +Y%); avoid headlines that only highlight large relative improvements on small baselines.
- **Tone down** superlatives (“substantially”, “dramatically”) unless numbers support them.
- **Clarify contribution:** frame the work as **incremental but practical** (cost, deployability) rather than SOTA-by-large-margin unless you have strong external benchmarks.
- **Side-by-side** with the strongest fair baselines under the same data and protocol.

---

## 5. Statistical significance testing

**Editor concern:** No statistical significance testing.

**Suggested changes:**

- **Multiple runs** with different random seeds (document seed count and variance).
- Report **mean ± std** (or confidence intervals) for key metrics (accuracy, macro-F1, AUC).
- Where applicable: **paired tests** (e.g., McNemar for classification) or bootstrap CIs for AUC differences — cite standard references.
- State clearly if the **test set is fixed** and repetitions are only on training variability.

---

## 6. Comparison with a real thermal camera system

**Editor concern:** Comparison with a real thermal camera is not adequately supported.

**Suggested changes:**

- If a **real thermal camera** was used anywhere: full **hardware model**, capture protocol, preprocessing, and alignment with RGB.
- If the “thermal camera” row in results is **literature or a separate experiment**, label it as such and avoid implying same pipeline as RGB/synthetic without evidence.
- Prefer **same test set** comparisons: proposed (RGB+synthetic) vs. RGB-only vs. real-thermal model when data exist.
- **Cost/latency** comparison can support “practical relevance” if thermal hardware is expensive or unwieldy—but keep claims **evidence-based**.

---

## 7. Cross-cutting: rigor, assumptions, and robustness

**Editor concern:** Strengthen experimental validation, clarify assumptions, improve robustness.

**Suggested changes:**

- **Assumptions subsection** in Methods (lighting, lesion visibility, domain shift, synthetic–real gap).
- **Failure cases** or qualitative examples where synthetic thermal misleads the model.
- **Reproducibility:** code/data availability statement, hyperparameters summarized in one table.
- **Ethics / deployment:** if claiming agricultural use, briefly note generalization to other cultivars or regions as future work.

---

## Suggested order of work

1. Real-thermal validation experiment (or strongest feasible proxy) + rewritten claims.  
2. Statistical reporting (seeds / CIs).  
3. Clarify physics-informed wording, calibration, ablations.  
4. Reframe relative vs. absolute improvements and thermal-camera comparisons.  
5. Dataset scale and model complexity discussion + limitations polish.

---

*Generated as a planning aid from the editor’s summary; adapt section names to your journal’s structure.*
