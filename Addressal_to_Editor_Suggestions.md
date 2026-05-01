# Addressal to Editor Suggestions

**Manuscript:** *Cross-Modal Knowledge Transfer for Cost-Effective Mango Disease Detection Using Synthetic Thermal Imaging*

This document provides a concise, point-by-point response to the editor’s comments and summarizes the corresponding revisions implemented in the updated manuscript (see `Paper/frontiers_clean_without highlights.tex`; highlighted changes are shown in `Paper/frontiers_highlighted.tex`).

---

## Comment 1
**Comment:** The manuscript presents an interesting and practically relevant approach for low-cost mango disease detection using cross-modal knowledge transfer and synthetic thermal imaging; however, it currently lacks sufficient scientific rigor for direct acceptance.

**Response:** We are grateful to the editor and the editorial team for the careful evaluation of our manuscript and for the constructive summary of concerns. We fully agree that clearer validation framing, tighter scope of claims, and stronger statistical reporting are essential for a rigorous presentation. Accordingly, we revised the manuscript to (i) explicitly bound the scope of claims to what is supported by the available public data protocol, (ii) add a dedicated rigor/assumptions discussion, and (iii) add paired statistical inference to avoid relying on point metrics alone.

**Implemented in manuscript:** Discussion subsections **Rigor, assumptions, and robustness** and **Justification of the RGB-only setting and scope of thermal comparison**; addition of paired inference in **Statistical testing** and **Paired statistical significance analysis**.

---

## Comment 2
**Comment:** The most critical concern is the absence of validation against real thermal data, which weakens the central claim that the synthetic maps approximate true thermal signals.

**Response:** We thank the editor for highlighting this central point. Acquiring paired RGB and calibrated LWIR imagery of diseased mango fruit at scale was not feasible within our institutional and field constraints (cost of scientific-grade thermal systems, orchard access aligned with mango phenology/seasonality, and absence of a suitable public paired benchmark aligned with MangoFruitDDS). Importantly, in the revised manuscript we **do not claim** pixel-level or radiometric equivalence between synthetic maps and measured thermal radiance.

In the revised manuscript we:
- **Reframe** the “Thermal Camera” row as a **literature-aligned reference/upper-bound context**, not a head-to-head benchmark on identical specimens (see subsection **Overall Performance Comparison** and the table titled **“Performance comparison of various methods on the test dataset MangoFruitDDS.”**).
- **Add a dedicated justification subsection** explaining why paired thermal ground truth is absent and why the method is explicitly aimed at **zero-incremental-hardware** deployment settings (Discussion).
- **Identify paired RGB–thermal acquisition** as explicit future work when such datasets become available (Future Work).

**Implemented in manuscript:** subsection **Overall Performance Comparison** and the Discussion subsection **Justification of the RGB-only setting and scope of thermal comparison**.

---

## Comment 3
**Comment:** The “physics-informed” model appears largely heuristic without proper calibration or justification, the dataset size is relatively small for the complexity of the deep learning architecture, and the reported performance gains are modest and somewhat overstated (relative vs. absolute improvement).

**Response:** We agree that the synthesis stage must not be presented as calibrated thermal physics when paired RGB–LWIR data are unavailable. The revised manuscript clarifies that the lesion-to-thermal transformation is a **physically motivated / physiology-inspired** synthesis that imposes interpretable structure (e.g., monotonic mapping from lesion confidence to pseudo-temperature surfaces with diffusion/noise terms), rather than a calibrated thermometric reconstruction.

We also address dataset size and performance framing as follows:
- **Dataset size vs. model complexity:** we add an explicit **Adequacy of sample size** paragraph in the Dataset section that (i) states the MangoFruitDDS split sizes, (ii) clarifies that MangoLeafBD is used for lesion pretraining/transfer, and (iii) justifies stability for comparative conclusions under stratification + augmentation + early stopping + regularization.
- **Modest gains / relative vs. absolute:** we keep gains stated as **absolute percentage points** and avoid overstated language; the baseline used for the 4.8-point comparison is explicitly identified (ViT-Base at 85.8%).

**Implemented in manuscript:**
- Subsection **Stage 2: Physics-Informed Thermal Synthesis** (clarified formulation and parameterization)
- Dataset section: paragraph **Adequacy of sample size**
- Abstract + Results framing: baseline stated explicitly and thermal reference contextualized.

---

## Comment 4
**Comment:** The study also does not include statistical significance testing, and the comparison with a real thermal camera system is not adequately supported.

**Response:** We thank the editor for pointing out this gap. The revised manuscript adds **paired inferential procedures** appropriate to a fixed test partition:
- **McNemar’s test** for paired correct/incorrect outcomes (two-sided, \(\alpha = 0.05\))
- **Bootstrap resampling** (\(B = 2000\)) to report a \(95\%\) percentile interval for \(\Delta\)macro-OvR AUROC

We also make the thermal-camera comparison transparent: the “Thermal Camera” row is now explicitly described as a **literature-aligned reference** (contextual ceiling), not a concurrently measured result on the same stratified test split.

**Implemented in manuscript:**
- Subsubsection **Statistical testing** (Methods)
- Subsubsection **Paired statistical significance analysis** and the table titled **“Paired evaluation on the MangoFruitDDS test split: Average Fusion (baseline) vs. Proposed Method (attention-based fusion model).”** (Results)
- Thermal reference framing in subsection **Overall Performance Comparison**, immediately preceding the table titled **“Performance comparison of various methods on the test dataset MangoFruitDDS.”**

---

## Comment 5
**Comment:** While the methodology is innovative and the application has strong potential, substantial revisions are needed to strengthen experimental validation, clarify assumptions, and improve the robustness of the results before the manuscript can be considered for publication.

**Response:** We appreciate this assessment and have revised the manuscript to clarify assumptions and improve robustness reporting while keeping claims evidence-based. In particular, we (i) add a dedicated rigor/assumptions discussion, (ii) explicitly scope claims to comparative classification performance on the available public protocol, and (iii) state the absence of paired thermal ground truth as a constraint motivating the “RGB-only + synthetic thermal” option for low-cost deployment rather than as an unsupported equivalence claim.

**Implemented in manuscript:** Discussion subsections **Rigor, assumptions, and robustness** and **Justification of the RGB-only setting and scope of thermal comparison**, plus the Future Work item on paired RGB–thermal benchmarking.

