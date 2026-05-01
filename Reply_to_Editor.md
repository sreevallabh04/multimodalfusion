# Reply to the Editor

**Manuscript:** "Cross-Modal Knowledge Transfer for Cost-Effective Mango Disease Detection Using Synthetic Thermal Imaging"

**Submitted to:** Frontiers in Agronomy

We are grateful to you and the editorial team for the time taken to evaluate our manuscript and for the thoughtful summary of concerns. We fully agree that clearer validation, tighter framing of claims, and stronger statistical support are essential for a rigorous presentation of this work. We have revised the manuscript accordingly and briefly address each theme from your letter below.

---

### On overall scientific rigor and scope of acceptance

We thank the editor for recognizing the practical relevance of the problem and the innovation of the proposed pipeline. We accept that the previous version did not fully separate **demonstrated algorithmic behaviour** on the available public protocol from **stronger claims** that would require instrumental thermal validation. The revised manuscript tightens language throughout (Abstract, Discussion, and a dedicated subsection on justification of the RGB-only setting) so that contributions are stated in line with what the data and benchmarks can actually support.

---

### On validation against real thermal data

We thank the editor for highlighting this central point. Acquiring paired RGB and calibrated LWIR imagery of diseased mango fruit at scale was not feasible within our institutional and field constraints (cost of scientific-grade thermal systems, orchard access aligned with mango phenology, and absence of a suitable public paired benchmark for MangoFruitDDS-style splits). We do not claim pixel-level or radiometric equivalence between synthetic maps and measured thermal radiance.

In the revised manuscript we: (i) **reframe** the thermal reference in the results table explicitly as a **literature-informed upper-bound context**, not a head-to-head measurement on identical specimens; (ii) **add a justification-oriented discussion** explaining why paired thermal ground truth is absent and why the method is aimed at **zero-incremental-hardware** deployment; and (iii) **identify paired acquisition** as explicit future work when data become available.

---

### On the “physics-informed” synthesis stage (heuristic vs. calibrated)

We agree that the term “physics-informed” must not overstate fidelity. The revised manuscript **clarifies** that parameters (baseline, metabolic scaling, diffusion, noise) are **heuristic priors** motivated by plant pathology and heat-diffusion intuition, **not** calibrated from co-registered thermal imagery. We state explicitly that the stage encodes **interpretable structure** (e.g., monotonic mapping from lesion confidence to pseudo-temperature patterns) rather than a full continuum heat-transfer inversion, and we connect this to the intentional constraint when paired RGB–thermal fruit data are unavailable.

---

### On dataset size vs. model complexity

We thank the editor for this concern. The revised text **clarifies the role of each data source**: curated fruit images for the classification task, with **leaf imagery used for lesion-detector pretraining** (transfer), **stratified splits**, augmentation, early stopping, and regularization. We **acknowledge** that MangoFruitDDS is moderate in scale and frame conclusions as **comparative** under a fixed protocol, while noting that larger field corpora would strengthen external validation.

---

### On reported performance gains (relative vs. absolute improvement)

We agree that framing matters. The manuscript now **reports both absolute and relative** improvements where relevant and **softens wording** so that gains (e.g., over strong RGB-only baselines) are not overstated. The abstract and discussion **define** the “best baseline” and **contextualize** performance against the literature-style thermal ceiling as a **ratio**, without implying experimental parity with a calibrated camera on our test split.

---

### On statistical significance testing

We thank the editor for this important gap. The revised manuscript includes **paired inferential procedures** appropriate to the fixed test partition (e.g., McNemar-style analysis and bootstrap summaries for AUROC differences, with full reporting in the main text and/or a dedicated table), so that improvements are not supported by point metrics alone. We state limitations of these tests (e.g., exchangeability **within** the fixed split, not across new orchards).

---

### On comparison with a real thermal camera system

We agree the comparison must be **transparent**. The thermal row is now described as a **reference level from the thermal plant-phenotyping literature**, not a concurrently measured row on our stratified split. We explain **why** a same-dataset thermal benchmark was not obtained and how readers should interpret the **cost–accuracy** narrative without equating it to controlled radiometric benchmarking.

---

### Closing

We believe these revisions materially strengthen experimental transparency, statistical support, and careful framing of assumptions. We hope the revised manuscript will be found suitable for further consideration. We remain happy to implement any additional adjustments the editorial team considers necessary.

Thank you again for your guidance.

Sincerely,  
The Authors
