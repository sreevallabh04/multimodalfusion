# Reply to Reviewer 3

**Manuscript:** "Cross-Modal Knowledge Transfer for Cost-Effective Mango Disease Detection Using Synthetic Thermal Imaging"

**Submitted to:** Frontiers in Agronomy

We sincerely thank Reviewer 3 for the positive assessment of our work and the detailed, constructive suggestions for improvement. We greatly appreciate the recommendation for acceptance after major revision, and we have carefully addressed every point raised. Below, we provide a detailed response to each comment.

---

### Comment 1: Minor edit errors; words written together

> *"The paper is well presented except that there are minor edit errors. Some words are written together and need reediting."*

**Author's response:**

We thank the reviewer for noting these errors. We have carefully proofread the entire manuscript and corrected all instances of words written together and other typographical issues. Specific corrections include:
- "classification.Our" → "classification. Our"
- "physics-informaed" → "physics-informed"
- "scenar:" → "scenario:"
- "lim-ited" → "limited"
- "the the" → "the"

The full manuscript has been thoroughly re-edited to eliminate such errors and improve overall readability.

---

### Comment 2: Abstract — cost-effectiveness not clear

> *"The abstract is good but the cost-effectiveness is not clear here. Authors should add a line or two to back the cost-effectiveness by for example summary of table 3."*

**Author's response:**

We sincerely thank the reviewer for this practical suggestion. The revised abstract now includes explicit cost-effectiveness figures drawn from the cost comparison table. The updated text reads: *"The proposed method achieves approximately 89% of the performance of a real thermal camera system (93.5%) while incurring zero hardware cost compared to \$25,000 for thermal cameras or \$50,000 for multi-spectral systems, making it directly deployable on commodity smartphones for low-cost crop health monitoring."* We believe this addition makes the cost-effectiveness claim concrete and verifiable directly from the abstract.

This change has been incorporated into the Abstract of the revised manuscript.

---

### Comment 3: Introduction — consider merging with related works

> *"The introduction is precise and straight to the point. But if the template allows, maybe the introduction could be merged with the related works."*

**Author's response:**

We thank the reviewer for this thoughtful suggestion. We considered merging the two sections; however, the Frontiers template and journal guidelines recommend maintaining separate Introduction and Related Work sections to allow for a structured presentation. We have, however, ensured that the Introduction now flows naturally into the Related Work section, with the Introduction establishing the problem context and motivation while the Related Work provides the critical discussion of prior approaches and identifies the specific gap our work addresses. Both sections have been substantially expanded in the revised manuscript.

---

### Comment 4: Include flowchart, workflow, and block diagram

> *"Include a flowchart, workflow and block diagram representation of the methodology to provide a pictorial overview for readers similar to figure 2 and 3."*

**Author's response:**

We sincerely thank the reviewer for this excellent suggestion. In the revised manuscript, we have made the following additions:

- **New Section 3.1 ("Overview of the Proposed Framework"):** This subsection provides a textual step-by-step description of the complete 3-stage pipeline before diving into technical details.
- **Figure 1** (`crossmap_architecture_diagram.png`): A comprehensive pipeline overview figure illustrating all three stages — (1) supervised lesion detector training on leaf images, (2) physics-informed thermal map generation for fruit images, and (3) dual-branch attention-based fusion for disease classification. The figure caption describes each stage explicitly.
- **Figure 2** (`Architecture.png`): A detailed block diagram of the fusion architecture showing the dual ConvNeXt-Tiny branches, learned gating mechanisms, 16-head multi-head attention module, channel attention, and progressive MLP fusion network.

Together, these additions provide the pictorial overview requested, enabling readers to grasp the methodology at a glance before reading the detailed descriptions. These changes have been incorporated into Section 3 of the revised manuscript.

---

### Comment 5: Experimental setup is weak — justify hyperparameters

> *"The experimental setup is weak. Authors should explain what informed the data size of 32 not 64 and learning rate of 0.001 and not any other. Why 32 epochs?"*

**Author's response:**

We greatly appreciate the reviewer for raising this important concern. The revised manuscript (Section 4.1) now provides detailed justifications for each hyperparameter choice:

- **Batch size of 32:** We explain that the MangoFruitDDS dataset contains 838 images, yielding approximately 587 training samples after the 70% split. With a batch size of 32, each epoch comprises approximately 18 gradient updates, providing sufficient stochasticity for effective generalization. We note that a larger batch size of 64 was evaluated during preliminary experiments but produced less stable training dynamics due to the reduced number of updates per epoch (approximately 9), consistent with the finding that smaller batches can act as implicit regularizers for small datasets (Noon et al., 2020).

- **Learning rate of 0.001:** We clarify that an initial learning rate of $10^{-3}$ was chosen as the widely recommended starting point for AdamW optimization with pretrained backbones (Loshchilov and Hutter, 2019). We also specify the cosine annealing schedule with warm restarts ($T_0 = 10$, $T_{\text{mult}} = 2$, $\eta_{\text{min}} = 10^{-6}$) that provides periodic learning rate increases to help escape local minima.

- **Training duration:** We wish to clarify that the model is **not** trained for a fixed number of 32 epochs. Instead, we employ **early stopping with a patience of 15 epochs** monitoring validation accuracy, allowing training to proceed up to 50 epochs. In practice, models typically converge between 30–40 epochs, as confirmed by the training accuracy and loss curves presented in Figure 11 of the revised manuscript. This approach avoids both underfitting (from too few epochs) and overfitting (from excessive training).

Additionally, we describe the regularization strategy (weight decay of $10^{-4}$, label smoothing of 0.1, gradient clipping with max norm of 1.0) and the fusion training protocol (RGB branch frozen for 15 epochs, then unfrozen with reduced learning rate). These revisions have been incorporated into Section 4.1 of the revised manuscript.

---

### Comment 6: Table of comparison for fair conclusion

> *"A table of comparison would be very useful for fair conclusion."*

**Author's response:**

We sincerely thank the reviewer for this valuable suggestion. We have added a new table (Table 4 in the revised manuscript) titled "Comparison with published state-of-the-art methods for fruit/mango disease detection." This table compares our method against five published approaches:

| Method | Dataset | Accuracy | Modality |
|--------|---------|----------|----------|
| Mohanty et al. (2016) | PlantVillage | 99.4% | RGB |
| Ferentinos (2018) | PlantVillage (ext.) | 99.5% | RGB |
| Singh et al. (2019) | Mango Leaf | 97.1% | RGB |
| Hassan et al. (2021) | PlantVillage | 96.5% | RGB |
| Ahmad et al. (2023) | Mango Fruit | 91.2% | RGB |
| **Proposed Method** | **MangoFruitDDS** | **87.3%** | **RGB + Synth. Thermal** |

We transparently acknowledge that direct comparison is challenging because methods use different datasets and experimental protocols, and we provide a detailed discussion explaining the differences in dataset difficulty. For instance, the higher accuracies on PlantVillage reflect its controlled laboratory conditions, whereas MangoFruitDDS presents a more challenging task with natural imaging conditions. This comparison has been added to Section 5.2 of the revised manuscript.

---

### Comment 7: Influence of flip/rotation on results

> *"What is the influence of the flip rotation on results?"*

**Author's response:**

We thank the reviewer for this insightful question. We have added a new subsection (Section 5.8, "Influence of Data Augmentation") with a dedicated table (Table 7) showing the incremental effect of each augmentation strategy:

| Augmentation Strategy | Accuracy |
|-----------------------|----------|
| No augmentation | 83.6% |
| + Horizontal/vertical flips | 84.8% (+1.2%) |
| + Flips + rotations (±15°) | 85.6% (+0.8%) |
| + Flips + rotations + photometric | 86.7% (+1.1%) |
| + All augmentations (full pipeline) | 87.3% (+0.6%) |

The results show that horizontal and vertical flips contribute a 1.2% accuracy gain, while rotations add a further 0.8%. We also explain that rotation limits were set conservatively (±15° for RGB, ±10° for thermal) to avoid introducing unrealistic orientations that could degrade the physics-informed structure of the synthetic thermal maps. These revisions have been incorporated into Section 5.8 of the revised manuscript.

---

### Comment 8: Missing SOTA comparison table in results

> *"The section results and analysis is missing table of comparison with other related or state of the art works."*

**Author's response:**

We thank the reviewer for reiterating this important point. As described in our response to Comment 6 above, we have added Table 4 (state-of-the-art comparison) to the Results and Analysis section (Section 5.2) of the revised manuscript. The table includes five published methods with their respective datasets, accuracies, and modalities, accompanied by a detailed discussion contextualizing the results.

---

### Comment 9: Discussion limited; title too broad

> *"The discussion section is limited. Work is limited to only mango disease but topic says agricultural disease detection, which is broader. Maybe topic should be modified for mango disease only."*

**Author's response:**

We sincerely thank the reviewer for this astute observation. We have addressed this concern in two ways:

1. **Title revised:** The title has been changed from "Cross-Modal Knowledge Transfer for Cost-Effective Multi-Modal Agricultural Disease Detection" to **"Cross-Modal Knowledge Transfer for Cost-Effective Mango Disease Detection Using Synthetic Thermal Imaging."** This accurately reflects the scope of the experimental evaluation.

2. **Discussion section substantially expanded:** The Discussion has been expanded from two brief paragraphs to four comprehensive subsections:
   - **Section 6.1 (Technical Contributions and Analysis):** Provides deeper analysis of the biological rationale for cross-organ transfer and interprets the per-class results, explaining why Stem/Rot (+15.3% F1) benefits most from synthetic thermal cues while Black Mould (+0.0%) does not.
   - **Section 6.2 (Practical Impact and Cost-Effectiveness):** Quantifies the cost-effectiveness advantage with an expanded comparison table now including Expert Field Diagnosis as an additional reference point.
   - **Section 6.3 (Scope and Generalizability):** A new subsection explicitly addressing the mango-specific scope while discussing the framework's generalizability to other fruit crops. We explain that the choice of mango was motivated by dataset availability and economic importance, and that the thermal synthesis and fusion modules are crop-agnostic.
   - **Section 6.4 (Limitations):** Expanded to five specific, clearly articulated limitations with supporting citations.

These revisions have been incorporated into Section 6 of the revised manuscript.

---

We are deeply grateful to Reviewer 3 for the constructive feedback and the recommendation for acceptance. We believe the revised manuscript now comprehensively addresses all the concerns raised, and we hope the revisions meet the reviewer's expectations.
