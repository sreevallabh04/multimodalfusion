# Reply to Editor Comments

**Manuscript:** "Cross-Modal Knowledge Transfer for Cost-Effective Mango Disease Detection Using Synthetic Thermal Imaging"

**Submitted to:** Frontiers in Agronomy

We sincerely thank the Editor for the thorough and constructive evaluation of our manuscript. We have carefully addressed each of the concerns raised, and the manuscript has undergone substantial revision accordingly. Below, we provide a point-by-point response to each comment.

---

### Comment 1: Introduction and Literature Review are underdeveloped

> *"The Introduction and Literature Review are currently underdeveloped. While the Introduction outlines the general background, motivation, and research question, it does not adequately engage with recent advancements in the field. Similarly, the Literature Review sections are overly brief and lack critical depth. The authors are encouraged to substantially expand these sections by incorporating and critically discussing more recent studies (preferably up to 2025) to better position their work within the current state of research."*

**Author's response:**

We sincerely thank the Editor for this important observation. We fully acknowledge that the previous versions of the Introduction and Literature Review were insufficient in scope and depth. In the revised manuscript, we have made the following changes:

- **Introduction (Section 1):** The Introduction has been expanded from a single short paragraph to five substantial paragraphs. It now engages with recent advancements including comprehensive reviews of vision-based machine learning for plant disease detection (Thakur et al., 2022), deep learning in precision agriculture (Coulibaly et al., 2022), and the limitations of single-modality approaches (Abade et al., 2021; Hasan et al., 2020). We have also incorporated discussion of knowledge distillation (Hinton et al., 2015; Li et al., 2023) and multimodal learning with transformers (Xu et al., 2023) to better position our cross-modal transfer approach within current research trends.

- **Related Work (Section 2):** Each subsection has been substantially expanded with critical discussion. The "Multi-Modal Agricultural Sensing" subsection now includes references to hyperspectral imaging (Nagasubramanian et al., 2019) and modern backbone architectures such as ConvNeXt (Liu et al., 2022). The "Cross-Modal Learning and Knowledge Transfer" subsection now critically discusses knowledge distillation frameworks, multimodal transformer surveys, and recent applications in crop disease recognition (Ji et al., 2023; Wang et al., 2022). The "Smartphone-Based Agricultural AI" subsection now includes recent mango-specific deep learning work by Ahmad et al. (2023) and identifies the specific gap our work addresses.

In total, approximately 15 new references (2019–2023) have been added to the bibliography, and both sections now provide the critical depth required to contextualize our contributions within the current state of research.

---

### Comment 2: Methodology and Results lack reproducibility detail

> *"The Methodology and Results sections lack the level of detail required for reproducibility. For instance, the manuscript states that two datasets were used with a 70:20:20 split, but it is unclear whether this split was applied independently to each dataset or to a combined dataset. Furthermore, the presentation of results does not clearly distinguish between datasets, making interpretation difficult. The manuscript also claims that the second dataset serves as a knowledge source for thermal synthesis; however, there is no clear evidence or description indicating that this dataset includes thermal imagery."*

**Author's response:**

We greatly appreciate the Editor for identifying these critical ambiguities. We have comprehensively revised the Methodology and Results sections to address each concern:

- **Data split clarification:** We have corrected the split ratio description (it is 70/15/15, not 70/20/20 as previously unclear) and explicitly state in Section 3.2 of the revised manuscript: *"Each dataset is independently split into 70% training, 15% validation, and 15% testing sets using stratified random sampling to preserve class distributions across splits. The splits are applied independently to each dataset: MangoLeafBD is split for training and evaluating the lesion detector, while MangoFruitDDS is split for training and evaluating the final fusion classifier."*

- **Dataset roles clarified:** We have added a new summary table (Table 1) that clearly distinguishes the two datasets by their number of images, classes, organ type, and specific role in the pipeline. Each dataset now has a dedicated paragraph with a bold heading (MangoFruitDDS as "Target Dataset" and MangoLeafBD as "Source Dataset for Knowledge Transfer").

- **MangoLeafBD and thermal imagery:** We now explicitly clarify in Section 3.2 that MangoLeafBD **does not contain any thermal imagery**. The revised text reads: *"Importantly, MangoLeafBD does not contain any thermal imagery; rather, it provides pathological knowledge about how diseases manifest visually in mango plant tissue."* We explain the biological rationale: fungal and bacterial infections produce similar cellular stress responses (chlorosis, necrosis, lesion formation) across different organs of the same plant species, making the learned lesion patterns transferable from leaves to fruits for synthetic thermal signature generation.

- **Results presentation:** All result tables and figures now explicitly reference the MangoFruitDDS test set, removing any ambiguity about which dataset results correspond to.

---

### Comment 3: Results not benchmarked against existing methods; baseline not defined

> *"In addition, the results are not benchmarked against existing methods. The abstract mentions a 4.8% improvement over a 'best' baseline, but the baseline itself is not defined or referenced, which undermines the credibility of this claim."*

**Author's response:**

We thank the Editor for this valid criticism. In the revised manuscript, we have addressed this in two ways:

- **Baseline explicitly defined:** The "best baseline" is now clearly identified as **RGB ViT-Base achieving 85.8% accuracy** throughout the manuscript — in the abstract, in the contributions list (Section 1), in the Results section (Section 5.1), and in the Conclusion. The table caption for Table 2 now explicitly states: *"The best RGB-only baseline is ViT-Base (85.8%). Our proposed method achieves 87.3%, a 4.8% absolute improvement."*

- **State-of-the-art comparison table added:** We have added a new table (Table 4 in the revised manuscript) titled "Comparison with published state-of-the-art methods for fruit/mango disease detection." This table includes results from Mohanty et al. (2016), Ferentinos (2018), Singh et al. (2019), Hassan et al. (2021), and Ahmad et al. (2023). We acknowledge that direct comparison is challenging due to different datasets and protocols, and we provide a detailed discussion explaining the differences in dataset difficulty and imaging conditions that account for the variation in reported accuracies across studies.

---

### Comment 4: Methodology and experimental design require substantial revision

> *"Overall, the methodology and experimental design require substantial revision and clarification. In its current form, the manuscript does not meet the standards of rigor and transparency necessary for publication."*

**Author's response:**

We sincerely appreciate the Editor's candid assessment, and we have undertaken a comprehensive revision of the manuscript to meet the required standards of rigor and transparency. Key improvements include:

- **Methodology restructured as a 3-stage pipeline** with a new "Overview of the Proposed Framework" subsection (Section 3.1) and corresponding figure providing a high-level workflow.
- **Feature extraction and thermal synthesis** are now described in full detail, including the supervised ResNet-18 lesion detector architecture, spatial attention formulation, physics-informed thermal model with equation-level breakdown of each component, and a step-by-step pseudocode algorithm.
- **Fusion architecture** is now fully specified: dual ConvNeXt-Tiny branches, 512-dimensional feature space, learned gating mechanisms, 16-head multi-head attention, channel attention, progressive MLP fusion with residual connections, and full equation set.
- **Experimental setup** now includes detailed justifications for all hyperparameter choices (batch size, learning rate, training duration), a complete description of per-modality data augmentation strategies, and the fusion training protocol (RGB pretraining, branch freezing, learning rate reduction).
- **New ablation study table** quantifying the contribution of each component, and a **new data augmentation ablation table** analyzing the influence of geometric and photometric transforms.
- **Discussion section** expanded to four subsections with deeper technical analysis, cost-effectiveness quantification, scope and generalizability discussion, and explicit limitations.

We believe the revised manuscript now meets the standards of rigor and transparency necessary for publication, and we are grateful for the Editor's guidance in improving our work.
