# Reply to Reviewer 1

**Manuscript:** "Cross-Modal Knowledge Transfer for Cost-Effective Mango Disease Detection Using Synthetic Thermal Imaging"

**Submitted to:** Frontiers in Agronomy

We sincerely thank Reviewer 1 for the careful and insightful review of our manuscript. The comments have been instrumental in improving the clarity and rigor of our work. Below, we provide a detailed point-by-point response to each concern raised.

---

### Comment 1: Thermal-like map generation process not sufficiently explained

> *"The manuscript mentions generating thermal-like maps from lesion features extracted from leaf images, but the process is not sufficiently explained. The authors should describe:*
> - *The specific feature extraction techniques used*
> - *How lesion features are transformed into thermal-like representations*
> - *Whether any supervised or unsupervised learning approach is used for map generation*
>
> *A detailed explanation of this step is necessary because it forms the core contribution of the study."*

**Author's response:**

We sincerely thank the reviewer for this critical observation. We fully agree that this step constitutes the core contribution of the study and warrants thorough explanation. The revised manuscript now provides complete detail in Sections 3.4 (Stage 1) and 3.5 (Stage 2):

**Specific feature extraction techniques (Section 3.4):** The lesion detector employs a **ResNet-18** backbone trained on the MangoLeafBD dataset. The model incorporates a **spatial attention mechanism** that weights feature map regions proportionally to disease likelihood. We now provide the explicit formulation: the spatial attention is computed as $A(x) = \sigma(\text{Conv}_{1 \times 1}(F(x))) \odot (1 - P_{\text{healthy}}(x))$, where $F(x)$ are intermediate feature maps and $P_{\text{healthy}}$ is the predicted probability of the healthy class. This produces a spatial probability map $P_{\text{lesion}}(x, y) \in [0, 1]$ encoding disease likelihood across tissue regions.

**Lesion-to-thermal transformation (Section 3.5):** The lesion probability maps are transformed into synthetic thermal signatures through a **physics-informed model** grounded in plant pathology. We now provide the complete thermal synthesis equation with a detailed breakdown of each component:
- $T_{\text{base}} = 0.3$: normalized baseline temperature of healthy tissue
- $\alpha = 0.7$: metabolic stress scaling factor, reflecting empirical observations that diseased regions exhibit temperature increases of up to 70% above baseline
- $\mathcal{M}(\cdot)$: metabolic dysfunction model mapping lesion probabilities to temperature elevations
- $\mathcal{D}(G_{\sigma})$: thermal diffusion simulation via Gaussian blur (kernel $15 \times 15$, $\sigma = 3.0$)
- $\mathcal{E}(\mathcal{N}(0, \beta))$: environmental noise ($\beta = 0.05$) modeling natural temperature fluctuations

Additionally, we have included **Algorithm 1**, a step-by-step pseudocode of the complete thermal synthesis pipeline, to ensure full reproducibility.

**Supervised vs. unsupervised (Section 3.4):** We now explicitly state (in bold) that the lesion detector is trained in a **"supervised"** manner on the MangoLeafBD dataset for leaf disease classification. The thermal map generation itself is a deterministic, physics-informed transformation applied to the supervised model's output — it does not involve any additional learning.

These revisions have been incorporated into Sections 3.4 and 3.5 of the revised manuscript.

---

### Comment 2: Fusion architecture and attention mechanism unclear

> *"The study states that RGB images and generated thermal maps are fused using an attention mechanism, but the architecture and implementation details are unclear. The authors should clarify:*
> - *The type of attention mechanism used*
> - *The deep learning architecture applied*
> - *How the fusion improves disease classification performance*
>
> *Providing a block diagram or architectural illustration would greatly improve clarity."*

**Author's response:**

We sincerely thank the reviewer for this valuable suggestion. We agree that the fusion architecture is a key component of our framework and deserved more detailed exposition. The revised manuscript now provides comprehensive detail in Section 3.6 (Stage 3):

**Type of attention mechanism:** The fusion module uses **16-head multi-head self-attention (MHA)**, inspired by the transformer architecture (Vaswani et al., 2017). Prior to multi-head attention, each modality branch passes through a **learned gating mechanism** — an MLP followed by sigmoid activation that performs per-modality self-attention, allowing each branch to suppress noisy or uninformative features before fusion. We now provide the full equation set (Equations 4–7) describing the gating, multi-head attention, channel attention, and progressive fusion steps.

**Deep learning architecture:** The revised manuscript specifies the complete dual-branch architecture:
- **RGB Branch:** ConvNeXt-Tiny backbone (pretrained on ImageNet), global average pooling, linear projection to 512-dimensional feature space
- **Thermal Branch:** Separate ConvNeXt-Tiny backbone (modified for single-channel input), MLP projector with residual connections to 512-dimensional space
- **Fusion:** Gated features are stacked along the modality dimension, processed by 16-head multi-head attention, refined via channel attention, then passed through a progressive MLP fusion network (1024 → 512 dimensions through three linear layers with residual connections) to produce the final classification output

**How fusion improves classification:** We now explain in Section 3.6 that the multi-head attention mechanism **automatically weights modality contributions based on disease-specific characteristics**. For diseases involving internal tissue decay (e.g., Stem/Rot), the thermal branch receives higher attention weights since these conditions produce thermal anomalies not visible in RGB. For diseases with prominent surface symptoms (e.g., Black Mould), the RGB branch naturally dominates. This is empirically confirmed in our per-class results (Table 3), where Stem/Rot shows the largest improvement (+15.3% F1) while Black Mould shows no additional benefit (+0.0%), exactly as expected from the attention mechanism's behavior.

**Block diagram / architectural illustration:** We now include two complementary figures:
- **Figure 1** (`crossmap_architecture_diagram.png`): High-level overview of the complete 3-stage pipeline showing how the lesion detector, thermal synthesis, and fusion stages connect
- **Figure 2** (`Architecture.png`): Detailed fusion architecture showing the dual ConvNeXt-Tiny branches, gating mechanisms, 16-head multi-head attention, and progressive MLP fusion

These revisions have been incorporated into Section 3.6 and the corresponding figures of the revised manuscript.
