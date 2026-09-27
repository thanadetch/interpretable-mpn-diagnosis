# Related work — what each paper does (verified by reading unless marked) and how to position against it

Rule: open the paper before stating what it does; use "to our knowledge", never "first".

## 1. For WR-TransMIL

Verdict after five search rounds (2026-09-05 → 09-27): no MIL model found that replaces the readout (class) token with a separately encoded whole-image embedding.

| work | what it does | difference | how to cite |
|---|---|---|---|
| TransMIL (Shao et al., NeurIPS 2021) | correlated MIL; learned "class token"; Nyström attention; PPEG | base model; class token is a learned constant | base |
| BERT (Devlin et al., 2019) / ViT (Dosovitskiy et al., 2021) | "classification token ([CLS])" / "[class] token" | origin of the token | terminology |
| **CrossViT** (Chen et al., ICCV 2021) | each branch's CLS (after summarising its branch) queries the other branch's patch tokens | CLS still starts from a learned constant; end-to-end multi-scale ViT | closest principle — cite |
| **CLIP** (Radford et al., 2021) | attention-pooling query conditioned on the average-pooled representation of the same image | query = mean of the local features; our patch-mean control is this variant and loses | cite beside the patch-mean control |
| Conditional DETR (ICCV 2021); Efficient DETR (2021) | content-conditioned / encoder-initialised queries in detection | detection | principle |
| DSMIL (Li et al., CVPR 2021) | query = one critical instance | instance-derived | baseline |
| CTMIL (npj Precis. Oncol. 2025); MsCAMIL (2026) | TransMIL descendants (deeper + coordinates + auxiliary loss; cross-magnification attention) | keep the learned class token | TransMIL descendants |
| HAG-MIL (2023) | removes the class token, aggregates all instances | opposite direction | related |
| **SEW** (arXiv 2024) | WSI thumbnail branch with its own learned CLS selects regions; KL loss ties local CLS tokens to thumbnail features | thumbnail guides via selection + loss; never the readout | closest "thumbnail + class token" work — cite |
| GMIC (Shen et al., MedIA 2021) | global features concatenated with the MIL output before the classifier | late fusion (our control) | related |
| MEGT, PTCMIL, ViTAGG-MIL | add context / prompt / learnable query tokens | additive tokens (our extra-token comparator) | related |
| MicroMIL (MICCAI 2025) | MIL for microscope images; bag = a patient's many images | no single whole-bag view exists | framing contrast |

Suggested sentence: *"Subsequent TransMIL variants retain the learned class token [CTMIL, MsCAMIL] or remove it [HAG-MIL], and context-aware MIL adds global information as extra tokens [MEGT, PTCMIL] or after aggregation [GMIC]. Outside MIL, queries conditioned on the input appear in detection [Conditional DETR, Efficient DETR] and in CLIP's attention pooling, whose query is the mean of the image's own features. To our knowledge, no MIL model uses a separately encoded whole-image embedding as its readout token."*

## 2. MIL baselines

ABMIL (Ilse et al., ICML 2018); CLAM (Lu et al., 2021); DSMIL (Li et al., 2021); DTFD-MIL (Zhang et al., 2022); TransMIL (Shao et al., 2021).

## 3. Domain and metric (open each before citing)

- European consensus grading of marrow fibrosis (Thiele et al., Haematologica 2005); WHO criteria (grade ≤ 1 prefibrotic vs ≥ 2 overt PMF).
- Automated reticulin fibrosis assessment (Virchows Archiv 2025): InceptionV3-based FCN on WSIs, κ .831 vs haematopathologists.
- Continuous Indexing of Fibrosis, CIF (2022): tile-level ranking on a 0–1 scale.
- BoMBR (bioRxiv 2024): annotated bone-marrow dataset for reticulin fibre segmentation.
- Inter-rater κ for fibrosis grading ≈ .76–.83 (Modern Pathology 2014 and a WHO-applicability study; verify).
- QWK (Cohen 1968); used in PANDA (Bulten et al., Nature Medicine 2022).
- Encoders: Virchow2 (Zimmermann et al., 2024), UNI/UNI2-h (Chen et al., Nature Medicine 2024), TITAN (Ding et al., 2024). Verify bibliographic details.
- Embedding-space vs patch-level augmentation: Zaffar et al., EmbAugmenter (ISBI 2023).

## 4. Thesis only (ASGAP)

SMILE (Lu et al., MICCAI COMPAY 2021): top-N instance selection + plain-softmax transformer — no sparsemax/entmax. MINN-SA (BMC Bioinformatics 2022): sparsemax on a non-gated scorer, TCR sequences. Also cite sparsemax (Martins & Astudillo 2016), α-entmax (Peters et al. 2019), learnable α (Correia et al. 2019).
