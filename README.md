# VGG Ablation — Texform vs. Natural Images

Measures how much information about image properties is carried by different
dimensions of VGG16 representations, and whether that information survives
texform scrambling.

## What it does

1. **Feature extraction** — loads natural/texform image pairs, extracts ~15
   pixel-level features per image (contrast, edge density, texture statistics,
   keypoint counts, etc.) plus two high-level semantic labels (`big`, `animal`).
2. **VGG16 activations** — passes images through a pretrained VGG16 and
   extracts activations from specified max-pooling layers, cropped to the
   central 8×8 spatial units across all channels.
3. **Dimensionality reduction** — reduces the high-dimensional activation
   tensor to 20 PCA dimensions per layer.
4. **Random ablation sweep** — iteratively ablates N randomly chosen PCA
   dimensions (N = 1…10) and, for each N, decodes all image features from the
   remaining dimensions using 10-fold cross-validated OLS regression. This is
   repeated 100 times per N to obtain stable mean ± SEM decoding curves.
5. **Visualization** — plots decoding accuracy vs. dimensions ablated for every
   layer and image type. Features whose accuracy drops significantly as more
   dimensions are removed (significant negative slope, p < 0.01) are separated
   from those that do not.
