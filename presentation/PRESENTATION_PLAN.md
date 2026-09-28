# XFeat paper presentation plan

## Presentation constraints

- Paper: **XFeat: Accelerated Features for Lightweight Image Matching**
- Venue: CVPR 2024
- Authors: Guilherme Potje, Felipe Cadar, Andre Araujo, Renato Martins, and Erickson R. Nascimento
- Planned duration: **10 minutes**
- Recommended length: **10 content slides plus a title slide**
- Audience goal: understandable to someone who knows basic computer vision but has not read the paper
- Main story: XFeat obtains a strong accuracy-speed trade-off by keeping spatial resolution useful for localization while putting very few channels in expensive early layers. One trained backbone supports both sparse and semi-dense matching.

## One-sentence explanation

XFeat is a lightweight CNN that detects, describes, and matches image locations quickly enough for CPU and embedded use, while offering either sparse keypoint matches or more numerous semi-dense matches from the same pretrained network.

## Recommended narrative

1. Explain what image matching is with two photographs of the same scene.
2. Explain why existing learned features can be too expensive for robots, AR, and SLAM.
3. State the paper's central design insight: high-resolution early feature maps are expensive, so use very few early channels and move capacity deeper into the network.
4. Introduce the three outputs: descriptor map, keypoint heatmap, and reliability map.
5. Explain the two inference modes, XFeat and XFeat*.
6. Explain training only at a high level.
7. Show experimental evidence across outdoor, indoor, homography, and localization tasks.
8. End with strengths, limitations, and the reproduction plan.

## Slide-by-slide structure

### Slide 1 - Title and paper identity (0:00-0:20)

**Title:** XFeat: Accelerated Features for Lightweight Image Matching

Include:

- CVPR 2024
- Authors and VeRLab
- Presenter names
- One visual example of matched points between two images

Opening line:

> This paper asks whether learned image matching can remain accurate while becoming fast enough for ordinary CPUs and small embedded devices.

### Slide 2 - What problem does image matching solve? (0:20-1:05)

Use two images of the same scene from different viewpoints or illumination.

Explain in simple terms:

- A correspondence connects the same physical point in two images.
- Good correspondences support camera pose estimation, panorama registration, Structure-from-Motion, visual localization, SLAM, robotics, and augmented reality.
- Changes in viewpoint, lighting, texture, and repeated patterns make matching difficult.

Avoid beginning with formulas. Establish the task visually.

### Slide 3 - The accuracy versus efficiency problem (1:05-1:50)

Use Figure 1 or redraw its main idea as a simple accuracy-versus-FPS chart.

Explain:

- Modern learned methods improve robustness but often increase compute and memory.
- Local-feature tasks need relatively high-resolution inputs for pixel-accurate geometry.
- Early CNN layers are especially expensive because their feature maps are large.
- The target is the top-right of the plot: high pose accuracy and high FPS.

Key equation, shown only if space permits:

`FLOPs = H_i x W_i x C_i x C_(i+1) x k^2`

Plain-language interpretation: computation grows with both image area and channel count.

### Slide 4 - XFeat's central architecture idea (1:50-2:50)

Use Figure 3 as the main visual.

Explain the design:

- Keep only **4 channels** in the first stage, where spatial resolution is largest.
- Increase capacity deeper in the network as resolution becomes smaller.
- Channel progression: **4, 8, 24, 64, 64, 128**.
- The full backbone has 23 convolutional layers but remains fast because expensive early layers are narrow.
- Fuse representations from 1/8, 1/16, and 1/32 scales into a descriptor map at 1/8 resolution.

Three outputs:

1. `F`: a 64-dimensional descriptor map at H/8 by W/8.
2. `K`: a keypoint heatmap.
3. `R`: a reliability map estimating whether descriptors are likely to match confidently.

Analogy: spend the computation budget where feature maps are small and therefore cheaper.

### Slide 5 - Two modes from one backbone (2:50-4:10)

Place sparse XFeat and semi-dense XFeat* side by side. Use Figure 2 and the repository diagrams:

- `diagrams/xfeat_sparse_matching.mmd`
- `diagrams/xfeat_semidense_matching.mmd`

#### Sparse XFeat

- Detect keypoints with non-maximum suppression.
- Score each keypoint using keypoint confidence multiplied by descriptor reliability.
- Retain up to 4,096 keypoints.
- Interpolate 64-D descriptors at those positions.
- Match using mutual nearest neighbors.

#### Semi-dense XFeat*

- Use reliable cells from the coarse H/8 by W/8 descriptor grid.
- Process two image scales and retain up to 10,000 features.
- Find mutual nearest-neighbor coarse matches.
- Concatenate each matched descriptor pair and pass it through a small MLP.
- Predict an offset inside an 8-by-8 cell and reject low-confidence refinements.

Clarify terminology:

- **Sparse** means selected interest points only.
- **Coarse** means one feature-grid location represents roughly an 8-by-8 input region.
- **Semi-dense** means substantially more image coverage than sparse matching without matching every pixel.
- Both modes use the same `xfeat.pt` backbone checkpoint.

### Slide 6 - Why the heads are efficient (4:10-4:55)

Use a simplified diagram rather than another dense architecture figure.

Keypoint head:

- Rearranges every 8-by-8 pixel block into 64 channels.
- Uses four inexpensive 1-by-1 convolutions.
- Predicts 65 classes per cell: 64 pixel positions plus one no-keypoint or dustbin class.
- Runs as a separate low-level branch so keypoint learning does not consume descriptor capacity.

Refinement head:

- Receives only already-matched 64-D descriptor pairs.
- Predicts 64 offset logits corresponding to an 8-by-8 cell.
- Does not need a high-resolution feature map.

Central message: XFeat spends computation only after the candidate set has been reduced.

### Slide 7 - How the model is trained (4:55-5:45)

Keep this slide visual and compact.

Training data:

- MegaDepth real image pairs: 60%
- Synthetically warped COCO images: 40%
- Images resized to approximately 800 by 600

Training setup:

- Supervised pixel correspondences
- Adam optimizer, initial learning rate 3e-4
- Batch size 10
- 160,000 iterations
- Learning-rate decay of 0.5 every 30,000 updates
- Approximately 36 hours on one RTX 4090
- Approximately 6.5 GB VRAM

Four learning objectives:

1. Descriptor matching with a bidirectional dual-softmax loss.
2. Reliability prediction from matching confidence.
3. Fine offset prediction for semi-dense refinement.
4. Keypoint detection distilled from ALIKE-Tiny keypoints.

Do not derive all equations during a 10-minute presentation. State what each loss teaches.

### Slide 8 - Evaluation design and metrics (5:45-6:25)

Use a compact dataset-to-task map:

| Dataset | Task | Main measure |
|---|---|---|
| MegaDepth-1500 | Outdoor relative camera pose | AUC at 5, 10, and 20 degrees; Acc@10 |
| ScanNet-1500 | Indoor relative camera pose | AUC at 5, 10, and 20 degrees |
| HPatches | Homography estimation | Mean Homography Accuracy at 3, 5, and 7 pixels |
| Aachen Day-Night | Visual localization | Percentage localized within position/rotation thresholds |

Explain metrics briefly:

- Higher AUC means more image pairs have small pose error.
- Acc@10 is the percentage with pose error below 10 degrees.
- MIR is the fraction of matches consistent with the estimated geometry.
- MHA measures whether an estimated homography places image corners near their correct positions.

Protocol details:

- MegaDepth maximum image dimension: 1,200 pixels.
- ScanNet: VGA resolution.
- XFeat: up to 4,096 features.
- XFeat*: up to 10,000 features, two scales, refinement confidence filtering.
- Robust geometry estimation uses RANSAC-family methods.

### Slide 9 - Main pose-estimation results (6:25-7:30)

Do not paste all of Tables 1 and 2 at full size. Show selected rows or a clear grouped chart.

#### MegaDepth-1500

| Method | AUC@5 | AUC@10 | AUC@20 | Acc@10 | MIR | Inliers | CPU FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| XFeat | 42.6 | 56.4 | 67.7 | 74.9 | 0.55 | 892 | 27.1 |
| XFeat* | 50.2 | 65.4 | 77.1 | 85.1 | 0.74 | 1,885 | 19.2 |
| ALIKE | 49.4 | 61.8 | 71.4 | 77.7 | 0.47 | 333 | 5.3 |
| SuperPoint | 37.3 | 50.1 | 61.5 | 67.4 | 0.35 | 495 | 3.0 |
| DISK* | 55.2 | 66.8 | 75.3 | 81.3 | 0.71 | 1,997 | 1.2 |

Interpretation:

- Sparse XFeat is about 5 times faster than ALIKE and about 9 times faster than SuperPoint on the reported CPU.
- XFeat* obtains the best reported AUC@20, Acc@10, and MIR in this table while remaining about 16 times faster than DISK*.
- DISK remains stronger at the strictest AUC@5 threshold.

#### ScanNet-1500

| Mode | AUC@5 | AUC@10 | AUC@20 |
|---|---:|---:|---:|
| XFeat | 16.7 | 32.6 | 47.8 |
| XFeat* | 18.4 | 34.7 | 50.3 |

Interpretation: both modes generalize strongly to indoor scenes even though they were not trained on ScanNet.

### Slide 10 - Other tasks and qualitative evidence (7:30-8:30)

Split the slide into two regions.

#### HPatches homography

XFeat results:

- Illumination MHA@3/5/7: **95.0 / 98.1 / 98.8**
- Viewpoint MHA@3/5/7: **68.6 / 81.1 / 86.1**

State that XFeat is highly competitive, especially under illumination changes, with a small compute footprint.

#### Aachen Day-Night localization

XFeat results:

- Day: **84.7 / 91.5 / 96.5**
- Night: **77.6 / 89.8 / 98.0**
- Thresholds: 0.25 m and 2 degrees; 0.5 m and 5 degrees; 5 m and 10 degrees

Use Figure 5 or Figure 8 for qualitative evidence of matching across viewpoint and illumination changes. Prefer showing XFeat and XFeat* examples rather than all baseline rows.

### Slide 11 - Takeaways, limitations, and our reproduction (8:30-10:00)

#### Contributions

1. A lightweight CNN whose channel allocation reduces high-resolution compute.
2. A separate minimal keypoint branch that protects descriptor capacity.
3. One backbone supporting both sparse and semi-dense matching.
4. A small descriptor-only refinement module that recovers pixel-level offsets.
5. Strong CPU efficiency without hardware-specific optimization.

#### Limitations and qualifications

- XFeat* is semi-dense rather than fully dense.
- Coarse descriptors and offset refinement reduce accuracy at the strictest pose-error thresholds.
- XFeat* needs an image pair during refinement, although coarse descriptors can be cached independently.
- Transformer matchers such as LoFTR remain more robust for highly ambiguous pairs and extreme viewpoint changes.
- Reported FPS depends on the exact CPU, resolution, software version, and timing protocol.
- Some evaluation details were refactored after publication, so reproduced values may be close rather than bit-identical.

#### Reproduction scope

- Use the authors' released `weights/xfeat.pt` for both XFeat and XFeat*.
- Reproduce Tables 1-3 and qualitative Figures 2 and 5 first.
- MegaDepth-1500, ScanNet-1500, and HPatches are already downloaded.
- Table 4 requires the complete HLoc Aachen setup and official evaluation submission.

Closing line:

> XFeat shows that careful allocation of computation can deliver useful learned matching on resource-limited hardware without giving up the option of denser correspondences.

## Essential paper knowledge

### Problem definition

Given two images, identify pairs of 2D points that correspond to the same physical scene locations. These correspondences allow geometric models such as homographies, essential matrices, and camera poses to be estimated.

### Core architectural insight

For a convolutional layer, computational cost scales with spatial area and input/output channels. Early layers process the largest spatial maps, so XFeat makes them exceptionally narrow and increases channel depth only after downsampling. This differs from uniformly shrinking a conventional VGG-style local-feature network.

### Descriptor and reliability heads

- Multi-resolution features at 1/8, 1/16, and 1/32 are projected, upsampled to 1/8, summed, and fused.
- The resulting descriptor map has 64 channels.
- A reliability head estimates whether each coarse descriptor is likely to produce a confident correspondence.

### Keypoint head

- Operates as a dedicated parallel branch on low-level image structure.
- Converts each 8-by-8 image cell into 64 channels.
- Predicts one of 64 within-cell locations or a dustbin class.
- The separate branch is motivated by the limited capacity of a very small backbone; forcing one representation to serve both dense description and keypoint regression degraded XFeat* in the ablation.

### Sparse XFeat inference

- Convert keypoint logits into a full-resolution heatmap.
- Apply non-maximum suppression.
- Score keypoints with keypoint confidence times reliability.
- Select the top 4,096.
- Bicubically interpolate descriptors from the 1/8 descriptor map.
- L2-normalize descriptors.
- Calculate cosine similarities and retain mutual nearest neighbors.

### Semi-dense XFeat* inference

- Sample reliable descriptors directly from the coarse map.
- Use two input scales to improve coverage.
- Keep up to 10,000 descriptors.
- Apply mutual nearest-neighbor matching.
- Concatenate each matched descriptor pair.
- Use an MLP to predict an 8-by-8 offset distribution.
- Convert it into a sub-cell offset and filter by confidence.

### Training

The model learns from known pixel correspondences. Descriptor similarity is trained bidirectionally. Reliability is trained to predict matching confidence. The refinement head learns the true within-cell offset. The keypoint head uses ALIKE-Tiny detections as teacher labels. These four objectives are linearly combined.

### Ablation conclusions

MegaDepth-1500 AUC@5:

| Strategy | XFeat | XFeat* |
|---|---:|---:|
| Default | 42.6 | 50.2 |
| No synthetic data | 41.5 | 33.9 |
| Smaller model | 37.4 | 40.7 |
| Joint keypoint extraction | 42.9 | 39.7 |
| No match refinement | N/A | 38.6 |

Interpretation:

- Synthetic COCO warps matter especially for XFeat* generalization.
- Uniformly reducing channel count damages accuracy.
- A separate keypoint branch matters mainly for semi-dense representation quality.
- Refinement is essential to XFeat*.

### Supplementary efficiency result

On an Orange Pi Zero 3 with a Cortex-A53 at 480 by 360, the reported extraction rates were:

- XFeat: 1.8 FPS
- ALIKE: 0.58 FPS
- SuperPoint: 0.16 FPS

### Comparison with learned matchers

On MegaDepth-1500 at 1,200 pixels, XFeat* reports 1.33 image pairs per second on an i7-6700K, compared with 0.31 for LightGlue, 0.06 for LoFTR, and 0.05 for Patch2Pix. LoFTR and LightGlue achieve higher pose accuracy, so this result demonstrates an efficiency trade-off rather than universal superiority.

## Presentation design guidance

- Prefer diagrams, matched-image examples, and highlighted numbers over paragraphs.
- Use a consistent color for XFeat and another for XFeat* across all slides.
- Explain each acronym at first use: CNN, MNN, NMS, AUC, MIR, SfM, and RANSAC.
- Avoid showing all seven equations. The FLOPs equation and a simple offset-refinement illustration are sufficient.
- Avoid presenting the entire related-work taxonomy. Mention SuperPoint, ALIKE, DISK, LoFTR, and LightGlue only when they clarify the comparison.
- Crop paper figures cleanly and retain their figure number or add a small citation footer.
- Use progressive disclosure on the architecture slide: backbone first, then `F`, `K`, and `R`.
- On result slides, highlight the relevant XFeat/XFeat* rows and gray out baselines.
- Keep claims tied to the reported hardware and protocol.

## Likely lecturer questions

### Are XFeat and XFeat* different trained models?

No. They use the same backbone checkpoint. They differ in feature selection, scale processing, matching density, and refinement during inference.

### Why is XFeat* called semi-dense?

It selects many reliable locations from a coarse grid but does not match every input pixel.

### Why use mutual nearest neighbors?

It accepts a match only when each descriptor chooses the other, reducing one-way ambiguous matches.

### Why does XFeat keep high input resolution but reduce early channels?

Geometry needs spatial precision. Reducing resolution too aggressively removes location detail, while reducing channels lowers cost without directly discarding pixel positions.

### Why does the descriptor map remain at 1/8 resolution?

It balances spatial detail with compute. Features from deeper 1/16 and 1/32 maps add context and are fused back at 1/8.

### How does XFeat* recover pixel precision from a coarse map?

The matched descriptor pair enters a learned MLP that predicts an offset inside the corresponding 8-by-8 cell.

### Why use synthetic COCO warps?

They provide diverse self-supervised transformations and reduce overfitting to outdoor landmark textures from MegaDepth.

### Is XFeat always more accurate than larger matchers?

No. LoFTR and LightGlue achieve stronger pose accuracy in the supplementary matcher comparison. XFeat's main contribution is its accuracy-efficiency balance.

### Why might reproduced results differ?

RANSAC is stochastic, the official repository was refactored, and timing depends on hardware and software configuration.

## Source and asset map

- Full paper: `research paper.pdf`
- Sparse pipeline diagram: `diagrams/xfeat_sparse_matching.mmd`
- Semi-dense pipeline diagram: `diagrams/xfeat_semidense_matching.mmd`
- Combined diagram preview: `diagrams/README.md`
- Official architecture implementation: `modules/model.py`
- Official sparse and semi-dense inference: `modules/xfeat.py`
- MegaDepth evaluation: `modules/eval/megadepth1500.py`
- ScanNet evaluation: `modules/eval/scannet1500.py`
- Released checkpoint: `weights/xfeat.pt`

## Figure selection from the paper

- Figure 1: motivation and speed-accuracy trade-off.
- Figure 2: intuitive comparison between sparse XFeat and semi-dense XFeat*.
- Figure 3: main network architecture and three outputs.
- Figure 4: descriptor-pair refinement module.
- Figure 5: best main-paper qualitative MegaDepth evidence.
- Figure 6: detailed backbone; use only if the audience needs architecture depth.
- Figure 7: timing breakdown; optional in a 10-minute talk.
- Figures 8 and 9: supplementary qualitative examples; useful as backup slides.

## Optional backup slides

1. Detailed backbone and channel dimensions from Figure 6.
2. Four training losses and equations.
3. Ablation table.
4. Detailed benchmark protocols.
5. Learned matcher comparison from Table 6.
6. Dataset and reproduction status.
