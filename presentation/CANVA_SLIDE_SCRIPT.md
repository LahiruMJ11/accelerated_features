# Canva slide content and speaker notes

Target: 11 slides, 10 minutes. Retain the current title slide and presenter names. Place the existing sparse and semi-dense flowcharts on slides 5 and 6. Use the existing white background, dark text and turquoise accents. Label all numerical results as reported by the paper, rather than our reproduced results.

Source throughout: Potje et al., *XFeat: Accelerated Features for Lightweight Image Matching*, CVPR 2024. Local source: `research paper.pdf` (14-page arXiv version).

## 1. XFeat: Accelerated Features for Lightweight Image Matching

**On slide:**

CVPR 2024

Guilherme Potje, Felipe Cadar, André Araujo, Renato Martins, Erickson R. Nascimento

Retain existing presenter names.

**Speaker notes — 20 seconds:** This paper asks whether learned image matching can remain accurate while running efficiently on an ordinary CPU. It introduces one network with two matching modes, sparse XFeat and semi-dense XFeat star.

## 2. Image matching

**On slide:**

A match links the same physical point in two photographs.

Matches help estimate camera motion and align images.

Viewpoint changes, lighting changes and repeated patterns make matching difficult.

**Visual:** Crop the upper XFeat example from Figure 2. Explain a single line before discussing the whole set.

**Speaker notes — 45 seconds:** Imagine taking two photographs of a building from different positions. The computer needs to know which window or corner in the first photograph corresponds to the same place in the second. Each connecting line represents a proposed correspondence. Enough correct correspondences let us estimate how the camera moved, build a 3D reconstruction, or locate the camera in an existing map. Similar windows can look alike, and lighting or viewpoint can change their appearance.

## 3. Accuracy and computational cost

**On slide:**

Precise geometry needs detailed images.

Large feature maps make early CNN layers expensive.

XFeat uses very few early channels and more channels deeper in the network.

**Visual:** A large, legible channel sequence: 4, 8, 24, 64, 64, 128. Label the direction as increasing network depth. Optional small formula: cost scales with image area × input channels × output channels.

**Speaker notes — 45 seconds:** A convolutional neural network transforms an image into feature maps. Channels represent different learned patterns. Early layers process many pixels, so using many channels there is expensive. Reducing the entire input image loses spatial detail needed for geometry. XFeat instead starts with only four channels and adds capacity after downsampling makes the spatial maps cheaper. The central contribution is careful allocation of computation.

## 4. The XFeat network

**On slide:**

F: 64 numbers describe each location on an H/8 × W/8 grid.

K: keypoint scores identify useful image locations.

R: reliability scores estimate whether descriptors can match confidently.

**Visual:** Figure 3, with its aspect ratio preserved. Keep labels large enough to read. Put explanations below the figure.

**Speaker notes — 60 seconds:** The network has a descriptor branch and a separate keypoint branch. The descriptor branch combines features from several resolutions to capture local appearance and wider context. Its final grid is eight times smaller along each image dimension. Each grid location has a vector of 64 values called a descriptor. A reliability map tells us which descriptors are promising. The keypoint branch rearranges each 8 by 8 image block into channels and uses inexpensive 1 by 1 convolutions. Keeping this branch separate helps the small descriptor network preserve its capacity. Source: Sections 3.1–3.2 and Figure 3.

## 5. Sparse matching: XFeat

**On slide / editable diagram:**

Two images

Detect keypoints and rank by keypoint score × reliability

Keep up to 4,096 keypoints per image

Sample a 64-value descriptor at each keypoint

Keep mutual nearest-neighbor matches

**Callout:** Mutual means A chooses B and B chooses A.

**Speaker notes — 75 seconds:** Sparse means we work at selected interest points rather than all pixels. Non-maximum suppression keeps strong local peaks and removes nearby duplicates. The score combines the keypoint confidence with descriptor reliability. We interpolate the coarse descriptor map at the selected positions. To match, compare descriptors across the two images. If point A's most similar descriptor is B, and B's most similar descriptor is A, keep that pair. This mutual check reduces ambiguous one-way matches, but does not guarantee correctness. A later geometric estimator rejects inconsistent matches. Source: Section 4, XFeat inference.

## 6. Semi-dense matching: XFeat*

**On slide / editable diagram:**

Extract coarse descriptors at two image scales

Keep up to 10,000 reliable features

Find mutual nearest-neighbor coarse matches

A small MLP predicts an offset inside an 8 × 8 cell

Keep confident refined matches

**Callout:** Same trained network. Different inference procedure.

**Speaker notes — 75 seconds:** Coarse means the feature grid has lower spatial resolution. One grid step corresponds to eight input pixels at that image scale. XFeat star selects reliable grid features directly instead of selecting keypoints first. It uses two scales and more features, providing wider coverage. After coarse matching, a small multilayer perceptron reads each pair of descriptors and predicts a location offset inside an 8 by 8 cell. This refines the match without constructing a large high-resolution feature map. It is called semi-dense because it covers many locations while still selecting only a subset. Source: Section 3.2, Figure 4 and Section 4 inference protocol. Paper scales: 0.65 and 1.3. Refinement confidence threshold: 0.2.

## 7. Training

**On slide:**

60% MegaDepth real image pairs

40% COCO images with synthetic warps

Learn descriptors, reliability, keypoints and refinement offsets

800 × 600 images, 10 pairs per batch

About 36 hours on one RTX 4090

**Speaker notes — 50 seconds:** Training uses examples where corresponding locations are known. Real MegaDepth pairs teach matching across views of outdoor scenes. Synthetic transformations of diverse COCO images provide extra correspondences and improve generalization. The descriptor objective makes true matches similar. Reliability learns matching confidence. Refinement learns the within-cell offset. The keypoint branch learns from ALIKE-Tiny keypoint detections. These objectives train the same model used by both modes. Training hardware requirements are much larger than inference requirements. Source: Sections 3.3 and 4, supplementary training details.

## 8. Evaluation tasks

**On slide / editable table:**

| Dataset | Task | Measurement |
|---|---|---|
| MegaDepth-1500 | Outdoor camera pose | Pose AUC at 5°, 10°, 20° |
| ScanNet-1500 | Indoor camera pose | Pose AUC at 5°, 10°, 20° |
| HPatches | Planar image alignment | Homography accuracy at 3, 5, 7 pixels |
| Aachen Day-Night | Camera localization | Within distance and rotation limits |

**Callout:** Higher is better. Smaller error thresholds are stricter.

**Speaker notes — 40 seconds:** Relative pose asks how the second camera moved from the first. AUC summarizes the pose-error distribution up to a threshold, rewarding smaller errors. It differs from the simple fraction below a threshold. HPatches evaluates an image-to-image planar transformation, called a homography. Aachen evaluates the camera position and orientation within a 3D map. Geometric estimation uses RANSAC-family methods to handle wrong matches. Source: Sections 4.1–4.3.

## 9. Pose results and CPU speed

**On slide / editable table:**

| Method | MegaDepth AUC@20° | ScanNet AUC@20° | CPU FPS |
|---|---:|---:|---:|
| SuperPoint | 61.5 | 36.7 | 3.0 |
| ALIKE | 71.4 | 25.9 | 5.3 |
| DISK* | 75.3 | 33.9 | 1.2 |
| XFeat | 67.7 | 47.8 | 27.1 |
| XFeat* | 77.1 | 50.3 | 19.2 |

**Footnote:** Paper-reported results, Tables 1–2. CPU timing at VGA on Intel i5-1135G7. MegaDepth accuracy uses maximum dimension 1,200 pixels. These are different resolution settings.

**Speaker notes — 65 seconds:** Sparse XFeat is about five times faster than ALIKE under the paper's timing protocol. XFeat star improves coverage and accuracy, with some additional cost. In these selected comparisons, it has the highest AUC at 20 degrees on both pose datasets. The ScanNet result is encouraging because the model did not train on ScanNet. The claim is a strong accuracy-speed balance. For example, DISK star still has higher MegaDepth AUC at the stricter five-degree threshold, 55.2 versus 50.2. Also, these FPS values are feature-extraction timings under the reported setup, not a guarantee of complete application frame rate.

## 10. Alignment, localization and example matches

**On slide:**

HPatches, XFeat MHA@5 pixels

Illumination: 98.1%     Viewpoint: 81.1%

Aachen, XFeat within 0.5 m and 5°

Day: 91.5%     Night: 89.8%

**Visual:** The XFeat / XFeat* top row of Figure 5. Label it as qualitative examples from the paper.

**Speaker notes — 60 seconds:** XFeat also performs well beyond relative pose. On HPatches, these numbers give the fraction of cases meeting the five-pixel corner-error threshold, averaged according to the benchmark. Viewpoint changes remain harder than illumination changes. On Aachen, the model supports localization both during the day and at night. Tables 3 and 4 report sparse XFeat results, so we should not invent XFeat star rows for them. The images show selected examples where matching succeeds under difficult changes. They illustrate behavior but do not replace the full benchmark. Source: Tables 3–4 and Figure 5.

## 11. Takeaways and our reproduction

**On slide:**

Efficient early layers make learned features practical on CPUs.

One network supports sparse keypoints and semi-dense refinement.

Accuracy depends on the task and error threshold.

Our next step: reproduce Tables 1–3 and examples from Figures 2 and 5 using the released weights.

**Small note:** Timing depends on hardware. RANSAC and software differences can change reproduced scores.

**Speaker notes — 65 seconds:** The main lesson is that architecture design can substantially reduce the cost of local feature extraction. XFeat uses a small early channel count, a separate keypoint branch and inexpensive refinement based on descriptor pairs. The star mode gives more coverage, but is still semi-dense. Larger learned matchers can achieve higher accuracy on difficult cases, and strict thresholds expose precision limitations. Our assignment will evaluate the released checkpoint on the exact benchmark protocols, beginning with MegaDepth, ScanNet and HPatches. We will clearly distinguish paper values from our own measurements. Aachen needs its separate localization pipeline and evaluation. This presentation explains the paper; reproduction results will follow in the final project presentation.

## Notes for editing the existing Canva deck

- The original title slide is slide 1.
- Move the existing sparse flowchart to slide 5 and the existing coarse matching flowchart to slide 6.
- Rename “The 2 main architectures” to “Sparse matching: XFeat”. The two modes share a network, so “two architectures” is misleading.
- Rename “Dense matching of the coarse feature map” to “Semi-dense matching: XFeat*”.
- Keep at most one main figure or diagram per slide and avoid placing the complete paper tables on the slide.
- Maintain the white/turquoise style of the supplied presentation.
- Use sources in speaker notes and a small visible paper citation on results slides.
