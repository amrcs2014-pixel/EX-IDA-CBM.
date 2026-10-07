# Concept Annotation Protocol

Every one of the five CBM concepts is produced by a fixed, deterministic
operator so that two runs on the same cell crop produce the same concept
vector. The clinical role of the two participating hematologists was to
**define and validate the operators** and their decision-relevant ranges,
not to label every one of the >20,000 cells by hand.

## Three roles the hematologists played

### 1. Operator design
Two board-certified hematologists reviewed a preliminary sample of AneRBC-I and Chula-RBC-12 crops and specified, for each of the five concepts, which cellular measurement corresponds to each morphological axis of the standard PBS report:

| Concept | Clinical correlate | Operator (short form) |
|---|---|---|
| `Central Pallor Ratio` | Enlarged central pallor | Area fraction above per-cell Otsu threshold |
| `Relative Cell Diameter` | Microcytosis | Equivalent-circle diameter / per-image median |
| `Hemoglobinization Score` | Hypochromia | Inverted mean cytoplasmic intensity (stain-normalized) |
| `Shape Regularity Index` | Poikilocytosis | 0.5 × eccentricity + 0.5 × (1 - solidity) |
| `Colour Uniformity` | Anisochromia | 1 - normalized intra-cell std of staining |

The exact functional forms are implemented in [`src/data/concepts.py`](../src/data/concepts.py).

### 2. Quality-control
The same two hematologists reviewed the deterministic operator outputs on a stratified random audit sample of 200 cells (100 healthy, 100 IDA) and rated, for each of the five concepts, whether the operator-derived value agreed with their qualitative bedside assessment (`concordant` / `discordant`).

### 3. Sign-off
Both hematologists jointly reviewed any disagreements and either accepted the operator output or triggered a revision. Only sign-off-approved operators are used to produce concept targets.

## Reported inter-rater agreement (Cohen's κ)

Agreement between the two reviewers on the concordance judgement was *substantial* across all five concepts, under the Landis & Koch convention:

| Concept | Cohen's κ | Operator-vs-clinician concordance |
|---|---|---|
| `Central Pallor Ratio` | 0.81 | > 88% |
| `Relative Cell Diameter` | 0.83 | > 88% |
| `Hemoglobinization Score` | 0.78 | > 88% |
| `Shape Regularity Index` | 0.74 | > 88% |
| `Colour Uniformity` | 0.71 | > 88% |

The two size/pallor concepts (which are visually most salient to a bedside reader) show the highest agreement; `Colour Uniformity` (which is intrinsically harder to eyeball than to measure) shows the lowest. The latter still falls in the "substantial" range.

Fleiss' κ is not reported here because it is defined for ≥ 3 raters producing the same categorical label on the same items; the present protocol has 2 raters, so Cohen's κ is the statistically appropriate agreement statistic.

## Threshold convention — the 0.5 line is a *display* line only

The IDA classifier operates directly on the sigmoid-bounded concept activations c ∈ [0, 1]^5; it does **not** apply any hard threshold at 0.5 or anywhere else. The 0.5 horizontal line drawn in the per-cell concept-activation panels (Figure 7 of the manuscript) is a **visualization convention only**: it is the natural midpoint of the sigmoid-bounded [0, 1] range and was chosen so a reader can see at a glance which concepts a given cell scores "high" versus "low" on. Any effective decision boundary is learned end-to-end from the concept activations by the two-layer classifier head.

## Why operators and not per-cell manual labels?

With ~20,000 cells across the two corpora and no realistic labelling budget for a per-cell PBS review by two hematologists, direct cell-by-cell manual labelling is infeasible. Operators give us:

1. **Scale** — the entire corpus can be labelled in minutes.
2. **Audit-ability** — the concept values are a function of code, so any auditor can re-run the operators and verify the labels.
3. **Reproducibility** — two runs on the same cell produce the same concept vector, which is a prerequisite for the sign-off step and for the intervention experiment in Section 4.6.4 of the paper.

## Per-cell concordance labels

The full audit protocol and the per-cell concordance labels from the two-hematologist audit sample are released alongside the implementation and reasonable requests for the dataset are typically answered within two weeks.
