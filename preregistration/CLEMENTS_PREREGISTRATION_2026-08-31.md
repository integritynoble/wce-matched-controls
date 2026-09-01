# Pre-registration: the Clements third external cohort

**Written 2026-08-31, BEFORE any model was evaluated on this cohort.**
Committed to git before the evaluation script was run, so the commit timestamp
precedes the existence of any Clements result. Nothing below may be changed
after results are seen; deviations, if any, will be recorded as deviations.

The point of writing this is not ceremony. This paper's central claim is that
underpowered, outcome-contingent analysis produces effects that dissolve on
extension. Adding a third cohort and then deciding what to report would be the
same error, committed by us, on the paper that criticises it.

---

## 1. Commitment

**The Clements result will be reported in the manuscript whatever it shows** —
including if it contradicts the in-domain and Galar findings, and including if
it is uninformative. It will be reported with its power stated.

## 2. Data

43 de-identified PillCam studies from Clements University Hospital, reviewed at
study level by Dr. Roopa Vemulapalli, sheet returned via Abirami Krishnamurthy
(`clements_study_review_sheet 1.csv`).

- **S04 has no entries and is excluded. n = 42 studies.**
- Study identifiers are GUIDs. Only the `pillcam files deid` set is touched;
  no patient name or MRN enters any artefact, manifest, or the manuscript.
- Frames extracted 2026-08-31 with `given_rapid_reader.py`, parameters fixed
  before any evaluation: `--stride 10 --quality-filter --dedup`.
  Yield: **37,482 frames** across 43 studies (median ~700/study).

Positive study counts, from the sheet, known before evaluation:

| finding | Kvasir class | positives / 42 |
|---|---|---|
| normal_no_significant_finding | Normal clean mucosa | 10 |
| angioectasia_AVM | Angiectasia | 8 |
| erosion | Erosion | 7 |
| active_bleeding | Blood - fresh | 4 |
| blood_or_hematin_no_active_bleeding | Blood - hematin | 4 |
| ulcer | Ulcer | 4 |

`other_finding_free_text` (18 studies) is **not** used for scoring. It records
findings outside the six-category scheme (erythematous mucosa, lymphangiectasia,
polyps, gastritis) and mapping it post hoc would be a free parameter.

## 3. Arms

Zero-shot. No model is retrained, no model sees a Clements frame during
training. EfficientNet-B0, 44 seeds per arm:

`prior`, `rgb`, `gauss`, `random_fixed`, `phi_dup`, `shuffled`, `zeros`

**`cross_image` is excluded for a technical reason recorded here in advance:
its 44 runs were completed but their checkpoints were not retained, and
zero-shot evaluation needs weights.** This is stated now, before results exist,
so it cannot later be mistaken for a results-dependent omission. The
consequence is explicit: on this cohort attribution is tested against the
content-free controls (C1–C4) but **not** against the donor-prior control (C5),
which is the strongest rung and the primary contrast elsewhere in the paper. If
the 44 `cross_image` models are retrained before submission, that contrast will
be added; if they are not, the limitation stands as written.

## 4. Aggregation rule — fixed in advance

The models are frame classifiers; the labels are study level. For study $s$ and
class $c$:

$$\text{score}(s, c) = \max_{f \in s} \; p_c(f)$$

The **maximum** frame probability over the study. Chosen because a finding is
present in a study if it is present in any frame, which is what a max
expresses. Mean and top-$k$ alternatives are **not** to be tried and reported;
if the max is shown to be a poor choice, that is a separate analysis reported
as exploratory.

## 5. Metric and contrasts

- Per finding: study-level ROC AUC over 42 studies.
- **Primary metric: macro AUC across the six findings**, each computed over all
  42 studies.
- Paired by seed: each arm contributes 44 macro-AUC values.
- **Primary contrast: `prior` − `rgb`** (utility on this cohort).
- **Secondary contrasts: `prior` − each of the five content-free controls**
  (attribution against matched nulls).
- Equivalence by TOST at **$\delta = 0.023$**, the same margin used throughout
  the paper, taken from the originally reported effect size.

## 6. Validity gate — evaluated before any contrast is read

A contrast between arms is meaningless if no arm performs above chance on this
cohort. Pre-specified:

> **If the best arm's study-level macro AUC is below 0.60, the cohort is
> reported as uninformative for attribution and no contrast is interpreted.**

This mirrors the gate applied to the Galar cohort and to the MedIA regime
sweep. It is a floor check; the ceiling case cannot arise here.

## 7. What this cohort can and cannot support

**Can:** whether the ordering of arms observed in-domain and on Galar recurs on
a third, independent, real-world hospital cohort acquired on different hardware
and read by a different clinician.

**Cannot:** a frame-level claim. These are study-level labels aggregated from
frame predictions, so the estimand differs from the rest of the paper and the
two are not interchangeable.

**Power, stated honestly.** With 4–10 positive studies per finding, a *single*
arm's AUC carries a confidence interval of roughly $\pm 0.2$. The arm contrast
is far better determined, because both arms are scored on the *same* 42
studies, so study-sampling error largely cancels and the paired difference is
driven by seed variation, where $n = 44$. We therefore expect the contrast to
be reportable while absolute AUCs are not. **This expectation is recorded
before seeing the data**; if the observed paired SE is large enough that
$\pm0.023$ cannot be resolved, we report the cohort as underpowered for
equivalence and give the interval rather than presenting a null.

## 8. Analysis code

`GI_project/code/Capsule-Endoscopy/eval_clements.py` and
`aggregate_clements.py`, written against this document and committed with it.
