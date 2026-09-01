# Clements third external cohort — results

**2026-09-01.** Analysis plan pre-registered in
`CLEMENTS_PREREGISTRATION_2026-08-31.md`, committed at `b567369` **before** any
model was evaluated on this cohort. Nothing below deviates from that plan.
Reproduce with `eval_clements.py` then `aggregate_clements.py`; values in
`clements_results.json`.

42 studies (S04 blank, dropped as pre-specified), 37,482 frames, 308 runs
(7 arms × 44 seeds), zero-shot. No model was retrained; no model has seen a
Clements frame.

---

## 1. Validity gate — passed, but only just

Pre-registered: *if the best arm's study-level macro AUC is below 0.60, the
cohort is reported as uninformative and no contrast is interpreted.*

> best arm (`rgb`) = **0.6031**, floor = 0.60.

**The gate passed by 0.0031.** It was fixed in advance and we honour it, so the
contrasts below are reported as interpretable. But a margin that thin is not a
clean pass, and the per-finding breakdown shows why it should be read with
care:

| finding | positives / 42 | AUC (prior arm) |
|---|---|---|
| Blood - hematin | 4 | 0.7382 ± 0.1275 |
| Normal clean mucosa | 10 | 0.6668 ± 0.0457 |
| Blood - fresh | 4 | 0.6210 ± 0.1073 |
| Ulcer | 4 | 0.5697 ± 0.1029 |
| Angiectasia | 8 | 0.5217 ± 0.0984 |
| **Erosion** | 7 | **0.4510 ± 0.0837** |

**Three of six findings sit at or below chance**, and the macro average is
carried by two. Transfer to this cohort is poor — much poorer than to Galar
(0.628 external against 0.743 in-domain). Any claim resting on it is
correspondingly weak.

## 2. Study-level macro AUC, all seven arms (n = 44 each)

| rank | arm | study-level macro AUC |
|---|---|---|
| 1 | rgb | 0.6031 ± 0.0446 |
| 2 | zeros | 0.6017 ± 0.0447 |
| 3 | random_fixed | 0.5987 ± 0.0446 |
| 4 | shuffled | 0.5974 ± 0.0427 |
| **5** | **prior** | **0.5947 ± 0.0337** |
| 6 | phi_dup | 0.5941 ± 0.0453 |
| 7 | gauss | 0.5940 ± 0.0475 |

**The prior ranks fifth of seven**, below the RGB baseline it exists to improve
and below three content-free controls.

## 3. Contrasts — every one equivalent at ±0.023

Paired by seed, TOST at the paper's standing margin.

| contrast | Δ | 90 % CI | p (TOST) | verdict |
|---|---|---|---|---|
| **prior − rgb** (primary) | **−0.0084** | [−0.0225, +0.0057] | 0.045 | **equivalent** |
| prior − zeros | −0.0070 | [−0.0194, +0.0055] | 0.018 | equivalent |
| prior − random_fixed | −0.0040 | [−0.0181, +0.0102] | 0.015 | equivalent |
| prior − shuffled | −0.0027 | [−0.0165, +0.0112] | 0.009 | equivalent |
| prior − gauss | +0.0008 | [−0.0122, +0.0138] | 0.003 | equivalent |
| prior − phi_dup | +0.0006 | [−0.0136, +0.0148] | 0.006 | equivalent |

The primary contrast is the weakest of the six: p = 0.045 against α = 0.05, and
its lower bound (−0.0225) sits just inside the margin (−0.023). Equivalence is
established, but marginally, and we say so rather than reporting a bare
"equivalent".

## 4. What this adds, and what it does not

**Adds.** A third cohort, independent of both Kvasir-Capsule and Galar: real
clinical PillCam studies from a different hospital, on different hardware, read
by a different gastroenterologist, with findings recorded at study level rather
than per frame. The ordering seen in-domain and on Galar recurs: the prior does
not beat its controls, and every contrast is equivalent at the pre-specified
margin. Three independent cohorts now agree.

**Does not add.** A frame-level claim — the estimand here is study level and is
not interchangeable with the rest of the paper. Nor does it add a strong
external test: with the gate passed by 0.003 and half the findings at chance,
this cohort discriminates barely at all, and an equivalence result obtained
where nothing discriminates is weaker than the same result obtained where the
baseline transfers well. Galar remains the stronger external evidence.

**Not tested here.** `cross_image`, the C5 donor-prior control and the primary
attribution contrast elsewhere in the paper. Its 44 runs completed but their
checkpoints were not retained, so zero-shot evaluation was impossible. This was
recorded in the pre-registration before results existed. Attribution on this
cohort is therefore tested against C1–C4 only.

## 5. Deviations from the pre-registration

None.
