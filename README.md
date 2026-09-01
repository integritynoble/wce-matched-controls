# Matched controls for physics-informed input channels in capsule endoscopy

Code and per-seed predictions accompanying two companion manuscripts:

- *"Matched Controls and Equivalence Testing for Physics-Informed Input
  Channels in Wireless Capsule Endoscopy"* (submitted, IEEE Transactions on
  Medical Imaging, 2026) — the empirical case study.
- *"What Does a Deterministic Representation Contribute? Matched Controls,
  Positive Controls, and the Regime Dependence of Physics-Informed Inputs"*
  (submitted, Medical Image Analysis, 2026) — the methodological framework.

This repository contains everything needed to check the papers' numbers without
retraining anything: the analytic prior, the six matched control priors, the
equivalence-testing code, the aggregation scripts, the positive-control
simulation, the influence-ratio calibration, and the **per-frame test
predictions from all 616 training runs**.

## Why this exists

A component added to a classifier is judged by the improvement it produces, and
that improvement is attributed to what the component was meant to contribute.
That attribution is licensed only if two alternatives are excluded: that
something of the same shape and scale carrying **none** of the intended content
would do as well, and that an effect of the observed size lies within
seed-to-seed noise.

This repository implements both checks.

## Layout

```
code/
  physics_prior.py       the analytic hemoglobin prior: P_blood and the radial
                         fluence map Phi, both computed in closed form from RGB
  control_priors.py      the six matched controls -- zeros, shuffled,
                         random_fixed, gauss, phi_dup, cross_image -- each matched to the
                         prior in tensor shape and scale but carrying no
                         optical content
  datasets_pi.py         the transform that emits the 5-channel tensor
  models_pi.py           the 5-channel classifier
  train_stage2_pi.py     the trainer used for every run
  compute_tost.py        two one-sided tests against a pre-specified margin
  aggregate_controls.py  the control-matrix tables
  aggregate_selection.py the checkpoint-selection-rule ablation
  regen_results_numbers.py  regenerates the numbers quoted in the manuscript

predictions/
  efficientnet_b0/<arm>_seed<N>.npz     308 runs
  convnext_tiny/<arm>_seed<N>.npz       308 runs

splits/
  kvasir_split_manifest_2026-05-18.json  the canonical patient-disjoint split
```

Each `.npz` holds `probs` (n_frames x 14 softmax outputs), `labels`, `paths`
and `class_names`. Nothing in the analysis needs the checkpoints themselves —
every table and test in the paper is recomputable from these files.

## Arms

| arm | channels 4-5 | what it isolates |
|---|---|---|
| `rgb` | none (3-channel) | the baseline |
| `prior` | P_blood, Phi | the analytic prior under test |
| `zeros` | all-zero | input widening alone |
| `shuffled` | prior maps, spatially permuted | prior statistics without spatial structure |
| `random_fixed` | fixed random field | a content-free channel of matched scale |
| `gauss` | smooth Gaussian field | content-free but spatially smooth |
| `phi_dup` | Phi duplicated | geometry without the chromophore term |

A control that matches or beats `prior` removes the attribution, because it
supplies the same tensor shape, scale and optimization perturbation with none of
the optical content.

## Reproducing the paper's numbers

```bash
python code/regen_results_numbers.py --runs_root predictions/
python code/aggregate_controls.py   --runs_root predictions/
python code/compute_tost.py         --runs_root predictions/ --margin 0.023
```

The equivalence margin of 0.023 is the effect size reported for this prior in
the earlier preprint; the TOST asks whether an effect that large can be ruled
out, rather than whether a difference failed to reach significance.

## Data

Kvasir-Capsule is available from the original authors
(<https://datasets.simula.no/kvasir-capsule/>) under a research-use licence and
is not redistributed here. `splits/` gives the exact patient-disjoint partition
used, so the split can be reconstructed exactly.

## Citation

Citation details will be added on acceptance.

## Licence

MIT for the code. The prediction files are released under CC BY 4.0.


## Added for the methodological companion (MedIA)

```
code/
  positive_control.py       Beer-Lambert simulation establishing that the
                            apparatus detects a real effect where one exists
                            (+0.248 AUC at N=150, +0.0006 at N=4000)
  calibrate_influence.py    tests whether the label-free influence ratio R
                            predicts label-based attribution. It does not
                            (Spearman rho = 0.02); the negative result and its
                            mechanism are reported in the paper
  aggregate_regime.py       the data-fraction x capacity sweep on real images,
                            with the floor/ceiling validity gate and the
                            comparability check against the published matrix
  make_phase_figure.py      the two-panel phase diagram, simulation vs real
  aggregate_attribution.py  the control hierarchy, in-domain and external
  aggregate_partitions.py   variance decomposition across ten patient-disjoint
                            partitions (sigma_seed vs sigma_partition)
  influence_ratio.py        the label-free diagnostic and its permutation null

results/
  positive_control_results.json   per-seed AUCs for every simulation cell
  pc_probs.npz, pc_results.json   per-sample predictions behind the calibration
  calibration_results.json        R vs Delta_attr for all 48 cells
  regime_results_n5.json          the real-image sweep at 5 seeds per cell
  canonical_reference.json        per-arm mean/sd/se over the canonical 44-seed
                                  matrix, used to gate comparability

preregistration/
  the Clements third-cohort analysis plan, committed before any model was
  evaluated on that cohort, and the results recording no deviations from it
```

Note that `calibrate_influence.py` reports a **negative** result about one of
our own proposed methods. It is included deliberately: a statistic whose scale
is confounded is dangerous only while the confound is undocumented.
