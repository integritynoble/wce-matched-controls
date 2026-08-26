# Matched controls for physics-informed input channels in capsule endoscopy

Code and per-seed predictions accompanying *"Matched Controls and Equivalence
Testing for Physics-Informed Input Channels in Wireless Capsule Endoscopy"*
(submitted, IEEE Transactions on Medical Imaging, 2026).

This repository contains everything needed to check the paper's numbers without
retraining anything: the analytic prior, the five content-free control priors,
the equivalence-testing code, the aggregation scripts, and the **per-frame test
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
  control_priors.py      the five matched controls -- zeros, shuffled,
                         random_fixed, gauss, phi_dup -- each matched to the
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
