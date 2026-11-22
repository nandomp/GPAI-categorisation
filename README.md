# GPAI-categorisation

Data and code for the paper *General Enough? Formalising General-Purpose AI for the EU AI Act* (under review).

The paper compares rules for deciding whether a model counts as general-purpose AI under Article 3(63) of the EU AI Act, using ability profiles for 19 language models and labels from a survey of 112 researchers.

## Contents

```
data/
  ability_profiles.csv   ability estimates for 19 models on 18 dimensions
  models.csv             family, release year, survey votes and label per model
rules.py                 aggregates, BDB statistics and threshold fitting
analysis.py              runs the analysis and writes results/
results/                 output of analysis.py
```

**Ability profiles.** One row per model, one column per dimension. The values are ability estimates on the 18 ADeLe dimensions (Zhou et al., 2026, *Nature* 652, 58–67, https://doi.org/10.1038/s41586-026-10303-2). Dimension codes follow the ADeLe rubrics and are listed in Appendix A of the paper.

**Survey labels.** `models.csv` gives, for each model, the number of respondents who answered GPAI, not GPAI or unsure (n = 112). A model is labelled GPAI or non-GPAI when at least two thirds of the definite answers agree. All 19 models reach this threshold (12 GPAI, 7 non-GPAI). Individual responses are not shared.

## Running

```
pip install -r requirements.txt
python analysis.py
```

It runs in a few seconds and prints the fitted BDB thresholds, the rule comparison, the models on which the rules disagree and the leave-one-out stability of the fitted thresholds. The same results are written as CSV files to `results/`.

## What is reproduced

- Table 2: summary statistics per model (`results/summary_stats.csv`)
- Table 5: in-sample accuracy, leave-one-out accuracy and uncertainty fraction for the eight rules (`results/rule_comparison.csv`)
- The fitted BDB thresholds and the seven models whose verdict depends on the rule (`results/verdicts.csv`)
- Appendix E.4: fold-by-fold thresholds under leave-one-out (`results/loocv_folds.csv`)

The bootstrap, conformal, elasticity and temporal analyses are not included in this release.

## Details

- Quartiles use linear interpolation between order statistics (NumPy's default).
- Thresholds are fitted over the observed values of the relevant statistic. Among values with equal agreement, the smallest is kept. For BDB the search runs over all combinations of observed lower quartiles, upper quartiles and balance values.
- A count rule "at least k of 18 dimensions at or above τ" is evaluated as "the k-th highest score is at or above τ", which lets leave-one-out refit τ in the same way as for the other rules.
- The uncertainty fraction adds Gaussian noise (σ = 0.25) to every dimension, 4,000 draws per model with seed 0, and counts the models whose probability of a GPAI verdict lies between 0.2 and 0.8.

## Licence

MIT
