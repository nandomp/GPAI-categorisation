from pathlib import Path

import numpy as np
import pandas as pd

from rules import (TAU, arithmetic, geometric, harmonic, minslack, make_four_domain,
                   balance, count_above, bdb_stats, bdb_predict, fit_threshold, fit_bdb)

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

profiles = pd.read_csv(DATA / "ability_profiles.csv").set_index("model")
models = pd.read_csv(DATA / "models.csv").set_index("model").loc[profiles.index]
X = profiles.values
y = (models["label"] == "GPAI").astype(int).values
names = list(profiles.index)
n = len(names)

four_domain = make_four_domain(list(profiles.columns))


def kth_largest(k):
    # "at least k dimensions >= tau" is the same as "k-th largest score >= tau"
    return lambda x: np.sort(x)[::-1][k - 1]


scalar_rules = {
    "4-domain mean": four_domain,
    "arithmetic mean": arithmetic,
    "geometric mean": geometric,
    "harmonic mean": harmonic,
    "MinSlack": minslack,
    ">=9/18": kth_largest(9),
    ">=13/18": kth_largest(13),
}
order = ["4-domain mean", "arithmetic mean", "geometric mean", "harmonic mean",
         "MinSlack", "BDB_q25", ">=9/18", ">=13/18"]

# Table 2
summary = pd.DataFrame({
    "mean": [arithmetic(x) for x in X],
    "geometric": [geometric(x) for x in X],
    "harmonic": [harmonic(x) for x in X],
    "min": X.min(axis=1),
    "q25": np.quantile(X, 0.25, axis=1),
    "q75": np.quantile(X, 0.75, axis=1),
    "max": X.max(axis=1),
    "balance": [balance(x) for x in X],
    "n_above_4": [count_above(x) for x in X],
    "label": models["label"].values,
}, index=names)
summary.round(3).to_csv(OUT / "summary_stats.csv")

# full-sample BDB fit and verdicts at tau = 4
stats = [bdb_stats(x) for x in X]
beta, delta, eps = fit_bdb(stats, y)

verdicts = {name: np.array([int(f(x) >= TAU) for x in X]) for name, f in scalar_rules.items()}
verdicts["BDB_q25"] = np.array([bdb_predict(s, beta, delta, eps) for s in stats])
verdict_table = pd.DataFrame({r: verdicts[r] for r in order}, index=names)
verdict_table.insert(0, "label", y)
verdict_table.to_csv(OUT / "verdicts.csv")
disagree = [m for m in names if verdict_table.loc[m, order].nunique() > 1]

# leave-one-out
loo_hits = {r: 0 for r in order}
folds = []
for i in range(n):
    tr = np.array([j for j in range(n) if j != i])
    for name, f in scalar_rules.items():
        s = np.array([f(x) for x in X])
        t = fit_threshold(s[tr], y[tr])
        loo_hits[name] += int(s[i] >= t) == y[i]
        if name == "arithmetic mean":
            tau_fold = t
    b, d, e = fit_bdb([stats[j] for j in tr], y[tr])
    loo_hits["BDB_q25"] += bdb_predict(stats[i], b, d, e) == y[i]
    folds.append(dict(held_out=names[i], beta=b, delta=d, eps=e, tau_mean=tau_fold))
folds = pd.DataFrame(folds)
folds.round(4).to_csv(OUT / "loocv_folds.csv", index=False)

# uncertainty under profile noise
rng = np.random.default_rng(0)
SIGMA, DRAWS = 0.25, 4000
noisy = X[:, None, :] + rng.normal(0, SIGMA, size=(n, DRAWS, X.shape[1]))
uncertain = {}
for name in order:
    p = []
    for i in range(n):
        if name == "BDB_q25":
            v = [bdb_predict(bdb_stats(z), beta, delta, eps) for z in noisy[i]]
        else:
            v = [scalar_rules[name](z) >= TAU for z in noisy[i]]
        p.append(np.mean(v))
    p = np.array(p)
    uncertain[name] = np.mean((p >= 0.2) & (p <= 0.8))

table = pd.DataFrame({
    "in_sample_acc": [np.mean(verdicts[r] == y) for r in order],
    "loocv_acc": [loo_hits[r] / n for r in order],
    "uncertainty_fraction": [uncertain[r] for r in order],
}, index=order)
table.round(3).to_csv(OUT / "rule_comparison.csv")

print(f"BDB_q25 fit: beta={beta:.4f}, delta={delta:.4f}, eps={eps:.4f}\n")
print(table.round(3).to_string(), "\n")
print(f"{len(disagree)} models change verdict across rules:", ", ".join(disagree), "\n")
print("LOOCV fold stability (mean, sd, min, max):")
for c in ["beta", "delta", "eps", "tau_mean"]:
    v = folds[c]
    print(f"  {c:9s} {v.mean():.3f}  {v.std(ddof=0):.3f}  [{v.min():.3f}, {v.max():.3f}]")
