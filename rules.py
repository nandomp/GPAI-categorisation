import numpy as np

TAU = 4.0
FOUR_DOMAINS = {"AS": ["AS"], "CE": ["CEc", "CEe"], "CL": ["CL"], "QL": ["QLl", "QLq"]}


def gini(x):
    s = np.sort(x)
    n = len(s)
    return 2 * np.sum(np.arange(1, n + 1) * s) / (n * s.sum()) - (n + 1) / n


def balance(x):
    return 1 - gini(x)


def arithmetic(x):
    return x.mean()


def geometric(x):
    return np.exp(np.log(x).mean())


def harmonic(x):
    return len(x) / np.sum(1 / x)


def minslack(x, lam=0.3):
    return x.min() + lam * (x.mean() - x.min())


def make_four_domain(columns):
    idx = {d: [columns.index(c) for c in cols] for d, cols in FOUR_DOMAINS.items()}

    def four_domain(x):
        return np.mean([x[i].mean() for i in idx.values()])

    return four_domain


def count_above(x, tau=TAU):
    return int((x >= tau).sum())


def bdb_stats(x):
    return np.quantile(x, 0.25), np.quantile(x, 0.75), balance(x)


def bdb_predict(stats, beta, delta, eps):
    q25, q75, bal = stats
    return int(q25 >= beta and q75 >= delta and bal >= eps)


def fit_threshold(scores, y):
    # smallest observed value that maximises agreement
    best_t, best_acc = None, -1
    for t in np.unique(scores):
        acc = np.mean((scores >= t).astype(int) == y)
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t


def fit_bdb(stats, y):
    q25, q75, bal = (np.array(s) for s in zip(*stats))
    best, best_acc = None, -1
    for b in np.unique(q25):
        for d in np.unique(q75):
            for e in np.unique(bal):
                pred = ((q25 >= b) & (q75 >= d) & (bal >= e)).astype(int)
                acc = np.mean(pred == y)
                if acc > best_acc:
                    best, best_acc = (b, d, e), acc
    return best
