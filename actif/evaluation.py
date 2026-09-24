"""Évaluation chronologique hors échantillon (walk-forward) et métriques.

Protocole : fenêtre d'entraînement croissante, réentraînement tous les `step`
jours, puis prévision du bloc suivant, jamais vu. Un `embargo` de quelques
jours sépare la fin de l'entraînement du début du test.

Les modèles apprennent le rendement divisé par la volatilité récente (connue
en t) : sans cela, les périodes agitées dominent l'erreur quadratique.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class WalkForward:
    initial_train: int = 1000
    step: int = 250
    embargo: int = 5

    def splits(self, n: int):
        train_end = self.initial_train
        while train_end + self.embargo < n:
            test_start = train_end + self.embargo
            test_end = min(test_start + self.step, n)
            yield np.arange(0, train_end), np.arange(test_start, test_end)
            train_end += self.step


def walk_forward_predict(factory, X, y, vol, wf: WalkForward) -> pd.Series:
    preds = pd.Series(np.nan, index=X.index, name="pred")
    y_scaled = y / vol
    for train, test in wf.splits(len(X)):
        model = factory().fit(X.iloc[train], y_scaled.iloc[train])
        preds.iloc[test] = model.predict(X.iloc[test]) * vol.iloc[test].to_numpy()
    return preds.dropna()


# --- métriques -------------------------------------------------------------

def r2_oos(y, p):
    """R² hors échantillon contre la prévision nulle (rendement attendu = 0)."""
    return 1 - np.sum((y - p) ** 2) / np.sum(y**2)


def diebold_mariano(y, p, lags=10):
    """Le modèle bat-il la prévision nulle ? Statistique DM avec variance HAC
    (Newey-West) et p-valeur unilatérale."""
    d = y**2 - (y - p) ** 2
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return stat, 1 - stats.norm.cdf(stat)


def strategy_returns(y, p, cost_bp):
    """Position = signe de la prévision ; coût payé à chaque changement de position."""
    pos = np.sign(p)
    turnover = np.abs(np.diff(pos, prepend=0))
    return pos * y - turnover * cost_bp / 1e4


def sharpe(pnl):
    return np.sqrt(252) * pnl.mean() / pnl.std()


def block_bootstrap_ci(pnl, stat=sharpe, block=20, n_boot=1000, seed=0):
    """Intervalle à 95 % par bootstrap par blocs (préserve l'autocorrélation)."""
    rng = np.random.default_rng(seed)
    n = len(pnl)
    n_blocks = int(np.ceil(n / block))
    values = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        values.append(stat(pnl[idx]))
    return np.percentile(values, [2.5, 97.5])


def max_drawdown(pnl):
    wealth = np.cumsum(pnl)
    return np.max(np.maximum.accumulate(wealth) - wealth)


def evaluate(y: pd.Series, p: pd.Series, vol: pd.Series, cost_bp: float) -> dict:
    y, p, vol = y.loc[p.index], p, vol.loc[p.index]
    ys, ps = (y / vol).to_numpy(), (p / vol).to_numpy()
    dm, dm_p = diebold_mariano(ys, ps)
    pnl = strategy_returns(y.to_numpy(), p.to_numpy(), cost_bp)
    lo, hi = block_bootstrap_ci(pnl)
    return {
        "R2_oos_%": 100 * r2_oos(ys, ps),
        "IC_spearman": stats.spearmanr(p, y).statistic,
        "taux_direction_%": 100 * np.mean(np.sign(p) == np.sign(y)),
        "DM_pvalue": dm_p,
        "sharpe_net": sharpe(pnl),
        "sharpe_IC95_bas": lo,
        "sharpe_IC95_haut": hi,
        "drawdown_max_%": 100 * max_drawdown(pnl),
        "rotation_annuelle": 252 * np.mean(np.abs(np.diff(np.sign(p.to_numpy())))) / 2,
    }
