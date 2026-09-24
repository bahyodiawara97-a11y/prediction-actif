"""Génération d'une série de rendements synthétique à signal faible.

Le rendement du jour t s'écrit :

    r_t = mu_t + sigma_t * sqrt(1 - R2) * eps_t
    mu_t = sigma_t * sqrt(R2) * s_{t-1}

- sigma_t suit un GARCH(1,1) : la volatilité se regroupe, comme sur un vrai marché ;
- s_{t-1} est un signal de variance ~1, connu à la clôture de t-1, qui combine
  un facteur exogène observable (x, AR(1) persistant) et un momentum non linéaire ;
- R2 est la part de variance (standardisée) réellement prévisible.

Comme mu_t est connu par construction, on dispose d'un « oracle » : la meilleure
prévision possible. Il sert de plafond pour juger les modèles, ce qu'aucune
donnée réelle ne permet.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    n_days: int = 5000
    r2: float = 0.01          # part de variance prévisible (0 = marché efficient)
    phi_x: float = 0.95       # persistance du facteur exogène
    w_x: float = 0.7          # poids du facteur exogène (le reste : momentum)
    mom_window: int = 20
    annual_vol: float = 0.20
    garch_alpha: float = 0.08
    garch_beta: float = 0.90
    seed: int = 0


# Écart-type de tanh(Z) pour Z ~ N(0,1) : normalise le momentum à variance ~1
_TANH_STD = 0.6278


def generate(cfg: SyntheticConfig = SyntheticConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_days
    var_long = cfg.annual_vol**2 / 252
    omega = var_long * (1 - cfg.garch_alpha - cfg.garch_beta)
    norm = np.hypot(cfg.w_x, 1 - cfg.w_x)

    x = np.zeros(n)
    sigma = np.zeros(n)
    mu = np.zeros(n)
    ret = np.zeros(n)
    z = np.zeros(n)  # rendements standardisés, pour le momentum

    u = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    x[0] = u[0]
    sigma[0] = np.sqrt(var_long)

    for t in range(n):
        if t > 0:
            x[t] = cfg.phi_x * x[t - 1] + np.sqrt(1 - cfg.phi_x**2) * u[t]
            shock = ret[t - 1] - mu[t - 1]
            sigma[t] = np.sqrt(omega + cfg.garch_alpha * shock**2
                               + cfg.garch_beta * sigma[t - 1] ** 2)
            lo = max(0, t - cfg.mom_window)
            mom = np.tanh(z[lo:t].sum() / np.sqrt(cfg.mom_window)) / _TANH_STD
            s = (cfg.w_x * x[t - 1] + (1 - cfg.w_x) * mom) / norm
            mu[t] = sigma[t] * np.sqrt(cfg.r2) * s
        ret[t] = mu[t] + sigma[t] * np.sqrt(1 - cfg.r2) * eps[t]
        z[t] = ret[t] / sigma[t]

    index = pd.bdate_range("2006-01-02", periods=n, name="date")
    return pd.DataFrame(
        {
            "ret": ret,
            "price": 100 * np.exp(np.cumsum(ret)),
            "x": x,          # observable par le modèle
            "mu": mu,        # espérance conditionnelle vraie (oracle uniquement)
            "sigma": sigma,  # volatilité vraie (oracle uniquement)
        },
        index=index,
    )
