"""Création de variables.

Règle unique : la ligne t ne contient que de l'information disponible à la
clôture du jour t. La cible de la ligne t est le rendement de t+1.
Le test tests/test_features.py vérifie cette règle en modifiant le futur.
"""

import numpy as np
import pandas as pd

# Colonnes de la série brute que le modèle a le droit de voir
OBSERVABLE = ["ret", "price", "x"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    r, price, x = df["ret"], df["price"], df["x"]
    vol20 = r.rolling(20).std()

    f = {}
    for k in range(1, 6):
        f[f"ret_lag{k}"] = r.shift(k - 1)          # ret_lag1 = rendement du jour t
    for w in (5, 20, 60):
        f[f"mom_{w}"] = r.rolling(w).sum()
    f["mom_20_z"] = f["mom_20"] / (vol20 * np.sqrt(20))
    f["vol_20"] = vol20
    f["vol_ratio"] = r.rolling(5).std() / r.rolling(60).std()
    f["ma_gap_50"] = np.log(price / price.rolling(50).mean())
    f["x"] = x
    f["x_diff5"] = x - x.shift(5)
    f["x_ma20"] = x.rolling(20).mean()
    return pd.DataFrame(f, index=df.index)


def make_dataset(df: pd.DataFrame):
    """Renvoie X, y (rendement de t+1), vol (échelle connue en t), mu (oracle)."""
    X = build_features(df[OBSERVABLE])
    y = df["ret"].shift(-1).rename("target")
    oracle = df["mu"].shift(-1).rename("oracle")
    data = pd.concat([X, y, oracle], axis=1).dropna()
    return data[X.columns], data["target"], data["vol_20"], data["oracle"]
