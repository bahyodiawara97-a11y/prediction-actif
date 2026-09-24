"""Modèles candidats. Chacun est une fabrique : un modèle neuf à chaque réentraînement."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class HistoricalMean(BaseEstimator, RegressorMixin):
    """Référence : prévoit la moyenne des rendements passés."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def ridge():
    # alpha choisi par validation chronologique interne, jamais par validation aléatoire
    return GridSearchCV(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": np.logspace(0, 5, 11)},
        cv=TimeSeriesSplit(n_splits=4),
        scoring="neg_mean_squared_error",
    )


def gradient_boosting():
    # Très régularisé : sur un signal faible, un modèle souple apprend surtout le bruit.
    # Pas d'early_stopping : sa validation interne est tirée au hasard, donc mélange les dates.
    return HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.03, max_depth=3,
        min_samples_leaf=200, l2_regularization=1.0, random_state=0,
    )


MODELS = {
    "moyenne_historique": HistoricalMean,
    "ridge": ridge,
    "gradient_boosting": gradient_boosting,
}
