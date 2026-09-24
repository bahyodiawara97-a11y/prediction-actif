import numpy as np
import pandas as pd
import pytest

from actif.data import SyntheticConfig, generate
from actif.evaluation import WalkForward, evaluate, walk_forward_predict
from actif.features import OBSERVABLE, build_features, make_dataset
from actif.models import MODELS


@pytest.fixture(scope="module")
def df():
    return generate(SyntheticConfig(n_days=2000, seed=1))


def test_features_n_utilisent_pas_le_futur(df):
    """On brouille tout ce qui suit la date t : les variables jusqu'à t ne doivent pas bouger."""
    t = 1200
    altered = df[OBSERVABLE].copy()
    rng = np.random.default_rng(42)
    altered.iloc[t + 1:] = rng.uniform(1, 2, altered.iloc[t + 1:].shape)
    before = build_features(df[OBSERVABLE]).iloc[: t + 1]
    after = build_features(altered).iloc[: t + 1]
    pd.testing.assert_frame_equal(before, after)


def test_cible_est_le_rendement_du_lendemain(df):
    X, y, _, _ = make_dataset(df)
    d = X.index[10]
    nxt = df.index[df.index.get_loc(d) + 1]
    assert y.loc[d] == df.loc[nxt, "ret"]


def test_walk_forward_chronologique_et_disjoint():
    wf = WalkForward(initial_train=100, step=30, embargo=5)
    seen = []
    for train, test in wf.splits(400):
        assert train.max() + wf.embargo < test.min()
        seen.extend(test)
    assert len(seen) == len(set(seen))            # chaque jour testé une seule fois
    assert seen == list(range(105, 400))          # et sans trou


def test_placebo_rien_a_decouvrir():
    """Sans signal, la ridge ne doit pas battre la prévision nulle de façon significative."""
    X, y, vol, _ = make_dataset(generate(SyntheticConfig(n_days=3000, r2=0.0, seed=3)))
    p = walk_forward_predict(MODELS["ridge"], X, y, vol, WalkForward())
    m = evaluate(y, p, vol, cost_bp=0)
    assert m["DM_pvalue"] > 0.05
    assert abs(m["IC_spearman"]) < 0.05


def test_signal_fort_est_retrouve():
    """Avec un signal nettement plus fort, le pipeline doit le détecter."""
    X, y, vol, _ = make_dataset(generate(SyntheticConfig(n_days=3000, r2=0.05, seed=3)))
    p = walk_forward_predict(MODELS["ridge"], X, y, vol, WalkForward())
    m = evaluate(y, p, vol, cost_bp=0)
    assert m["DM_pvalue"] < 0.01
    assert m["IC_spearman"] > 0.1
