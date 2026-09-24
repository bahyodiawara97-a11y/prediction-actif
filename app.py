"""Démo interactive : streamlit run app.py"""

from dataclasses import replace

import matplotlib.pyplot as plt
import streamlit as st

from actif.data import SyntheticConfig
from actif.evaluation import WalkForward
from actif.run import make_figure, run_experiment

st.set_page_config(page_title="Prédiction d'un actif financier", layout="wide")

st.title("Prédiction d'un actif financier")
st.markdown(
    "Série synthétique à signal faible : on **connaît** la part prévisible, "
    "donc on peut comparer chaque modèle à un **oracle** (la meilleure prévision possible). "
    "Toutes les prévisions sont faites hors échantillon, dans l'ordre chronologique."
)

with st.sidebar:
    st.header("Paramètres")
    r2 = st.slider("Part de variance prévisible (%)", 0.0, 5.0, 1.0, 0.25,
                   help="0 % = marché efficient : rien n'est prévisible.") / 100
    cost_bp = st.slider("Coût par transaction (points de base)", 0.0, 20.0, 2.0, 0.5)
    seed = st.number_input("Graine aléatoire (change la série)", 0, 999, 0)
    n_days = st.select_slider("Nombre de jours", [2000, 3000, 5000, 8000], 5000)
    placebo = st.checkbox("Lancer aussi le contrôle placebo (série sans signal)", True)


@st.cache_data(show_spinner=False)
def compute(r2, cost_bp, seed, n_days):
    cfg = SyntheticConfig(n_days=n_days, r2=r2, seed=seed)
    table, preds, y, _ = run_experiment(cfg, WalkForward(), cost_bp)
    return table, preds, y


COLS = {
    "R2_oos_%": "R² hors échantillon (%)",
    "IC_spearman": "Corrélation de rang (IC)",
    "taux_direction_%": "Bonne direction (%)",
    "DM_pvalue": "p-valeur Diebold-Mariano",
    "sharpe_net": "Sharpe net",
    "sharpe_IC95_bas": "Sharpe IC95 bas",
    "sharpe_IC95_haut": "Sharpe IC95 haut",
    "rotation_annuelle": "Allers-retours / an",
}


def show_table(table):
    t = table[list(COLS)].rename(columns=COLS)
    t.insert(0, "Bat la prévision nulle", table["DM_pvalue"].lt(0.05).map({True: "oui", False: "non"}))
    st.dataframe(t.style.format(precision=3, subset=list(COLS.values())), use_container_width=True)


with st.spinner("Entraînement walk-forward en cours…"):
    table, preds, y = compute(r2, cost_bp, seed, n_days)

st.subheader("Série avec signal")
if r2 == 0:
    st.info("Part prévisible à 0 % : la série est du bruit pur. Aucun modèle ne devrait battre la prévision nulle.")
else:
    best = table.drop(index="oracle")["R2_oos_%"].idxmax()
    ceiling = table.loc["oracle", "R2_oos_%"]
    got = table.loc[best, "R2_oos_%"]
    c1, c2, c3 = st.columns(3)
    c1.metric("R² plafond (oracle)", f"{ceiling:.2f} %")
    c2.metric(f"R² du meilleur modèle ({best})", f"{got:.2f} %")
    c3.metric("Part du signal récupérée", f"{100 * got / ceiling:.0f} %" if ceiling > 0 else "—")
show_table(table)

fig = make_figure(preds, y, cost_bp)
st.pyplot(fig)
plt.close(fig)

if placebo:
    st.subheader("Contrôle placebo : même protocole, série sans aucun signal")
    st.caption("Un protocole honnête ne doit rien y « découvrir » : aucun modèle ne devrait battre la prévision nulle.")
    with st.spinner("Placebo en cours…"):
        ptable, _, _ = compute(0.0, cost_bp, seed, n_days)
    show_table(ptable)

with st.expander("Comment lire ces résultats"):
    st.markdown(
        "- **R² hors échantillon** : part de l'erreur supprimée par rapport à la prévision « rendement = 0 ». "
        "Négatif = le modèle fait pire que ne rien prévoir.\n"
        "- **p-valeur Diebold-Mariano** : probabilité d'observer un tel avantage par hasard. "
        "Sous 0,05, le modèle bat la prévision nulle de façon significative.\n"
        "- **Sharpe net** : rendement / risque annualisé de la stratégie « acheter si la prévision est positive, "
        "vendre sinon », après coûts. L'intervalle IC95 vient d'un bootstrap par blocs.\n"
        "- À surveiller : le gradient boosting a souvent un Sharpe positif mais un R² négatif. "
        "Il surapprend le bruit, ce qu'un Sharpe seul ne montre pas."
    )
