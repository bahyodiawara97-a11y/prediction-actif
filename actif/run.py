"""Point d'entrée : python -m actif.run [--r2 0.01] [--seed 0] [--cost-bp 2]

Lance deux expériences :
1. la série avec signal : chaque modèle est comparé à l'oracle et aux références ;
2. un contrôle placebo, la même série sans signal (R2 = 0) : un protocole
   honnête ne doit rien y « découvrir ».
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import SyntheticConfig, generate
from .evaluation import WalkForward, evaluate, strategy_returns, walk_forward_predict
from .features import make_dataset
from .models import MODELS


def run_experiment(cfg: SyntheticConfig, wf: WalkForward, cost_bp: float):
    X, y, vol, oracle = make_dataset(generate(cfg))
    preds = {name: walk_forward_predict(f, X, y, vol, wf) for name, f in MODELS.items()}
    oos = preds["ridge"].index
    if cfg.r2 > 0:  # sans signal, l'oracle prévoit 0 partout : rien à mesurer
        preds["oracle"] = oracle.loc[oos]
    table = pd.DataFrame({n: evaluate(y, p, vol, cost_bp) for n, p in preds.items()}).T
    return table, preds, y, vol


def make_figure(preds, y, cost_bp):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name, p in preds.items():
        yy = y.loc[p.index]
        pnl = strategy_returns(yy.to_numpy(), p.to_numpy(), cost_bp)
        style = "--" if name == "oracle" else "-"
        axes[0].plot(p.index, np.cumsum(pnl), style, label=name)
        if p.nunique() > 20:  # une prévision quasi constante n'a pas de corrélation lisible
            axes[1].plot(p.index, p.rolling(250).corr(yy), style, label=name)
    axes[0].set_title(f"Stratégie « signe de la prévision », net de {cost_bp} pb par transaction")
    axes[0].set_ylabel("rendement log cumulé")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_title("Corrélation prévision / réalisé, glissante sur 250 jours")
    axes[0].legend()
    fig.tight_layout()
    return fig


def plot(preds, y, cost_bp, out: Path):
    fig = make_figure(preds, y, cost_bp)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r2", type=float, default=0.01)
    ap.add_argument("--n-days", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cost-bp", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=Path("resultats"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    cfg = SyntheticConfig(n_days=args.n_days, r2=args.r2, seed=args.seed)
    wf = WalkForward()

    table, preds, y, vol = run_experiment(cfg, wf, args.cost_bp)
    placebo, _, _, _ = run_experiment(replace(cfg, r2=0.0), wf, args.cost_bp)

    fmt = lambda t: t.round(3).to_markdown()
    report = (
        f"# Résultats hors échantillon\n\n"
        f"Série : {cfg.n_days} jours, R² prévisible vrai = {100 * cfg.r2:.2f} %, "
        f"graine {cfg.seed}. Walk-forward : entraînement initial {wf.initial_train} j, "
        f"réentraînement tous les {wf.step} j, embargo {wf.embargo} j. "
        f"Coûts : {args.cost_bp} pb par transaction. "
        f"{len(preds['ridge'])} jours évalués hors échantillon.\n\n"
        f"## Série avec signal\n\n{fmt(table)}\n\n"
        f"## Contrôle placebo (même série, R² vrai = 0)\n\n{fmt(placebo)}\n"
    )
    (args.out / "metriques.md").write_text(report)
    table.to_csv(args.out / "metriques.csv")
    placebo.to_csv(args.out / "metriques_placebo.csv")
    plot(preds, y, args.cost_bp, args.out / "performance.png")
    print(report)


if __name__ == "__main__":
    main()
