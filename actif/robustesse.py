"""Répète l'expérience sur plusieurs graines : un résultat sur une seule série
synthétique peut être de la chance. python -m actif.robustesse [--n-seeds 10]"""

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .data import SyntheticConfig
from .evaluation import WalkForward
from .run import run_experiment

COLS = ["R2_oos_%", "IC_spearman", "sharpe_net", "DM_pvalue"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--r2", type=float, default=0.01)
    ap.add_argument("--cost-bp", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=Path("resultats"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    rows = []
    for seed in range(args.n_seeds):
        cfg = SyntheticConfig(r2=args.r2, seed=seed)
        for label, c in (("signal", cfg), ("placebo", replace(cfg, r2=0.0))):
            table, *_ = run_experiment(c, WalkForward(), args.cost_bp)
            for model, m in table[COLS].iterrows():
                rows.append({"serie": label, "modele": model, "graine": seed, **m})
        print(f"graine {seed} terminée")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "robustesse_detail.csv", index=False)
    g = df.groupby(["serie", "modele"])
    summary = g[COLS[:3]].mean().round(3)
    summary["significatif_%"] = (100 * g["DM_pvalue"].apply(lambda p: (p < 0.05).mean())).round(0)
    text = (
        f"# Robustesse sur {args.n_seeds} graines\n\n"
        "Moyennes des métriques hors échantillon. `significatif_%` : part des graines où le "
        "modèle bat la prévision nulle au test de Diebold-Mariano (p < 0,05). "
        "Sur le placebo, c'est un taux de fausses découvertes, attendu autour de 5 % ou moins.\n\n"
        f"{summary.to_markdown()}\n"
    )
    (args.out / "robustesse.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
