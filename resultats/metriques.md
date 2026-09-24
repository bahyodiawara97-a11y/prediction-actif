# Résultats hors échantillon

Série : 5000 jours, R² prévisible vrai = 1.00 %, graine 0. Walk-forward : entraînement initial 1000 j, réentraînement tous les 250 j, embargo 5 j. Coûts : 2.0 pb par transaction. 3935 jours évalués hors échantillon.

## Série avec signal

|                    |   R2_oos_% |   IC_spearman |   taux_direction_% |   DM_pvalue |   sharpe_net |   sharpe_IC95_bas |   sharpe_IC95_haut |   drawdown_max_% |   rotation_annuelle |
|:-------------------|-----------:|--------------:|-------------------:|------------:|-------------:|------------------:|-------------------:|-----------------:|--------------------:|
| moyenne_historique |     -0.081 |        -0.022 |             48.844 |       0.955 |       -0.339 |            -0.983 |              0.223 |          179.831 |               0.064 |
| ridge              |      0.847 |         0.072 |             51.919 |       0.001 |        0.938 |             0.37  |              1.458 |           50.318 |              33.438 |
| gradient_boosting  |      0.114 |         0.05  |             50.673 |       0.408 |        0.446 |            -0.099 |              0.948 |           48.68  |              71.488 |
| oracle             |      1.468 |         0.098 |             52.376 |       0     |        1.096 |             0.527 |              1.608 |           39.723 |              22.484 |

## Contrôle placebo (même série, R² vrai = 0)

|                    |   R2_oos_% |   IC_spearman |   taux_direction_% |   DM_pvalue |   sharpe_net |   sharpe_IC95_bas |   sharpe_IC95_haut |   drawdown_max_% |   rotation_annuelle |
|:-------------------|-----------:|--------------:|-------------------:|------------:|-------------:|------------------:|-------------------:|-----------------:|--------------------:|
| moyenne_historique |     -0.02  |        -0.017 |             50.928 |       0.695 |        0.164 |            -0.353 |              0.669 |           92.042 |               0.128 |
| ridge              |     -0.032 |        -0.021 |             50.724 |       0.789 |        0.171 |            -0.331 |              0.681 |           77.997 |               8.199 |
| gradient_boosting  |     -1.307 |        -0.018 |             49.276 |       1     |       -0.459 |            -0.929 |             -0.003 |          184.33  |              99.993 |
