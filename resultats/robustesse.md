# Robustesse sur 10 graines

Moyennes des métriques hors échantillon. `significatif_%` : part des graines où le modèle bat la prévision nulle au test de Diebold-Mariano (p < 0,05). Sur le placebo, c'est un taux de fausses découvertes, attendu autour de 5 % ou moins.

|                                   |   R2_oos_% |   IC_spearman |   sharpe_net |   significatif_% |
|:----------------------------------|-----------:|--------------:|-------------:|-----------------:|
| ('placebo', 'gradient_boosting')  |     -1.307 |        -0.001 |       -0.22  |                0 |
| ('placebo', 'moyenne_historique') |     -0.033 |        -0.016 |       -0.014 |                0 |
| ('placebo', 'ridge')              |     -0.07  |        -0.01  |       -0.032 |                0 |
| ('signal', 'gradient_boosting')   |     -0.1   |         0.058 |        0.663 |                0 |
| ('signal', 'moyenne_historique')  |     -0.024 |        -0.016 |        0.064 |                0 |
| ('signal', 'oracle')              |      1.369 |         0.1   |        1.262 |              100 |
| ('signal', 'ridge')               |      0.862 |         0.079 |        1.005 |               80 |
