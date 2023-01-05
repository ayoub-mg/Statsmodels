# Attributs & Statistiques Résultantes

Les différentes méthodes du paquet `statsmodels` ayant trait aux modèles linéaires renvoient des objets comprenant plusieurs attributs, qui permettent de synthétiser le résultat de l'ajustement ou du test statistique sous forme de quantités numériques ou de réalisations de statistiques.

## Attributs

Voici une description des attributs communs à toutes les classes de régression `statsmodels` :

- `params` : `array`
    >Les paramètres du modèle (*intercept* en tête).

- `fittedvalues` : `array`
    >Les valeurs prédites par le modèle pour chaque observation.

- `pvalues` : `array`
    >Les valeurs du seuil critique pour chaque paramètre.

- `f_pvalue` : `float`
    >La valeur du seuil critique ($p$-value) du test de Fisher global.

- `pinv_wexog` : `array`
    >Le pseudo-inverse de la matrice explicative d'ordre $p\times n$. Approximativement égal à $\left(\mathbf X^{T}\Sigma^{-1}\mathbf X\right)^{-1}\mathbf X^{T}\Psi$, où $\Psi$ est défini comme $\Psi\Psi^{T}=\Sigma^{-1}$.

- `cholsimgainv` : `array`
    >La matrice triangulaire supérieur $\Psi^T$ vérifiant $\Psi\Psi^{T}=\Sigma^{-1}$.


- `df_model` : `float` 
    >Le nombre de degrés de liberté du modèle, égal à $p-1$, où $p$ est le nombre de régresseurs. Il est à noter que l'*intercept* est aussi traité en tant que degré de liberté.


- `df_resid` : `float`
    >Le nombre de degrés de liberté des résidus, c'est-à-dire $n-p$, où $n$ est le nombre d'observations et $p$ celui des paramètres. La constante à l'origine est pareillement comptabilisé.


- `llf` : `float`
    >La valeur de la fonction de vraisemblance du modèle ajusté.


- `nobs` : `float`
    >Le nombre d'observations de l'échantillon.

- `normalized_cov_params` : `array`
    >Matrice d'ordre $p\times p$ de valeur $(\mathbf X^{T}\Sigma^{-1}\mathbf X)^{-1}$.


- `sigma` : `array`
    >La matrice de variance-covariance des termes d'erreur $\mathbf{\varepsilon}\sim \mathcal N\left(\mathbf 0,\Sigma\right)$.


- `wexog` : `array`
    >La matrice explicative modifiée $\Psi^{T}\mathbf X$.


- `wendog` : `array`
    >La matrice de réponse modifiée $\Psi^{T}\mathbf y$.

- `conf_int(alpha)` : `array`
    >L'intervalle de confiance sur chacun des paramètres au niveau de confiance `alpha`.

- `centered_tss` : `float`
    >La variance totale (somme des carrés des écarts à la moyenne) du modèle.





## Statistiques Résultantes

En invoquant la méthode `summary()` d'un objet encapsulant le résultat d'une fonction d'ajustement d'un modèle statistique, plusieurs statistiques sont affichées sous format textuel, au premier rang desquelles :

- `R-squared` : Coefficient de détermination linéaire de la régression, noté $R^2$, qui mesure la qualité de la prédiction de l'ajustement d'un modèle linéaire. Il correspond à la proportion de la variation observée de la variable de réponse expliquée par le(s) variable(s) explicative(s).

    > $${\displaystyle R^{2}=1-{\dfrac {\sum _{i=1}^{n}\left(y_{i}-{\hat {y_{i}}}\right)^{2}}{\sum _{i=1}^{n}\left(y_{i}-{\bar {y}}\right)^{2}}}}$$

- `Adj. R-squared` : Permet de contourner le phénomène d'inflation de $R^2$ lors de l'ajout de variables explicatives au modèle. Utile pour la comparaison entre modèle de niveaux différents.
    >$${\displaystyle {\bar {R}}^{2}=1-(1-R^{2}){n-1 \over n-p}}$$

- `F-statistic` : Réalisation de la statistique de Fisher mesurant la significativité de l'ajustement du modèle par rapport au modèle naïf (ajustement par la moyenne).
  >$$\mathrm F = \frac{\mathrm{MCE}}{\mathrm{MCR}}$$


- `Prob (F-statistic)` : Probabilité que le modèle complet n'est pas meilleur que le modèle naïf. Plus il est faible, le mieux l'échantillon s'ajuste au modèle.


- `Log-Likelihood` : Probabilité conditionnelle que le modèle est adapté.

- `AIC` : Critère d'information d'Akaike estimant l'erreur de prédiction, et donc ipso-facto la qualité d'ajustement du modèle statistique.
    >$${\displaystyle \mathrm {AIC} \,=\,2k-2\ln({\hat {L}})}$$
        où $\hat L$ est le maximum de vraisemblance du modèle.

- `Df Residuals` : Degrés de liberté des résidus entre les valeurs observées et prédites.

- `BIC` : Critère d'information Bayésien, pénalisant le nombre de paramètres plus fortement que $\rm AIC$.
    >$${\mathrm {BIC}}=-2\ln(\hat L)+\ln(n)k$$

- `Df Model` : Nombre de paramètres du modèle.

- `Coefficient Constant` : *Intercept* de $\mathbf y$.

- `Independent Coefficient` : Représente la quantité à ajouter à la réponse en modifiant les variables indépendantes d'une unité.

- `Standard Error` : Erreur standard (écart-type) de l'estimation des paramètres.

- `P>|t|` : Valeur du seuil critique ($p$-value), associée au test d'hypothèse nulle $\mathrm H_0 : \beta_j=0$.

- `[.025 - .975]` : Intervalle de confiance encapsulant $95\%$ des estimations des paramètres.

- `Skew` : Mesure de l'asymétrie de la distribution de probabilité des résidus. Une asymétrie négative indique que la queue est plus longue à gauche et que la concentration des données se trouve à droite. Une valeur positive insinue que la queue est plus longue à droite. 0 indique que les queues sont équilibrées.

- `Kurtosis` : Décrit la forme de la distribution des résidus en s'intéressant aux queues et non sur le pic. Si la valeur est forte, cela signifie qu'il y a plus de valeurs aberrantes. Si la valeur est inférieure à 3, cela signifie qu'il y a moins de valeurs aberrantes. Une valeur de 3 indique une distribution normale. Les valeurs supérieures à 3 indiquent un plus grand nombre de valeurs aberrantes.