# Mise en place de l'environnement

La façon la plus simple d'installer `statsmodels` est de le charger en tant que composant de la distribution Anaconda, l'environnement multi-plateforme dédié à la science des données et à l'apprentissage automatique, laquelle vise à simplifier la gestion des paquets et de déploiement. C'est la méthode d'installation préconisée pour la plupart des utilisateurs.

Il est à noter que les compatibilités en question concernent la version stable 0.13.0 du paquet.

## Versions Python

Les versions de Python prises en charge par le module `statsmodels` sont Python 3.7, 3.8, et 3.9.

## Anaconda


`statsmodels` est disponible au travers du gestionnaire des packages **conda** fourni par Anaconda. La dernière version peut être installée au moyen de :

```
conda install -c conda-forge statsmodels
```

## PyPI (pip)

La version la plus récente peut être récupérée grâce à `pip` :

```
pip install statsmodels
```

## Dépendances Systématiques

Les dépendances minimales requises sont :

- `Python` ≥ 3.7
- `NumPy` ≥ 1.17
- `SciPy` ≥ 1.3
- `Pandas` ≥ 1.0
- `Patsy` ≥ 0.5.2

Compte tenu du long cycle de publication, `statsmodels` suit une politique souple pour les dépendances : les dépendances minimales sont publiées chaque an et demi à deux ans. La prochaine mise à jour planifiée des versions minimales est prévue pour le premier semestre 2023.


## Dépendances Optionnelles

- `cvxopt` est important pour la technique de régularisation des modèles d'apprentissage. 
- `Matplotlib` ≥ 3 est souvent sollicité pour les tracés et les graphiques.
- `X-12-ARIMA` ou `X-13ARIMA-SEATS` peuvent être utilisés pour l'analyse des séries temporelles.
- `pytest` framework de tests logiciels.
- `IPython` ≥ 6.0 nécessaire pour builder les fichiers localement ou pour utiliser des notebooks.
- `joblib` ≥ 1.0 utile pour l'accélération des estimations de certains modèles.
- `jupyter` est requis pour l'exécution des calepins Jupyter.


```{Note}
Il est à noter que toutes les manipulations entreprises dans cet ouvrage utilisent uniquement la suite des librairies fournies avec la distribution Anaconda. Les autres dépendances couvertes ci-dessus font figure d'un inventaire à titre informatif.
```