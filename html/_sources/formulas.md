# Langage de Description des Modèles

Depuis la version 0.5.0, `statsmodels` permet aux utilisateurs de spécifier des modèles statistiques en utilisant des formules de type R. Implicitement, `statsmodels` utilise le paquet **patsy** pour convertir les formules et les données passées en paramètres en matrices de régression qui seront utilisées dans l'ajustement du modèle.

Le langage de description `patsy` est assez puissant pour schématiser la relation entre plusieurs variables au sein d'un modèle. Plus spécifiquement, l’idée revient à exprimer une relation *fonctionnelle*, symbolisée par l’opérateur `~`, entre une variable réponse `y` et une ou plusieurs variables explicatives.

Disons, pour simplifier, que `y` est une variable d’intérêt (numérique ou catégorique selon le type de modèle), `x` une variable numérique et que `u` et `v` sont des variables catégorielles (qualitatives). Voici les principales relations auxquelles on peut s’intéresser dans un modèle statistique :

   - `y ~ x` : régression simple, correspondant à $y=ax+b$.
   - `y ~ x + 0` ou `y ~ x - 1`: idem avec suppression du terme d’ordonnée à l’origine (*intercept*), ou $y=ax$.
   - `y ~ u + v` : régresse avec deux effets principaux indépendants, *i.e.* $y=\beta u+\gamma v$.
   - `y ~ u * v` : idem avec interaction ou multiplication (équivalent à `1 + u + v + u:v`).
   - `y ~ u / v` : idem en considérant une relation d’emboîtement (équivalent à `1 + u + v + u %in% v`).
   - `y ~ I(x + u + v)` : permet de prendre le terme tel qu'il est sans interprétation. Le `+` correspond à l'addition dans Python, non celle de `patsy`.
   - `y ~ C(x)` : permet de forcer le traitement de la variable numérique `x` comme une catégorie.



```{admonition} Pour enchérir
:class: seealso 
Une description intègrale du langage de formulation des modèles est disponible dans la [documentation de **patsy**](https://patsy.readthedocs.io/en/latest/).
```