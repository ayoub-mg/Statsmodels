---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# Système d'importation


`statsmodels` propose deux tournures pour importer ses fonctions et ses classes :

1. [Importation de l'API pour interagir avec les différentes fonctionnalités du paquet](#importation-de-lapi)
   <br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Permet l'auto-complétion via la touche `tab`.

2. [Importation directe des fonctions et classes](#importation-directe)
    <br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• Épargne d'importer des modules et des méthodes non utilisés.




## Importation de l'API

Pour un usage interactif, le mécanisme d'importation recommandé est :

```{code-cell}
:tags: ["remove-output"]
import statsmodels.api as sm
```

Importer `statsmodels.api` chargera la plupart des structures publiques de `statsmodels`. La plupart des fonctions et des classes sont ainsi disponibles à un ou deux niveaux du chemin de l'import (*i.e.* un ou deux `.`), sans que l'espace de nommage `sm` soit trop encombré.


L'interface de programmation API de `statsmodels` comporte les interfaces suivantes :

- `statsmodels.api` : Modèles et méthodes d'échantillonnage. Souvent chargée avec `import statsmodels.api as sm`.
- `statsmodels.tsa.api` : Méthodes et modèles de traitement des séries temporelles. Généralement importé moyennant `import statsmodels.tsa.api as tsa`.
- `statsmodels.formula.api` : Interface spécifiant les modèles au travers des formules analogues à celles du langage R et des dataframes. Peut être chargée par `import statsmodels.formula.api as smf`.


Pour répertorier les fonctions et classes offertes, il est possible d'utiliser la commande `dir` : 

```{code-cell}
:tags: ["output_scroll"]
dir(sm)
```

## Importation directe


Les sous-modules `statsmodels` sont classés par thème (*e.g.* `discret` pour les modèles de choix discrets, ou `tsa` pour l'analyse des séries temporelles). L'arborescence du paquet s'apparente à ce qui suit :

```
statsmodels/
    __init__.py
    api.py
    discrete/
        __init__.py
        discrete_model.py
        tests/
            results/
    tsa/
        __init__.py
        api.py
        tsatools.py
        stattools.py
        arima_process.py
        vector_ar/
            __init__.py
            var_model.py
            tests/
                results/
        tests/
            results/
    stats/
        __init__.py
        api.py
        stattools.py
        tests/
    tools/
        __init__.py
        tools.py
        decorators.py
        tests/
```



### Exemples

Classes & Fonctions :
```python
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.tools import rank, add_constant
```

Modules :
```python
from statsmodels.datasets import macrodata
import statsmodels.stats import diagnostic
```

Modules sous pseudonymes :
```python
import statsmodels.regression.linear_model as lm
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
```