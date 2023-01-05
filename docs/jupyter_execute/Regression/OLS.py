#!/usr/bin/env python
# coding: utf-8

# # Exemple de Régression par les Moindres Carrés (OLS)

# La méthode canonique du paquet `statsmodels` pour l'ajustement des modèles de régression linéaire selon le critère des moindres carrés (OLS) est la méthode `OLS`.
# 
# Les paramètres pris en charge par cette fonction :
# - `endog` : `array-like`
#     >Une variable de réponse endogène sous forme d'objet compatible avec les arrays à une dimension `numpy`.
# 
# - `exog` : `array-like`
#     >Un tableau $n\times k$ où $n$ est le nombre d'observations et $k$ est le nombre de régresseurs. Un terme *intercept* n'est pas inclus par défaut et doit être spécifié (au moyen de `add_constant`).
# 
# - `missing` : `str`
#     >Les options disponibles sont `none`, `drop`, et `raise`. Si `none`, aucune vérification des valeurs `nan` n'est effectuée. `drop` fait que toutes les observations avec des `nan` sont abandonnées. `raise` signale une erreur. La valeur par défaut est `none`.

# On commence par charger les paquets nécessaires, entre autres `matplotlib.pyplot` pour le traçage, `numpy` pour la manipulation des matrices et vecteurs et la génération aléatoire, ainsi que `pandas` pour le stockage des données.

# In[1]:


# Affichage avec la bibliothèque graphique intégrée à Notebook
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

np.random.seed(7890)


# ## Estimation Linéaire par les Moindres Carrés

# On commence par générer un échantillon de $n=100$ observations du modèle $y_i=1+0.1x_i+10x_i^2+\varepsilon_i$ où $\varepsilon_i\sim\mathcal N(0,1)$. Cet échantillon servera de référence pour l'ajustement ci-après du modèle de régression linéaire de la variable $y$ sur les variables $x$ et $x^2$.

# In[2]:


n = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x ** 2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=n)


# Aucune constante n'est ajoutée au modèle. Il faut entasser une colonne de $1$ manuellement.

# In[3]:


X = sm.add_constant(X)
y = np.dot(X, beta) + e


# Il s'agit maintenant d'ajuster le modèle de régression linéaire par moindres carrés sur le jeu de donnéees en enchaînant la méthode `fit()` sur le modèle résultant de l'appel `OLS()`. Puis, on affiche le résultat par la méthode `summary()`.

# In[4]:


res = sm.OLS(y, X).fit()
res.summary()


# Les quantités d'intérêt peuvent être extraites directement du modèle ajusté. `dir(results)` pour obtenir une liste complète. Voici quelques exemples :

# In[5]:


print("Estimations des Paramètres : ", res.params)
print("Coefficient de détermination R2 : ", res.rsquared)


# Force est de constater que le coefficient de détermination est proche de 1. Cela traduit que la proportion de la variabilité observée de $y$ expliquée par le modèle de régression linéaire $\beta_0+\beta_1x+\beta_2x$ est plus de $99\%$, confirmant ainsi le modèle théorique de base employé pour générer l'échantillon.
# 
# L'estimation des paramètres selon le critère des moindres carrés étant très précise, les paramètres retrouvés avoisinent les valeurs théoriques $\beta_0=1$, $\beta_1=0.1$ et $\beta_2=10$. 

# ## Estimation Linéaire en les paramètres par les Moindres Carrés

# ```{admonition} Rappel
# :class: tip
# La désignation *linéaire* renvoie à la relation entre les paramètres $\boldsymbol\beta$ et la réponse $y$. $y$ est une fonction linéaire par rapport à $\boldsymbol\beta$, mais pas forcément de(s) variable(s) indépendante(s) (*e.g.* le modèle $y=e^x\beta+\varepsilon$ est bel et bien linéaire).
# ```

# On simule pour cette section un jeu de données artificel avec une relation non-linéaire entre $x$ et $y$ : $y_i=5+0.5x_i+0.5\sin(x_i)-0.02(x_i-5)^2+\varepsilon_i$ avec des erreurs $\varepsilon_i\sim\mathcal N(0,0.5^2)$. C'est bien un modèle linéaire, car il l'est pour les paramètres.

# In[6]:


sig = 0.5
x = np.linspace(0, 20, n)
X = np.column_stack((x, np.sin(x), (x-5)**2, np.ones(n)))
beta = np.array([0.5, 0.5, -0.02, 5.])
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=n)


# In[7]:


res2 = sm.OLS(y, X).fit()
res2.summary()


# On peut extraire des quantités d'intérêt par la même façon :

# In[8]:


print("Estimation des Paramètres : ", res2.params)
print("Erreurs Standards :", res2.bse)
print("Valeurs Prédites : ", res2.predict())
print("Coefficient de Détermination : ", res2.rsquared)


# Plus de $92\%$ de variabilité observée de $y$ est recensée par le modèle, ce dernier s'adapte significativement bien aux données. Les estimations des paramètres sont au voisinage des paramètres théoriques réels :  $\beta_1=0.5$, $\beta_2=0.5$, $\beta_3=-0.02$ et $\beta_0=5$. 

# Il est possible de tracer un graphique pour comparer les valeurs sans bruit (*i.e.* sans $\varepsilon$) du modèle aux prédictions par les moindres carrés ordinaires. Les intervalles de prédictions sont construits à l'aide de la commande `wls_prediction_std`.

# In[9]:


pfstd, iv_sup, iv_inf = wls_prediction_std(res2)
fig, ax = plt.subplots(figsize=(15,6))
    
ax.plot(x, y, 'o', label = 'Valeurs Réelles')
ax.plot(x, y_true, 'b-', label='Valeurs Sans Bruit')
ax.plot(x, res2.fittedvalues, 'r--', label='Valeurs Ajustées')
ax.plot(x, iv_sup, 'g--', label="Intervalle de Prédiction")
ax.plot(x, iv_inf, 'g--')
ax.legend(loc="best")


# Les valeurs ajustées sont à peu près identiques aux valeurs sans bruit, ce qui confirme la significativité de l'ajustement du modèle susmentionné aux données. L'intervalle de prédiction encapsule aussi l'écrasante majorité des valeurs observée, de par sa définition.
