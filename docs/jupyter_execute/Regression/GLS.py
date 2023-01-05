#!/usr/bin/env python
# coding: utf-8

# # Exemple de Régression par les Moindres Carrés Généralisés (GLS)

# La méthode canonique du paquet `statsmodels` pour l'ajustement des modèles de régression linéaire selon le critère des moindres carrés généralisés (GLS) est la méthode `GLS`.
# 
# Les paramètres requis sont :
# - `endog` : `array-like`
#     >Une variable de réponse endogène sous forme d'objet compatible avec les arrays à une dimension `numpy`.
# 
# - `exog` : `array-like`
#     >Un tableau $n\times k$ où $n$ est le nombre d'observations et $k$ est le nombre de régresseurs. Un terme *intercept* n'est pas inclus par défaut et doit être spécifié par l'utilisateur (au moyen de `add_constant`).
# 
# - `sigma` : `scalar` ou `array`
#     >Un tableau ou un scalaire de type `numpy` désignant la matrice de variance-covariance pondérée $\Sigma$. La valeur par défaut est `None`. Si `sigma` est un scalaire, il est supposé que `sigma` est une matrice diagonale $n\times n$ avec le scalaire donné comme valeur de chaque élément diagonal. Si `sigma` est un vecteur de longueur $n$, alors `sigma` est supposé être une matrice diagonale avec la valeur donnée sur la diagonale.
# 

# On commence par charger les paquets nécessaires, principalement `matplotlib.pyplot` pour le traçage, et `numpy` pour la manipulation des matrices et vecteurs et la génération aléatoire.

# In[1]:


import numpy as np
import statsmodels.api as sm


# ```{note}
# `statsmodels` offre la possibilité de charger des jeux de données classiques identiques à ceux du logiciel R, via le sous-module `datasets` pour réaliser des tests, comparer des modèles étudiés, ou pour des tutoriels d'apprentissage.
# ```
# 
# Le jeu de données qu'on utilisera pour cette manipulation est **Longley**. C'est une réalisation de séries temporelles de diverses variables macroéconomiques américaines connues pour être fortement colinéaires. On en extrait la variable de réponse.

# In[2]:


data = sm.datasets.longley.load()
data.exog = sm.add_constant(data.exog)
n=len(data.exog)
print(data.exog.head())


# Pour déterminer le paramètre `sigma`, on estime d'emblée les résidus qui feront figure des erreurs $(\varepsilon_i)_i$, avec la régression par le critère des moindres carrés OLS.
# 
# ```{admonition} Mise en garde
# :class: warning
# Cette technique d'estimation de la matrice $\Sigma$ est dite **Feasible Generalized Least Squares** (**FGLS**), et se prête mieux aux échantillons de très grande taille.
# ```

# In[3]:


ols_resid = sm.OLS(data.endog, data.exog).fit().resid


# Selon la documentation du jeu de données **Longley**, les termes d'erreur suivent un processus stochastiques de type $\mathrm{AR}(1)$, avec une tendance : $\varepsilon_i = \beta_0 + \rho\varepsilon_{i-1} + \eta_i$ où $\eta \sim \mathcal 
# N(\mathbf 0,\Sigma^2)$.
# $\rho$ est par conséquent l'autocorrélation des résidus, et peut être facilement estimée en estimant les paramètres de la régression des résidus par les mêmes résidus retardés ($\varepsilon_i\sim\varepsilon_{i-1}$).

# In[4]:


resid_fit = sm.OLS(np.asarray(ols_resid)[1:], sm.add_constant(np.asarray(ols_resid)[:-1])).fit()
rho = resid_fit.params[1]
print(rho)


# Puisque les termes d'un processus $\mathrm{AR}(1)$ ont une forte corrélation avec les termes voisins, la matrice de variance-covariance pondérée $\Sigma$ peut être déterminée comme suit :
# 
# $${\begin{bmatrix}1&\rho&\cdots &\rho^{n-1}\\\rho&1&\cdots &\rho^{n-2}\\\vdots &\vdots &\ddots &\vdots \\\rho^{n-1}&\rho^{n-2}&\cdots &1\end{bmatrix}}$$

# In[5]:


sigma = rho**np.array([[abs(j-i) for j in range(n) ]for i in range(n)])
gls_model = sm.GLS(data.endog, data.exog, sigma=sigma)
gls_results = gls_model.fit()
print(gls_results.summary())


# La qualité d'ajustement du modèle linéaire par le critère des moindres carrés généralisés, *i.e.* $R^2=99\%$, confirme l'origine des données du jeu **Longley**. 
