#!/usr/bin/env python
# coding: utf-8

# # Analyse de la Variance (ANOVA)

# En statistique, l'**analyse de la variance** (terme souvent abrégé par le terme anglais ANOVA : *Analysis of Variance*) est un ensemble de modèles statistiques utilisés pour vérifier si les moyennes des groupes proviennent d'une même population. Les groupes correspondent aux modalités d'une variable qualitative et les moyennes sont calculés à partir d'une variable continue. Ce test s'applique lorsque l'on mesure une ou plusieurs variables explicatives catégorielle (appelées alors facteurs de variabilité, leurs différentes modalités étant parfois appelées « niveaux ») qui ont de l'influence sur la loi d'une variable continue à expliquer.
# ```{admonition} Autrement dit
# :class: note
# L'analyse de la variance permet d'étudier le comportement d'une variable quantitative à expliquer en fonction d'une ou de plusieurs variables qualitatives, aussi appelées nominales catégorielles.
# ```

# La méthode à utiliser pour réaliser des analyses de la variance du paquet `statsmodels` est `anova_lm`, en précisant les modèles statistiques à comparer en paramètres.

# Pour examiner l'utilisation de cette méthode, on 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from urllib.request import urlopen
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import pandas as pd
pd.set_option("display.width", 100)
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm


# Le jeu de données utilisé constitue un ensemble de $46$ observations des salaires d'un groupe d'employés selon leur expérience, laquelle est une variable numérique variant dans $[0,20]$.

# In[2]:


url = "http://stats191.stanford.edu/data/salary.table"
fh = urlopen(url)
salary_table = pd.read_table(fh)
salary_table.to_csv("salary.table")

E = salary_table.E
M = salary_table.M
X = salary_table.X
S = salary_table.S


# In[3]:


plt.figure(figsize=(20, 12))
symbols = ["D", "^"]
colors = ["r", "g", "blue"]
factor_groups = salary_table.groupby(["E", "M"])
for values, group in factor_groups:
    i, j = values
    plt.scatter(group["X"], group["S"], marker=symbols[j], color=colors[i - 1], s=144)
plt.xlabel("Expérience")
plt.ylabel("Salaire")


# On ajuste un modèle linéaire des moindres carrés ordinaires sur les données en question, moyennant la spécification par les formules (type R).

# In[4]:


formula = "S ~ C(E) + C(M) + X"
lm = ols(formula, salary_table).fit()
print(lm.summary())


# On récupère l'influence sur la régression de chaque observation au travers de la méthode `get_influence()` :

# In[5]:


infl = lm.get_influence()
print(infl.summary_table())


# À présent, on trace le graphique des résidus pour chaque groupe séparément :

# In[6]:


resid = lm.resid
plt.figure(figsize=(20, 12))
for values, group in factor_groups:
    i, j = values
    group_num = i * 2 + j - 1 
    x = [group_num] * len(group)
    plt.scatter(x,resid[group.index],marker=symbols[j],color=colors[i - 1],s=144,edgecolors="black")
plt.xlabel("Groupes")
plt.ylabel("Résidus")


# Ensuite, on ajuste un modèle pour le comparer au premier via l'analyse de la variance.

# In[7]:


interX_lm = ols("S ~ C(E) * X + C(M)", salary_table).fit()
print(interX_lm.summary())


# À présent, on compare les deux modèles élaborés avec la fonction `anova_lm`

# In[8]:


from statsmodels.stats.api import anova_lm

table1 = anova_lm(lm, interX_lm)
print(table1)

interM_lm = ols("S ~ X + C(E)*C(M)", data=salary_table).fit()

table2 = anova_lm(lm, interM_lm)
print(table2)


# ```{admonition} Conclusion
# :class: tip
# Dès lors que la première valeur du seuil critique ($p$-value) est supérieure à $5\%$, alors le deuxième modèle `interX_lm` est significativement meilleur. Le troisième modèle `interM_lm` ne contribue pas significativement mieux par rapport au modèle de base, dès lors que sa $p$-valeur est inférieure à $5\%$.
# ```
