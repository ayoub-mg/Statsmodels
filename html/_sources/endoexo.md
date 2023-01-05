
# `endog` ou `exog` ?

La librairie Statsmodels utilise par convention les termes `endog` et `exog` pour désigner les variables observées intervenant dans un problème d'estimation.


Une variable est dite **endogène** lorsque sa valeur est déterminée par les caractéristiques internes du modèle; elle est dite **exogène** lorsque sa valeur est déterminée par des conditions extérieures au modèle. Les variables endogènes sont aussi appelées variables expliquées ou de sortie, tandis que les variables exogènes correspondent aux variables explicatives ou encore d'entrée du modèle.


L'endogénéité se réfère généralement à une situation dans laquelle une des variables explicatives est corrélée avec le terme d'erreur. La distinction entre les variables endogènes et exogènes vient des modèles d'équations simultanées, où on sépare les variables entre celles qui sont déterminées par le modèle et celles qui sont prédéterminées.


En guise d'exemple, si la variable explicative $x$ est corrélée avec le terme d'erreur dans un modèle de régression, l'estimation du coefficient de régression dans une régression par le critère des moindres carrés (OLS) sera biaisée; toutefois, si la corrélation ne dépend que de la mesure $i$, l'estimation du coefficient reste convergente. 

Auquel cas, supposons que le modèle devant être estimé est :

$$
\begin{aligned}
{\displaystyle y_{i}=\alpha +\beta x_{i}+\gamma z_{i}+\varepsilon _{i}}
\end{aligned}
$$

Si $\mathbb Cov(x,\varepsilon)\neq 0$, alors le régresseur $x$ n'est plus exogène, et l'hypothèse standard d'orthogonalité dans le théorème de Gauss-Markov ne tient plus. Par conséquent, $y$ et $x$ sont toutes les deux endogènes à notre modèle.

```{admonition} Note
:class: tip
Cette corrélation peut survenir quand il y a des erreurs de mesure sur les variables expliquées, ou alors si une variable omise agit à la fois sur la variable expliquée et sur une (ou des) variable(s) explicative(s). 
```
