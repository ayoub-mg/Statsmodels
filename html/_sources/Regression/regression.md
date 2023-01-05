# Régression Linéaire

La régression linéaire est une approche consistant à modéliser la relation entre une variable de réponse et une ou plusieurs variables explicatives (également appelées respectivement variables dépendantes et indépendantes). Le cas d'une seule variable explicative est appelé **régression linéaire simple**; pour plusieurs, le modèle correspond à une **régression linéaire multiple**.


Étant donné un échantillon $\{y_{i},\,x_{i1},\ldots ,x_{ip}\}_{i=1}^{n}$ de $n$ unités statistiques, un modèle de régression linéaire suppose que la relation entre la variable de réponse $y$ et le vecteur de $p$ variables régresseurs $x$ est linéaire. Cette relation est modélisée à travers un terme d'erreur ou de bruit $\varepsilon$ représentée par une variable aléatoire non observée.

Le modèle s'écrit ainsi :

$${\displaystyle y_{i}=\beta _{0}+\beta _{1}x_{i1}+\cdots +\beta _{p}x_{ip}+\varepsilon _{i}=\mathbf {x} _{i}^{T}{\boldsymbol {\beta }}+\varepsilon _{i},\qquad i=1,\ldots ,n}$$

où $^T$ dénote la transposée, dans le sens où $\mathbf {x} _{i}^{T}{\boldsymbol {\beta }}$ correspond au produit scalaire des vecteurs $\mathbf x_i$ et $\boldsymbol{\beta}$.

Ce système d'équations est souvent écrit sous forme matricielle ${\displaystyle \mathbf {y} =\mathbf {X} {\boldsymbol {\beta }}+{\boldsymbol {\varepsilon }}}$, où

$${\displaystyle \mathbf {y} ={\begin{bmatrix}y_{1}\\y_{2}\\\vdots \\y_{n}\end{bmatrix}},\quad} {\displaystyle \mathbf {X} ={\begin{bmatrix}\mathbf {x} _{1}^{ {T}}\\\mathbf {x} _{2}^{ {T}}\\\vdots \\\mathbf {x} _{n}^{ {T}}\end{bmatrix}}={\begin{bmatrix}1&x_{11}&\cdots &x_{1p}\\1&x_{21}&\cdots &x_{2p}\\\vdots &\vdots &\ddots &\vdots \\1&x_{n1}&\cdots &x_{np}\end{bmatrix}}}$$

et

$${\displaystyle {\boldsymbol {\beta }}={\begin{bmatrix}\beta _{0}\\\beta _{1}\\\beta _{2}\\\vdots \\\beta _{p}\end{bmatrix}},\quad {\boldsymbol {\varepsilon }}={\begin{bmatrix}\varepsilon _{1}\\\varepsilon _{2}\\\vdots \\\varepsilon _{n}\end{bmatrix}}\sim\mathcal N(\mathbf 0,\Sigma).}$$


Selon les propriétés de la matrice de variance-covariance $\Sigma$, trois classes sont disponibles pour l'estimation des paramètres :
- `OLS` : [méthode des moindres carrés ordinaires](#moindres-carrés-ordinaires-ols) pour des erreurs *i.i.d.* $\Sigma=\sigma^2\mathbf I$.
- `WLS` : [méthode des moindres carrés pondérés](#moindres-carrés-pondérés-wls) pour des résidus hétéroscédastiques $\Sigma=\mathrm{diag}(\sigma_1^2,\sigma_2^2,...,\sigma_n^2)$.
- `GLS` : [méthode des moindres carrés généralisés](#moindres-carrés-généralisés-gls) pour une matrice $\Sigma$ arbitraire.




## Moindres Carrés Ordinaires (OLS)

Les coefficients $\boldsymbol\beta$ sont déterminés selon le sens des problèmes de minimisation quadratique. Il s'agit de minimiser la somme des carrés des écarts, entre chaque point du nuage de régression et son projeté, parallèlement à l'axe des ordonnées, sur la droite de régression :

$${\hat {\boldsymbol {\beta }}}={\underset {\boldsymbol {\beta }}{\operatorname {arg\,min} }}\,S({\boldsymbol {\beta }})$$

où $S(\boldsymbol{\beta})$ est la somme des carrés des erreurs :

$${\displaystyle S({\boldsymbol {\beta }})=\sum _{i=1}^{n}{\biggl (}y_{i}-\sum _{j=1}^{p}X_{ij}\beta _{j}{\biggr )}^{2}={\bigl \|}\mathbf {y} -\mathbf {X} {\boldsymbol {\beta }}{\bigr \|}^{2}}$$


Ce problème de minimisation a une solution unique, pour peu que les $p$ colonnes de la matrice ${\displaystyle \mathbf {X} }$ soient linéairement indépendantes, donnée par la résolution des équations dites normales :

$${\displaystyle (\mathbf {X} ^{\mathsf {T}}\mathbf {X} ){\hat {\boldsymbol {\beta }}}=\mathbf {X} ^{{T}}\mathbf {y}}$$ 
ou encore 

$${\displaystyle {\hat {\boldsymbol {\beta }}}=\left(\mathbf {X} ^{{T}}\mathbf {X} \right)^{-1}\mathbf {X} ^{{T}}\mathbf {y}}$$

```{admonition} Propriété
:class: tip
**Cette estimation possède l'erreur quadratique minimale dans la famille des estimateurs linéaires sans biais des paramètres**.
```
## Moindres Carrés Pondérés (WLS)

Un cas particulier de moindres carrés généralisés est les moindres carrés pondérés; utile lorsque tous les éléments non diagonaux de $\Sigma$ la matrice de covariance des résidus sont nulles, et les variances des observations (le long de la diagonale de la matrice de covariance) ne sont pas égales (hétéroscédasticité). 

La fonction de coût à minimiser s'écrit alors :

$${S(\boldsymbol\beta)=\sum_{i=1}^nW_{ii}r_i(\boldsymbol{\beta})^2}$$

où $r_i(\boldsymbol{\beta})$ sont les résidus du modèle ajusté à l'échantillon ${r_i(\boldsymbol{\beta})=y_i-f(x_i,\boldsymbol\beta)}$ et $W_{ii}=\frac1{\sigma_i^2}$ est le poids.

Dès lors que les erreurs d'observation ne sont pas corrélées et que la matrice de poids, $W=\Sigma^{-1}$, est diagonale, les équations d'annulation du gradient $S(\boldsymbol\beta)$ peuvent être écrites comme suit :

$${\displaystyle \mathbf {\left(X^{\textsf {T}}WX\right){\hat {\boldsymbol {\beta }}}=X^{\textsf {T}}Wy}}$$

L'estimateur résultant $\hat{\boldsymbol\beta}$ est **BLUE** (Best Linear Unbiased Estimator), c'est-à-dire à variance minimale dans la classe des estimateurs sans biais. 


## Moindres Carrés Généralisés (GLS)


Les moindres carrés généralisés sont une technique d'estimation des paramètres d'un modèle de régression linéaire lorsqu'il existe un certain degré de corrélation entre ses résidus.

Les paramètres $\boldsymbol\beta$ sont estimés en minimisant la distance carrée de Mahalanobis du vecteur résiduel, *i.e.* :

$${\displaystyle \mathbf {\hat {\beta }} ={\underset {b}{\operatorname {argmin} }}\,(\mathbf {y} -\mathbf {X} \mathbf {b} )^{ {T}}\mathbf {\Sigma } ^{-1}(\mathbf {y} -\mathbf {X} \mathbf {b} )={\underset {b}{\operatorname {argmin} }}\,\mathbf {y} ^{{T}}\,\mathbf {\Sigma } ^{-1}\mathbf {y} +(\mathbf {X} \mathbf {b} )^{{T}}\mathbf {\Sigma } ^{-1}\mathbf {X} \mathbf {b} -\mathbf {y} ^{{T}}\mathbf {\Sigma } ^{-1}\mathbf {X} \mathbf {b} -(\mathbf {X} \mathbf {b} )^{{T}}\mathbf {\Sigma } ^{-1}\mathbf {y}}$$

C'est une forme quadratique en $\mathbf b$. En prenant le gradient de cette quantité, et l'annuler, on obtient :

$${\displaystyle \mathbf {\hat {\beta }} =\left(\mathbf {X} ^{{T}}\mathbf {\Sigma } ^{-1}\mathbf {X} \right)^{-1}\mathbf {X} ^{{T}}\mathbf {\Sigma } ^{-1}\mathbf {y} }$$

```{admonition} Propriétés
:class: tip
**Les estimateurs GLS sont sans biais, convergents, efficaces et asymptotiquement normaux. Le théorème de Gauss-Markov tient encore, ils constituent les estimateurs non biaisés à erreurs quadratiques moyennes les plus faibles**.
```