# Modèles Linéaires Généralisés


Dans un **modèle linéaire générale** :


$$\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}, \quad \mathbb{E} \left( \boldsymbol{\varepsilon} | \mathbf{X}\right) = \boldsymbol{0},\quad \mathbb{V}{\rm ar}\left( \boldsymbol{\varepsilon} | \mathbf{X} \right) = \mathbb{E} \left( \boldsymbol{\varepsilon} \boldsymbol{\varepsilon}^T \right)= \mathbf{\Sigma} 
$$

La variable de réponse $\mathbf y$ est décrite via une relation linéaire des variables explicatives + un terme de bruit.

Bien que cette modélisation ait servi pour décrire une panoplie de relations, elle laisse à désirer eu égard à d'autres types de spécifications :

- Lorsque les variations de $\mathbf y$ sont restreintes à des valeurs entières discrètes;
- Si la variance de $\mathbf y$ dépend de la moyenne.

Le **modèle linéaire généralisé** (GLM) est une généralisation souple des modèles linéaires généraux. Il consiste en :


1. Un prédicteur linéaire : $\eta=\mathbf X\boldsymbol{\beta}$

2. Une fonction de lien $g$, reliant l'espérance de la réponse au prédicteur :  
   
   $${\displaystyle \mathbb {E} (\mathbf y|\mathbf {X} )={\boldsymbol {\mu }}=g^{-1}(\eta)=g^{-1}(\mathbf {X} {\boldsymbol {\beta }})}$$

3. Une fonction variance $\mathrm V$, qui dépend de la loi de $\mathbf y$ (supposée de la famille exponentielle) :
   
   $${\displaystyle \mathbb Var (\mathbf y|\mathbf {X} )=\operatorname {V} ({\boldsymbol {\mu }})=\operatorname {V} (g^{-1}(\mathbf {X} {\boldsymbol {\beta }})).}$$


Chaque $y_i$ est supposé être généré à partir d'une distribution particulière de la famille exponentielle, où la fonction de densité de probabilité s'écrit comme suit :

$$f(y_i) = \exp\left( \dfrac{y_i \theta_i - b(\theta_i)}{a_i(\phi)} + c(y_i, \phi) \right)
$$

où :
- $\theta_i$ est le paramètre de position;
- $\phi$ est le paramètre d'échelle;
- $a_i(⋅)$, $b(⋅)$ et $c(⋅,⋅)$ sont des fonctions connues.

```{admonition} Remarque
:class: note
Un modèle linéaire où $y_i$ est linéairement dépendent de $X_i$ (*e.g.* $y_i=X_iβ+\varepsilon_i$) est différent d'un modèle linéaire généralisé où $μ_i$ est liée linéairement à $X_i$ (*e.g.* $μ_i=X_iβ$).
```


Nous présentons ci-dessous les fonctions de liaison et les modèles qui en résultent pour certains cas :


| Loi de $y$| Nom | Fonction de lien | Modèle | Prédiction $\hatμ$ | Valeurs de $y$
|-----------|----------------------------|------------------|--------|--------------------|---------------
| Normale   |         Identité           |       $g(μ)=μ$   |  $μ=\mathbf X\boldsymbol{β}$ | $\hatμ=\mathbf X\hat{\boldsymbol{β}}$  |    $y\in]-∞,+∞[$                |
| Exponentielle  |          Inverse négatif          |       $g(μ)=-μ^{-1}$       |   $-μ^{-1}=\mathbf X\boldsymbol{β}$  |        $\hatμ=-\left(\mathbf X\hat{\boldsymbol{β}}\right)^{-1}$     | $y\in[0,∞[$
|   Poisson  |     Log       |       $g(μ)=\log(μ)$     |   $\log(\mu)=\mathbf X\boldsymbol{\beta}$   | $\hat\mu=\exp(\mathbf X\hat{\boldsymbol{\beta}})$      |  $y\in\{0,1,2,...\}$
|       Binomiale   |        Logit            |     $g(\mu) = \log\left(\dfrac{\mu}{1-\mu}\right)$   | $\log\left(\dfrac{\mu}{1-\mu}\right) = \mathbf{X} \boldsymbol{\beta}$     |         $\widehat{\mu} = \dfrac{\exp\left(\mathbf{X} \widehat{\boldsymbol{\beta}}\right)}{1 + \exp\left(\mathbf{X} \widehat{\boldsymbol{\beta}}\right)}$      |$y\in\{0,1\}$




```{admonition} Hypothèses du GLM
:class: tip
- La relation entre les variables dépendantes et indépendantes peut être non linéaire;
- La variable de réponse peut avoir une distribution non-normale;
- La méthode du maximum de vraisemblance peut être utilisée pour estimer les paramètres;
- Les erreurs sont indépendantes mais peuvent avoir une distribution non-normale.
```