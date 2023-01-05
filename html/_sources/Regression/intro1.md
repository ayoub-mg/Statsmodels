# Introduction

Les modèles linéaires sont une famille de modèles statistiques dans lesquels on cherche à exprimer une variable aléatoire ${\mathbf {Y}}$ à expliquer en fonction de variables explicatives ${\mathbf X}$ sous forme d'un opérateur linéaire. Le modèle linéaire général est ainsi une généralisation de la régression linéaire multiple au cas de plusieurs variables expliquées.


Le modèle linéaire général est donné selon la formule :

$$ {\displaystyle \mathbf {Y} =\mathbf {X} \mathbf {B} +\mathbf {U} } $$


où $\mathbf Y$ est une matrice d'observations multivariées, $\mathbf X$ est une matrice de variables explicatives indépendantes ($\mathrm{rg}(\mathbf X)=p$), $\mathbf B$ est une matrice de paramètres inconnus à estimer et $\mathbf U$ est une matrice contenant des erreurs ou du bruit ($\mathbb E[\mathbf U]=\mathbf 0$).

Selon la forme de la matrice $\mathbf X$, on est dans le cas de la régression linéaire ($\mathbf X$ est alors composée de la variable constante $\mathbf 1$ et des $p$ variables explicatives) ou dans le cas du modèle factoriel ($\mathbf X$ est composée des variables indicatrices associées aux niveaux du (ou des) facteur(s)).

On suppose généralement que les erreurs ne sont pas corrélées entre les mesures et qu'elles suivent une loi normale multivariée $\mathcal N(\mathbf 0,\sigma^2\mathbf I)$. Si ce n'est pas le cas, des modèles linéaires généralisés peuvent être utilisés pour atténuer les hypothèses sur  $\mathbf Y$ et  $\mathbf U$ en incorporant d'autres familles de distributions au premier rang desquelles les lois exponentielles.

```{admonition} Intérêt des Modèles Linéaires
:class: tip
Les modèles linéaires présentent l'avantage d'être versatiles et applicables à de nombreux domaines de modélisation. Ce faisant, les estimations deviennent plus commodes à calculer, et possèdent des propriétés très utiles en terme de biais et de convergence. 
```