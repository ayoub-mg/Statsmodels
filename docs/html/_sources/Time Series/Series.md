# Séries Temporelles

Une **série temporelle** ou **série chronologique** est une suite d’observations $(x_1,…,x_n)$ d’un phénomène physique faites au cours du temps : consommation d’électricité, cours du pétrole, population marocaine, rythme cardiaque, relevé d’un sismographe, trafic Internet, ventes de téléphones mobiles, hauteurs des crues du nil, température des océans, concentration en dioxyde de carbone de l’atmosphère, taux de glucose dans le sang, côte de popularité du président, etc. Il s’agit d’une suite finie de valeurs réelles, indicées par un temps continu ou discret régulier, typiquement d’un signal échantillonné à une fréquence fixe.


Une idée serait de tracer le graphe de la série temporelle $t→x_t$, puis de déterminer une famille de fonctions qui ont la même allure, puis enfin de déterminer la meilleure fonction en minimisant un critère d’ajustement comme les moindres carrés par exemple, avec pénalisation du critère optimisé par la complexité de la fonction utilisée. L’incorporation du bruit dans cette approche conduit aux modèles additifs stochastiques du type :

$$x_t = d_t + z_t$$

où $t→ d_t$ est une fonction déterministe et où $z_t$ est un bruit aléatoire.

La partie déterministe d'une série temporelle peut être décomposée en :
+ **Tendance** : une fonction $t\rightarrow m_t$ qui varie lentement, correspondant à une évolution à long et moyen terme de la série.
+ **Saisonnalité** : une fonction $t\rightarrow s_t$ correspondant à un phénoméne périodique à l’intérieur d’une année, et qui se reproduit de façon plus ou moins permanente d’une année sur l’autre. Ces variations sont dues au rythme des saisons.

Le bruit est supposé être **stationnaire**, c’est-à-dire que ses caractéristiques statistiques comme son espérance et sa covariance ne varient pas au cours du temps. Dans cette approche stochastique, $(x_t)_{t∈\mathcal T}$ est modélisée par une trajectoire d’un processus stochastique $(X_t)_{t∈\mathcal T}$, *i.e.* une famille de variables aléatoires.

```{admonition} Remarque
:class: note
La stationnarité de la partie aléatoire est utile pour effectuer une prévision par translation. Si par exemple on observe $Z_1, …, Z_t$ alors on peut estimer la matrice de covariance de ce vecteur aléatoire, qui est aussi par stationnarité une estimation de la structure de covariance de $Z_{1+1}, …, Z_{t+1}$, ce qui permet de prédire le futur $Z_{t+1}$ en utilisant une projection par moindres carrés sur les observations $Z_2, …, Z_t$.
```

## Processus $\rm{ARMA}$

Les processus $\mathrm{ARMA}(p, q)$ (*Auto Regressive Moving Average*) forment une classe de processus stationnaires paramétrés incluant à la fois un mécanisme d’autoregression linéaire ($\rm AR$) d’ordre $p$, et de moyenne mobile ou ajustée ($\rm MA$) d’ordre $q$.

### Bruit Blanc $\rm BB$
Un processus stationnaire $(Z_t)_{t∈\mathbb Z}$ est un bruit blanc si :

$$\forall t\in\mathbb Z\quad:\quad \mathbb E(Z_t)= m,\quad \mathbb V(Z_t)=σ^2\\\forall(t,t')\in\mathbb Z^2,t\neq t'\quad:\quad \mathbb Cov(Z_t,Z_t')=0 
$$
C'est donc une suite de variables aléatoires non-corrélées, de moyennes et variances constantes. Si l’espérance est nulle, le bruit blanc est dit centré. Si les variables sont gaussiennes, le bruit blanc est dit gaussien.

### Processus $\rm AR$
Un processus $(X_t)_{t∈\mathbb Z}$ est $\mathrm{AR}(p)$ pour un entier $p ≥ 0$ lorsqu’il est stationnaire, et solution de l’équation de récurrence :

$$X_t = Z_t +∑_{k=1}^pφ_kX_{t−k}
$$

où $(Z_t)_{t∈\mathbb Z}$ est un bruit blanc $\mathrm{BB}(0, σ^2)$ et où $φ ∈ \R^p$ est un vecteur fixe. On dit que $p$ est l’ordre
du processus et $(φ, σ^2)$ ses paramètres.

### Processus $\rm MA$
Un processus $(X_t)_{t∈\mathbb Z}$ est $\mathrm{MA}(q)$ pour un entier $q ≥ 0$ lorsqu’il est stationnaire, et solution de l’équation de récurrence :

$$X_t = Z_t +∑_{k=1}^qθ_kZ_{t−k}
$$

où $(Z_t)_{t∈\mathbb Z}$ est un $\mathrm{BB}(0, σ^2)$ et où $θ ∈ \mathbb R^q$ est un vecteur fixe. On dit que $q$ est l’ordre du processus et que $(θ, σ^2)$ sont ses paramètres.

### Processus $\rm ARMA$
Soient $p,q\in\mathbb N$, $φ\in\mathbb R^p$ et $θ\in\mathbb R^q$ des coefficients fixés, et $(Z_t)_{t∈\mathbb Z}\sim\mathrm{BB}(0, σ^2)$. On dit que $(X_t)_{t∈\mathbb Z}$ est un processus $\mathrm{ARMA}(p, q)$, ou $\mathrm{ARMA} d’ordre $(p,q)$, lorsqu’il est stationnaire et vérifie l’équation de récurrence linéaire suivante a :

$$∀t∈\mathbb Z,\quad X_t =∑_{k=1}^pφ_kX_{t−k}+ Z_t +∑_{k=1}^qθ_kZ_{t−k}.
$$

De plus :
- si $θ ≡ 0$ ou $q = 0$, alors on dit qu’il s’agit d’un processus $\mathrm{AR}(p)$;
- si $φ ≡ 0$ ou $p = 0$, alors on dit qu’il s’agit d’un processus $\mathrm{MA}(q)$.