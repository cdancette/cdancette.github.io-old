---
layout: post
title: "Fonctionnement d'un réseau de neurone artificiel"
keywords: machine learning, réseau, neurone, apprentissage, back, tutoriel, artificial neural network
---

<script type="text/javascript"
    src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
Une explication simple sur le principe de fonctionement d'un réseau de neurone, outil de machine learning qui permet d'approximer des fonctions.

Le réseau de neurone est une méthode utilisée en machine learning, pour la regression ou la classification de données.
La régression est l'approximation d'une fonction réelle (par exemple les prix d'un logement, que l'on veut approximer en fonction de sa localisation géographique). La classification consiste à attribuer des labels à des données : par exemple, en fonction de la couleur et de la forme d'un champignon, déterminer s'il est toxique ou pas (2 labels ici : Toxique, non toxique).

## Le neurone.

Un neurone est tout simplement une fonction, dotée de paramètres.

Cette fonction est $$ output = activation(w_1 x1 + w_2  x2 + ... + w_n x_n) $$.

Les entrées (input) seront fournies par les données, ou bien par les sorties des neurones précédents.
La sortie (output), est un nombre réel.

![Schéma d'un neurone]({{ site.baseurl }}assets/neuron.png)

## Le réseau

Un réseau est constitué habituellement de couches successives de neurones placés en parallèle.

Chaque neurone prend son entrée depuis la sortie des neurones de la couche suivante.
On appelle "fully connected layer" lorsque pour chaque neurone, sa sortie est reliée à tous les neurones de la couche suivante. 

![Réseau de neurone]({{ site.baseurl }}assets/Neural_network.svg)

#### Les fonctions d'activation

Si on utilise la fonction identité ($$ f(x) = x $$), alors le neurone représentera une simple combinaison linéaire des input, et des poids.

En général, on utilise des fonctions non linéaires. Deux fonctions parmi les plus courantes sont la sigmoide, et reLu :

La sigmoïde 

$$ sig(x) = \frac{1}{1 + e^{-x})} $$


![sigmode]({{ site.baseurl }}assets/sigmoid.svg){: .center-image }

La fonction ReLU

$$ relu(x) = max(0, x) $$


En vert sur l'image

![sigmode]({{ site.baseurl }}assets/relu.png){: .center-image }


Ces différentes fonctions peuvent avoir une influence sur la performance du réseau de neurone à généraliser à partir des données qui lui sont fournies.

## FeedForward

Un réseau de neurone est utilisé en regression ou classification. Pour l'utiliser, on utilise l'opération de "forward" : il s'agit de calculer la sortie du réseau de neurone en fonction de l'entrée, en appliquant couche par couche les fonctions des neurones.

Pour obtenir une bonne valeur de sortie, il est donc nécessaire de configurer les paramtres des neurones (appelés "poids") pour qu'à chaque sample (une instance de nos données), une valeur convenable lui soit associée.

## Backpropagation

L'entrainement d'un réseau de neurone est effectué habituellement par l'algorithme de backpropagation, basé sur la descente du gradient.

Pour chaque sample (input, value), on calcule le loss $$L$$ (par exemple le carré de la différence entre la sortie du réseau de neurone).

Pour chaque poids $$w$$, on calcule le gradient de cette fonction $$ L $$ en fonction des poids du neurone.
On commence par la dernière couche, puis les couches précedentes grâce à la relation de chaine.
On obtient ainsi pour chaque poids $$w$$ la valeur du gradient au point donnée par le sample.

$$ \delta_L = \frac{\partial L}{\partial w}(value) $$

Puis on applique l'algorithme de la descente du gradient pour modifier légerement les poids du réseau de neurone

$$ w  = w + \lambda \delta_L $$

$$\lambda $$ est appelé le learning rate.

Pour avoir un détail des calculs, vous pouvez consulter le site suivant : [http://neuralnetworksanddeeplearning.com/chap2.html](http://neuralnetworksanddeeplearning.com/chap2.html)
