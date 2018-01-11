---
layout: post
title: "Reinforcement learning en python sur un jeu simple grâce au Q-learning, Partie 3"
keywords: reinforcement learning, jeu, machine learning, réseau, neurone, apprentissage, renforcement, tutoriel, neural network, deep learning
excerpt_separator: "<!-- more -->"
---

![Le jeu](/assets/qlearning2/capture.gif) 

Cet article est la suite de [{% post_url 2017-08-20-reinforcement-learning-part2 %}]({% post_url 2017-08-20-reinforcement-learning-part2 %}).

Dans cette troisième partie, nous allons étudier une variante plus complexe du jeu précédent : le terrain est modifié à chaque partie. Nous n'allons pas pouvoir stocker et visiter tous les états pour entrainer l'agent. Le réseau de neurone apprendra alors a généraliser, pour obtenir une fonction de valeur Q convenable.

<!-- more -->

Tous les codes présentés ici peuvent être trouvés  sur [github](https://github.com/cdancette/machine-learning-projects/blob/master/q-learning/q-learning-part3.ipynb).


# Le jeu

Le jeu est identique aux versions précédentes. Une seule différence : nous allons regénérer un terrain à chaque nouvelle partie. Ainsi
Le seul changement sera la manière d'encoder l'état du jeu.

Ainsi, la fonction `get_state` sera réécrite de la manière suivante : 

{% highlight python linenos %}
    def _get_state(self):
        x, y = self.position
        if self.alea:
            return np.reshape([self._get_grille(x, y) for (x, y) in
                    [self.position, self.end, self.hole, self.block]], (1, 64))
        return flatten(self._get_grille(x, y))
{% endhighlight %}

Ici, au lieu de renvoyer un tableau de taille 16 contenant juste la position de notre agent marqué par un 1, nous devons également indiquer la position des éléments sur le terrain.

Les éléments à indiquer sont : notre position `self.position`, la position de l'arrivée `self.end`, la position du trou `self.hole`, et la position du mur `self.block`.

Au lieu d'un tableau de taille 16 contenant des 0, avec un 1 pour marquer la position de l'agent, nous avons maintenant 4 tableaux similaires. L'état du jeu sera la concaténation de ces 4 tableaux (obtenu avec `np.reshape`).
L'état est donc un tableau de taille 4x16 = 64 cases, et il contiendra quatre 1, et les autres éléments seront des 0.

Ici pour encoder les états, nous avons utilisé ce qui est appelé un [One Hot encoder](https://en.wikipedia.org/wiki/One-hot) : Pour chaque élément, il y a 16 possibilités, et on encode cela en utilisant un tableau de 16 cases, en mettant un 1 pour représenter la position de l'élément. Ce n'est pas une représentation très compacte : les 16 cases possibles peuvent être encodées sur 4 bits, il nous suffirait donc de 4 cases dans notre tableau pour encoder la position de chaque élément. Mais le réseau devrait alors apprendre à décoder cet encodage en plus, et aurait donc surement besoin d'un entrainement plus long, ou bien de plus de couches. L'encodage que nous avons choisi ici est extrèmement simple (mais il n'est pas le plus compact possible, toutes les entrées possibles ne sont pas du tout utilisés).


Si vous avez aimé cet article, n'hésitez pas à m'envoyer un <a href="mailto:contact@cdancette.fr">mail</a>.

## Resources (en anglais)

- https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
- http://outlace.com/rlpart3.html
- https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html
- https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149
