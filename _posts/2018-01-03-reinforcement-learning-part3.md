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

# A venir : Partie 2 : environnement qui change entre chaque partie + réseaux de neurones

Si vous avez aimé cet article, n'hésitez pas à m'envoyer un <a href="mailto:contact@cdancette.fr">mail</a>.

## Resources (en anglais)

- https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
- http://outlace.com/rlpart3.html
- https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html
- https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149
