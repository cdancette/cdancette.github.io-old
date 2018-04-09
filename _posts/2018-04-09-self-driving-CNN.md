---
layout: post
title: "[Tutoriel] Conduite autonome par imitation grâce à un réseau de convolution (CNN)"
keywords: machine learning, CNN, convolution, autonomous driving, conduite autonome, neural network, réseau de neurone, vision par ordinateur, image recognition
excerpt_separator: "<!-- more -->"
---

![car-running](/assets/self-driving-1/car-running.gif)

<!-- 
TODO

- expliquer le réseau de convolution 
- quel modèle choisir ? Expliquer les loss ?
 -->
Nous allons appliquer un algorithme d'apprentissage supervisé (réseaux de convolution), pour commander la direction d'une voiture dans une simulation 2D: [https://gym.openai.com/envs/CarRacing-v0/](https://gym.openai.com/envs/CarRacing-v0/). Nous allons présenter le fonctionnement d'un réseau de convolution, comment créer le dataset et l'utiliser pour l'entrainement de notre réseau, puis comment utiliser gym pour récupérer la sortie de notre réseau de neurone afin de controler la simulation.

<!-- more -->

L'idée générale que nous allons utiliser est celui du classifieur supervisé. Nous allons entrainer un réseau de neurone convolutionel à classifier des images du jeu, selon trois labels : gauche, droite et tout droit. Nous convertirons ensuite ces commandes en instructions pour le simulateur, qui les executera.

Nous allons donc voir toutes les étapes nécessaires : Création du dataset d'entrainement, Apprentissage du réseau de neurone, et utilisation du modèle entrainé pour controller la voiture.

Tout le code de ce tutoriel est disponible ici : [https://github.com/cdancette/supervised-self-driving](https://github.com/cdancette/supervised-self-driving).


## Installation des packages nécessaires

Il nous faut certains packages pour faire fonctionner notre réseau de neurone.

### Installation de gym

Suivre les instructions sur [https://github.com/openai/gym#installation](https://github.com/openai/gym#installation).

Résumé des instructions :  
- cloner le repo gym : `git clone https://github.com/openai/gym.git`
- `cd gym`
-  installation gym, with the box2d environments : `pip install -e '.[box2d]'`

### Installation de pytorch
Pytorch est le framework de deep learning que nous allons utiliser. Il permet de construire des réseaux de neurones très simplement.

Suivre les instructions sur [http://pytorch.org/](http://pytorch.org/).

## L'environnement

Pour ce tutorial, nous allons utiliser la librairie [gym](https://github.com/openai/gym) développée par OpenAI. Elle fournit des environnements (des jeux simples) pour développer des algorithmes d'apprentissage par renforcement.  Vous pouvez voir la liste des environnements ici : [https://gym.openai.com/envs/](https://gym.openai.com/envs/).

L'environnement que nous allons utiliser est CarRacing-v0 ([https://gym.openai.com/envs/CarRacing-v0/](https://gym.openai.com/envs/CarRacing-v0/)).
Il s'agit de conduire une voiture sur un circuit, l'objectif étant d'avancer en restant sur la piste, qui contient de nombreux virages. L'entrée de l'algorithme (l'état fourni par l'environnement) est uniquement l'image affichée par l'environnement : on voit la voiture, et le terrain autour d'elle. 

![jeu](/assets/self-driving-1/car-racing.png)

Il s'agit ainsi de conduire la voiture en analysant cette image. 

Nous allons utiliser cette librairie de manière détournée : Elle est conçue pour faire du reinforcement learning. L'objectif est en principe d'utiliser les *rewards* (récompenses) fournies par l'environnement pour apprendre la stratégie optimale sans action de l'utilisateur. Ici, nous n'utiliserons pas ces récompenses.

De plus, nous allons faire du *end-to-end learning*, ce qui veut dire que le réseau de neurone va nous donner en sortie de manière directe les commandes pour naviguer la voiture. Il ne s'agit pas d'un module de détection de route, qui sera ensuite analysé par un autre programme (la plupart des vrais systèmes de conduite autonome sont faits ainsi). Ici, le réseau de neurone prend en entrée la matrice du terrain, et sort une commande à executer (tourne à gauche, tourne à droite, continue tout droit), sans aucun programme intermédiaire. 

Pour utiliser l'environnement, il faut l'importer comme ceci :

```python
import gym
env = gym.make('CarRacing-v0').env
```

On peut alors accéder à plusieurs fonctions utiles : 

* `env.reset()` permet de redémarrer l'environnement
* `env.step(action)` permet d'effectuer l'action `action`. Cette fonction retourne un tuple `state, reward, done, info` contenant l'état du jeu après l'action, la récompense obetenue, `done` indique si le jeu est terminé, et `info` contient des données de debug. 
* `env.render()` permet d'afficher la fenètre du jeu.

Ici, l'état `state` qui sera renvoyé par `env.step(action)` est l'image affichée sur l'écran (la matrice des pixels). C'est cette donnée que l'on utilisera pour diriger notre voiture. 

On peut créer une simulation controlable avec les boutons de la souris de la manière suivante : 

```python
import gym
import numpy as np
import imageio
import os
from pyglet.window import key

env = gym.make('CarRacing-v0').env
env.reset()

if __name__=='__main__':
    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    env.reset()
    while True:
        s, r, done, info = env.step(a)
        env.render()
        if done:
            env.reset()
    env.close()
```
Voir le fichier complet [manual_drive.py](https://github.com/cdancette/supervised-self-driving/blob/master/manual_drive.py)

La voiture est contrôlable avec les flèches du clavir.

Pour la suite, on veut entrainer un réseau de neurone qui prendra en entrée l'image du jeu, et en sortie, renverra la commande à envoyer (gauche, droite, tout droit). On va se concentrer dans un premier temps sur le contrôle de la **direction**. Le contrôle de la vitesse sera encore à effectuer avec les touches de la souris. 

## Création du dataset

La première étape pour entrainer notre réseau de neurone est de créer un jeu de données.
Il s'agit d'enregistrer un ensemble d'images accompagnées de leur *label*. Nous allons représenter les actions possibles avec des entiers : 

- 0 pour indiquer d'aller à gauche
- 1 pour indiquer d'aller à droite
- 2 pour indiquer d'aller tout droit

Ainsi, on va enregistrer un ensemble de **3000** images dans un dossier, accompagnées d'un fichier `labels.txt` indiquant sur chaque ligne `<chemin de l'image> label`. Nous avons 3 labels, nous sauvegardons donc 1000 images de chaque label pour le training set. Pour le testing set, on va sauvegarder 600. 

On remplace la boucle du code précédent par ce code: 

```python

LEFT = 0
RIGHT = 1
GO = 2
ACTIONS = [LEFT, RIGHT, GO]

# function to convert action array a to the action id
def action_to_id(a):
    if all(a == [-1, 0, 0]): return LEFT
    elif all(a == [1, 0, 0]): return RIGHT
    else:
        return GO

folder = "train_set"

images = os.path.join(folder, "images")
labels = os.path.join(folder, "labels.txt")
os.makedirs(images, exist_ok=True)

a = np.array([0.0, 0.0, 0.0])

samples_each_classes = 1000
# dictionnaire comptant le nombre d'images déja enregistrées pour chaque classe
samples_saved = {0: 0, 1: 0, 2: 0}  
i = 0
file_labels = open(labels, 'w')
while True:
    s, r, done, info = env.step(a)
    action_id = action_to_id(a)
    if samples_saved[action_id] < samples_each_classes:
        samples_saved[action_id] += 1
        samples_each_classes
        imageio.imwrite(os.path.join(folder, 'images', 'img-%s.jpg' % i ), s)
        file_labels.write('%s %s\n' % ('img-%s.jpg' % i, action_id))
        file_labels.flush()
        i += 1
        print(samples_saved)
    env.render()
```
Voir le fichier complet [record_dataset.py](https://github.com/cdancette/supervised-self-driving/blob/master/record_dataset.py)

Il faut créer un *train set*, qui servira a entrainer le réseau, et un *test set*, qui servira a évaluer ses performances pendant l'entrainement, pour savoir à quel moment l'interrompre. En effet, étant donné le nombre assez faible d'images que l'on utilise (3000), il y a un risque d'*overfitting*.[lien], c'est à dire que le réseau perdra en pouvoir de généralisation pour être meilleur dans les cas particuliers du *training set*. C'est une situation que l'on veut éviter, puisqu'on veut utiliser notre modèle par la suite dans des situations qu'il n'a pas vu. La technique d'interruption de l'entrainement avant la convergence est appelé *early stopping*.

On lance donc deux fois le script en modifiant les variable `folder` (dossier d'arrivée du dataset), et `samples_each_classes` (nombre d'exemples que l'on souhaite enregistrer).

## Entrainement du modèle avec pytorch


Pytorch est une librairie python de calcul matriciel et de deep learning. Elle consiste d'une part en un équivalent de numpy, mais utilisable à la fois sur CPU et sur GPU. Et d'autre part, en une librairie qui permet de calculer le gradiant de chaque opération effectuée sur les données, de manière à appliquer l'algorithme de backpropagation (voir le post ({% post_url 2017-10-08-neural-nets %}), à la base de l'entrainement des réseaux de neurones. Pytorch possède également un ensemble de modules à assembler, ce qui permet de créer très simplement des réseaux de neurones.

Dans pytorch, l'objet de base est le `module`. Chaque module est une fonction, ou un assemblage de fonctions pytorch, qui prend en entrée des Tenseurs (matrices contenant des données), et ressort un autre tenseur. L'ensemble des opérations effectuées dans ce module sera enregistré, car le graphe d'opération est nécessaire pour l'algorithme de backpropagation.

### Définition du modèle

On définit notre modèle comme ceci :

```python
class CustomModel(nn.Module):
    """
    from alexnet
    """
    def __init__(self):
        super(CustomModel, self).__init__()
        num_classes = 3
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(576, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, num_classes),
        )

    def forward(self, input):
        input = self.convnet(input)
        input = input.view(input.size(0), -1)
        return self.classifier(input)

```
Voir le fichier complet [model.py](https://github.com/cdancette/supervised-self-driving/blob/master/model.py)

Ce modèle est inspiré de l'architecture d'AlexNet[lien], un des premiers grand succès des réseaux de neurone pour la classification d'images.

On peut voire deux parties ici 

**La fonction __init__** : ici on définit l'architecture du réseau. Notre réseau est composé de deux parties : `self.convnet` et `self.classifier`. La partie `convnet` est la partie convolutionnelle : c'est elle qui se charge de faire l'analyse de l'image, et de reconnaitre les formes (voir le poste {% post_url 2018-01-08-convolutional-neural-net %}). 
Elle est composée de deux couches de convolution (reconnaissance de patterns), suivies d'une non-linéarité (ReLU), et d'une couche de pooling (qui permet de rendre la sortie invariante aux translations).

La seconde partie est le 'classifieur', il prend la sortie du réseau de convolution, et en ressort un vecteur de taille `num_classes = 3` qui représente le score de chaque action à effectuer. 

L'appel `nn.Sequential` permet de créer les couches en succession. L'entrée passera successivement par toutes ces couches, l'entrée d'une couche étant la sortie de la précédente.

[expliquer la couche de convolution, et les chiffres]

**La fonction forward**
C'est cette fonction qui sera appelée par pytorch au moment de l'appel de notre module. 
On remarque le passage d'une entrée 2D à une entrée 1D entre les deux parties convnet et classifier grâce à la fonction `input = input.view(input.size(0), -1)` (la première dimension étant le nombre d'images dans un batch). C'est un raccourci pour `input = input.view(input.size(0), input.size(1) * input.size(2) * input.size(3)`

L'entrée aura en effet 4 dimensions : la première pour le batch, les 2 suivantes pour les dimensions x et y de l'image, et la dernière pour le nombre de *channels* de l'image : Ce sera 3 pour les 3 couleurs à l'entrée du réseau, puis chaque convolution créera de nouveaux *channels* tout en réduisant la taille x et y. Ainsi, au fur et a mesure des couches, la 1ere dimension restera fixe (le nombre d'image dans le batch), mais les deux suivantes vont diminuer, et la 3eme (channels) va augmenter.

## Entrainement

### Préparation des données

Nous allons créer une classe `Dataset`, qui sera utilisée par pytorch pour charger notre jeu de donnée en mémoire grâce à sa classe `DataLoder`. Ces fonctions sont expliquées en détail dans [lien vers le tuto avancé sur data processing]

Tout d'abord, nous allons définir les transformations qui seront utilisées pour pré-traiter les images, afin de les donner en entrée au réseau de neurone. 


```python
from torchvision import transforms

transform_driving_image = transforms.Compose([
    transforms.CenterCrop(72),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

Cette transformation effectue les actions suivantes :

**Rogne l'image**  : `transforms.CenterCrop(72)`

Pour ne garder qu'un carré de taille 72 pixels, centré de la même manière que l'image. En effet, l'image que nous obtenons de l'environnement est comme ceci : 
![écran original](/assets/self-driving-1/screen-92px.png)
Nous pouvons voir que l'écran affiche une barre d'indication sur la vitesse et les commandes de direction et accéleration. Si nous ne la masquons pas, le CNN risque d'apprendre à associer les commandes que nous lui donnons, avec ces indications (c'est en effet le meilleur indicateur pour déduire la commande à effectuer à partir de l'écran). 

Après rognage, l'image obtenue est ci-dessous. Le CNN sera forcé d'analyser la route et la position de la voiture afin 
![écran rogné](/assets/self-driving-1/screen-cropped-72px.png)

On remarque que les images fournies au CNN sont de bien moins bonne qualité que celle affichées par l'environnement lors du jeu. Elles ne font en effet que 96 pixels de coté. Cela va suffir au réseau de neurone pour analyser les formes, et rendra l'entrainement beaucoup plus rapide (car beaucoup moins de neurones seront nécessaires).

**Transforme la matrice en Tensor pytorch** `transforms.ToTensor()`

Le tenseur est l'objet de base dans pytorch pour stocker des données. C'est l'analogue à une matrice numpy, sauf qu'il peut être stocké sur CPU, ou sur GPU. Nous devons transformer notre image en tenseur pytorch avant de la donner en entrée au réseau de neurone.

On pourrait aussi utiliser la fonction `tensor.from_numpy(numpy_array)` pour transformer un array numpy en *Tensor*.

**Normalisation** : `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

Les images fournies par PIL ont des données comprises entre 0 et 1. Ici, nous soustrayons 0.5, et divisons par 0.5 afin d'avoir des données comprises entre -1 et 1, ce qui est plus efficace pour l'entrainement d'un réseau de neurone (données centrées en 0 et de variance proche de 1).

### On peut alors créer la classe Dataset

```python
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torchvision import transforms
import os

def load_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

transform_driving_image = transforms.Compose([
    transforms.CenterCrop(72),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class CustomDataset:
    def __init__(self, dataset_path):
        self.images = os.path.join(dataset_path, "images")
        with open(os.path.join(dataset_path, "labels.txt"), 'r') as f:
            lines = [l.strip().split() for l in f.readlines()]
            lines = [[f, int(label)] for (f, label) in lines]
            self.labels = lines
        self.transform = transform_driving_image
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_name, label = self.labels[index]
        return self.transform(load_image(os.path.join(self.images, image_name))), torch.LongTensor([label])


def get_dataloader(dataset_path, batch_size):
    dataset = CustomDataset(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

LEFT = 0
RIGHT = 1
GO = 2
ACTIONS = [LEFT, RIGHT, GO]
```
Voir le fichier complet [data.py](https://github.com/cdancette/supervised-self-driving/blob/master/data.py)


**La fonction __len__** doit renvoyer la longueur du dataset. Ici c'est le nombre total d'images.

**La fonction __getitem__(self, index)** doit renvoyer l'objet d'index `index`. Ici, nous chargeons l'image correspondant à cet index, nous lui appliquons les transformations, puis nous renvoyons la matrice ainsi que les labels (sous forme de Tensor).

**Directions**
On a encodé les directions dans trois variables `LEFT`, `RIGHT` et `GO`, qui nous serviront dans les différents modules.

### Code pour l'entrainement du réseau de neurone

Ce code est tiré du tutoriel [http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

L'idée générale est la suivante : 

A chaque epoch, on entraine sur l'ensemble du dataset de train, puis on évalue sur le dataset de training. Les données sont chargées grâce à un *DataLoader*, fourni par pytorch (on lui donne en argument l'objet `Dataset` qu'on a créé précedemment).

Quelques étapes importantes:

**Wrapper** les `Tensors` dans des `Variables` : En pytorch, il est nécessaire de faire cette étape `data = Variable(tensor)`, car c'est l'objet `Variable` qui va garder en mémoire le gradiant de cette variable en fonction de la loss finale. Une variable est en fait une combinaison de deux tensors, celui des données et celui des gradiants.

**Backpropagation**

Pour effectuer la backpropagation en pytorch, les étapes suivantes sont nécessaires :
- `optimizer.zero_grad()` à chaque itération de la boucle. Cela remet à zero les gradiants de chaque paramètre.
- `loss.backward()` : cela va calculer les gradiants pour chaque variable par backpropagation en fonction de la loss, et les stocker dans l'objet Variable
- `optimizer.step()` : Modifie chaque paramètre de notre modèle (poids des réseaux) de manière à minimiser la loss.


```python
import copy 

import torch
from torch.autograd import Variable
from torch import optim, nn

from model import CustomModel
from data import CustomDataset, get_dataloader

def train(model, criterion, train_loader, test_loader, max_epochs=50, 
          learning_rate=0.001):
    
    dataloaders = {
        "train":train_loader, "val": test_loader
    }

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_acc = 0
    for epoch in range(max_epochs):
        print('Epoch {}/{}'.format(epoch, max_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['val', 'train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                labels = labels.view(labels.size(0))

                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / len(dataloaders[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, "models2/model-%s.weights" % epoch)

    print('Training complete')
    print('Best val Acc: {:4f}'.format(best_acc))

if __name__=='__main__': 
    num_classes = 3
    model = CustomModel()

    train_path = "train_set"
    test_path = "test_set"
    train_loader = get_dataloader(train_path, batch_size=8)
    test_loader = get_dataloader(test_path, batch_size=30)

    loss = nn.CrossEntropyLoss()
    train(model, loss, train_loader, test_loader)

```
Voir le fichier complet [train.py](https://github.com/cdancette/supervised-self-driving/blob/master/train.py)


## Conduite de la voiture grâce à notre modèle


Nous avons maintenant notre modèle qui est entrainé. Nous allons maintenant l'utiliser pour automatiser la direction de la voiture. 

```python
import gym
import numpy as np
import sys
import torch
from torch.autograd import Variable
import PIL
from torch.nn import Softmax
from pyglet.window import key

from model import CustomModel
from data import transform_driving_image

id_to_steer = {
    LEFT: -1,
    RIGHT: 1,
    GO: 0,
}

if __name__=='__main__':

    if len(sys.argv) < 2:
        sys.exit("Usage : python drive.py path/to/weights")
    # load the model
    model_weights = sys.argv[1]
    model = CustomModel()
    model.load_state_dict(torch.load(model_weights))

    env = gym.make('CarRacing-v0').env
    env.reset()

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    env.reset()
    
    # initialisation
    for i in range(50):
        env.step([0, 0, 0])
        env.render()
    
    i = 0
    while True:
        s, r, done, info = env.step(a)
        s = s.copy()
        # We transform our numpy array to PIL image
        # because our transformation takes an image as input
        s  = PIL.Image.fromarray(s)  
        input = transform_driving_image(s)
        input = Variable(input[None, :], volatile=True)
        output = Softmax()(model(input))
        _, index = output.max(1)
        index = index.data[0]
        a[0] = id_to_steer[index] * output.data[0, index] * 0.3  # lateral acceleration
        env.render()
    env.close()
```
Voir le fichier complet [drive.py](https://github.com/cdancette/supervised-self-driving/blob/master/drive.py)

Regardons plus précisément ce qui se passe dans la boucle : 

```python
s, r, done, info = env.step(a)
s = s.copy()        
s  = PIL.Image.fromarray(s)  
```

On récupère la matrice de pixels, et on la lit en utilisant PIL (pour qu'elle soit au même format que les images lues par le dataloader pendant le training)

```python
input = transform_driving_image(s)
```
On lui applique les mêmes transformations que dans le dataset (rognage des cotés de l'image, transformation en `Tensor`, et normalisation entre -1 et 1.)

```
input = Variable(input[None, :], volatile=True)
```
On convertit le `Tensor` en `Variable` pour le donner en entrée au réseau de neurone. L'argument `volatile=True` permet d'économiser de la mémoire, en disant au réseau de ne pas sauvegarder les opérations effectuées (utile quand on ne veut pas entrainer le modèle avec ces exemples).

```python
output = Softmax()(model(input))
_, index = output.max(1)  # index is a tensor
index = index.data[0]  # get the integer inside the tensor
```

On donne l'image au réseau, on récupère la sortie. C'est un tenseur de taille 3, chaque entrée correspond au score de chaque action (gauche, droite ou tout droit). L'action à choisir sera celle qui a le score le plus haut (on la passe dans un Softmax pour avoir une sortie entre 0 et 1).
On récupère l'action avec la fonction `max` qui renvoie la valeur max, et son index.


```python
a[0] = id_to_steer[index] * output.data[0, index] * 0.3  # lateral acceleration
env.render()
```

`a[0]` est l'accéleration latérale. On lui donne la valeur 0, 1 ou -1 selon l'action choisie par le réseau de neurone. On multiplie cette action par un coefficient de 0.3 pour éviter les actions trop brusques, et également par la probabilité de l'action donnée par le réseau (cela permet d'avoir des actions plus importantes si le réseau est sur de son action, et moins importante lorsque le réseau hésite).

**Après le lancement,** il faut contrôler la vitesse de la voiture avec les touches haut et bas du clavier. La direction sera choisie par le réseau de neurone

![car-running](/assets/self-driving-1/car-running.gif)

## Conclusion
Notre réseau reconnait les formes pour maintenir la voiture sur la trajectoire voulue. 
C'est une sorte de classifieur qui indique juste si la voiture est bien placée, trop à droite ou trop à gauche. Nous transmettons alors cette commande au simulateur. Tout ceci s'effectue en temps réel.

## Pour aller plus loin

### Controle de l'accéleration
Le controle de la voiture n'est pas total ici : le réseau controle uniquement l'accéleration latérale (la direction droite / gauche) de la voiture, mais ne controle pas l'accéleration (donc la vitesse).  Le problème est qu'il est impossible de deviner la vitesse de la voiture en regardant une seule image, donc il ne peut pas controller l'accéleration pour maintenir une vitesse convenable.

- utiliser la barre de vitesse qui est sous l'image (celle que l'on a masquée). Mais il faudrait garder masqué la barre de direction, qui induit en erreur le classifieur de direction;

- Donner au réseau plusieurs images successives, au lieu d'une seule. De cette manière, le réseau pourrait déduire la vitesse de la voiture

- Demander au réseau de controller uniquement la vitesse, et non l'accéleration (il faut alors coder un système de rétrocontrole qui va maintenir la vitesse demandée) : cette approche n'est pas vraiment *end-to-end* mais peut être plus simple si on a des données externes correctes sur la vitesse actuelle (on pourrait modifier l'environnement pour les fournir en plus de l'état).

### Data augmentation

Pour améliorer la performance des classifieurs, la meilleure méthode est d'augmenter la quantité de données. Mais ici, c'est assez long car les données doivent être enregistrées en jouant au jeu manuellement. Une manière d'augmenter artificiellement la quantité de données est appellée *data augmentation*. Il s'agit d'effectuer des transformations aux images, qui ne modifieront pas le labels (ou le modifieront de manière déterminée).

On peut par exemple prendre l'image symétrique par rapport à l'axe vertical. Les labels gauche / droite seront alors inversés, et on multiplie par 2 la quantité de données de manière immédiate. D'autres transformations possibles peuvent etre de déformer un peu l'image ou de modifier légèrement les couleurs (ici les couleurs sont fixes dans l'environnement, donc cela sera surement moins efficace ici que sur des vraies images).

### Analyse des patterns reconnus par le réseau

On peut aussi regarder dans les couches pour avoir des informations sur ce que les neurones ont appris. Des packages comme [https://github.com/utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) permettent de le faire.