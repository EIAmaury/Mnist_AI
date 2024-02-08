# Machine learning using CNN and Autoencoders

### Description:
Ce projet a pout bur de débruiter des images liées à la librairie mnist pour classifier ces images dans un second temps. L'objectif est de concevoir un modèle de machine
leanring efficace et précis dans le débruitage et classification de données. Ce projet est constitué de deux modèles d'intellignece artifiel. Dans un preimer un modèle d'autoencodeur va apprendre à débruiter les images de la librairie mnist. 
Dans un second temps, un modèle de classification classifie les différentes images en sortie de l'auto-encodeur.

![Mnist AI](184b7cb84d7b456c96a0bdfbbeaa5f14_XL.jpg)

### Sommmaire:
1. Préréquis
2. Installation
3. Structure du Code

### 1.Préréquis
* Python 3.8 or higher
* Tensorflow 2.x librairies

### 2.Installation
```
git clone repository_link.git
cd repository
```
### 3.Structure du Code

**1. Preparation des donnees:**
* Création des données de test et d'entrainement
* Normalisation des données
* Affichage des données bruitées

**2. Modèle autoencoder**
* Création de couche de convolution, de MaxPooling et de UpSampling
* Entrainement sur les données bruitées
* Affichage des résultats

**3.Modèle CNN (convolution neural network)**
* Encodage des données labélisés
* Création de couche de convolution, de MaxPooling et de Dense
* Affichage de la précision

  

