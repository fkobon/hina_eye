#  Hina Eye

##  Description
**Hina Eye** est un projet réalisé dans le cadre de mon master 1 pour le cours de mathématiques du traitement de signal, avec pour thème : **"Recherche de Personne Disparue en appliquant les techniques de Traitement d'Images en mathématique du signal"**.

Cette première partie du projet se concentre sur l'entraînement d'un modèle de reconnaissance faciale sur des images fixes. Le projet utilise des techniques de traitement d'images pour détecter et reconnaître les visages. Un modèle de réseau neuronal convolutif (CNN) ou un modèle basé sur HOG est appliqué pour la reconnaissance faciale, avec pour objectif de développer une solution pouvant être utilisée pour aider à la recherche de personnes disparues.

##  Prérequis
Avant de commencer, assurez-vous d'avoir installé les éléments suivants :
-  **Python 3.x**
-  **pip** (Python package manager)
-  **Virtualenv** (optionnel mais recommandé pour isoler l'environnement)
-  **Flask** et **Flask-CORS** (pour l'application web)
-  **face_recognition**, **Pillow**, **OpenCV**, **NumPy**

##  Étapes d'exécution
###  1. Cloner le projet
Commencez par cloner le projet depuis votre dépôt GitHub :
```bash
git clone https://github.com/fkobon/hina_eye.git
cd hina_eye
```

###  2. Créer et activer un environnement virtuel (optionnel mais recommandé)
Il est fortement recommandé de créer un environnement virtuel pour isoler les dépendances de votre projet :

```bash
# Créer un environnement virtuel
python3 -m venv venv

# Activer l'environnement virtuel
# Sur macOS/Linux
source venv/bin/activate

# Sur Windows
venv\Scripts\activate
```

###  3. Installation des dépendances
Ensuite, installez toutes les dépendances du projet avec `pip` en exécutant la commande suivante :

```bash
pip install -r requirements.txt
```

Le fichier `requirements.txt` doit contenir les dépendances suivantes :
```
face_recognition
Pillow
opencv-python
numpy
Flask
Flask-CORS
```

###  4. Préparation des datasets
-  **Dataset d'entraînement** : Les images d'entraînement doivent être placées dans un répertoire `training`. Chaque sous-dossier du répertoire `training` doit correspondre à une classe (nom de la personne) et contenir des images de cette personne. Exemple :

```markdown
training/
├── personne1/
│ ├── image1.jpg
│ ├── image2.jpg
├── personne2/
├── image1.jpg
├── image2.jpg
```

-  **Dataset de validation** : Les images de validation doivent être placées dans un répertoire `validation` similaire à celui du dataset d'entraînement.

###  5. Lancer l'entraînement
####  Option 1 : Entraînement avec le répertoire `training` par défaut
Si vous avez placé vos images dans le répertoire `training`, vous pouvez lancer l'entraînement simplement avec la commande suivante :
```bash
python3 hina_eye.py --train
```

####  Option 2 : Entraînement avec plusieurs datasets
Si vous avez plusieurs datasets dans des répertoires distincts, vous pouvez les passer en argument de la commande `--train` :
```bash
python3 hina_eye.py --train /chemin/vers/dataset1 /chemin/vers/dataset2
```
####  Option 3 : Utiliser un modèle spécifique (HOG ou CNN)
Par défaut, le modèle `hog` est utilisé. Si vous souhaitez utiliser un modèle plus précis, comme `cnn` (qui nécessite un GPU), vous pouvez le spécifier avec l'option `-m` :
```bash
python3 hina_eye.py --train /chemin/vers/dataset1 -m  cnn
```

###  6. Validation du modèle
Après l'entraînement, vous pouvez valider le modèle sur un dataset de validation. Par défaut, le répertoire `validation` est utilisé, mais vous pouvez aussi spécifier un autre répertoire.

```bash
# Validation avec le répertoire par défaut
python3 hina_eye.py --validate

# Validation avec un dataset personnalisé
python3 hina_eye.py --validate /chemin/vers/dataset_validation
```

###  7. Test du modèle
Pour tester le modèle sur une image individuelle, utilisez l'option `--test` en spécifiant le chemin de l'image à tester :

```bash
python3 hina_eye.py --test -f /chemin/vers/image_a_tester.jpg
```

Cela affichera l'image avec des boîtes de délimitation autour des visages et des points d'intérêt (yeux, nez, bouche).

###  8. Application Web de test
L'application web simple permet de tester la reconnaissance faciale via une interface utilisateur.

####  Étape 1 : Démarrer le serveur Flask
Pour exécuter l'application web, accédez au répertoire **`client_web_app`** et lancez le serveur Flask.
```bash
cd client_web_app
python3 app.py
```
Le serveur Flask sera accessible sur `http://localhost:9999`. L'application écoute sur ce port et reçoit les images envoyées par l'interface utilisateur pour effectuer la reconnaissance faciale.

####  Étape 2 : Utilisation de l'interface web

1. Ouvrez le fichier **`index.html`** dans votre navigateur :
```bash
open index.html  # Pour macOS
start index.html # Pour Windows
```
Ou visitez directement `http://localhost:9999` si Flask est configuré pour servir l'interface web.

2. Sélectionnez une image via l'interface et cliquez sur **"Envoyer l'image"** pour lancer la reconnaissance faciale.
3. Le résultat apparaîtra à l'écran avec les visages reconnus, ou un message d'erreur si aucun visage n'est détecté.

###  9. Nettoyage automatique du dossier `processed`
Avant chaque entraînement, le contenu du dossier `processed` (où les images prétraitées sont stockées) est automatiquement supprimé afin d'éviter que des fichiers d'anciens entraînements ne perturbent les résultats du nouvel entraînement.

##  Structure du projet
Voici à quoi doit ressembler la structure des fichiers du projet :
```markdown
├── hina_eye.py       # Script principal pour l'entraînement, la validation et les tests
├── requirements.txt  # Liste des dépendances du projet
├── training/         # Répertoire contenant les images d'entraînement
├── validation/       # Répertoire contenant les images de validation
├── processed/        # Répertoire temporaire pour stocker les images prétraitées (automatiquement nettoyé)
├── output/           # Répertoire de sortie pour les encodages du modèle
├── client_web_app/   # Répertoire contenant l'application web
│ ├── app.py          # Script Flask pour l'application web
│ ├── index.html      # Interface web pour l'upload et la reconnaissance faciale
│ ├── static/         # Répertoire contenant les fichiers statiques
│ │ ├── app.js        # Script JavaScript pour l'application web
│ │ ├── styles.css    # Styles CSS pour l'interface web
└── README.md         # Ce fichier
```

##  Remarques importantes
-  **Modèle HOG** : Utilise uniquement le CPU et est plus rapide, mais moins précis. Il est recommandé pour les machines sans GPU.
-  **Modèle CNN** : Nécessite un GPU, est plus précis mais plus lent. Assurez-vous que votre machine est configurée pour utiliser CUDA si vous choisissez ce modèle.

##  Problèmes courants

-  **Problèmes de dépendances** : Si vous rencontrez des problèmes d'installation avec OpenCV ou face_recognition, essayez de mettre à jour vos outils (pip, setuptools) et vérifiez que vous utilisez une version compatible de Python (généralement Python 3.x).

-  **Utilisation d'un GPU** : Si vous utilisez le modèle CNN et rencontrez des problèmes avec votre GPU, vérifiez que CUDA et cuDNN sont correctement configurés sur votre machine.

##  Contact

Si vous avez des questions, des suggestions ou des problèmes, veuillez me contacter à l'adresse suivante : [atse.kobon@iua.ci](mailto:atse.kobon@iua.ci).

##  Mémoire

Le mémoire lié à ce projet est disponible [ici](https://firca.ci/fx/memoire_m1_fx_math_signal_hina_eyes.pdf)

Enseignant : [Dr. Ghislain Pandry](https://scholar.google.com/citations?user=Q1G3CooAAAAJ&hl=fr), Chercheur Traitement Signal et Image. 
