import argparse
import pickle
from collections import Counter
from pathlib import Path
import shutil
import face_recognition
from PIL import Image, ImageDraw, ImageEnhance
import cv2
import numpy as np

# Chemin par défaut pour les encodages
CHEMIN_ENCODAGES_PAR_DEFAUT = Path("output/encodages.pkl")
COULEUR_BOITE_DELIMITATION = "blue"
COULEUR_TEXTE = "white"

# Création des répertoires s'ils n'existent pas déjà
Path("output").mkdir(exist_ok=True)
Path("processed").mkdir(exist_ok=True)

# Configuration des arguments en ligne de commande
parser = argparse.ArgumentParser(description="Reconnaître des visages dans une image")
parser.add_argument("--train", nargs="+", help="Chemins vers les répertoires des datasets d'entraînement")
parser.add_argument("--validate", action="store", nargs="?", const="validation", help="Valider le modèle entraîné (chemin du répertoire facultatif)")
parser.add_argument("--test", action="store_true", help="Tester le modèle avec une image inconnue")
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Quel modèle utiliser pour l'entraînement : hog (CPU), cnn (GPU)",
)
parser.add_argument("-f", action="store", help="Chemin vers une image avec un visage inconnu")
args = parser.parse_args()

# Fonction pour supprimer le contenu du dossier `processed`
def supprimer_contenu_dossier(dossier="processed"):
    """
    Supprime tous les fichiers et sous-dossiers dans le dossier spécifié.
    """
    dossier_path = Path(dossier)
    if dossier_path.exists() and dossier_path.is_dir():
        shutil.rmtree(dossier_path)
    # Recréer le dossier après suppression
    dossier_path.mkdir(exist_ok=True)

# Fonction pour normaliser l'éclairage d'une image
def normaliser_eclairage(image_np):
    """
    Normalise l'éclairage d'une image en utilisant l'algorithme CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image_np)
    return image_clahe

# Fonction pour prétraiter les images et les sauvegarder dans le dossier `processed`
def pretraiter_images(dossiers_entree, dossier_sortie="processed", taille=(800, 800)):
    """
    Charge les images des répertoires dans `dossiers_entree`, applique un prétraitement (redimensionnement, conversion en niveaux de gris,
    suppression du bruit, normalisation de l'éclairage) et les enregistre dans le dossier `dossier_sortie`.
    """
    for dossier_entree in dossiers_entree:
        # Parcours des images dans le dossier d'entrée
        for chemin_fichier in Path(dossier_entree).glob("*/*"):
            try:
                # Chargement de l'image en utilisant Pillow (PIL)
                image = Image.open(chemin_fichier)

                # Redimensionnement de l'image
                image = image.resize(taille)

                # Conversion en niveaux de gris
                image = image.convert("L")

                # Convertir l'image en format numpy pour OpenCV
                image_np = np.array(image)

                # Filtrage du bruit (Denoising)
                image_np = cv2.fastNlMeansDenoising(image_np, None, 10, 7, 21)

                # Normalisation de l'éclairage
                image_np = normaliser_eclairage(image_np)

                # Conversion de l'image numpy vers Pillow (PIL) après normalisation
                image = Image.fromarray(image_np)

                # Optionnel : Amélioration de la netteté
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(2.0)

                # Création du chemin de sortie
                nom_fichier = chemin_fichier.name
                dossier_parent = Path(dossier_sortie) / chemin_fichier.parent.name
                dossier_parent.mkdir(exist_ok=True, parents=True)
                chemin_sortie = dossier_parent / nom_fichier

                # Sauvegarde de l'image prétraitée dans `processed`
                image.save(chemin_sortie)
                print(f"Image {chemin_sortie} sauvegardée.")

            except Exception as e:
                print(f"Erreur lors du traitement de l'image {chemin_fichier}: {e}")

# Fonction pour encoder les visages connus dans les images du dossier `processed`
def encoder_visages_connus(
    modele: str = "hog", emplacement_encodages: Path = CHEMIN_ENCODAGES_PAR_DEFAUT, dossier_entree="processed"
) -> None:
    """
    Charge les images du répertoire de formation prétraité et construit un dictionnaire de leurs noms et encodages.
    """
    noms = []
    encodages = []
    correspondances_correctes = 0
    total_visages = 0

    # Parcours des images dans le répertoire d'entrée
    for chemin_fichier in Path(dossier_entree).glob("*/*"):
        nom = chemin_fichier.parent.name
        image = face_recognition.load_image_file(chemin_fichier)

        # Détection des emplacements des visages
        emplacements_visages = face_recognition.face_locations(image, model=modele)
        encodages_visages = face_recognition.face_encodings(image, emplacements_visages)

        # Ajout des encodages et des noms
        for encodage in encodages_visages:
            noms.append(nom)
            encodages.append(encodage)

        # Comparaison avec les visages déjà encodés pour évaluer la correspondance
        for encodage in encodages_visages:
            total_visages += 1
            nom_reconnu = _reconnaitre_visage(encodage, {"noms": noms, "encodages": encodages})
            if nom_reconnu == nom:
                correspondances_correctes += 1

    # Sauvegarde des encodages dans un fichier pickle
    encodages_noms = {"noms": noms, "encodages": encodages}
    with emplacement_encodages.open(mode="wb") as f:
        pickle.dump(encodages_noms, f)

    # Calcul et affichage du score
    if total_visages > 0:
        score = (correspondances_correctes / total_visages) * 100
        print(f"Score d'entraînement : {score:.2f}% de visages reconnus correctement")
    else:
        print("Aucun visage n'a été trouvé dans le répertoire de formation prétraité.")

# Fonction pour reconnaître les visages dans une image donnée
def reconnaitre_visages(
    emplacement_image: str,
    modele: str = "hog",
    emplacement_encodages: Path = CHEMIN_ENCODAGES_PAR_DEFAUT,
    extraire_points_interet=False
) -> None:
    """
    Étant donné une image inconnue, obtenir les emplacements et encodages de tout visage,
    et les comparer aux encodages connus pour trouver des correspondances possibles.
    """
    # Chargement des encodages connus
    with emplacement_encodages.open(mode="rb") as f:
        encodages_charges = pickle.load(f)

    # Chargement de l'image d'entrée
    image_entree = face_recognition.load_image_file(emplacement_image)

    # Détection des visages et calcul de leurs encodages
    emplacements_visages_entree = face_recognition.face_locations(image_entree, model=modele)
    encodages_visages_entree = face_recognition.face_encodings(image_entree, emplacements_visages_entree)

    # Conversion en image Pillow pour dessiner
    image_pillow = Image.fromarray(image_entree)
    dessin = ImageDraw.Draw(image_pillow)

    # Comparaison des visages détectés avec ceux connus
    for boite, encodage_inconnu in zip(emplacements_visages_entree, encodages_visages_entree):
        nom = _reconnaitre_visage(encodage_inconnu, encodages_charges)
        if not nom:
            nom = "Inconnu"
        _afficher_visage(dessin, boite, nom)

        # Si extraire_points_interet est activé, afficher les points d'intérêt du visage
        if extraire_points_interet:
            points_interet = face_recognition.face_landmarks(image_entree)
            for points in points_interet:
                for feature, coords in points.items():
                    dessin.line(coords, fill="green", width=2)  # Dessine les points d'intérêt en vert

    # Affichage de l'image avec les annotations
    del dessin
    image_pillow.show()

# Fonction pour reconnaître un visage à partir de son encodage
def _reconnaitre_visage(encodage_inconnu, encodages_charges):
    """
    Étant donné un encodage inconnu et tous les encodages connus, trouver l'encodage connu avec le plus de correspondances.
    """
    correspondances = face_recognition.compare_faces(encodages_charges["encodages"], encodage_inconnu)
    votes = Counter(
        nom
        for correspondance, nom in zip(correspondances, encodages_charges["noms"])
        if correspondance
    )
    if votes:
        return votes.most_common(1)[0][0]

# Fonction pour afficher le visage avec une boîte de délimitation et une légende
def _afficher_visage(dessin, boite, nom):
    """
    Dessine des boîtes de délimitation autour des visages, une zone de légende et des légendes textuelles.
    """
    haut, droite, bas, gauche = boite
    dessin.rectangle(((gauche, haut), (droite, bas)), outline=COULEUR_BOITE_DELIMITATION)
    gauche_texte, haut_texte, droite_texte, bas_texte = dessin.textbbox((gauche, bas), nom)
    dessin.rectangle(
        ((gauche_texte, haut_texte), (droite_texte, bas_texte)),
        fill=COULEUR_BOITE_DELIMITATION,
        outline=COULEUR_BOITE_DELIMITATION,
    )
    dessin.text(
        (gauche_texte, haut_texte),
        nom,
        fill=COULEUR_TEXTE,
    )

# Fonction pour valider le modèle sur un ensemble d'images
def valider(modele: str = "hog", dossier_validation="validation"):
    """
    Exécute reconnaitre_visages sur un ensemble d'images avec des visages connus pour valider les encodages.
    """
    for chemin_fichier in Path(dossier_validation).rglob("*"):
        if chemin_fichier.is_file():
            reconnaitre_visages(emplacement_image=str(chemin_fichier.absolute()), modele=modele)

# Point d'entrée principal
if __name__ == "__main__":
    if args.train:
        dossiers_train = args.train if args.train else ["training"]
        # Supprimer le contenu du dossier `processed` avant l'entraînement
        supprimer_contenu_dossier("processed")
        # Prétraitement des images avant l'entraînement
        pretraiter_images(dossiers_entree=dossiers_train, dossier_sortie="processed", taille=(800, 800))
        # Entraînement avec les images prétraitées
        encoder_visages_connus(modele=args.m, dossier_entree="processed")
    if args.validate:
        dossier_validation = args.validate if args.validate else "validation"
        valider(modele=args.m, dossier_validation=dossier_validation)
    if args.test:
        reconnaitre_visages(emplacement_image=args.f, modele=args.m, extraire_points_interet=True)