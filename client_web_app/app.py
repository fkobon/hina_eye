from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
from PIL import Image
import io
import numpy as np
import pickle
from pathlib import Path

app = Flask(__name__)

CORS(app)

# Chemin vers le fichier encodages.pkl
ENCODAGES_MODELE = Path(__file__).parent.parent / "output" / "encodages.pkl"

# Charger les encodages connus
try:
    with open(ENCODAGES_MODELE, "rb") as f:
        data = pickle.load(f)
except Exception as e:
    raise KeyError(f"Erreur lors du chargement du fichier encodages.pkl : {str(e)}")

if 'encodages' in data and 'noms' in data:
    known_face_encodings = data['encodages']
    known_face_names = data['noms']
else:
    raise KeyError("Le fichier encodages.pkl ne contient pas les clés 'encodages' et 'noms'.")

@app.route("/upload", methods=["POST"])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "Aucune image trouvée."}), 400

    try:
        # Charger l'image envoyée
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))

        # Redimensionner l'image pour accélérer le traitement
        img.thumbnail((800, 800))

        # Convertir en numpy array
        img_np = np.array(img)

        # Utiliser HOG pour une détection rapide
        face_locations = face_recognition.face_locations(img_np, model="hog")

        # Vérifier s'il y a des visages
        if not face_locations:
            return jsonify({"error": "Aucun visage détecté dans l'image."}), 200

        # Extraire les encodages des visages
        face_encodings = face_recognition.face_encodings(img_np, face_locations)

        recognized_faces = []
        for face_encoding in face_encodings:
            # Comparer les visages détectés aux encodages connus
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Inconnu"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]

            recognized_faces.append(name)

        return jsonify({"recognized_faces": recognized_faces})

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la reconnaissance faciale : {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999)
