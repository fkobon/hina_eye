document.getElementById('uploadButton').addEventListener('click', function() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];

    // Journaliser l'étape de sélection du fichier
    if (!file) {
        console.error("Aucun fichier sélectionné.");
        alert("Veuillez sélectionner une image.");
        return;
    }

    console.log("Fichier sélectionné :", file);

    // Vérifier le type du fichier (format d'image valide)
    const validImageTypes = ['image/jpeg', 'image/png'];
    if (!validImageTypes.includes(file.type)) {
        console.error("Type de fichier non valide. Types acceptés: JPEG, PNG.");
        alert("Veuillez télécharger une image au format JPEG ou PNG.");
        return;
    }

    // Créer un objet FormData pour envoyer le fichier au backend
    const formData = new FormData();
    formData.append('file', file);

    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '<p class="loading">Analyse en cours...</p>'; // Afficher un message de chargement

    console.log("Début de l'envoi de l'image au backend...");

    // Envoyer l'image au backend via une requête POST
    fetch('http://127.0.0.1:9999/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log("Réponse reçue du backend :", response);

        // Vérifier si la réponse est correcte (status code 200-299)
        if (!response.ok) {
            console.error(`Erreur HTTP : ${response.status}`);
            throw new Error(`Erreur lors de la communication avec le serveur : ${response.statusText}`);
        }

        return response.json();
    })
    .then(data => {
        console.log("Données reçues du serveur :", data);

        // Vider la zone des résultats avant d'afficher les nouveaux résultats
        resultDiv.innerHTML = '';

        // Vérifier si une erreur est retournée par le backend
        if (data.error) {
            console.error("Erreur retournée par le serveur :", data.error);
            resultDiv.innerHTML = `<p class="recognized-face">Erreur : ${data.error}</p>`;
            return;
        }

        // Afficher les résultats des visages reconnus
        if (data.recognized_faces.length > 0) {
            console.log(`${data.recognized_faces.length} visages reconnus.`);
            data.recognized_faces.forEach(face => {
                const faceDiv = document.createElement('div');
                faceDiv.classList.add('recognized-face');
                faceDiv.textContent = `Visage reconnu : ${face}`;
                resultDiv.appendChild(faceDiv);
            });
        } else {
            console.log("Aucun visage reconnu.");
            resultDiv.innerHTML = '<p class="recognized-face">Aucun visage reconnu.</p>';
        }
    })
    .catch(error => {
        console.error('Erreur lors de l’envoi de la requête ou du traitement :', error);
        resultDiv.innerHTML = '<p class="recognized-face">Erreur lors de la reconnaissance faciale. Veuillez réessayer.</p>';
    });
});
