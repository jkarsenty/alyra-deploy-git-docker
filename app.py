import logging
import json
from flask import Flask, request, jsonify, Response
from model import TextClassifier
from datetime import datetime


# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
#app.debug = True # Enable debug mode for development

classifier = TextClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    """Analyse le sentiment d'un texte avec le modèle ML"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Données JSON requises'}), 400
        
        text = data.get('text', '')
        if not text or not text.strip():
            return jsonify({'error': 'Texte vide ou manquant'}), 400
        
        # Log de la requête
        logger.info(f"Prédiction ML demandée pour: {text[:50]}...")
        
        result = classifier.predict(text)
        
        response = {
            'text': text,
            'prediction': result,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction ML: {str(e)}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil avec informations sur l'API"""
    try : 
        model_info = classifier.get_model_info()
    except :
        model_info = 'modèle manuellement défini'
        
    data = {
        'message': 'API de Classification de Sentiment ML',
        'version': '0.1.0',
        'status': 'healthy',
        'model': model_info,
        'endpoints': {
            'predict': "POST /predict - Analyse le sentiment d'un texte"
        }
    }

    return Response(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type='application/json; charset=utf-8'
    )



if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)