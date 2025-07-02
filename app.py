import logging
from flask import Flask, request, jsonify
from model import TextClassifier

app = Flask(__name__)
#app.debug = True # Enable debug mode for development

classifier = TextClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    print(f"Received text: {text}")  # Debugging line to check received text
    
    if not text:
        return jsonify({'error': 'Pas de texte fourni'}), 400
    
    result = classifier.predict(text)
    
    return jsonify({
        'text': text,
        'sentiment': result,
        'status': 'success'
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'API IA fonctionne!', 'status': 'healthy'})

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)