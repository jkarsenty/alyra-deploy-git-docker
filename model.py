import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from datetime import datetime

class TextClassifier:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.model_path = 'sentiment_model.pkl'
        
        # Statistiques d'usage
        self.stats = {
            'total_predictions': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'last_prediction': None
        }
        
        # Charger le modèle s'il existe, sinon créer et entraîner
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.train_model()
    
    def train_model(self):
        """Entraîne le modèle avec des données d'exemple"""
        print("Entraînement du modèle ML...")
        
        # Données d'entraînement d'exemple (en réalité, vous utiliseriez un vrai dataset)
        training_data = [
            # Positif
            ("Ce film est excellent", "positif"),
            ("J'adore ce produit", "positif"),
            ("Superbe qualité", "positif"),
            ("Très satisfait de mon achat", "positif"),
            ("Parfait, je recommande", "positif"),
            ("Génial, exactement ce que je cherchais", "positif"),
            ("Service client fantastique", "positif"),
            ("Livraison rapide et efficace", "positif"),
            ("Très bon rapport qualité prix", "positif"),
            ("Je suis ravi de cet achat", "positif"),
            
            # Négatif
            ("Ce produit est nul", "négatif"),
            ("Je déteste ce service", "négatif"),
            ("Qualité décevante", "négatif"),
            ("Très mauvaise expérience", "négatif"),
            ("N'achetez pas ce produit", "négatif"),
            ("Service client horrible", "négatif"),
            ("Livraison très lente", "négatif"),
            ("Produit défectueux", "négatif"),
            ("Je regrette cet achat", "négatif"),
            ("Très déçu du résultat", "négatif"),
            
            # Neutre
            ("Le produit est correct", "neutre"),
            ("Rien d'exceptionnel", "neutre"),
            ("C'est un produit standard", "neutre"),
            ("Conforme à la description", "neutre"),
            ("Produit moyen", "neutre"),
            ("Pas mal sans plus", "neutre"),
            ("Convenable pour le prix", "neutre"),
            ("Ordinaire", "neutre"),
        ]
        
        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]
        
        # Création du pipeline ML
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',  # Vous pouvez ajouter les stop words français
                max_features=1000,
                ngram_range=(1, 2)  # Unigrammes et bigrammes
            )),
            ('classifier', MultinomialNB(alpha=1.0))
        ])
        
        # Entraînement
        self.model.fit(texts, labels)
        self.is_trained = True
        
        # Sauvegarde du modèle
        self.save_model()
        print("Modèle entraîné et sauvegardé avec succès!")
    
    def save_model(self):
        """Sauvegarde le modèle entraîné"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self):
        """Charge le modèle depuis le fichier"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            print("Modèle chargé avec succès!")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            self.train_model()  # Fallback: réentraîner
    
    def predict(self, text):
        """Prédiction avec le modèle ML"""
        if not self.is_trained or self.model is None:
            return {
                'sentiment': 'neutre',
                'confidence': 0.0,
                'error': 'Modèle non entraîné'
            }
        
        if not text or not text.strip():
            return {
                'sentiment': 'neutre',
                'confidence': 0.0,
                'error': 'Texte vide'
            }
        
        try:
            # Prédiction
            prediction = self.model.predict([text])[0]
            
            # Probabilités pour calculer la confiance
            proba = self.model.predict_proba([text])[0]
            confidence = float(np.max(proba))
            
            # Obtenir toutes les probabilités
            classes = self.model.classes_
            probabilities = {cls: float(prob) for cls, prob in zip(classes, proba)}
            
            # Mise à jour des statistiques
            self._update_stats(prediction)
            
            return {
                'sentiment': prediction,
                'confidence': round(confidence, 3),
                'probabilities': probabilities,
                'model_type': 'ML (Naive Bayes + TF-IDF)'
            }
            
        except Exception as e:
            return {
                'sentiment': 'neutre',
                'confidence': 0.0,
                'error': f'Erreur de prédiction: {str(e)}'
            }
    
    def retrain_with_feedback(self, text, true_label):
        """Permet de réentraîner avec du feedback utilisateur"""
        # En production, vous ajouteriez ces données à votre dataset
        # et réentraîneriez périodiquement
        print(f"Feedback reçu: '{text}' -> {true_label}")
        # Ici vous pourriez implémenter l'apprentissage incrémental
        pass
    
    def get_model_info(self):
        """Informations sur le modèle"""
        if not self.is_trained:
            return {'status': 'non entraîné'}
        
        return {
            'status': 'entraîné',
            'type': 'Naive Bayes + TF-IDF',
            'classes': list(self.model.classes_) if hasattr(self.model, 'classes_') else [],
            'features': self.model.named_steps['tfidf'].get_feature_names_out()[:10].tolist() if hasattr(self.model.named_steps['tfidf'], 'get_feature_names_out') else [],
            'model_size': f"{os.path.getsize(self.model_path) / 1024:.1f} KB" if os.path.exists(self.model_path) else "N/A"
        }
    
    def _update_stats(self, sentiment):
        """Met à jour les statistiques d'usage"""
        self.stats['total_predictions'] += 1
        self.stats[f'{sentiment}_count'] += 1
        self.stats['last_prediction'] = datetime.now().isoformat()
    
    def get_stats(self):
        """Retourne les statistiques d'usage"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Remet à zéro les statistiques"""
        self.stats = {
            'total_predictions': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'last_prediction': None
        }