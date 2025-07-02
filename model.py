class TextClassifier:
    def __init__(self):
        # Modèle simple basé sur des mots-clés pour commencer
        self.positive_words = ['bon', 'super', 'génial', 'excellent', 'parfait', 'love', 'great']
        self.negative_words = ['mauvais', 'nul', 'horrible', 'terrible', 'hate', 'bad']
    
    def predict(self, text):
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positif'
        elif negative_count > positive_count:
            return 'négatif'
        else:
            return 'neutre'