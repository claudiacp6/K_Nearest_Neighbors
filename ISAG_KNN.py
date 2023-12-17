#Modelo KNN
#Estrutura do modelo:

# 1. Código de inicialização na classe KNN;
# 2. Função fit;
# 3. Método Predict;
# 4. Normalização dos dados.

import numpy as np

class KNN:
    
    def __init__(self, k=3, normalize=False):
        self.k = k
        self.normalize = normalize
        self.X_train = None
        self.y_train = None
  
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        if self.normalize:
            X_test = self.normalize_data(X_test)

        predictions = []
        for x in X_test:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[indices]
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predictions.append(predicted_label)

        return np.array(predictions)
        
    def normalize_data(self, data):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        return (data - min_vals) / (max_vals - min_vals)
