from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

class SVMModel:
    def __init__(self, kernel='linear', C=1.0):
        self.model = SVC(kernel=kernel, C=C, probability=True)
        self.vectorizer = None  # Để truy xuất tên từ

    def train(self, X_train, y_train, vectorizer=None):
        self.model.fit(X_train, y_train)
        self.vectorizer = vectorizer

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
