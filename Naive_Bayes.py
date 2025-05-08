from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

class NaiveBayesModel:
    def __init__(self):
        self.model = MultinomialNB()
        self.vectorizer = None  # Thêm vectorizer để giải mã các feature

    def train(self, X_train, y_train, vectorizer=None):
        """
        Huấn luyện mô hình và lưu vectorizer nếu cần để giải thích mô hình
        """
        self.model.fit(X_train, y_train)
        self.vectorizer = vectorizer  # Lưu lại vectorizer để truy xuất tên từ

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

 