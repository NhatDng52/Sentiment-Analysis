from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

class NaiveBayesModel:
    def __init__(self):
        # Khởi tạo mô hình Naive Bayes
        self.model = MultinomialNB()
    
    def train(self, X_train, y_train):
        """
        Huấn luyện mô hình Naive Bayes với dữ liệu X_train và nhãn y_train
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """
        Dự đoán nhãn cho dữ liệu X_test
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        Đánh giá mô hình trên tập test
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
    
    def save_model(self, path):
        """
        Lưu mô hình vào file
        """
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        """
        Tải mô hình từ file
        """
        self.model = joblib.load(path)