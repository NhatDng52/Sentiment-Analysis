#import
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from Dataset import Dataset
from sklearn.preprocessing import LabelEncoder  # Thêm phần mã hóa nhãn
from Naive_Bayes import NaiveBayesModel
from SVM import SVMModel
from BERT import BertModel
from sklearn.metrics import classification_report, accuracy_score, f1_score
#=============================BEGIN LOAD DATASET=============================



dataset = Dataset()
train_data_pd = dataset.get_train().to_pandas()
test_data_pd = dataset.get_test().to_pandas()

#=============================END LOAD DATASET=============================
#=============================BEGIN PREPROCESS=============================



vietnamese_stopwords = [
    'và', 'là', 'các', 'một', 'những', 'được', 'của', 'trong', 'với', 'cho', 
    'tôi', 'bạn', 'chúng', 'ta', 'thì', 'có', 'đã', 'này', 'ở', 'vì', 'nên', 
    'khi', 'đó', 'để', 'ra', 'vào', 'trên', 'dưới', 'hay', 'hoặc', 'rằng', 
    'mà', 'như', 'lúc', 'cũng', 'cùng', 'nữa', 'còn', 'sẽ', 'các', 'cái', 
    'ấy', 'nhưng', 'vẫn', 'từng', 'mọi', 'hết', 'bị', 'làm', 'điều'
]



label_encoder = LabelEncoder()
train_data_pd['label'] = label_encoder.fit_transform(train_data_pd['sentiment'])
vectorizer = TfidfVectorizer(stop_words=vietnamese_stopwords)

raw_texts = train_data_pd['sentence']  # Chuỗi input
labels = train_data_pd['label']    # Nhãn đã mã hóa

encoded_texts = vectorizer.fit_transform(train_data_pd['sentence'])

test_encoded_texts = vectorizer.transform(test_data_pd['sentence'])  # Chỉ transform cho tập test
test_labels = label_encoder.transform(test_data_pd['sentiment'])  # Mã hóa nhãn cho tập test
#===============================END PREPROCESS=============================
#===============================BEGIN TRAINING=============================


            #-------------------NAIVE BAYES-------------------
naive_bayes_model = NaiveBayesModel()  # Tạo mô hình Naive Bayes
naive_bayes_model.train(encoded_texts, labels)  # Huấn luyện mô hình với dữ liệu đã được vector hóa và nhãn
  
            
            
            #-------------------SUPPORT VECTOR MACHINE-------------------
svm_model = SVMModel(kernel='linear')  # Tạo mô hình SVM với kernel là linear
svm_model.train(encoded_texts, labels)  # Huấn luyện mô hình với dữ liệu đã được vector hóa và nhãn      

            
            #-------------------BIDIRECTIONAL ENCODER REPRESENTATIONS FROM TRANSFORMERS-------------------

bert_model = BertModel(num_labels= 3)

"Train một lần xong tham số sẽ được lưu ở thư mục ./results, lúc đó cần comment train lại và sửa bên model lấy tham số từ mục results"
# bert_model.train(raw_texts, labels)





#==================================END TRAINING=============================
#==================================BEGIN EVALUATION=============================
def evaluate_model(name, true_labels, pred_labels):
    print(f"\n================= {name} =================")
    print("Accuracy:", accuracy_score(true_labels, pred_labels))
    print("F1 Macro :", f1_score(true_labels, pred_labels, average='macro'))
    print("F1 Micro :", f1_score(true_labels, pred_labels, average='micro'))
    print("F1 Weighted :", f1_score(true_labels, pred_labels, average='weighted'))

    print("\nClassification Report:")
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=[str(c) for c in label_encoder.classes_]
    ))





NB_predictions = naive_bayes_model.predict(test_encoded_texts)
SVM_predictions = svm_model.predict(test_encoded_texts)
BERT_predictions = bert_model.predict(test_data_pd['sentence'].tolist())
evaluate_model("Naive Bayes", test_labels, NB_predictions)
evaluate_model("SVM", test_labels, SVM_predictions)
evaluate_model("BERT", test_labels, BERT_predictions)



#===========================END EVALUATION=============================