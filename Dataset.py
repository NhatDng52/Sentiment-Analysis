from datasets import load_dataset
import pandas as pd
 
class Dataset:
    def __init__(self):
        self.dataset = load_dataset("./vietnamese_students_feedback", trust_remote_code=True)

    def get_train(self):
        return self.dataset["train"]

    def get_test(self):
        return self.dataset["test"]

    def get_validation(self):
        return self.dataset["validation"]

if __name__ == "__main__":
    dataset = Dataset()
    # in câu đầu vã nhãn đầu tiên của tập train
    for i in range(30):
        print("First sentence and label in train set:")
        print("Sentence:", dataset.get_train()[i]["sentence"])
        print("Sentiment:", dataset.get_train()[i]["sentiment"])
        print("-" * 40)
    # Gộp cả 3 tập train, test, validation lại thành 1 dataframe
    all_data = pd.concat([
        dataset.get_train().to_pandas(),
        dataset.get_test().to_pandas(),
        dataset.get_validation().to_pandas()
    ], ignore_index=True)

    # Số mẫu tổng cộng
    num_samples = len(all_data)
    print(f"Total number of samples: {num_samples}")

    # Thống kê số lượng mỗi sentiment
    label_counts = all_data["sentiment"].value_counts().sort_index()
    print("\nLabel counts:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")

    # Độ dài trung bình của các câu
    sentence_lengths = all_data["sentence"].apply(len)
    avg_len = sentence_lengths.mean()
    print(f"\nAverage sentence length: {avg_len:.2f} words")