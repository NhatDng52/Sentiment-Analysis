from datasets import load_dataset

# Tải dataset bằng script custom
dataset = load_dataset("./vietnamese_students_feedback", trust_remote_code=True)

# In thử vài dòng
for example in dataset["train"].select(range(30)):
    print("Sentence:", example["sentence"])
    """- `sentiment`: Sentiment class, with values 0 (negative), 1 (neutral) and 2 (positive)."""
    print("Sentiment:", example["sentiment"])
    #print("Topic:", example["topic"])
    print("-" * 40)
    
class Dataset:
    def __init__(self):
        self.dataset = dataset

    def get_train(self):
        return self.dataset["train"]

    def get_test(self):
        return self.dataset["test"]

    def get_validation(self):
        return self.dataset["validation"]
