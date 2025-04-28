from Dataset import Dataset












if __name__ == "__main__":
    dataset = Dataset()
    train_data = dataset.get_train()
    
    # Print the type of the dataset
    print("train type:", type(train_data))
    
    # List all attributes and methods of the train dataset
    print("Attributes and methods of train_data:")
    print(dir(train_data))