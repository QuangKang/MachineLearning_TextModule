from datasets import load_dataset

def load_data(dataset_name: str = "ag_news"):
    ds = load_dataset(dataset_name)
    train = ds["train"]
    test = ds["test"]

    train_texts = train["text"]
    train_labels = train["label"]
    test_texts = test["text"]
    test_labels = test["label"]

    return train_texts, train_labels, test_texts, test_labels