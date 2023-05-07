from datasets import load_dataset
import numpy as np
import sys


def main():
    raw_datasets = load_dataset(
        "amazon_reviews_multi",
        "en",
        cache_dir=sys.argv[1]
    )

    # raw_datasets.rename_columns_({"review_body": "text", "starts": "label"})
    amz2 = raw_datasets.filter(lambda x: x["stars"] != 3)
    cls_train = np.where(np.array(amz2["train"]["stars"]) > 3, 1, 0)
    cls_valid = np.where(np.array(amz2["validation"]["stars"]) > 3, 1, 0)
    cls_test = np.where(np.array(amz2["test"]["stars"]) > 3, 1, 0)

    amz2["train"] = amz2["train"].add_column("label", cls_train)
    amz2["validation"] = amz2["validation"].add_column("label", cls_valid)
    amz2["test"] = amz2["test"].add_column("label", cls_test)

    amz2 = amz2.rename_columns({"review_title": "sentence1", "review_body": "sentence2"})
    amz2 = amz2.remove_columns([
        i for i in amz2["train"].column_names
        if i != "sentence1" and i != "sentence2" and i != "label"
    ])
    amz2.save_to_disk(sys.argv[2])

    amz5 = raw_datasets.rename_columns({
        "review_title": "sentence1",
        "review_body": "sentence2",
        "stars": "label"
    })
    amz5 = amz5.remove_columns([
        i for i in amz5["train"].column_names
        if i != "sentence1" and i != "sentence2" and i != "label"
    ])
    amz5.save_to_disk(sys.argv[3])


if __name__ == "__main__":
    main()
