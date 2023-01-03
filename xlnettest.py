#!/usr/bin/env python3
import utils
import argparse
import json
import pandas as pd
import numpy as np
from transformers import XLNetForSequenceClassification, XLNetTokenizer
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        help="The input data (expected in jsonl format).",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="The classified output in jsonl format.",
        required=False,
    )

    return parser.parse_args()


def load_input(df):
    df = utils.load_data("data/train.jsonl")
    train_inputs = []
    for i in df["postText"]:
        n = utils.preprocess_data(i)
        train_inputs.append(n)
    train_labels = utils.encode_labels(df["tags"])
    return train_inputs, train_labels


def use_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

def finetune(train_inputs, train_labels):
    model = XLNetForSequenceClassification.from_pretrained("xlnet-large-cased")
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in train_inputs]
    return model


def predict(model, df):
    df = load_input(df)
    labels = ["phrase", "passage", "multi"]

    uuids = list(df["uuid"])
    texts = list(df["text"])
    predictions = model.predict(texts)[1]

    for i in range(len(df)):
        yield {"uuid": uuids[i], "spoilerType": labels[np.argmax(predictions[i])]}


def run_baseline(input_file, output_file):
    with open(output_file, "w") as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_baseline(args.input, args.output)

# Convert the dataset to tensors
train_inputs = tokenizer(train_df.text, return_tensors="pt").input_ids
train_labels = torch.tensor(train_df.label)
valid_inputs = tokenizer(valid_df.text, return_tensors="pt").input_ids
valid_labels = torch.tensor(valid_df.label)

# Fine-tune the model
model.fit(train_inputs, train_labels, valid_inputs, valid_labels, epochs=5)
