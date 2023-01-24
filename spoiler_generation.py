#!/usr/bin/env python3
from src import utils
import argparse
import json
from typing import Tuple
import pandas as pd
from transformers import QuestionAnsweringPipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
from tqdm import tqdm

MODEL_NAME = 'distilbert-base-cased-distilled-squad'


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


def load_input(input_file: str) -> pd.DataFrame:
    """
    Load input data
    Args:
        input_file(str): Input data in jsonl format

    Returns:
        pd.DataFrame: DataFrame containing input features: uuid, postText, targetParagraphs and tags
    """
    return utils.get_qa_features(input_file)


def get_device():
    return torch.cuda.current_device() if torch.cuda.is_available() else -1


def initialize_qa_pipeline(model_name: str) -> Tuple[AutoModelForQuestionAnswering, AutoTokenizer]:
    """
    Load Model and Tokenizer and set up QuestionAnsweringPipeline
    Args:
        model_name(str): Model name to use for predicting spoilers

    Returns:
        transformers.QuestionAnsweringPipeline: Pipeline with QuestionAnswering Tokenizer and Model

    """
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return QuestionAnsweringPipeline(model, tokenizer, device=get_device())


def predict(input_file: str):
    input_features = load_input(input_file)
    pipeline = initialize_qa_pipeline(MODEL_NAME)

    uuids = list(input_features["uuid"])
    spoiler_types = list(input_features["tags"])
    spoilers = list(tqdm(map(lambda x: utils.spoiler_generator(pipeline, x['postText'], x['targetParagraphs'], x['tags']),
                        input_features.to_dict('records'))))

    for i in range(len(uuids)):
        yield {"uuid": uuids[i], "spoilerType": spoiler_types[i], "spoiler": spoilers[i]}


def run(input_file, output_file):
    with open(output_file, "w") as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + "\n")


def main():
    args = parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()

# %%
