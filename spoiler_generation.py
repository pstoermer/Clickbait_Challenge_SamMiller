import sys
sys.path.append('./src')
from src import utils
import argparse
import json
from typing import Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from transformers import QuestionAnsweringPipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch

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
    return utils.get_qa_features("data/train.jsonl")


def use_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)


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

    return QuestionAnsweringPipeline(model, tokenizer)


def predict(input_file: str):
    input_features = load_input(input_file)
    pipeline = initialize_qa_pipeline(MODEL_NAME)

    uuids = list(input_features["uuid"])
    clickbaits = list(input_features['postText'])
    contexts = list(input_features["targetParagraphs"])
    with ThreadPoolExecutor() as executor:
        spoilers = list(executor.map(utils.run_spoiler_generator,
                                     [pipeline] * len(input_features),
                                     input_features.to_dict('records')))

    for i in range(len(uuids)):
        yield {"uuid": uuids[i], "spoiler": spoilers[i]}


def run(input_file, output_file):
    with open(output_file, "w") as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + "\n")


def main():
    args = parse_args()
    run(args.input, args.output)

if __name__ == "__main__":
    main()

#%%
