#!/usr/bin/env python3
import argparse
import json
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, XLNetTokenizer
import torch
from keras.utils import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()


def load_input(df):
    if type(df) != pd.DataFrame:
        df = pd.read_json(df, lines=True)
    
    ret = []
    for _, i in df.iterrows():
        ret += [{'text': ' '.join(i['postText']) + ' - ' + i['targetTitle'] + ' ' + ' '.join(i['targetParagraphs']), 'uuid': i['uuid']}]
    
    return pd.DataFrame(ret)


def use_cuda():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def predict(df):
    df = load_input(df)
    uuids = list(df['uuid'])
    decoding_array = {  
                1: "passage",
                2 : "phrase",
                0 : "multi"
                }
    model = AutoModelForSequenceClassification.from_pretrained("EstrixDS/XLNet_SemEval_Task1")
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    # Create sentence and label lists
    sentences = df['text'].values

    # We need to add special tokens at the beginning and end of each sentence for XLNet to work properly
    sentences = [sentence[0] + " [SEP] [CLS]" for sentence in sentences]
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]


    MAX_LEN = 256
    # Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask) 

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    
    batch_size = 32  


    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    # Prediction on test set

    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions = []

    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        
        # Store predictions and true labels
        predictions.append(logits)
        
        for z in predictions[0]:
            label = np.where(z == z.max())
            decoded_label = decoding_array[label[0][0]]
    for i in range(len(df)):
        yield {'uuid': uuids[i], 'spoilerType': decoded_label}


def run_baseline(input_file, output_file):
    with open(output_file, 'w') as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)