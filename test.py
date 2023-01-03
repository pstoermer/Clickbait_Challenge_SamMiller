from src import utils
from transformers import XLNetForSequenceClassification, XLNetTokenizer

df = utils.load_data("data/train.jsonl")
text = []
for i in df["postText"]:
    n = utils.preprocess_data(i)
    text.append(n)
test = utils.encode_labels(df["tags"])
tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased", do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in text]
print(tokenized_texts)
print(type(tokenized_texts))
print(len(tokenized_texts))
for i, z in zip(text, tokenized_texts):
    print(i)
    print(z)
    print("\n")
