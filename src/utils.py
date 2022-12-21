import json
import pandas as pd
import string
import re

'''
import nltk

nltk.download('wordnet')
'''

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.preprocessing import LabelEncoder

from conf import preprocess_config


# Import and set up stopwords, lemmatizer and stemmer
nltk_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def load_data(file_path: str) -> pd.DataFrame:
    '''
    Open json lines file and parse each line into a DataFrame
    Args:
        file_path (str): Path to jsonl lines file

    Returns:
        data (pd.DataFrame): DataFrame containing the jsonl file data

    '''
    #
    with open(file_path, encoding='utf-8') as f:
        data = pd.DataFrame(json.loads(line) for line in f)
    return data


def preprocess_data(text: str, config: dict = preprocess_config) -> str:
    '''
    Preprocess input text according to config
    Parameters are set in the preprocess_conf dictionary inside "conf.py"

    Args:
        text (str): Input Text to preprocess
        config (dict): Preprocessing configuration (set in conf.py)

    Returns:
        tokens (list): Preprocessed tokenized text (List of strings)

    '''
    # convert spoiler to strings; join multiple list items by '\n'
    if type(text) == list:
        if len(text) <= 1:
            text = text[0]
        else:
            text = config['spoiler_join_char'].join(text)
    # set string to lowercase
    if config['lowercase']:
        text = text.lower()
    # special characters
    special_chars = string.punctuation
    # keep currency symbols (€/$)
    if config['keep_currency']:
        special_chars = special_chars.replace('$', '')
        special_chars = special_chars.replace('€', '')
    # keep hashtags
    if config['keep_hashtag']:
        special_chars = special_chars.replace('#', '')
    # remove punctuation
    if config['punctuation']:
        text = text.translate(str.maketrans('', '', string.punctuation))
    # remove all numbers
    if config['remove_numbers']:
        text = re.sub(r'\d+', '', text)
    # replace all numbers by 0
    if config['replace_numbers']:
        text = re.sub(r'\d+', '0', text)
    # remove starting and ending whitespaces
    if config['strip_whitespace']:
        text = text.strip()

    # tokenize string
    tokens = word_tokenize(text)

    # remove nltk stopwords
    if config['nltk_stopwords']:
        tokens = [token for token in tokens if not token in nltk_stopwords]
    # lemmatize words (WordNetLemmatizer)
    if config['lemmatizer']:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # stem words (Porter Stemmer)
    if config['stemmer']:
        tokens = [stemmer.stemmer(token) for token in tokens]
    if config['join_tokens']:
        tokens = ' '.join(tokens)

    return tokens


def encode_labels(label_column: pd.Series) -> pd.Series:
    '''
    Encode labels
    Args:
        label_column (pd.Series): Series containing labels (str) inside lists

    Returns:
        label_column (pd.Series): Series containing encoded labels (int)

    '''
    # get label string from list
    label_column = label_column.apply(lambda x: x[0])
    # set up encoder
    le = LabelEncoder()
    # fit and transform labels to label encodings
    label_column = le.fit_transform(label_column)
    return label_column
