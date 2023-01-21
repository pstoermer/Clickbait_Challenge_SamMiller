import json
from typing import List

import pandas as pd
import string
import re

import spacy
import transformers

nlp = spacy.load("en_core_web_sm")

from nltk.translate.bleu_score import sentence_bleu

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
        text = text[0] if len(text) <= 1 else config['spoiler_join_char'].join(text)
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
        tokens = [token for token in tokens if token not in nltk_stopwords]
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

def improve_question(question: str) -> str:
    '''
    Improve a question (clickbait) by removing spaCy stopwords and starting it with an interrogative (if there is one within the question).
    Args:
        question(str): The question (clickbait) to be improved

    Returns:
        improved_question(str): Improved version of question (clickbait)

    '''

    # Set to lowercase
    question = question.lower()

    # Tokenize
    doc = nlp(question)
    tokens = [token.text for token in doc]

    interrogative = next(
        (
            token
            for token in tokens
            if token
               in [
                   "how",
                   "what",
                   "when",
                   "where",
                   "who",
                   "which",
                   "whom",
                   "whose",
                   "why",
               ]
        ),
        None,
    )
    # add special condition for questions concerning amounts (how much...?, how many...?)
    if interrogative == 'how':
        if 'much' in tokens:
            interrogative = 'how much'
        elif 'many' in tokens:
            interrogative = 'how many'

    # Get spaCy's stopword list
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    # Remove stopwords from question tokens
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Re-create the question:
    # Start question with interrogative (if there is one)
    improved_question = interrogative + " " if interrogative else ""
    # add remaining tokens
    improved_question += " ".join([token for token in filtered_tokens if token != interrogative])
    return improved_question

def postprocess_phrase_spoiler(result: str) -> str:
    '''
    Post-process phrase spoilers limiting them to a maximum of 5 words/token
    Args:
        result(str): Raw answer (spoiler) predicted by the QuestionAnswering Model
    Returns:
        result(str): Postprocessed phrase answer (spoiler) restricted to 5 words/tokens

    '''
    words = result.split()
    return ' '.join(words[:5]) if len(words) > 5 else result

def postprocess_passage_spoiler(result: str) -> str:
    '''
    Post-process passage spoilers limiting them to a maximum of 5 words/token
    Args:
        result(str): Raw answer (spoiler) predicted by the QuestionAnswering Model
    Returns:
        result(str): Postprocessed phrase answer (spoiler) restricted to 5 words/tokens

    '''
    words = result.split()
    return ' '.join(words[:5]) if len(words) > 5 else result
def get_passage_spoiler(pipeline:transformers.QuestionAnsweringPipeline, question:str, context:str, max_loops:int) -> dict:
    '''
    Get passage spoilers (at least 5 words/tokens) by improving the question (clickbait) and using a question answering
    pipeline. If the returned answer contains less than 5 words/tokens, the sentence that contained the answer is removed
    and the pipeline is run again.

    Args:
        pipeline (transformers.QuestionAnsweringPipeline):  Pipeline to extract answer(spoiler) to question (clickbait)
                                                            from context
        question(str):  Question(clickbait) to extract answer(spoiler) for
        context:        Context for clickbait spoiler generation
        max_loops (int): Maximum number of times to run the pipeline before returning None (to prevent infinite loop)
    Returns:
        answer(dict): Answer (passage spoiler) containing at least 5 words/tokens
    '''

    answer = None
    question = improve_question(question)
    while not answer or len(answer['answer'].split()) < 5 or max_loops == 0:
        answer = pipeline(question, context)
        context = re.sub(rf'[^.?!]*(?<=[.?\s!]){answer["answer"]}(?=[\s.?!])[^.?!]*[.?!]', '', context)
        max_loops -= 1
    return answer

def calculate_bleu(true_spoilers: List[str], pred_spoilers: List[str]):
    return sum(
        sentence_bleu([true_spoiler.split(' ')], pred_spoiler.split(' '))
        for true_spoiler, pred_spoiler in zip(true_spoilers, pred_spoilers)
    )

