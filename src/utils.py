import json
from typing import List, Union, Dict, Any, Tuple

import pandas as pd
import string
import re

import spacy
import transformers

nlp = spacy.load("en_core_web_sm")

import nltk

# nltk.download('wordnet')
# nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.preprocessing import LabelEncoder

from .conf import preprocess_config

# Import and set up stopwords, lemmatizer and stemmer
nltk_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def load_data(file_path: str) -> pd.DataFrame:
    """
    Open json lines file and parse each line into a DataFrame
    Args:
        file_path (str):        Path to jsonl lines file

    Returns:
        data (pd.DataFrame):    DataFrame containing the jsonl file data

    """
    #
    with open(file_path, encoding='utf-8') as f:
        data = pd.DataFrame(json.loads(line) for line in f)
    return data


def preprocess_data(text: str, config: dict = preprocess_config) -> str:
    """
    Preprocess input text according to config
    Parameters are set in the preprocess_conf dictionary inside "conf.py"

    Args:
        text (str):     Input Text to preprocess
        config (dict):  Preprocessing configuration (set in conf.py)

    Returns:
        tokens (list):  Preprocessed tokenized text (List of strings)

    """
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
    """
    Encode labels
    Args:
        label_column (pd.Series): Series containing labels (str) inside lists

    Returns:
        label_column (pd.Series): Series containing encoded labels (int)

    """
    # get label string from list
    label_column = label_column.apply(lambda x: x[0])
    # set up encoder
    le = LabelEncoder()
    # fit and transform labels to label encodings
    label_column = le.fit_transform(label_column)
    return label_column


def improve_question(question: str) -> str:
    """
    Improve a question (clickbait) by removing spaCy stopwords and starting it with an interrogative
    (if there is one within the question).
    Args:
        question(str):          The question (clickbait) to be improved

    Returns:
        improved_question(str): Improved version of question (clickbait)

    """

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


def get_qa_features(file_path: str):
    """
    Extract features from dataset and transform them to list of strings
    Args:
        file_path (str):                Path to clickbait data

    Returns:
        input_features (pd.DataFrame):  Dataframe containing input features (clickbait texts, contexts and spoiler type)
    """
    data = load_data(file_path)

    data['targetParagraphs'] = data.targetParagraphs.apply(lambda x: ' '.join(x))
    data['postText'] = data.postText.apply(lambda x: improve_question(x[0].strip()))
    data['tags'] = data.tags.apply(lambda x: x[0])

    data['spoiler'] = data.apply(lambda x: '\n'.join(x['spoiler']) if x['tags'] != 'multi' else x, axis=1)

    return data[['uuid', 'postText', 'targetParagraphs', 'tags']]


def postprocess_phrase_spoiler(result: str) -> str:
    """
    Post-process phrase spoilers limiting them to a maximum of 5 words/token
    Args:
        result(str):    Raw answer (spoiler) predicted by the QuestionAnswering Model
    Returns:
        result(str):    Postprocessed phrase answer (spoiler) restricted to 5 words/tokens

    """
    words = result.split()
    return ' '.join(words[:5]) if len(words) > 5 else result


def postprocess_spoilers(result: str) -> str:
    """
    Post-process phrase spoilers removing punctuation.

    Args:
        result (str): Raw spoiler text predicted by the QuestionAnswering Model

    Returns:
        str: Preprocessed spoiler text without punctuation

    """
    return result.strip(string.punctuation)


def get_passage_spoiler(pipeline: transformers.QuestionAnsweringPipeline, clickbait: str, context: str,
                        max_loops: int = 5) -> Dict[str, Union[str, int]]:
    """
    Get passage spoilers (at least 5 words/tokens) by improving the question (clickbait) and using a question answering
    pipeline. If the returned answer contains less than 5 words/tokens, the sentence that contained the answer is removed
    and the pipeline is run again.

    Args:
        pipeline (transformers.QuestionAnsweringPipeline):  Pipeline to extract spoiler to clickbait
                                                            from context
        clickbait(str):      clickbait to extract spoiler for
        context:            Context for clickbait spoiler generation
        max_loops (int):    Maximum number of times to run the pipeline before returning None (to prevent infinite loop)
                            default: 10

    Returns:
        spoiler(Dict[str, Union[str, int]]):    Passage spoiler of format:
                                                {
                                                    'score': List[float]
                                                    'start': start index in context (List[int]),
                                                    'end': end index in context (List[int]),
                                                    'answer': spoiler text containing max 5 words/tokens (List[str])
                                                }
    """

    spoiler = None
    while not spoiler or len(spoiler['answer'].split()) < 5 or max_loops == 0:
        spoiler = pipeline(clickbait, context)
        spoiler['answer'] = spoiler['answer'].strip(string.punctuation)
        context = re.sub(rf'[^.?!]*(?<=[.?\s!]){spoiler["answer"]}(?=[\s.?!])[^.?!]*[.?!]', '', context, count=1)
        max_loops -= 1
    return spoiler


def get_multi_spoiler(pipeline: transformers.QuestionAnsweringPipeline, clickbait: str,
                      context: str) -> Dict[str, List[Union[str, int]]]:
    """
    Get multi spoilers (list of top 5 answers) by using a question answering pipeline.
    For each clickbait and context: get 5 answers
    After each answer the sentence that contained the answer is removed and the pipeline is run again.

    Args:
        pipeline (transformers.QuestionAnsweringPipeline):  Pipeline to extract answer(spoiler) to question (clickbait)
                                                            from context
        clickbait(str):      Question(clickbait) to extract answer(spoiler) for
        context(str):            Context for clickbait spoiler generation

    Returns:
        spoiler(Dict[str, List[Union[str,int]]]):   Multi spoiler of format:
                                                    [
                                                        {
                                                            'score': List[float]
                                                            'start': start index in context (List[int]),
                                                            'end': end index in context (List[int]),
                                                            'answer': spoiler text containing max 5 words/tokens (List[str])
                                                        }, ...
                                                    ]
    """

    multi_spoilers: Dict[str, List[Union[str, int]]] = {
        'score': [],
        'start': [],
        'end': [],
        'answer': []

    }

    for i in range(5):
        spoiler = pipeline(clickbait, context)
        spoiler['answer'] = spoiler['answer'].strip(string.punctuation)
        context = context.replace([i for i in re.split(r'[.?!]', context) if spoiler['answer'] in i][0], '')

        multi_spoilers['score'].append(spoiler['score'])
        multi_spoilers['start'].append(spoiler['start'])
        multi_spoilers['end'].append(spoiler['end'])
        multi_spoilers['answer'].append(spoiler['answer'])

    return multi_spoilers


def get_phrase_spoiler(pipeline: transformers.QuestionAnsweringPipeline, clickbait: str, context: str) -> Dict[
    str, Union[str, int]]:
    """
    Get phrase spoilers (max 5 words/tokens) by using a question answering pipeline.

    Args:
        pipeline (transformers.QuestionAnsweringPipeline):  Pipeline to extract spoiler to clickbait from context
        clickbait(str):                                     Clickbait to extract answer(spoiler) for
        context:                                            Context for clickbait spoiler generation

    Returns:
        spoiler(Dict[str, Union[str, int]]):    Phrase spoiler of format:
                                                {
                                                    'score': float
                                                    'start': start index in context (int),
                                                    'end': end index in context (int),
                                                    'answer': spoiler text containing max 5 words/tokens (str)
                                                }
    """

    return pipeline(clickbait, context, postprocess=postprocess_phrase_spoiler)


def spoiler_generator(pipeline: transformers.QuestionAnsweringPipeline, clickbait: str, context: str,
                      spoiler_type: str) -> Tuple[Union[str, int], Union[str, int], Union[str, int]]:
    """
    Perform spoiler generation depending on the spoiler type.

    Args:
        pipeline (transformers.QuestionAnsweringPipeline):  Pipeline to extract answer(spoiler) to question (clickbait)
                                                            from context
        clickbait(str):      Question(clickbait) to extract answer(spoiler) for
        context(str):            Context for clickbait spoiler generation
        spoiler_type(str):

    Returns:
        spoiler (Union[Dict[str, Union[str, int]], List[Dict[str, Union[str, int]]]]): spoiler dictionary or list of
                                                                        spoiler dict (multi spoiler) in the format:
                                                            {
                                                                'score': float
                                                                'start': start index in context (int),
                                                                'end': end index in context (int),
                                                                'answer': spoiler text (str)
                                                            }

    """
    if spoiler_type == 'passage':
        spoiler = get_passage_spoiler(pipeline=pipeline, clickbait=clickbait, context=context, max_loops=10)
    elif spoiler_type == 'multi':
        spoiler = get_multi_spoiler(pipeline=pipeline, clickbait=clickbait, context=context)
    elif spoiler_type == 'phrase':
        spoiler = get_phrase_spoiler(pipeline=pipeline, clickbait=clickbait, context=context)
    return spoiler['answer'], spoiler['start'], spoiler['end']


def run_spoiler_generator(pipeline: transformers.QuestionAnsweringPipeline,
                          row: pd.DataFrame) -> Tuple[Union[str, int], Union[str, int], Union[str, int]]:
    """
    Run spoiler generator on dataset

    Args:
        pipeline (transformers.QuestionAnsweringPipeline): QuestionAnsweringPipeline
        row (pd.DataFrame): Row of Dataset containing one clickbait, context and spoilertype

    Returns:

    """
    return spoiler_generator(pipeline, row['postText'], row['targetParagraphs'], row['tags'])
