import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from time import perf_counter
from time import sleep
from contextlib import contextmanager
from typing import Callable
import string
import re

stop_words = set(stopwords.words("english"))

def fill_string_nulls(df : pd.DataFrame):
    '''
        Fill nulls with empty string.
    '''
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].fillna("")

def fill_string_spaces(df : pd.DataFrame):
    '''
        Trim empty spaces.
    '''
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].str.replace("\s+", " ", regex=True)

def trim_strings(df : pd.DataFrame):
    '''
        Trim all string columns.
    '''
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].str.strip()

def generate_word_frequencies_from_column(df : pd.DataFrame, column : str):
    # Find all words in the column
    df = df.drop_duplicates(subset=[column])

    # Tokenize words into array of words and explode array
    all_words = df[column].str.findall("\\b\\w\\w+\\b").explode(column)
    all_words = all_words.str.lower()

    # Get word counts. We have column in column string for word and column 'count' for frequency 
    word_frequencies = all_words.value_counts().reset_index()
    
    # stemmer = PorterStemmer()

    # # Apply porter stemmer on column
    # all_words[column] = all_words[column].apply(lambda x: stemmer.stem(x))

    # # Now find any duplicates resulting from stemmer and do group by sum to
    # # resolve duplicates and restore our 'column' and 'count' dataframe
    # all_words = all_words.groupby(column)["count"].sum().reset_index()

    # Remove stopwords and words which do not contain letters (ex. 2016, 2017, 2018)
    uninterested_words_filter = (
        (word_frequencies[column].isin(stop_words)) |
        ~(word_frequencies[column].str.contains("[a-zA-Z]"))
    )
    word_frequencies = word_frequencies.loc[
        ~uninterested_words_filter
    ]
    escaped_puncs = re.escape(string.punctuation)
    # There should be no punctuations present in any of the words
    assert len(
        word_frequencies[word_frequencies[column].str.contains('|'.join(escaped_puncs.split()))]
    ) == 0, \
    "There should be no punctuations in the analyzed words"
    return word_frequencies.set_index(column)


@contextmanager
def catchtime() -> Callable[[], float]:
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()
    print(f"Total time elapsed: {t2 - t1:.4f}s")
    
