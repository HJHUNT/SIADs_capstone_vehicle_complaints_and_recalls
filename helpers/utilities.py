import os
import string
from typing import Callable
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from time import perf_counter
from time import sleep
from contextlib import contextmanager
from typing import Callable
import string
from time import perf_counter
import re

def get_dataset_dir():
    root_dir = os.path.dirname(__file__).rsplit(os.sep,1)[0]
    return os.path.join(root_dir, "Datasets")

@contextmanager
def catchtime() -> Callable[[], float]:
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()
    print(f"Total time elapsed: {t2 - t1:.4f}s")

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
    df = df.copy()
    df = df.drop_duplicates(subset=[column])

    # Tokenize words into array of words and explode array
    all_words = df[column].str.findall("\\b\\w\\w+\\b").explode(column)
    all_words = all_words.str.lower()
    all_words.reset_index(inplace=True)

    # Get document counts per word. 
    # We have column in column string for word and column 'count' for frequency 
    word_frequencies = all_words.groupby(column)["index"].nunique().reset_index()
    word_frequencies.rename(
        {
            "index" : "count"
        }, axis=1
    )
    
    # stemmer = PorterStemmer()

    # # Apply porter stemmer on column
    # all_words[column] = all_words[column].apply(lambda x: stemmer.stem(x))

    # # Now find any duplicates resulting from stemmer and do group by sum to
    # # resolve duplicates and restore our 'column' and 'count' dataframe
    # all_words = all_words.groupby(column)["count"].sum().reset_index()

    # Remove stopwords and words which do not contain letters (ex. 2016, 2017, 2018)
    stop_words = set(stopwords.words("english"))
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

def find_columns_defining_uniqueness(df : pd.DataFrame, duplicates_allowed=0,
                                     columns_defining_uniqueness=["ODINO"]):
    '''
        Find columns that defines a dataframe's uniqueness
        based on duplicate threshold requirements.

        TODO: WHILE LOOP IS UNNECCESARY. 

        df : DataFrame
        duplicates_allowed : how many ODINOs we want in our final dataset
    '''
    prev_length = -1
    columns_defining_uniqueness = ["ODINO"]
    while len(df) > 0 and len(df["ODINO"].unique()) > duplicates_allowed:
        # Find column with max unique counts that's not already selected
        unique_counts_per_odino = df.groupby("ODINO").nunique().sum(axis=0).sort_values(ascending=False)
        max_unique_count = -1
        max_unique_count_index = -1
        for i, column in enumerate(unique_counts_per_odino.index):
            if column in columns_defining_uniqueness:
                continue
            else:
                max_unique_count = unique_counts_per_odino.loc[column]
                max_unique_count_index = i
                break
        
        if max_unique_count == -1:
            break

        if max_unique_count > len(df["ODINO"].unique()):
            columns_defining_uniqueness.append(unique_counts_per_odino.index[max_unique_count_index])
            df = df.loc[df.duplicated(subset=columns_defining_uniqueness, keep=False)]
        else:
            # No more column additions brings value
            break
    return columns_defining_uniqueness




