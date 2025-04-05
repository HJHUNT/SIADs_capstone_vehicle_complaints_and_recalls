# In[1]
from sentence_transformers import SentenceTransformer
from sklearn.conftest import fetch_rcv1_fxt
from sklearn.feature_extraction.text import TfidfVectorizer
from all_minilm import HuggingFaceClassifier
from sklearn.model_selection import train_test_split
from typing import Union
import sys
import os
from numpy import extract
import pandas as pd
import pickle
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize

curdir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curdir)
sys.path.insert(0, parent_dir)
from helpers.pickle_decorator import pickle_io
from helpers.utilities import (
    fill_string_nulls
)


class Preprocesser:
    def __init__(
        self,
        text_column,
        custom_clean_name,
        custom_vectorizer_name,
        cleaning_type="nltk",
        csv_name="test_agg.csv",
        vectorizer=None,
        vectorizer_params=dict(
            ngram_range=(1,2),
            min_df=20,
            max_df=0.7,
            binary=True
        ),
        train_size=0.8,
        validation_size=0.1,
        extra_stopwords=[],
        random_state=42,
        is_stem=True,
        rerun=False
    ):
        root_dir = os.path.dirname(__file__).rsplit(os.sep,1)[0]
        self.dataset_dir = os.path.join(root_dir, "Datasets")
        self.csv_path = os.path.join(self.dataset_dir, csv_name)
        self.vectorizer = vectorizer(
            **vectorizer_params
        )
        self.vectorizer_params = vectorizer_params
        self.extra_stopwords = extra_stopwords
        self.cleaned_text_column = text_column # if no data cleaning is done
        self.text_column = text_column
        self.custom_clean_name = custom_clean_name
        self.custom_vectorizer_name = custom_vectorizer_name
        self.train_size = train_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.cleaning_type = cleaning_type
        self.is_stem = is_stem
        self.rerun = rerun
    @staticmethod
    def str_cat(df, columns):
        first_column = columns[0]
        other_columns = columns[1:]
        return df[first_column].str.cat(
            df[columns],
            sep=". "
        )

    def read_csv(self, csv_path, nrows=None):
        if nrows:
            self.df = pd.read_csv(csv_path, nrows=nrows)
        else:
            self.df = pd.read_csv(csv_path)

        # temporarily add IS_RECALL. This
        # should go into combine_dataset.py instead
        self.df["IS_RECALL"] = np.where(
            self.df["IS_COMPLAINT"] == 1, 0 , 1
        )
        self.df["CDESCR_AND_COMPONENT"] = np.where(
            self.df["IS_COMPLAINT"] == 1,
            Preprocesser.str_cat(self.df, ["CDESCR", "COMPDESC"]),
            Preprocesser.str_cat(self.df, ["CONSEQUENCE_DEFECT", "COMPDESC"])
        )
        return self.df
    
    def lowercase_string(self, text_column):
        self.df[text_column] = self.df[text_column].str.lower()
        
    def sentence_remover():
        '''
            TODO: Explore whether we can remove certain sentences
            from text.
        '''
        pass
    
    def nltk_clean_text(self, cleaned_text_column, text_column, is_stem=True):
        '''
            Remove stopwords, also option to use Porter stemmer.
        '''
        self.lowercase_string(text_column)
        fill_string_nulls(self.df)
        self.df[cleaned_text_column] = self.df[text_column].apply(
            self.process_text,
            stopwords=set(stopwords.words("english")) | set(self.extra_stopwords),
            is_stem=is_stem    
        )
        self.cleaned_text_column = cleaned_text_column
        self.text_column = text_column
        return self.df
    
    def train_test_split(self):
        # split the df_complaints dataframe into a test, train, and validation set with a 70/20/10 split
        self.df_train, self.df_test = train_test_split(self.df, test_size=(1-self.train_size), random_state=self.random_state)
        self.df_test, self.df_validation = train_test_split(self.df_test, test_size=(self.validation_size/(1-self.train_size)), random_state=self.random_state)
        self.df_train.reset_index(drop=True)
        self.df_test.reset_index(drop=True)
        self.df_validation.reset_index(drop=True)
    
    def vectorize(
        self,
        vectorizer_name=None,
    ):        
        basedir = os.path.normpath(
            f"{self.dataset_dir}/preprocessing/{type(self.vectorizer).__name__}/{vectorizer_name}"
        )
        # Path defaults
        train_file_path=os.path.join(basedir, f"{vectorizer_name}_train.pkl")
        validation_file_path=os.path.join(basedir, f"{vectorizer_name}_val.pkl")
        test_file_path=os.path.join(basedir, f"{vectorizer_name}_test.pkl")
        vectorizer_file_path=os.path.join(basedir, f"{vectorizer_name}_vectorizer.pkl")
        
        self.x_train_vect = pickle_io(
            lambda: self.vectorizer.fit_transform(self.df_train[self.cleaned_text_column]),
            file_path=train_file_path,
            metadata=self.vectorizer_params, rerun=self.rerun
        )()
        self.x_test_vect = pickle_io(
            lambda: self.vectorizer.transform(self.df_test[self.cleaned_text_column]),
            file_path=validation_file_path,
            metadata=self.vectorizer_params, rerun=self.rerun
        )()
        self.x_validation_vect = pickle_io(
            lambda: self.vectorizer.transform(self.df_validation[self.cleaned_text_column]),
            file_path=test_file_path,
            metadata=self.vectorizer_params, rerun=self.rerun
        )()
        self.vectorizer = pickle_io(
            lambda: self.vectorizer,
            file_path=vectorizer_file_path,
            metadata=self.vectorizer_params, rerun=self.rerun
        )()


    def preprocess(
        self, 
        cleaning_type : Union["nltk"] = "nltk",
        nrows=None
    ):
        # Read CSV
        print("Reading CSV")
        self.read_csv(self.csv_path, nrows=nrows) # this could be optimized

        # Text processing
        print("Cleaning Text")
        if cleaning_type == "nltk":
            self.df = pickle_io(
                lambda: self.nltk_clean_text(text_column=self.text_column, cleaned_text_column=f"CLEANED_{self.text_column}",
                                             is_stem=self.is_stem),
                file_path=os.path.join(self.dataset_dir, "preprocessing", self.custom_clean_name, f"{self.custom_clean_name}.pkl"),
                metadata={
                    "stopwords" : list(set(stopwords.words("english")) | set(self.extra_stopwords)),
                    "train_size" : round(self.train_size, 2),
                    "test_size" : round(1 - self.train_size, 2),
                    "validation" : round(self.validation_size, 2)
                },
                rerun=self.rerun
            )()
        
        print("Vectorizing")
        self.train_test_split()
        self.vectorize(vectorizer_name=self.custom_vectorizer_name)
        
    @staticmethod
    def process_text(text, **kwargs):
        """
        Process text by tokenizing, removing stop words, and stemming.
        Lambda function.
        """
        #print(text)
        stop_words = set(kwargs["stopwords"])

        # stopwords.words("english") + extra_stopwords
        stemmer = PorterStemmer()
        
        # make sure the text is not empty and or nan
        if not text or pd.isna(text):
            return ''


        # tokenize the text and remove stop words and stem the words
        if kwargs["is_stem"] is True:
            content_cleaned = [stemmer.stem(word) for word in word_tokenize(text.lower()) if word not in stop_words]
        else:
            content_cleaned = [word for word in word_tokenize(text.lower()) if word not in stop_words]

        # remove the punctuation from the content_cleaned list
        content_cleaned = [word for word in content_cleaned if word.isalnum()]
        
        return ' '.join(content_cleaned)

# In[2]
if __name__ == "__main__":
    p = Preprocesser(
        custom_clean_name="nltk_test",
        custom_vectorizer_name="tfidf_test",
        text_column="CDESCR",
        vectorizer=TfidfVectorizer
    )
    p.preprocess(
        nrows=10
    )
