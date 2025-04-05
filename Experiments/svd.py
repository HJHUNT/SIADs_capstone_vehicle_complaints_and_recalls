# create a streamlit app that allows users to input a text query and get the top 5 most similar documents from the corpus using the TextClassifier class from import os
import json
import os
import pickle
import pandas as pd
import numpy as np
import re
import os
import sys
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from typing import Literal, Union
from preprocessing import Preprocesser
curdir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curdir)
sys.path.insert(0, parent_dir)
from helpers.pickle_decorator import pickle_io
from helpers.utilities import get_dataset_dir

class TextClassifier:
    def __init__(self, column_name:str, subdirectory, dataset_dir=None, svd_n_dimensions=384,
                 random_state=42, rerun=False
                 ):
        # set the random state for reproducibility
        if dataset_dir is None:
            self.dataset_dir = get_dataset_dir()

        self.column_name = column_name
        self.column_name_cleaned = column_name + "_CLEANED"
        self.subdirectory = subdirectory
        self.random_state = random_state
        self.num_dimensions = svd_n_dimensions
        self.desired_save_path = os.path.join(
            self.dataset_dir,
            "svd",
            self.subdirectory
        )
        self.rerun = rerun

        # create variables to store the training, test, and validation sets for class functions
        self.lsa = None

    def transform(self, vect, filename=""):
        lsa_vectors = pickle_io(
            lambda: self.lsa.transform(vect),
            file_path=os.path.join(self.desired_save_path, filename),
            metadata={"svd_dimensions" : self.num_dimensions}
        )()
        setattr(self, filename, lsa_vectors) # dynamic self attribute
        return lsa_vectors
    
    def fit_transform(self, vect, filename=""):
        self.lsa = TruncatedSVD(self.num_dimensions)

        ## Write out vectors AND the model
        self.lsa_vectorized_train = pickle_io(
            lambda: self.lsa.fit_transform(vect),
            file_path=os.path.join(self.desired_save_path, filename),
            metadata={"svd_dimensions" : self.num_dimensions}
        )()
        pickle_io(
            lambda: self.lsa,
            file_path = os.path.join(self.desired_save_path, "lsa.pkl")
        )()
        
        return self.lsa_vectorized_train
    
    def find_similar(self, df, 
        query_text:str, 
        lsa_vectors : np.array,
        vectorizer,
        text_processing_function=lambda x:x,
        keep : Literal[
            "complaint",
            "recall",
            "all"
        ] = "all",
        similarity_fn=cosine_similarity,
        top=5):
        """
            p is preprocesser object is used to process query text.
            Find the most similar complaint to a query text in the training set
        """
        # process the query text
        query_text_cleaned = text_processing_function(query_text)
        # vectorize the query text
        query_vectorized = vectorizer.transform(query_text_cleaned)
        # reduce the dimensionality of the query vector
        query_vectorized_lsa = self.lsa.transform(query_vectorized)
        # find the cosine similarity between the query vector and all the complaints in the training set
        
        if keep == "recall":
            filter_number = 0
        elif keep == "complaint":
            filter_number = 1
        else:
            filter_number = None
    
        if keep == "all":
            indices = self.df_train.index
            cosine_similarities_query = similarity_fn(query_vectorized_lsa, self.lsa_vectorized_train)
        else:
            indices = df.loc[df["IS_COMPLAINT"] == filter_number].index
            cosine_similarities_query = similarity_fn(query_vectorized_lsa, self.lsa_vectorized_train[indices])
    
        # get the indices of the most similar complaints
        most_similar_indices_query = cosine_similarities_query.squeeze().argsort()[:-top-1:-1]

        # get the most similar complaints
        most_similar_elements = df.loc[indices].iloc[most_similar_indices_query]
        most_similar_elements["similarity"] = cosine_similarities_query.squeeze()[most_similar_indices_query]
        return most_similar_elements

        # # A functionalize the process of finding similar complaints to a query text
    # def find_similar_complaint(self, query_text:str):
    #     """
    #     Find the most similar complaint to a query text in the training set
    #     """
    #     # process the query text
    #     query_text_cleaned = TextClassifier.process_text(query_text)
    #     # vectorize the query text
    #     query_vectorized = self.vectorizer.transform([" ".join(query_text_cleaned)])
    #     # reduce the dimensionality of the query vector
    #     query_vectorized_lsa = self.lsa.transform(query_vectorized)
    #     # find the cosine similarity between the query vector and all the complaints in the training set
    #     cosine_similarities_query = cosine_similarity(query_vectorized_lsa, self.lsa_vectorized_train)
    #     # get the index of the most similar complaint in the training set
    #     most_similar_index_query = cosine_similarities_query.argmax()
    #     print(most_similar_index_query)
    #     # get the most similar complaint in the training set
    #     most_similar_complaint_train_query = self.df_train.iloc[most_similar_index_query]
    #     return most_similar_complaint_train_query
    
    # def run_training_pipeline(self, subgroup=""):
    #     # Load dataset in self.df
    #     self.read_dataset()

    #     # Process DataFrame and run SVD
    #     self.process_dataframe(subgroup=subgroup)

# run the below code if main script
# if __name__ == "__main__":
#     # https://www.nhtsa.gov/nhtsa-datasets-and-apis#recalls
#     # read in C:\Repo\SIADs_Audio_Text_SRS\Example\COMPLAINTS_RECEIVED_2025-2025.txt into a pandas dataframe, where the columns are RCL
#     df_complaints = pd.read_csv("C:\\Repo\\SIADs_Audio_Text_SRS\\Datasets\\COMPLAINTS_RECEIVED_2025-2025.txt", sep='\t', header=None, index_col=0)
#     df_complaints.columns = ['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'CRASH', 'FAILDATE', 'FIRE', 'INJURED', 'DEATHS', 'COMPDESC', 'CITY', 'STATE', 'VIN', 'DATEA', 'LDATE', 'MILES', 'OCCURENCES', 'CDESCR', 'CMPL_TYPE', 'POLICE_RPT_YN', 'PURCH_DT', 'ORIG_OWNER_YN', 'ANTI_BRAKES_YN', 'CRUISE_CONT_YN', 'NUM_CYLS', 'DRIVE_TRAIN', 'FUEL_SYS', 'FUEL_TYPE',
#                 'TRANS_TYPE', 'VEH_SPEED', 'DOT', 'TIRE_SIZE', 'LOC_OF_TIRE', 'TIRE_FAIL_TYPE', 'ORIG_EQUIP_YN', 'MANUF_DT', 'SEAT_TYPE', 'RESTRAINT_TYPE', 'DEALER_NAME', 'DEALER_TEL', 'DEALER_CITY', 'DEALER_STATE', 'DEALER_ZIP', 'PROD_TYPE', 'REPAIRED_YN', 'MEDICAL_ATTN', 'VEHICLES_TOWED_YN']

#     # create a list of unique manufacturers in the "MFR_NAME" column
#     list_of_manufacturers = df_complaints["MFR_NAME"].unique()

#     # testing out the functionilzed code

#     # call the TextClassifier class and create an instance of it as text_classifier
#     # pass in the df_complaints dataframe and the "CDESCR" column
#     text_classifier = TextClassifier(df_complaints, "CDESCR")
#     # process the text in the "CDESCR" column
#     text_classifier.process_dataframe()

#     # use one of the complaints in the test set as a query to find the most similar complaint in the training set
#     #complaint_test_query = text_classifier.df_test["CDESCR"].iloc[5]
#     #complaint_test_query = text_classifier.df_test["CDESCR"].iloc[4]
#     #complaint_test_query = "Car won't start and makes a clicking noise"
#     complaint_test_query = "Battery dies after a few days of not driving the car"

#     print(complaint_test_query)
#     # find the most similar complaint to the complaint test
#     most_similar_complaint = text_classifier.find_similar_complaint(complaint_test_query)
#     # print the most similar complaint with the below columns
#     print(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CDESCR"]])
#     print(most_similar_complaint["CDESCR"])