# create a streamlit app that allows users to input a text query and get the top 5 most similar documents from the corpus using the TextClassifier class from import os
import os
import pickle
import pandas as pd
import numpy as np
import re
import os
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

class TextClassifier:
    def __init__(self, dataset_path, column_name:str, svd_n_dimensions=384,
                 ngram_range=(1,1), min_df=1, max_df=1.0, binary=False
                 ):
        # set the random state for reproducibility
        self.random_state = 42
        # create a TfidfVectorizer object
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            binary=binary
        )
        # Set Tfidf metadata
        self.num_dimensions = svd_n_dimensions
        self.n_gram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.binary = binary

        self.dataset_path = os.path.abspath(dataset_path) # normalize
        self.dataset_dir, self.dataset_basename = self.dataset_path.rsplit(
            os.sep, maxsplit=1
        )
        self.column_name = column_name
        self.column_name_cleaned = column_name + "_CLEANED"
        # create variables to store the training, test, and validation sets for class functions
        self.column_name_cleaned = None
        self.df_train = None
        self.df_test = None
        self.df_validation = None
        self.x_train_vect = None
        self.x_test_vect = None
        self.x_validation_vect = None
        self.lsa = None
        self.vectorized_train = None
        self.vectorized_test = None
        self.vectorized_validation = None

    def read_dataset(self):
        """Read the dataset from the csv file"""
        self.df = pd.read_csv(self.dataset_path)

    # create a function that will be called from a apply lambda function to process the text from a dataframe with the "CDESCR" column
    def process_text(text):
        """
        Process text by tokenizing, removing stop words, and stemming
        """
        #print(text)
        stop_words = set(stopwords.words("english"))
        stemmer = PorterStemmer()
        
        # make sure the text is not empty and or nan
        if not text or pd.isna(text):
            return []

        # tokenize the text and remove stop words and stem the words
        content_cleaned = [stemmer.stem(word) for word in word_tokenize(text.lower()) if word not in stop_words]

        # remove the punctuation from the content_cleaned list
        content_cleaned = [word for word in content_cleaned if word.isalnum()]
        
        return content_cleaned

    def load_pickle_if_paths_exist(BASE_PATH, basenames_to_check : list):
        pickles = []
        for basename_to_check in basenames_to_check:
            if os.path.exists(f"{BASE_PATH}/{basename_to_check}"):
                with open(f"{BASE_PATH}/{basename_to_check}", "rb") as f:
                    pickles.append(pickle.load(f))

        return pickles
    
    def process_dataframe(self, train_size=0.7, test_size=0.2, validation_size=0.1, subgroup=""):
        """
        Process the text in the "CDESCR" column and create a new column "CDESCR_CLEANED" with the processed text
        """
        basename = (
            f"{self.column_name}_{self.num_dimensions}",
            f"tfidf_{str(self.n_gram_range)}_min_df_{self.min_df}"
            f"max_df_{self.max_df}_binary_{self.binary}"
        ).join("_")
        # get current working directory
        cwd = os.getcwd()
        # change the Example folder to Datasets folder in the cwd path
        desired_save_path = os.path.join(self.dataset_dir, subgroup)
        # desired_save_path = cwd + "\\Datasets"
        # create a folder path to save the pickle files
        if not os.path.exists(desired_save_path):
            os.makedirs(desired_save_path)

        # check to see if there is a pickle file for the dataframe with the processed text
        if os.path.exists(desired_save_path + '//' + basename + "_df.pkl"):
            # load the pickle file
            with open(desired_save_path + '//' + basename + "_df.pkl", "rb") as f:
                self.df = pickle.load(f)
        else:
            # process the text in the "CDESCR" column and create a new column "CDESCR_CLEANED" with the processed text
            self.df[self.column_name_cleaned] = self.df[self.column_name].apply(lambda x: TextClassifier.process_text(x))
            # create a pickle file for the dataframe with the processed text
            with open(desired_save_path + '//' + basename + "_df.pkl", "wb") as f:
                pickle.dump(self.df, f)

        # split the df_complaints dataframe into a test, train, and validation set with a 70/20/10 split
        self.df_train, self.df_test = train_test_split(self.df, test_size=(1-train_size), random_state=self.random_state)
        self.df_test, self.df_validation = train_test_split(self.df_test, test_size=(validation_size/(1-train_size)), random_state=self.random_state)

        # check to see if there is a pickle file for the vectorizer
        if (
            os.path.exists(desired_save_path + '//' + basename + "_vectorizer.pkl") 
            and os.path.exists(desired_save_path + '//' + basename + "_x_train_vect.pkl") 
            and os.path.exists(desired_save_path + '//' + basename + "_x_test_vect.pkl") 
            and os.path.exists(desired_save_path + '//' + basename + "_x_validation_vect.pkl")
        ):
            # load the pickle file
            with open(desired_save_path + '//' + basename + "_vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(desired_save_path + '//' + basename + "_x_train_vect.pkl", "rb") as f:
                self.x_train_vect = pickle.load(f)
            with open(desired_save_path + '//' + basename + "_x_test_vect.pkl", "rb") as f:
                self.x_test_vect = pickle.load(f)
            with open(desired_save_path + '//' + basename + "_x_validation_vect.pkl", "rb") as f:
                self.x_validation_vect = pickle.load(f)
        else:
            # fit the vectorizer on the "CDESCR_CLEANED" column and transform the "CDESCR_CLEANED" column into a vectorized format
            # apply a lambda function to join the list of words in the "CDESCR_CLEANED" column into a string
            self.x_train_vect = self.vectorizer.fit_transform(self.df_train[self.column_name_cleaned].apply(lambda x: " ".join(x)))
            self.x_test_vect = self.vectorizer.transform(self.df_test[self.column_name_cleaned].apply(lambda x: " ".join(x)))
            self.x_validation_vect = self.vectorizer.transform(self.df_validation[self.column_name_cleaned].apply(lambda x: " ".join(x)))        
            #print(self.x_train_vect.shape, self.x_test_vect.shape, self.x_validation_vect.shape)
            # create a pickle file for the vectorizer
            with open(desired_save_path + '//' + basename + "_vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
            with open(desired_save_path + '//' + basename + "_x_train_vect.pkl", "wb") as f:
                pickle.dump(self.x_train_vect, f)
            with open(desired_save_path + '//' + basename + "_x_test_vect.pkl", "wb") as f:
                pickle.dump(self.x_test_vect, f)
            with open(desired_save_path + '//' + basename + "_x_validation_vect.pkl", "wb") as f:
                pickle.dump(self.x_validation_vect, f)

        # check to see if there is a pickle file for the lsa
        if os.path.exists(desired_save_path + '//' + basename + "_lsa.pkl") and os.path.exists(desired_save_path + '//' + basename + "_complaints_vectorized_train.pkl") and os.path.exists(desired_save_path + '//' + basename + "_complaints_vectorized_test.pkl") and os.path.exists(desired_save_path + '//' + basename + "_complaints_vectorized_validation.pkl"):
            # load the pickle file
            with open(desired_save_path + '//' + basename + "_lsa.pkl", "rb") as f:
                self.lsa = pickle.load(f)
            with open(desired_save_path + '//' + basename + "_complaints_vectorized_train.pkl", "rb") as f:
                self.complaints_vectorized_train = pickle.load(f)
            with open(desired_save_path + '//' + basename + "_complaints_vectorized_test.pkl", "rb") as f:
                self.complaints_vectorized_test = pickle.load(f)
            with open(desired_save_path + '//' + basename + "_complaints_vectorized_validation.pkl", "rb") as f:
                self.complaints_vectorized_validation = pickle.load(f)
        else:
            # perform LSA on the vectorized data to reduce the dimensionality
            self.lsa = TruncatedSVD(n_components=self.num_dimensions, random_state=self.random_state)
            self.complaints_vectorized_train = self.lsa.fit_transform(self.x_train_vect)
            self.complaints_vectorized_test = self.lsa.transform(self.x_test_vect)
            self.complaints_vectorized_validation = self.lsa.transform(self.x_validation_vect)
            
            # create a pickle file for the vectorized training data
            with open(desired_save_path + '//' + self.column_name + "_lsa.pkl", "wb") as f:
                pickle.dump(self.lsa, f)
            with open(desired_save_path + '//' + self.column_name + "_complaints_vectorized_train.pkl", "wb") as f:
                pickle.dump(self.complaints_vectorized_train, f)
            with open(desired_save_path + '//' + self.column_name + "_complaints_vectorized_test.pkl", "wb") as f:
                pickle.dump(self.complaints_vectorized_test, f)
            with open(desired_save_path + '//' + self.column_name + "_complaints_vectorized_validation.pkl", "wb") as f:
                pickle.dump(self.complaints_vectorized_validation, f)

        return self.df, self.df_train, self.df_test, self.df_validation, self.lsa, self.vectorizer
    
    # A functionalize the process of finding similar complaints to a query text
    def find_similar_complaint(self, query_text:str):
        """
        Find the most similar complaint to a query text in the training set
        """
        # process the query text
        query_text_cleaned = TextClassifier.process_text(query_text)
        # vectorize the query text
        query_vectorized = self.vectorizer.transform([" ".join(query_text_cleaned)])
        # reduce the dimensionality of the query vector
        query_vectorized_lsa = self.lsa.transform(query_vectorized)
        # find the cosine similarity between the query vector and all the complaints in the training set
        cosine_similarities_query = cosine_similarity(query_vectorized_lsa, self.complaints_vectorized_train)
        # get the index of the most similar complaint in the training set
        most_similar_index_query = cosine_similarities_query.argmax()
        print(most_similar_index_query)
        # get the most similar complaint in the training set
        most_similar_complaint_train_query = self.df_train.iloc[most_similar_index_query]
        return most_similar_complaint_train_query
    
    def find_similar(self, query_text:str, 
                                    dataset_to_check : Literal[
                                        "train", 
                                        "validation",
                                        "test"    
                                    ] = "train",
                                    similarity_fn=cosine_similarity,
                                    top=5):
        """
        Find the most similar complaint to a query text in the training set
        """
        # process the query text
        query_text_cleaned = TextClassifier.process_text(query_text)
        # vectorize the query text
        query_vectorized = self.vectorizer.transform([" ".join(query_text_cleaned)])
        # reduce the dimensionality of the query vector
        query_vectorized_lsa = self.lsa.transform(query_vectorized)
        # find the cosine similarity between the query vector and all the complaints in the training set
        
        if dataset_to_check == "train":
            cosine_similarities_query = similarity_fn(query_vectorized_lsa, self.complaints_vectorized_train)
        elif dataset_to_check == "validation":
            cosine_similarities_query = similarity_fn(query_vectorized_lsa, self.complaints_vectorized_validation)
        elif dataset_to_check == "test":
            cosine_similarities_query = similarity_fn(query_vectorized_lsa, self.complaints_vectorized_test)
    
        # get the index of the most similar complaints
        most_similar_indices_query = cosine_similarities_query.squeeze().argsort()[:-top-1:-1]

        # get the most similar complaints
        most_similar_elements = self.df_train.iloc[most_similar_indices_query]
        most_similar_elements["cosine_similarities"] = cosine_similarities_query.squeeze()[most_similar_indices_query]
        return most_similar_elements
    
    def run_training_pipeline(self, subgroup=""):
        # Load dataset in self.df
        self.read_dataset()

        # Process DataFrame and run SVD
        self.process_dataframe(subgroup=subgroup)

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