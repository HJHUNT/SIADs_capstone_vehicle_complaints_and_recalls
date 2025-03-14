import os
import pickle
import time
from tqdm import tqdm
from nltk import download
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# check to see if the nltk data has been downloaded in the virtual environment
if not os.path.exists(os.path.join(os.path.expanduser("~"), "nltk_data")):
    # download the nltk data
    download('stopwords')
    download('punkt_tab')

class TextClassifier:
    def __init__(self, df, column_name:str):
        # set the random state for reproducibility
        self.random_state = 42
        # create a TfidfVectorizer object
        self.vectorizer = TfidfVectorizer()
        self.classifier_kmeans = KMeans(n_clusters=150, random_state=self.random_state, n_init=100, max_iter=1000)
        self.classifier_RFC = RandomForestClassifier(random_state=self.random_state)
        # set the number of dimensions to reduce the vectorized data to
        self.num_dimensions = 384   # medium number of dimensions for better performance
        self.df = df
        self.column_name = column_name
        self.column_name_cleaned = column_name + "_CLEANED"
        # create variables to store the training, test, and validation sets for class functions
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
        # save path off of current working directory
        self.desired_save_path = os.getcwd() + "\\Datasets"

    # create a function that will be called from a apply lambda function to process the text from a dataframe with the "CDESCR" column
    @staticmethod
    def process_text(text):
        """
        Process text by tokenizing, removing stop words, and stemming
        """
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
    
    def process_dataframe(self, train_size=0.7, test_size=0.2, validation_size=0.1):
        """
        Process the text in the "CDESCR" column and create a new column "CDESCR_CLEANED" with the processed text
        """
        start_time = time.time()
        total_steps = 10
        progress_bar = tqdm(total=total_steps, desc="Processing DataFrame", unit="step")

        # create a folder path to save the pickle files
        if not os.path.exists(self.desired_save_path):
            os.makedirs(self.desired_save_path)

        progress_bar.update(1)

        # check to see if there is a pickle file for the dataframe with the processed text
        if os.path.exists(self.desired_save_path + '//' + self.column_name + "_df.pkl"):
            # load the pickle file
            with open(self.desired_save_path + '//' + self.column_name + "_df.pkl", "rb") as f:
                self.df = pickle.load(f)
        else:
            # process the text in the "CDESCR" column and create a new column "CDESCR_CLEANED" with the processed text
            self.df[self.column_name_cleaned] = self.df[self.column_name].apply(lambda x: TextClassifier.process_text(x))
            # create a pickle file for the dataframe with the processed text
            with open(self.desired_save_path + '//' + self.column_name + "_df.pkl", "wb") as f:
                pickle.dump(self.df, f)

        progress_bar.update(1)

        # split the df_complaints dataframe into a test, train, and validation set with a 70/20/10 split
        self.df_train, self.df_test = train_test_split(self.df, test_size=(1-train_size), random_state=self.random_state)
        self.df_test, self.df_validation = train_test_split(self.df_test, test_size=(validation_size/(1-train_size)), random_state=self.random_state)

        progress_bar.update(1)

        # check to see if there is a pickle file for the vectorizer
        if os.path.exists(self.desired_save_path + '//' + self.column_name + "_vectorizer.pkl") and os.path.exists(self.desired_save_path + '//' + self.column_name + "_x_train_vect.pkl") and os.path.exists(self.desired_save_path + '//' + self.column_name + "_x_test_vect.pkl") and os.path.exists(self.desired_save_path + '//' + self.column_name + "_x_validation_vect.pkl"):
            # load the pickle file
            with open(self.desired_save_path + '//' + self.column_name + "_vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(self.desired_save_path + '//' + self.column_name + "_x_train_vect.pkl", "rb") as f:
                self.x_train_vect = pickle.load(f)
            with open(self.desired_save_path + '//' + self.column_name + "_x_test_vect.pkl", "rb") as f:
                self.x_test_vect = pickle.load(f)
            with open(self.desired_save_path + '//' + self.column_name + "_x_validation_vect.pkl", "rb") as f:
                self.x_validation_vect = pickle.load(f)
        else:
            # fit the vectorizer on the "CDESCR_CLEANED" column and transform the "CDESCR_CLEANED" column into a vectorized format
            # apply a lambda function to join the list of words in the "CDESCR_CLEANED" column into a string
            self.x_train_vect = self.vectorizer.fit_transform(self.df_train[self.column_name_cleaned].apply(lambda x: " ".join(x)))
            self.x_test_vect = self.vectorizer.transform(self.df_test[self.column_name_cleaned].apply(lambda x: " ".join(x)))
            self.x_validation_vect = self.vectorizer.transform(self.df_validation[self.column_name_cleaned].apply(lambda x: " ".join(x)))        
            # create a pickle file for the vectorizer
            with open(self.desired_save_path + '//' + self.column_name + "_vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
            with open(self.desired_save_path + '//' + self.column_name + "_x_train_vect.pkl", "wb") as f:
                pickle.dump(self.x_train_vect, f)
            with open(self.desired_save_path + '//' + self.column_name + "_x_test_vect.pkl", "wb") as f:
                pickle.dump(self.x_test_vect, f)
            with open(self.desired_save_path + '//' + self.column_name + "_x_validation_vect.pkl", "wb") as f:
                pickle.dump(self.x_validation_vect, f)

        progress_bar.update(1)

        # check to see if there is a pickle file for the lsa
        if os.path.exists(self.desired_save_path + '//' + self.column_name + "_lsa.pkl") and os.path.exists(self.desired_save_path + '//' + self.column_name + "_complaints_vectorized_train.pkl") and os.path.exists(self.desired_save_path + '//' + self.column_name + "_complaints_vectorized_test.pkl") and os.path.exists(self.desired_save_path + '//' + self.column_name + "_complaints_vectorized_validation.pkl"):
            # load the pickle file
            with open(self.desired_save_path + '//' + self.column_name + "_lsa.pkl", "rb") as f:
                self.lsa = pickle.load(f)
            with open(self.desired_save_path + '//' + self.column_name + "_complaints_vectorized_train.pkl", "rb") as f:
                self.complaints_vectorized_train = pickle.load(f)
            with open(self.desired_save_path + '//' + self.column_name + "_complaints_vectorized_test.pkl", "rb") as f:
                self.complaints_vectorized_test = pickle.load(f)
            with open(self.desired_save_path + '//' + self.column_name + "_complaints_vectorized_validation.pkl", "rb") as f:
                self.complaints_vectorized_validation = pickle.load(f)
        else:
            # perform LSA on the vectorized data to reduce the dimensionality
            self.lsa = TruncatedSVD(n_components=self.num_dimensions, random_state=self.random_state)
            self.complaints_vectorized_train = self.lsa.fit_transform(self.x_train_vect)
            self.complaints_vectorized_test = self.lsa.transform(self.x_test_vect)
            self.complaints_vectorized_validation = self.lsa.transform(self.x_validation_vect)
            
            # create a pickle file for the vectorized training data
            with open(self.desired_save_path + '//' + self.column_name + "_lsa.pkl", "wb") as f:
                pickle.dump(self.lsa, f)
            with open(self.desired_save_path + '//' + self.column_name + "_complaints_vectorized_train.pkl", "wb") as f:
                pickle.dump(self.complaints_vectorized_train, f)
            with open(self.desired_save_path + '//' + self.column_name + "_complaints_vectorized_test.pkl", "wb") as f:
                pickle.dump(self.complaints_vectorized_test, f)
            with open(self.desired_save_path + '//' + self.column_name + "_complaints_vectorized_validation.pkl", "wb") as f:
                pickle.dump(self.complaints_vectorized_validation, f)

        progress_bar.update(1)
        progress_bar.close()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds.")

        return self.df, self.df_train, self.df_test, self.df_validation, self.lsa, self.vectorizer
    
    # A functionalized process of finding similar complaints to a query text
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
    
    # A functionalized process for fitting a kmeans model to the training data
    def fit_kmeans(self,col_name:str):
        """
        Fit a KMeans model to the training data
        """
        # check to see if there is a pickle file for the classifier
        if os.path.exists(self.desired_save_path + '//' + self.column_name + "_classifier_kmeans.pkl") and os.path.exists(self.desired_save_path + '//' + self.column_name + "_classifier_RFC.pkl"):
            # load the pickle file
            with open(self.desired_save_path + '//' + self.column_name + "_classifier_kmeans.pkl", "rb") as f:
                self.classifier_kmeans = pickle.load(f)
            with open(self.desired_save_path + '//' + self.column_name + "_classifier_RFC.pkl", "rb") as f:
                self.classifier_RFC = pickle.load(f)
        else:
            # fit the KMeans model to the training data
            self.classifier_kmeans.fit(self.complaints_vectorized_train, self.df_train[col_name])
            # fit the Random Forest Classifier model to the training data
            self.classifier_RFC.fit(self.complaints_vectorized_train, self.df_train[col_name])
            # create a pickle file for the classifier
            with open(self.desired_save_path + '//' + self.column_name + "_classifier_kmeans.pkl", "wb") as f:
                pickle.dump(self.classifier_kmeans, f)
            with open(self.desired_save_path + '//' + self.column_name + "_classifier_RFC.pkl", "wb") as f:
                pickle.dump(self.classifier_RFC, f)
        return self.classifier_kmeans

    # predict the cluster off the query text
    def predict_cluster(self, query_text:str):
        """
        Predict the cluster of a query text
        """
        # process the query text
        query_text_cleaned = TextClassifier.process_text(query_text)
        # vectorize the query text
        query_vectorized = self.vectorizer.transform([" ".join(query_text_cleaned)])
        # reduce the dimensionality of the query vector
        query_vectorized_lsa = self.lsa.transform(query_vectorized)
        # predict the kmeans cluster of the query text
        cluster_kmeans = self.classifier_kmeans.predict(query_vectorized_lsa)
        # predict the Random Forest Classifier cluster of the query text
        cluster_RFC = self.classifier_RFC.predict(query_vectorized_lsa)
        # convert the kmenas predicted cluster value back to the original value from the "COMPDESC" column
        cluster_kmeans_pred = LabelEncoder().fit(self.df_train["COMPDESC"]).inverse_transform(cluster_kmeans)
        # convert the Random Forest Classifier predicted cluster value back to the original value from the "COMPDESC" column
        cluster_RFC_pred = LabelEncoder().fit(self.df_train["COMPDESC"]).inverse_transform(cluster_RFC)
        # return the predicted cluster
        return cluster_kmeans_pred, cluster_RFC_pred
    
# run the below code if main script
if __name__ == "__main__":
    # https://www.nhtsa.gov/nhtsa-datasets-and-apis#recalls
    # read in C:\Repo\SIADs_Audio_Text_SRS\Example\COMPLAINTS_RECEIVED_2025-2025.txt into a pandas dataframe, where the columns are RCL
    df_complaints = pd.read_csv("C:\\Repo\\SIADs_Audio_Text_SRS\\Datasets\\COMPLAINTS_RECEIVED_2025-2025.txt", sep='\t', header=None, index_col=0)
    df_complaints.columns = ['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'CRASH', 'FAILDATE', 'FIRE', 'INJURED', 'DEATHS', 'COMPDESC', 'CITY', 'STATE', 'VIN', 'DATEA', 'LDATE', 'MILES', 'OCCURENCES', 'CDESCR', 'CMPL_TYPE', 'POLICE_RPT_YN', 'PURCH_DT', 'ORIG_OWNER_YN', 'ANTI_BRAKES_YN', 'CRUISE_CONT_YN', 'NUM_CYLS', 'DRIVE_TRAIN', 'FUEL_SYS', 'FUEL_TYPE',
                'TRANS_TYPE', 'VEH_SPEED', 'DOT', 'TIRE_SIZE', 'LOC_OF_TIRE', 'TIRE_FAIL_TYPE', 'ORIG_EQUIP_YN', 'MANUF_DT', 'SEAT_TYPE', 'RESTRAINT_TYPE', 'DEALER_NAME', 'DEALER_TEL', 'DEALER_CITY', 'DEALER_STATE', 'DEALER_ZIP', 'PROD_TYPE', 'REPAIRED_YN', 'MEDICAL_ATTN', 'VEHICLES_TOWED_YN']

    # state encode the COMPDESC values and create a new column in the dataframe called COMPDESC_StateEncoded
    df_complaints["COMPDESC_StateEncoded"] = LabelEncoder().fit_transform(df_complaints["COMPDESC"])

    # create a list of unique manufacturers in the "MFR_NAME" column
    list_of_manufacturers = df_complaints["MFR_NAME"].unique()

    # testing out the functionilzed code

    # call the TextClassifier class and create an instance of it as text_classifier
    # pass in the df_complaints dataframe and the "CDESCR" column
    text_classifier = TextClassifier(df_complaints, "CDESCR")
    # process the text in the "CDESCR" column
    text_classifier.process_dataframe()
    # fit a KMeans model to the training data
    text_classifier.fit_kmeans("COMPDESC_StateEncoded")

    # use one of the complaints in the test set as a query to find the most similar complaint in the training set
    #complaint_test_query = "Battery dies after a few days of not driving the car"
    #complaint_test_query = "loss of power steering"
    complaint_test_query = "Wheel sounds like it is scraping against something when driving"

    print(complaint_test_query)
    # find the most similar complaint to the complaint test
    most_similar_complaint = text_classifier.find_similar_complaint(complaint_test_query)
    # print the most similar complaint with the below columns
    print(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CDESCR", "COMPDESC"]])
    print(most_similar_complaint["CDESCR"])
    # predict the cluster of the query text
    cluster_kmeans_pred, cluster_RFC_pred = text_classifier.predict_cluster(complaint_test_query)
    print(cluster_kmeans_pred)
    print(cluster_RFC_pred)

    # plot the clusters of the training data for the KMeans model
    # create a dataframe with the cluster predictions
    df_train_clustered = text_classifier.df_train.copy()
    df_train_clustered["Cluster"] = text_classifier.classifier_kmeans.labels_
    # plot the clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x="COMPDESC_StateEncoded", y="Cluster", data=df_train_clustered, palette="viridis")
    # plot the query text in the plot as a red, with a legend with the predicted cluster
    plt.scatter(x=most_similar_complaint["COMPDESC_StateEncoded"], y=cluster_kmeans_pred, color="red", label="Query Text")
    plt.legend("Query Text")
    plt.title("Clusters of the Training Data")
    plt.show()
    plt.close()
    