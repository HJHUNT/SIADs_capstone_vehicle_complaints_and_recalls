import os
import pickle
import time
from tqdm import tqdm
from nltk import download
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
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import sys
from Experiments.preprocessing import Preprocesser
from Experiments.classifier import Classifier
from sklearn.linear_model import SGDClassifier


# check to see if the nltk data has been downloaded in the virtual environment
if not os.path.exists(os.path.join(os.path.expanduser("~"), "nltk_data")):
    # download the nltk data
    download("stopwords")
    download("punkt_tab")

class TextClassifier:
    def __init__(self, df, column_name: str="CDESCR"):
        # set the number of dimensions to reduce the vectorized data to
        self.num_dimensions = 384  # medium number of dimensions for better performance
        self.df = df
        self.compdesc = "COMPDESC"
        self.cdescr = "CDESCR"
        self.compdesc_state_encoded = self.compdesc + "_StateEncoded"
        self.compdesc_condensed = self.compdesc + "_CONDENSED"
        self.compdesc_condensed_state_encoded = self.compdesc_condensed + "_StateEncoded"

        # state encode the COMPDESC values and create a new column in the dataframe called COMPDESC_StateEncoded
        self.compdesc_encoder = LabelEncoder().fit_transform(self.df[self.compdesc])
        # create a new column in the dataframe called COMPDESC_StateEncoded
        self.df[self.compdesc_state_encoded] = LabelEncoder().fit_transform(self.df[self.compdesc])

        # call the condense_component_description function to condense the component description in the dataframe by removing any text after a colon or slash
        self.compdesc_list_condensed, self.compdesc_dict = self.condense_component_description(self.df, self.compdesc)
        # use the compdesc_dict to look up "COMPDESC" against the keys of the dict and assign the value to a new column in the dataframe called "COMPDESC_CONDENSED"
        self.df[self.compdesc_condensed] = self.df[self.compdesc].apply(lambda x: self.compdesc_dict.get(x))
        # state encode the COMPDESC values and create a new column in the dataframe called COMPDESC_StateEncoded
        self.label_condensed_encoder = LabelEncoder().fit(self.df[self.compdesc_condensed])
        self.df[self.compdesc_condensed_state_encoded] = LabelEncoder().fit_transform(self.df[self.compdesc_condensed])
        # drop duplicates from the dataframe based on the "COMPDESC" and "CDESCR" columns
        self.df.drop_duplicates([self.compdesc_condensed, self.cdescr], inplace=True)
        print("Dataframe shape after dropping duplicates: ", self.df.shape)
        print("Unique values in the condensed component description: ", len(self.df[self.compdesc_condensed].unique()))

        # model parameters
        # set the random state for reproducibility
        self.random_state = 42
        # create a TfidfVectorizer object
        self.vectorizer = TfidfVectorizer()
        self.desired_clusters = len(self.df[self.compdesc_condensed].unique())
        self.classifier_kmeans = KMeans(n_clusters=self.desired_clusters, random_state=self.random_state, n_init=100, max_iter=1000)
        self.classifier_RFC = RandomForestClassifier(random_state=self.random_state)

        # # create a pickle file for the label encoder
        self.column_name = column_name
        self.column_name_cleaned = column_name + "_CLEANED"
        self.column_name_cleaned_vect = column_name + "_CLEANED_VECT"
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
        content_cleaned = [
            stemmer.stem(word)
            for word in word_tokenize(text.lower())
            if word not in stop_words
        ]


        # remove the punctuation from the content_cleaned list
        content_cleaned = [word for word in content_cleaned if word.isalnum()]


        return content_cleaned


    def process_dataframe(self, train_size=0.7, test_size=0.2, validation_size=0.1):
        """
        Process the text in the "CDESCR" column and create a new column "CDESCR_CLEANED" with the processed text
        """
        start_time = time.time()
        total_steps = 5
        progress_bar = tqdm(total=total_steps, desc="Processing DataFrame", unit="step")

        # create a folder path to save the pickle files
        if not os.path.exists(self.desired_save_path):
            os.makedirs(self.desired_save_path)

        # check to see if there is a pickle file for the dataframe with the processed text
        if os.path.exists(self.desired_save_path + "//" + self.column_name + "_df.pkl"):
            # load the pickle file
            with open(self.desired_save_path + "//" + self.column_name + "_df.pkl", "rb") as f:
                self.df = pickle.load(f)
        else:
            # process the text in the "CDESCR" column and create a new column "CDESCR_CLEANED" with the processed text
            self.df[self.column_name_cleaned] = self.df[self.column_name].apply(lambda x: TextClassifier.process_text(x))
            # create a pickle file for the dataframe with the processed text
            with open(self.desired_save_path + "//" + self.column_name + "_df.pkl", "wb") as f:
                pickle.dump(self.df, f)

        progress_bar.update(1)

        self.df_train, self.df_test = train_test_split(self.df, test_size=(1 - train_size), random_state=self.random_state)
        self.df_test, self.df_validation = train_test_split(self.df_test, test_size=(validation_size / (1 - train_size)), random_state=self.random_state)

        progress_bar.update(1)

        # check to see if there is a pickle file for the vectorizer
        
        if (os.path.exists(self.desired_save_path + "//" + self.column_name + "_vectorizer.pkl") and os.path.exists(self.desired_save_path + "//" + self.column_name + "_x_train_vect.pkl") and os.path.exists(self.desired_save_path + "//" + self.column_name + "_x_test_vect.pkl") and os.path.exists(self.desired_save_path + "//" + self.column_name + "_x_validation_vect.pkl")):
            # load the pickle file
            with open(self.desired_save_path + "//" + self.column_name + "_vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(self.desired_save_path + "//" + self.column_name + "_x_train_vect.pkl", "rb") as f:
                self.x_train_vect = pickle.load(f)
            with open(self.desired_save_path + "//" + self.column_name + "_x_test_vect.pkl", "rb") as f:
                self.x_test_vect = pickle.load(f)
            with open(self.desired_save_path + "//" + self.column_name + "_x_validation_vect.pkl", "rb") as f:
                self.x_validation_vect = pickle.load(f)
        else:
            # fit the vectorizer on the "CDESCR_CLEANED" column and transform the "CDESCR_CLEANED" column into a vectorized format
            # apply a lambda function to join the list of words in the "CDESCR_CLEANED" column into a string
            self.x_train_vect = self.vectorizer.fit_transform(self.df_train[self.column_name_cleaned].apply(lambda x: " ".join(x)))
            self.x_test_vect = self.vectorizer.transform(self.df_test[self.column_name_cleaned].apply(lambda x: " ".join(x)))
            self.x_validation_vect = self.vectorizer.transform(self.df_validation[self.column_name_cleaned].apply(lambda x: " ".join(x)))
            # create a pickle file for the vectorizer
            with open(self.desired_save_path + "//" + self.column_name + "_vectorizer.pkl", "wb") as f:
                pickle.dump(self.vectorizer, f)
            with open(self.desired_save_path + "//" + self.column_name + "_x_train_vect.pkl", "wb") as f:
                pickle.dump(self.x_train_vect, f)
            with open(self.desired_save_path + "//" + self.column_name + "_x_test_vect.pkl", "wb") as f:
                pickle.dump(self.x_test_vect, f)
            with open(self.desired_save_path + "//" + self.column_name + "_x_validation_vect.pkl", "wb") as f:
                pickle.dump(self.x_validation_vect, f)

        progress_bar.update(1)

        # check to see if there is a pickle file for the lsa
        if (os.path.exists(self.desired_save_path + "//" + self.column_name + "_lsa.pkl") and os.path.exists(self.desired_save_path + "//" + self.column_name + "_complaints_vectorized_train.pkl") and os.path.exists(self.desired_save_path + "//" + self.column_name + "_complaints_vectorized_test.pkl") and os.path.exists(self.desired_save_path + "//" + self.column_name + "_complaints_vectorized_validation.pkl")):
            # load the pickle file
            with open(self.desired_save_path + "//" + self.column_name + "_lsa.pkl", "rb") as f:
                self.lsa = pickle.load(f)
            with open(self.desired_save_path + "//" + self.column_name + "_complaints_vectorized_train.pkl", "rb") as f:
                self.complaints_vectorized_train = pickle.load(f)
            with open(self.desired_save_path + "//" + self.column_name + "_complaints_vectorized_test.pkl", "rb") as f:
                self.complaints_vectorized_test = pickle.load(f)
            with open(self.desired_save_path + "//" + self.column_name + "_complaints_vectorized_validation.pkl", "rb") as f:
                self.complaints_vectorized_validation = pickle.load(f)
        else:
            # perform LSA on the vectorized data to reduce the dimensionality
            self.lsa = TruncatedSVD(n_components=self.num_dimensions, random_state=self.random_state)
            self.complaints_vectorized_train = self.lsa.fit_transform(self.x_train_vect)
            self.complaints_vectorized_test = self.lsa.transform(self.x_test_vect)
            self.complaints_vectorized_validation = self.lsa.transform(self.x_validation_vect)
            # create a pickle file for the vectorized training data
            with open(self.desired_save_path + "//" + self.column_name + "_lsa.pkl", "wb") as f:
                pickle.dump(self.lsa, f)
            with open(self.desired_save_path + "//" + self.column_name + "_complaints_vectorized_train.pkl", "wb") as f:
                pickle.dump(self.complaints_vectorized_train, f)
            with open(self.desired_save_path + "//" + self.column_name + "_complaints_vectorized_test.pkl", "wb") as f:
                pickle.dump(self.complaints_vectorized_test, f)
            with open(self.desired_save_path + "//" + self.column_name + "_complaints_vectorized_validation.pkl", "wb") as f:
                pickle.dump(self.complaints_vectorized_validation, f)

        progress_bar.update(1)

        self.df_train[self.column_name_cleaned_vect] = self.complaints_vectorized_train.tolist()

        
        self.fit_kmeans(self.compdesc_condensed_state_encoded)
        progress_bar.update(1) 

        progress_bar.close()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds.")

        return (self.df, self.df_train, self.df_test, self.df_validation, self.lsa, self.vectorizer)

    # A functionalized process of finding similar complaints to a query text
    def find_similar_complaint(self, query_text: str, top_n=1):
        """
        Find the most similar complaint to a query text in the training set
        """
        self.rank_count = top_n
        # process the query text
        query_text_cleaned = TextClassifier.process_text(query_text)
        # vectorize the query text
        query_vectorized = self.vectorizer.transform([" ".join(query_text_cleaned)])
        # reduce the dimensionality of the query vector
        query_vectorized_lsa = self.lsa.transform(query_vectorized)
        # find the cosine similarity between the query vector and all the complaints in the training set
        cosine_similarities_query = cosine_similarity(query_vectorized_lsa, self.complaints_vectorized_train)
        # get the index of the most similar complaint in the training set
        #most_similar_index_query = cosine_similarities_query.argmax()
        most_similar_index_query = cosine_similarities_query.squeeze().argsort()[:-top_n-1:-1]
        print(most_similar_index_query)
        # get the most similar complaint in the training set
        most_similar_complaint_train_query = self.df_train.iloc[most_similar_index_query]
        # keep track of the top to decrease the rank count of the most similar complaints add this as a column to the datafram
        most_similar_complaint_train_query["rank"] = range(1, top_n + 1)
        return most_similar_complaint_train_query

    # A functionalized process for fitting a kmeans model to the training data
    def fit_kmeans(self, col_name: str):
        """
        Fit a KMeans model to the training data
        """
        # check to see if there is a pickle file for the classifier
        if os.path.exists(self.desired_save_path + "//" + self.column_name + "_classifier_kmeans.pkl") and os.path.exists(self.desired_save_path + "//" + self.column_name + "_classifier_RFC.pkl"):
            # load the pickle file
            with open(self.desired_save_path + "//" + self.column_name + "_classifier_kmeans.pkl", "rb") as f:
                self.classifier_kmeans = pickle.load(f)
            with open(self.desired_save_path + "//" + self.column_name + "_classifier_RFC.pkl","rb") as f:
                self.classifier_RFC = pickle.load(f)
        else:
            # fit the KMeans model to the training data
            self.classifier_kmeans.fit(self.complaints_vectorized_train, self.df_train[col_name])
            # fit the Random Forest Classifier model to the training data
            self.classifier_RFC.fit(self.complaints_vectorized_train, self.df_train[col_name])
            # create a pickle file for the classifier
            with open(self.desired_save_path + "//" + self.column_name + "_classifier_kmeans.pkl", "wb") as f:
                pickle.dump(self.classifier_kmeans, f)
            with open(self.desired_save_path + "//" + self.column_name + "_classifier_RFC.pkl", "wb") as f:
                pickle.dump(self.classifier_RFC, f)
        return self.classifier_kmeans

    # predict the cluster off the query text
    def predict_cluster(self, query_text: str):
        """
        Predict the cluster of a query text
        """
        self.query_text = query_text
        # process the query text
        query_text_cleaned = TextClassifier.process_text(query_text)
        # vectorize the query text
        self.query_vectorized = self.vectorizer.transform([" ".join(query_text_cleaned)])
        # reduce the dimensionality of the query vector
        self.query_vectorized_lsa = self.lsa.transform(self.query_vectorized)
        # predict the kmeans cluster of the query text
        cluster_kmeans = self.classifier_kmeans.predict(self.query_vectorized_lsa)
        # predict the Random Forest Classifier cluster of the query text
        cluster_RFC = self.classifier_RFC.predict(self.query_vectorized_lsa)
        # convert the kmenas predicted cluster value back to the original value from the "COMPDESC" column
        #self.cluster_kmeans_pred = self.label_encoder.inverse_transform(cluster_kmeans)
        self.cluster_kmeans_pred = self.label_condensed_encoder.inverse_transform(cluster_kmeans)
        
        #(LabelEncoder().fit(self.df_train["COMPDESC"]).inverse_transform(cluster_kmeans))
        # convert the Random Forest Classifier predicted cluster value back to the original value from the "COMPDESC" column
        #self.cluster_RFC_pred = self.label_encoder.inverse_transform(cluster_RFC)
        self.cluster_RFC_pred = self.label_condensed_encoder.inverse_transform(cluster_RFC)
        #(LabelEncoder().fit(self.df_train["COMPDESC"]).inverse_transform(cluster_RFC))
        # return the predicted cluster
        return self.cluster_kmeans_pred, self.cluster_RFC_pred, self.query_vectorized_lsa
        
    # Function to plot clusters
    def plot_clusters(self, labels, title, query_vectorized,fig, ax, most_similar_complaint_df):
        '''
        Plot clusters using matplotlib
        '''
        # Reduce dimensions of the vetrorized text to 2 for visualization
        X_pca = PCA(n_components=2).fit_transform(self.complaints_vectorized_train)
        # Reduce dimensions of the vetrorized text for the query text for visualization
        #X_pca_query = PCA(n_components=2).fit_transform(query_vectorized)
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=2, label="Training Text in Vectorized Space")
        # plot blue dots for the most similar complaints to the query text
        #ax.scatter(most_similar_complaint_df['CDESCR_CLEANED'][:, 0], most_similar_complaint_df['CDESCR_CLEANED'][:, 1], color="blue", label="Most Similar Complaints to Query Text")
        sns.scatterplot(x=most_similar_complaint_df['CDESCR_CLEANED_VECT'].apply(lambda x: x[0]), y=most_similar_complaint_df['CDESCR_CLEANED_VECT'].apply(lambda x: x[1]), color="blue", label="Most Similar Complaints to Query Text")
        # print a red spot for where the query text is
        #plt.scatter(X_pca_query[:, 0], X_pca_query[:, 1], color="red", label="Query Text")
        ax.scatter(query_vectorized[:, 0], query_vectorized[:, 1], color="red", label="Query Text in Vectorized Space")
        ax.set_title(title)
        ax.set_xlabel('PCA Reduced Dementionality of vectorized complaints text')
        ax.set_ylabel('PCA Reduced Dementionality of vectorized complaints text')
        ax.legend()
        #plt.show()
        return fig
    
        # Function to plot clusters using Altair
    def plot_clusters_alt(self, labels, title, query_vectorized, most_similar_complaint_df):
        # Reduce dimensions of the vectorized text to 2 for visualization
        pca_train_data = PCA(n_components=2).fit_transform(self.complaints_vectorized_train)
        #query_pca = PCA(n_components=2).fit_transform(query_vectorized.toarray())

        self.df_train['x_pca_train_data'] = pca_train_data[:, 0]
        self.df_train['y_pca_train_data'] = pca_train_data[:, 1]
        self.df_train['labels_num'] = labels
        #self.df_train['labels_words'] = self.label_encoder.inverse_transform(labels)
        self.df_train['labels_words'] = self.label_condensed_encoder.inverse_transform(labels)
        # copy the self.df_train to a new dataframe called filtered_df and only keep entries that equal the predicted cluster
        cluster_df = self.df_train.copy()
        no_cluster_df = self.df_train.copy()
        if "KMeans" in title:
            cluster_df = cluster_df[cluster_df['labels_words'] == self.cluster_kmeans_pred[0]]
            no_cluster_df = no_cluster_df[no_cluster_df['labels_words'] != self.cluster_kmeans_pred[0]]
        elif "Random Forest Classifier" in title:
            cluster_df = cluster_df[cluster_df['labels_words'] == self.cluster_RFC_pred[0]]
            no_cluster_df = no_cluster_df[no_cluster_df['labels_words'] != self.cluster_RFC_pred[0]]
        else:
            return None
        
        title = "Vector Space Graph"

        temp_df = pd.DataFrame()
        temp_df['query_vectorized_x'] = query_vectorized[:, 0]
        temp_df['query_vectorized_y'] = query_vectorized[:, 1]
        temp_df['Query Text'] = self.query_text
        # add the query text to the dataframe

        most_similar_complaint_df['x'] = most_similar_complaint_df['CDESCR_CLEANED_VECT'].apply(lambda x: x[0])
        most_similar_complaint_df['y'] = most_similar_complaint_df['CDESCR_CLEANED_VECT'].apply(lambda x: x[1])

        #

        # FF0000 is red
        most_sim = alt.Chart(most_similar_complaint_df).mark_point(size=125).transform_calculate(color='"Most Similar Complaints"').encode(
            x=alt.X('x', axis=None),
            y=alt.Y('y', axis=None),
            color=alt.Color("color:N", scale=alt.Scale(range=["#fd7f6f"]), legend=alt.Legend(title="Document Search", symbolLimit=0, titleFontSize=10, labelFontSize=10)),
            fill=alt.value("#fd7f6f"),
            #color=alt.Color(legend=alt.Legend(title="Most Similar Complaints to Query Text", symbolLimit=0)),
            tooltip= ["NHTSAs ID", "MANUFACTURER", "MAKE", "MODEL", "YEAR", "COMPONENT DESCRIPTION","ISSUE TYPE", 'rank']
        )

        # create a Altaire chart for the query text "#0bb4ff" is light blue
        query = alt.Chart(temp_df).mark_point(size=125).transform_calculate(color='"Query Text"').encode(
            x=alt.X('query_vectorized_x', axis=None),
            y=alt.Y('query_vectorized_y',  axis=None),
            color=alt.Color("color:N", scale=alt.Scale(range=["#0bb4ff"]), legend=alt.Legend(title="User Entered", symbolLimit=0, titleFontSize=10, labelFontSize=10)),
            fill=alt.value("#0bb4ff"),
            tooltip=['Query Text']
        )

        # Create an Altair chart for the cluster of the training data
        # lable should say "cluster" in the legend
        PCA_no_cluster = alt.Chart(no_cluster_df).mark_point(size=75).encode(
            x=alt.X('x_pca_train_data', axis=None),
            y=alt.Y('y_pca_train_data', axis=None),
            color=alt.Color('labels_words', scale=alt.Scale(scheme='greys'),legend=alt.Legend(columns=4, title="All Component Groups", symbolLimit=0, titleFontSize=10, labelFontSize=10, orient="bottom")),
            tooltip=['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'COMPDESC', 'CDESCR']
        ).properties(title=title, width=500, height=500)#.interactive()

        # Create an Altair chart for the cluster of the training data
        # lable should say "cluster" in the legend
        PCA_cluster = alt.Chart(cluster_df).mark_point(size=75).encode(
            x=alt.X('x_pca_train_data', axis=None, title='X-axis'),
            y=alt.Y('y_pca_train_data', axis=None, title='Y-axis'),
            color=alt.Color('labels_words', scale=alt.Scale(scheme='accent'),legend=alt.Legend(title="Classification Prediction", symbolLimit=0, titleFontSize=10, labelFontSize=10)),
            # fill the points in with the scale color
            fill=alt.value("#77DD77"),
            tooltip=['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'COMPDESC', 'CDESCR']
        ).properties(title=title, width=500, height=500)#.interactive()
                     #title, width=500, height=500)#.interactive()

        #chart = PCA_cluster + query + most_sim 

        # use alt.layer() to create a layered chart
        chart = alt.layer(PCA_no_cluster, PCA_cluster, most_sim, query).resolve_scale(color='independent').configure_legend(titleFontSize=20, labelFontSize =18, gradientLength=400, gradientThickness=30, symbolSize = 130,)
        #.configure_legend(titleFontSize=20, labelFontSize =18, gradientLength=400, gradientThickness=30, symbolSize = 130,)

        #chart.configure(background='#FFFFFF') 
    
        #chart.save('filename.html')

        return chart
    
    def condense_component_description(self, df, column_name):
        """
        Condense the component description in the dataframe by removing any text after a colon or slash
        """
        # get the unique values in the column
        compdesc_list = df[column_name].unique()

        # condense the compdesc_list with common text like the main text using regex like ELECTRICAL SYSTEM, there are multiple instances of this in the list with : separators and other text
        compdesc_list_condensed = []
        compdesc_dict = {}

        # loop through the compdesc_list and check to see if there is a : or / in the text
        for text in compdesc_list:
            # check to see if ther is a : in the text
            if ":" in text:
                # split the words on the : and / separator and then loop through the words
                temp = text.split(":")[0]
                words = temp.split(r'/')
                # loop through the temp list and append the words to the compdesc_list_condensed list
                for word in words:
                    # check to see if the word is not empty and not a number
                    compdesc_list_condensed.append(word)
                    compdesc_dict[text] = word
            elif r'/' in text:
                # split the words on the : and / separator and then loop through the words
                words = text.split(r'/')
                # loop through the temp list and append the words to the compdesc_list_condensed list
                for word in words:
                    # check to see if the word is not empty and not a number
                    compdesc_list_condensed.append(word)
                    compdesc_dict[text] = word
            else:
                # append the text to the list
                compdesc_list_condensed.append(text)
                compdesc_dict[text] = text
                
        # remove duplicates from the list
        compdesc_list_condensed = list(set(compdesc_list_condensed))

        # return the condensed list and the dictionary
        return compdesc_list_condensed, compdesc_dict
    
    def get_keys_from_value(dictionary, target_value):
        for key, value in dictionary.items():
            # check to see if the value is a string and if it contains the target_value
            if isinstance(value, str):
                if value in target_value:
                    return key
            else:
                # check to see if the value is a number and if it is equal to the target_value
                if target_value == value:
                    return key
                else:
                    pass
            return target_value
        
    def get_text_processed_and_regression(self, complaint_query):
        # create a list of stopwords from nltk.corpus.stopwords
        recall_stopwords = ["crash", "risk", "increasing", "increase", "increases", "increased", "may", "could",
        "injury", "equipment", "loss", "resulting", "condition", "occur", "result", "event", "labels", "possibly"]

        complaint_stopwords = ["engine", "unknown", "car", "driving", "issue", "dealer", "failed", "problem",
        "dealership", "issues", "times", "service", "back", "safety", "recall", "due", "like"]

        # preprocess the data using the Preprocesser class
        p = Preprocesser(
            "CDESCR_AND_COMPONENT",
            csv_name="test_agg.csv",
            custom_clean_name="nltk_stopwords_cdescr_and_components",
            custom_vectorizer_name="tfidf_binary_unigram_bigram_cdescr_and_component",
            extra_stopwords=recall_stopwords + complaint_stopwords,
            vectorizer=TfidfVectorizer,
            vectorizer_params=dict(
                ngram_range=(1,2),
                min_df=20,
                max_df=0.7,
                binary=True
            ),
            is_stem=True,
            rerun=False)
        
        p.preprocess()

        c = Classifier(
            classifier=SGDClassifier,
            classifier_params=dict(
                random_state=42,
                loss="log_loss"
            ),
            custom_classifier_name="lr_cdescr_and_components",
            X_train = p.x_train_vect,
            y_train = p.df_train["IS_RECALL"],
            X_test = p.x_validation_vect,
            y_test = p.df_validation["IS_RECALL"],
            rerun=False
        )
        c.fit()
        lr_prediction = c.predict(
            complaint_query,
            p.vectorizer,
            p.process_text,
            text_process_params=dict(
                stopwords=set(stopwords.words("english")) | set(p.extra_stopwords),
                is_stem=True
            ),
            is_proba=True
        ).flatten()

        lr_prediction = (lr_prediction * 100).round(2)
        return lr_prediction
    
    def plot_regression_bar_chart(self, complaint_query, fig, ax):
        lr_prediction = self.get_text_processed_and_regression(complaint_query)
        # create a bar chart with the prediction values
        y = 0
        # yellow bar for the complaint prediction
        ax.barh(y, lr_prediction[0], label="Probability of Complaint", color="#FFEE8C")
        # red bar for the recall prediction
        # print the percentage of the bar in the center of the bar  ax.bar_label(p, label_type='center')
        ax.barh(y, lr_prediction[1], left=lr_prediction[0], label="Probability of Recall", color="#fd7f6f")
        ax.tick_params(axis="y", colors="none")
        ax.tick_params(axis="x", colors="none")
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        x_ticks = [0, lr_prediction[0], 100]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(["0%", f"{lr_prediction[0]}%", "100%"])
        for i in range(2):
            if i == 0 and x_ticks[i] < 7.5:
                ax.text(x=x_ticks[i] + 2, y=0.8, s=f"{lr_prediction[i]}%", ha="left", va='center', color='black', fontweight=600)
            else:
                ax.text(x=x_ticks[i] + 2, y=0, s=f"{lr_prediction[i]}%", ha="left", va='center', color='white', fontweight=600)
        ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.9), ncols=2,facecolor='none', framealpha=0.0)

        # Set the figure background color to transparent
        fig.patch.set_alpha(0.0)  # Makes the entire figure transparent
        fig.patch.set_facecolor('none')  # Ensure it's transparent, not just white

        # Set the axes background color to transparent
        ax.set_facecolor('none')  # Makes the plot area transparent
        
        return fig
        


# run the below code if main script
if __name__ == "__main__":
    # change c:\Repo\SIADs_Audio_Text_SRS\src which is where the current script is to c:\Repo\SIADs_Audio_Text_SRS\ which is the root directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Current working directory: ", os.getcwd())
    # C:\Repo\SIADs_Audio_Text_SRS\streamlit_GUI.py
    # cd to the helpers directory
    sys.path.append(os.path.join(os.getcwd(), "helpers"))
    from utilities import get_dataset_dir

    DATASET_DIR = get_dataset_dir()
    # https://www.nhtsa.gov/nhtsa-datasets-and-apis#recalls
    # read in C:\Repo\SIADs_Audio_Text_SRS\Example\COMPLAINTS_RECEIVED_2025-2025.txt into a pandas dataframe, where the columns are RCL

    #df_complaints = pd.read_csv("C:\\Repo\\SIADs_Audio_Text_SRS\\Datasets\\COMPLAINTS_RECEIVED_2025-2025.txt", sep="\t", header=None, index_col=0)
    #df_complaints.columns = ['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'CRASH', 'FAILDATE', 'FIRE', 'INJURED', 'DEATHS', 'COMPDESC', 'CITY', 'STATE', 'VIN', 'DATEA', 'LDATE', 'MILES', 'OCCURENCES', 'CDESCR', 'CMPL_TYPE', 'POLICE_RPT_YN', 'PURCH_DT', 'ORIG_OWNER_YN', 'ANTI_BRAKES_YN', 'CRUISE_CONT_YN', 'NUM_CYLS', 'DRIVE_TRAIN', 'FUEL_SYS', 'FUEL_TYPE',
    #          'TRANS_TYPE', 'VEH_SPEED', 'DOT', 'TIRE_SIZE', 'LOC_OF_TIRE', 'TIRE_FAIL_TYPE', 'ORIG_EQUIP_YN', 'MANUF_DT', 'SEAT_TYPE', 'RESTRAINT_TYPE', 'DEALER_NAME', 'DEALER_TEL', 'DEALER_CITY', 'DEALER_STATE', 'DEALER_ZIP', 'PROD_TYPE', 'REPAIRED_YN', 'MEDICAL_ATTN', 'VEHICLES_TOWED_YN']

    df_complaints = pd.read_csv(f"{DATASET_DIR}\\test_no_agg.csv")

    retrive_top_n_docs = 10
    # set the target column to be the "CDESCR" column
    traget_col = "CDESCR"
    state_encode = "COMPDESC_StateEncoded"
    # state encode the COMPDESC values and create a new column in the dataframe called COMPDESC_StateEncoded
    df_complaints[state_encode] = LabelEncoder().fit_transform(df_complaints["COMPDESC"])

    # state encode the COMPDESC values and create a new column in the dataframe called COMPDESC_StateEncoded
    df_complaints["COMPDESC_StateEncoded"] = LabelEncoder().fit_transform(df_complaints["COMPDESC"])

    # create a list of unique manufacturers in the "MFR_NAME" column
    # list_of_manufacturers = df_complaints["MFR_NAME"].unique()

    # call the TextClassifier class and create an instance of it as text_classifier
    # pass in the df_complaints dataframe and the "CDESCR" column
    text_classifier = TextClassifier(df_complaints, traget_col)


    state_encode = "COMPDESC_CONDENSED_StateEncoded"
    # call the condense_component_description function to condense the component description in the dataframe by removing any text after a colon or slash
    compdesc_list_condensed, compdesc_dict = text_classifier.condense_component_description(df_complaints, "COMPDESC")
    # use the compdesc_dict to look up "COMPDESC" against the keys of the dict and assign the value to a new column in the dataframe called "COMPDESC_CONDENSED"
    df_complaints["COMPDESC_CONDENSED"] = df_complaints["COMPDESC"].apply(lambda x: compdesc_dict.get(x))
    # state encode the COMPDESC values and create a new column in the dataframe called COMPDESC_StateEncoded
    df_complaints["COMPDESC_CONDENSED_StateEncoded"] = LabelEncoder().fit_transform(df_complaints["COMPDESC_CONDENSED"])


    # process the text in the "CDESCR" column
    text_classifier.process_dataframe()
    # fit a KMeans model to the training data
    text_classifier.fit_kmeans(state_encode)

    # use one of the complaints in the test set as a query to find the most similar complaint in the training set
    complaint_test_query = "Battery dies after a few days of not driving the car"
    # complaint_test_query = "loss of power steering"
    #complaint_test_query = "Wheel sounds like it is scraping against something when driving"


    print(complaint_test_query)
    # find the most similar complaint to the complaint test
    most_similar_complaint = text_classifier.find_similar_complaint(complaint_test_query, retrive_top_n_docs)
    # print all the columns of the most similar complaint
    print("here", most_similar_complaint[text_classifier.column_name_cleaned_vect].head())

    # print the most similar complaint with the below columns
    print(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CDESCR", "COMPDESC","IS_COMPLAINT"]])
    print(most_similar_complaint["CDESCR"])
    # predict the cluster of the query text
    #print(text_classifier.compdesc_list_condensed)
    print(text_classifier.desired_clusters)
    cluster_kmeans_pred, cluster_RFC_pred, query_vectorized = text_classifier.predict_cluster(complaint_test_query)
    print(cluster_kmeans_pred)
    print(cluster_RFC_pred)

    # plot the clusters of the training data for the KMeans model
    # create a dataframe with the cluster predictions
    df_train_clustered = text_classifier.df_train.copy()
    df_train_clustered["Cluster"] = text_classifier.classifier_kmeans.labels_

    # Calculate silhouette scores for each clustering algorithm
    silhouette_score_kmeans = silhouette_score(text_classifier.complaints_vectorized_train, text_classifier.classifier_kmeans.labels_)
    print(f"Silhouette Score for KMeans: {silhouette_score_kmeans}")
    silhouette_score_RFC = silhouette_score(text_classifier.complaints_vectorized_train, text_classifier.classifier_RFC.predict(text_classifier.complaints_vectorized_train))
    print(f"Silhouette Score for Random Forest Classifier: {silhouette_score_RFC}")

    # pickel the most similar complaint
    #with open(text_classifier.desired_save_path + "//" + "most_similar_complaint.pkl", "wb") as f:
    #    pickle.dump(most_similar_complaint, f)

    # create a Matplotlib figure and axes
    fig1, ax1 = plt.subplots()
    # plot the clusters of the training data for the KMeans model
    text_classifier.plot_clusters(text_classifier.classifier_kmeans.labels_, 'KMeans Clusters of the Training Data', query_vectorized, fig1, ax1, most_similar_complaint)
    # create a Matplotlib figure and axes
    #fig2, ax2 = plt.subplots()
    # plot the clusters of the training data for the Random Forest Classifier model
    #text_classifier.plot_clusters(text_classifier.classifier_RFC.predict(text_classifier.complaints_vectorized_train), 'Random Forest Classifier Clusters of the Training Data', query_vectorized, fig2, ax2, most_similar_complaint)

    # plot the clusters of the training data for the KMeans model
    plt.show()

    # call the plot_clusters_alt function to plot the clusters using Altair
    #chart = text_classifier.plot_clusters_alt(text_classifier.classifier_kmeans.labels_, 'KMeans Clusters of the Training Data', query_vectorized, most_similar_complaint)
    # alt.renderers.enable("html")
    #chart = text_classifier.plot_clusters_alt(text_classifier.classifier_RFC.predict(text_classifier.complaints_vectorized_train), 'Random Forest Classifier Clusters of the Training Data', query_vectorized, most_similar_complaint)
    # find the top value in text_classifier.classifier_kmeans.labels_
    #print(max(text_classifier.classifier_kmeans.labels_))
    #print(max(text_classifier.classifier_RFC.predict(text_classifier.complaints_vectorized_train)))
    #print(max(text_classifier.df_train[state_encode]))          
