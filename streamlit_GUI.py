# import Text_Query.py to use the functions
from src.Text_Query import *
from src.Extracted_Text import wav_to_text_and_tokenize
from src.Extracted_Text import record_audio
from src.speech_rec import *
import streamlit as st
import sys
import os
from sklearn.linear_model import SGDClassifier
from helpers.utilities import get_dataset_dir
from Experiments.preprocessing import Preprocesser
from Experiments.classifier import Classifier
print("Current working directory: ", os.getcwd())

DATASET_DIR = get_dataset_dir()
n = 25
# create a list from 1 to 15
desired_top_complaints = list(range(1, n))

# https://www.nhtsa.gov/nhtsa-datasets-and-apis#recalls
# read in os.getcwd() + Example\COMPLAINTS_RECEIVED_2025-2025.txt into a pandas dataframe, where the columns are RCL

# df_complaints.columns = ['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'CRASH', 'FAILDATE', 'FIRE', 'INJURED', 'DEATHS', 'COMPDESC', 'CITY', 'STATE', 'VIN', 'DATEA', 'LDATE', 'MILES', 'OCCURENCES', 'CDESCR', 'CMPL_TYPE', 'POLICE_RPT_YN', 'PURCH_DT', 'ORIG_OWNER_YN', 'ANTI_BRAKES_YN', 'CRUISE_CONT_YN', 'NUM_CYLS', 'DRIVE_TRAIN', 'FUEL_SYS', 'FUEL_TYPE',
#             'TRANS_TYPE', 'VEH_SPEED', 'DOT', 'TIRE_SIZE', 'LOC_OF_TIRE', 'TIRE_FAIL_TYPE', 'ORIG_EQUIP_YN', 'MANUF_DT', 'SEAT_TYPE', 'RESTRAINT_TYPE', 'DEALER_NAME', 'DEALER_TEL', 'DEALER_CITY', 'DEALER_STATE', 'DEALER_ZIP', 'PROD_TYPE', 'REPAIRED_YN', 'MEDICAL_ATTN', 'VEHICLES_TOWED_YN']

df_complaints = pd.read_csv(f"{DATASET_DIR}\\test_no_agg.csv")

# create a list of unique manufacturers in the "MFR_NAME" column
#list_of_manufacturers = df_complaints["MFR_NAME"].unique()

# call the TextClassifier class and create an instance of it as text_classifier
# pass in the df_complaints dataframe and the "CDESCR" column
text_classifier = TextClassifier(df_complaints, "CDESCR")

# process the text in the "CDESCR" column
text_classifier.process_dataframe()


# use one of the complaints in the test set as a query to find the most similar complaint in the training set
# complaint_test_query = text_classifier.df_test["CDESCR"].iloc[5]
# complaint_test_query = text_classifier.df_test["CDESCR"].iloc[4]
# complaint_test_query = "Car won't start and makes a clicking noise"
default_complaint_test_query = "Battery dies after a few days of not driving the car"

# set the page configuration to wide and make it dark mode
st.set_page_config(layout="wide", page_title="Complaint Finder", page_icon="🚗", initial_sidebar_state="expanded")

# Title of the web app
st.sidebar.title("Complaint Finder")

audio_value = st.sidebar.audio_input("Record a voice query")
if audio_value:
    # Convert audio to text
    extracted_text = audio_to_text(audio_value)
    print("Extracted Text:")
    print(extracted_text)
    # Display the extracted text in the sidebar
    default_complaint_test_query = extracted_text

# Create a text box for the user to input a complaint
complaint_query = st.sidebar.text_area("Enter a complaint:", default_complaint_test_query)

# create a select box for the user to select a desired amount of top complaints to display
top_complaints_n = st.sidebar.selectbox("Desired top complaints to display:", desired_top_complaints)

# Create a button for the user to click to find the most similar complaint
if st.sidebar.button("Search and Classify"):

    # find the most similar complaint to the complaint test
    most_similar_complaint = text_classifier.find_similar_complaint(complaint_query, n)
    # print the most similar complaint with the below columns make the text bold
    st.header("Document Match:")
    st.subheader("Cosine Similarity")

    # draw a line on the left column
    # only show the row of the top complaints that the user selected as top_complaints_n
    print(most_similar_complaint.columns)

    # loop through the top_complaints_n and display the most similar complaints
    for i in range(top_complaints_n):
        # if i is the first complaint, then append the word Top similar doc
        if i == 0 and top_complaints_n == 1:
            st.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC","IS_COMPLAINT"]].iloc[i])
            st.write("DESCRIPTION OF THE COMPLAINT: \n" + most_similar_complaint["CDESCR"].iloc[i])
        else:
            # if i is the first complaint, then append the word Top similar doc
            if i == 0:
                expander = st.expander("Doc " + str(i+1) + " - Top similar")
            # elif get the last complaint in the list, then append the word Least similar doc
            elif i == top_complaints_n-1:
                expander = st.expander("Doc " + str(i+1) + " - Least similar")
            else:
                expander = st.expander("Doc " + str(i+1))
            expander.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC","IS_COMPLAINT"]].iloc[i])
            expander.write("DESCRIPTION OF THE COMPLAINT: \n" + most_similar_complaint["CDESCR"].iloc[i])
    
    st.write("---")
    st.header("Regression")
    st.subheader("Classification of Complaint/Recall")
    fig, ax = plt.subplots(figsize=(10, 1), dpi=100)
    fig = text_classifier.plot_regression_bar_chart(complaint_query, fig, ax)
    st.pyplot(fig)

    # inbed a plot with a cluster visualization onto the streamlit page by calling the plot_clusters methodright_col.pyplot(kmeans_fig)
    st.write("---")

    ## Predict Cluster
    text_classifier.fit_kmeans(text_classifier.compdesc_condensed_state_encoded)
    cluster_kmeans_pred, cluster_RFC_pred, query_vectorized = text_classifier.predict_cluster(complaint_query)

    st.header("Classification:")
    
    kmeans_pred = "Prediction: " + cluster_kmeans_pred[0]
    RFC_pred = "Prediction: " + cluster_RFC_pred[0]
    kmeans_pred_tab_text = "Kmeans Cluster " + kmeans_pred
    RFC_pred_tab_text = "Random Forest Cluster " + RFC_pred

    tab1, tab2 = st.tabs([RFC_pred_tab_text, kmeans_pred_tab_text])

    with tab1:
        st.header("Random Forest")
        st.write(RFC_pred)
        RFC_fig = text_classifier.plot_clusters_alt(text_classifier.classifier_RFC.predict(text_classifier.complaints_vectorized_train), 'Random Forest Classifier Clusters of the Training Data', query_vectorized, most_similar_complaint.head(top_complaints_n))
        st.altair_chart(RFC_fig, use_container_width=None, theme=None, selection_mode=None)

    with tab2:
        st.subheader("Kmeans")
        st.write(kmeans_pred)
        kmeans_fig = text_classifier.plot_clusters_alt(text_classifier.classifier_kmeans.labels_, 'KMeans Clusters of the Training Data', query_vectorized, most_similar_complaint.head(top_complaints_n))
        st.altair_chart(kmeans_fig, use_container_width=None, theme=None, selection_mode=None)

