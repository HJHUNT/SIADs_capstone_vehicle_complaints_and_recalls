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
# read in C:\Repo\SIADs_Audio_Text_SRS\Example\COMPLAINTS_RECEIVED_2025-2025.txt into a pandas dataframe, where the columns are RCL

# df_complaints.columns = ['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'CRASH', 'FAILDATE', 'FIRE', 'INJURED', 'DEATHS', 'COMPDESC', 'CITY', 'STATE', 'VIN', 'DATEA', 'LDATE', 'MILES', 'OCCURENCES', 'CDESCR', 'CMPL_TYPE', 'POLICE_RPT_YN', 'PURCH_DT', 'ORIG_OWNER_YN', 'ANTI_BRAKES_YN', 'CRUISE_CONT_YN', 'NUM_CYLS', 'DRIVE_TRAIN', 'FUEL_SYS', 'FUEL_TYPE',
#             'TRANS_TYPE', 'VEH_SPEED', 'DOT', 'TIRE_SIZE', 'LOC_OF_TIRE', 'TIRE_FAIL_TYPE', 'ORIG_EQUIP_YN', 'MANUF_DT', 'SEAT_TYPE', 'RESTRAINT_TYPE', 'DEALER_NAME', 'DEALER_TEL', 'DEALER_CITY', 'DEALER_STATE', 'DEALER_ZIP', 'PROD_TYPE', 'REPAIRED_YN', 'MEDICAL_ATTN', 'VEHICLES_TOWED_YN']
df_complaints = pd.read_csv(f"{DATASET_DIR}\\test_no_agg.csv")
# create a list of unique manufacturers in the "MFR_NAME" column
#list_of_manufacturers = df_complaints["MFR_NAME"].unique()

# call the TextClassifier class and create an instance of it as text_classifier
# pass in the df_complaints dataframe and the "CDESCR" column
text_classifier = TextClassifier(df_complaints, "CDESCR")

#state_encode = "COMPDESC_CONDENSED_StateEncoded"
# call the condense_component_description function to condense the component description in the dataframe by removing any text after a colon or slash
#compdesc_list_condensed, compdesc_dict = text_classifier.condense_component_description(df_complaints, "COMPDESC")
# use the compdesc_dict to look up "COMPDESC" against the keys of the dict and assign the value to a new column in the dataframe called "COMPDESC_CONDENSED"
#df_complaints["COMPDESC_CONDENSED"] = df_complaints["COMPDESC"].apply(lambda x: compdesc_dict.get(x))
# state encode the COMPDESC values and create a new column in the dataframe called COMPDESC_StateEncoded
#df_complaints["COMPDESC_CONDENSED_StateEncoded"] = LabelEncoder().fit_transform(df_complaints["COMPDESC_CONDENSED"])

# process the text in the "CDESCR" column
text_classifier.process_dataframe()


# use one of the complaints in the test set as a query to find the most similar complaint in the training set
# complaint_test_query = text_classifier.df_test["CDESCR"].iloc[5]
# complaint_test_query = text_classifier.df_test["CDESCR"].iloc[4]
# complaint_test_query = "Car won't start and makes a clicking noise"
default_complaint_test_query = "Battery dies after a few days of not driving the car"


#print(complaint_test_query)
#print(complaint_test_query)
# find the most similar complaint to the complaint test
#most_similar_complaint = text_classifier.find_similar_complaint(complaint_test_query)
#most_similar_complaint = text_classifier.find_similar_complaint(complaint_test_query)
# print the most similar complaint with the below columns
#print(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CDESCR", "COMPDESC"]])
#print(most_similar_complaint["CDESCR"])
#print(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CDESCR", "COMPDESC"]])
#print(most_similar_complaint["CDESCR"])

# set the page configuration to wide and make it dark mode
st.set_page_config(layout="wide", page_title="Complaint Finder", page_icon="🚗", initial_sidebar_state="expanded")


#col_1, col_2 = st.columns(2)

#left_col, right_col = st.columns(2)

# Title of the web app
st.sidebar.title("Complaint Finder")

audio_value = st.sidebar.audio_input("Record a voice query")
if audio_value:
    #st.audio(audio_value)
    # Convert audio to text
    extracted_text = audio_to_text(audio_value)
    print("Extracted Text:")
    print(extracted_text)
    # Display the extracted text in the sidebar
    #st.sidebar.write("Extracted Text:", extracted_text)
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
    #col_1.header("Cosine Similarity")
    #col_1.subheader("Document Match:")
    st.header("Document Match:")
    st.subheader("Cosine Similarity")

    # draw a line on the left column
    # only show the row of the top complaints that the user selected as top_complaints_n
    #col_1.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC"]].iloc[0])#top_complaints_n-1])
    #left_col.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC"]])
    #col_1.write("DESCRIPTION OF THE COMPLAINT: " + most_similar_complaint["CDESCR"].iloc[top_complaints_n-1])
    print(most_similar_complaint.columns)

    # loop through the top_complaints_n and display the most similar complaints
    for i in range(top_complaints_n):
        # if i is the first complaint, then append the word Top similar doc
        if i == 0 and top_complaints_n == 1:
            #col_1.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC"]].iloc[i])
            #col_1.write("DESCRIPTION OF THE COMPLAINT: \n" + most_similar_complaint["CDESCR"].iloc[i])
            st.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC","IS_COMPLAINT"]].iloc[i])
            st.write("DESCRIPTION OF THE COMPLAINT: \n" + most_similar_complaint["CDESCR"].iloc[i])
        else:
            if i == 0:
                #expander = col_1.expander("Doc " + str(i+1) + " - Top similar")
                expander = st.expander("Doc " + str(i+1) + " - Top similar")
            # elif i is the last complaint, then append the word Least similar doc
            elif i == top_complaints_n-1:
                #expander = col_1.expander("Doc " + str(i+1) + " - Least similar")
                expander = st.expander("Doc " + str(i+1) + " - Least similar")
            else:
                #expander = col_1.expander("Doc " + str(i+1))
                expander = st.expander("Doc " + str(i+1))
            expander.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC","IS_COMPLAINT"]].iloc[i])
            expander.write("DESCRIPTION OF THE COMPLAINT: \n" + most_similar_complaint["CDESCR"].iloc[i])
    #expander = col_1.expander("Doc 1")
    #expander.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC"]].iloc[0])
    #expander.write("DESCRIPTION OF THE COMPLAINT: " + most_similar_complaint["CDESCR"].iloc[0])#top_complaints_n-1])
    
    # find the user input complaint classification
    # fit a KMeans model to the training data
    
    # output the classification of the query to the right of the input text box
    #col_2.header("Kmeans")
    #col_2.subheader("Classification Prediction:")
    #col_2.write(cluster_kmeans_pred[0])
    

    # create a Matplotlib figure and axes
    #fig1, ax1 = plt.subplots()
    #kmeans_fig = text_classifier.plot_clusters(text_classifier.classifier_kmeans.labels_, 'KMeans Clusters of the Training Data', query_vectorized, fig1, ax1, most_similar_complaint.head(top_complaints_n))
    #col_2.pyplot(kmeans_fig, use_container_width=True)
    recall_stopwords = ["crash", "risk", "increasing", "increase", "increases", "increased", "may", "could",
    "injury", "equipment", "loss", "resulting", "condition", "occur", "result", "event", "labels", "possibly"]

    complaint_stopwords = ["engine", "unknown", "car", "driving", "issue", "dealer", "failed", "problem",
    "dealership", "issues", "times", "service", "back", "safety", "recall", "due", "like",
    ]
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
        rerun=False
    )
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
    fig, ax = plt.subplots(figsize=(10, 1), dpi=100)
    y = 0
    ax.barh(y, lr_prediction[0], label="Probability of Complaint", color="#FFD700")
    ax.barh(y, lr_prediction[1], left=lr_prediction[0], label="Probability of Recall", color="#FF0000")
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
            ax.text(x=x_ticks[i] + 2, y=0.8, s=f"{lr_prediction[i]}%", ha="left", va='center', color='black',
                fontweight=600)
        else:
            ax.text(x=x_ticks[i] + 2, y=0, s=f"{lr_prediction[i]}%", ha="left", va='center', color='white',
                fontweight=600)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, -0.9), ncols=2,
              facecolor='none', framealpha=0.0)

    # background_color = st.get_option("theme.backgroundColor")
    # Set the figure background color to transparent
    fig.patch.set_alpha(0.0)  # Makes the entire figure transparent
    fig.patch.set_facecolor('none')  # Ensure it's transparent, not just white

    # Set the axes background color to transparent
    ax.set_facecolor('none')  # Makes the plot area transparent
    
    st.write("---")

    st.header("Regression")
    st.subheader("Classification of Complaint/Recall")
    st.pyplot(fig)

    # inbed a plot with a cluster visualization onto the streamlit page by calling the plot_clusters methodright_col.pyplot(kmeans_fig)
    #col_2.write("---")
    #col_2.header("Random Forest")
    #col_2.subheader("Classification Prediction:")
    #col_2.write(cluster_RFC_pred[0])
    st.write("---")


    ## Predict Cluster
    text_classifier.fit_kmeans(text_classifier.compdesc_condensed_state_encoded)
    cluster_kmeans_pred, cluster_RFC_pred, query_vectorized = text_classifier.predict_cluster(complaint_query)

    st.header("Classification:")
    # st.write("Classification of the query: ", query_classification)
    # Create a Matplotlib figure and axes
    #fig2, ax2 = plt.subplots()
    #RFC_fig = text_classifier.plot_clusters(text_classifier.classifier_RFC.predict(text_classifier.complaints_vectorized_train), 'Random Forest Classifier Clusters of the Training Data', query_vectorized, fig2, ax2, most_similar_complaint.head(top_complaints_n))
    #col_2.pyplot(RFC_fig, use_container_width=True)
    #st.pyplot(RFC_fig, use_container_width=True)
    
    kmeans_pred = "Prediction: " + cluster_kmeans_pred[0]
    RFC_pred = "Prediction: " + cluster_RFC_pred[0]
    kmeans_pred_tab_text = "Kmeans Cluster " + kmeans_pred
    RFC_pred_tab_text = "Random Forest Cluster " + RFC_pred

    tab1, tab2 = st.tabs([RFC_pred_tab_text, kmeans_pred_tab_text])

    with tab1:
        st.header("Random Forest")
        st.write(RFC_pred)
        # Use the native Altair theme.
        RFC_fig = text_classifier.plot_clusters_alt(text_classifier.classifier_RFC.predict(text_classifier.complaints_vectorized_train), 'Random Forest Classifier Clusters of the Training Data', query_vectorized, most_similar_complaint.head(top_complaints_n))
        st.altair_chart(RFC_fig, use_container_width=None, theme=None, selection_mode=None)

    with tab2:
        st.subheader("Kmeans")
        st.write(kmeans_pred)
        # This is the default. So you can also omit the theme argument.
        kmeans_fig = text_classifier.plot_clusters_alt(text_classifier.classifier_kmeans.labels_, 'KMeans Clusters of the Training Data', query_vectorized, most_similar_complaint.head(top_complaints_n))
        #col_2.altair_chart(kmeans_fig, use_container_width=False)
        st.altair_chart(kmeans_fig, use_container_width=None, theme=None, selection_mode=None)
    
    recall_stopwords = ["crash", "risk", "increasing", "increase", "increases", "increased", "may", "could",
    "injury", "equipment", "loss", "resulting", "condition", "occur", "result", "event", "labels", "possibly"]

    complaint_stopwords = ["unknown", "car", "driving", "issue", "dealer", "failed", "problem",
    "dealership", "issues", "times", "service", "back", "safety", "recall", "due", "like",
    ]
    
# run the streamlit app by running the below command in the terminal
# streamlit run streamlit_GUI.py
