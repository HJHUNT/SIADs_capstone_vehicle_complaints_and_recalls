# import Text_Query.py to use the functions
from Text_Query import *
import streamlit as st

n = 25
# create a list from 1 to 15
desired_top_complaints = list(range(1, n))


# https://www.nhtsa.gov/nhtsa-datasets-and-apis#recalls
# read in C:\Repo\SIADs_Audio_Text_SRS\Example\COMPLAINTS_RECEIVED_2025-2025.txt into a pandas dataframe, where the columns are RCL
df_complaints = pd.read_csv("C:\\Repo\\SIADs_Audio_Text_SRS\\Datasets\\COMPLAINTS_RECEIVED_2025-2025.txt", sep="\t", header=None, index_col=0)
df_complaints.columns = ['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'CRASH', 'FAILDATE', 'FIRE', 'INJURED', 'DEATHS', 'COMPDESC', 'CITY', 'STATE', 'VIN', 'DATEA', 'LDATE', 'MILES', 'OCCURENCES', 'CDESCR', 'CMPL_TYPE', 'POLICE_RPT_YN', 'PURCH_DT', 'ORIG_OWNER_YN', 'ANTI_BRAKES_YN', 'CRUISE_CONT_YN', 'NUM_CYLS', 'DRIVE_TRAIN', 'FUEL_SYS', 'FUEL_TYPE',
            'TRANS_TYPE', 'VEH_SPEED', 'DOT', 'TIRE_SIZE', 'LOC_OF_TIRE', 'TIRE_FAIL_TYPE', 'ORIG_EQUIP_YN', 'MANUF_DT', 'SEAT_TYPE', 'RESTRAINT_TYPE', 'DEALER_NAME', 'DEALER_TEL', 'DEALER_CITY', 'DEALER_STATE', 'DEALER_ZIP', 'PROD_TYPE', 'REPAIRED_YN', 'MEDICAL_ATTN', 'VEHICLES_TOWED_YN']


# state encode the COMPDESC values and create a new column in the dataframe called COMPDESC_StateEncoded
df_complaints["COMPDESC_StateEncoded"] = LabelEncoder().fit_transform(df_complaints["COMPDESC"])


# create a list of unique manufacturers in the "MFR_NAME" column
list_of_manufacturers = df_complaints["MFR_NAME"].unique()

# call the TextClassifier class and create an instance of it as text_classifier
# pass in the df_complaints dataframe and the "CDESCR" column
text_classifier = TextClassifier(df_complaints, "CDESCR")
# process the text in the "CDESCR" column
text_classifier.process_dataframe()
# fit a KMeans model to the training data
text_classifier.fit_kmeans("COMPDESC_StateEncoded")

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


col_1, col_2 = st.columns(2)

left_col, right_col = st.columns(2)

# Title of the web app
st.sidebar.title("Complaint Finder")

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

    # loop through the top_complaints_n and display the most similar complaints
    for i in range(top_complaints_n):
        # if i is the first complaint, then append the word Top similar doc
        if i == 0 and top_complaints_n == 1:
            #col_1.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC"]].iloc[i])
            #col_1.write("DESCRIPTION OF THE COMPLAINT: \n" + most_similar_complaint["CDESCR"].iloc[i])
            st.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC"]].iloc[i])
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
            expander.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC"]].iloc[i])
            expander.write("DESCRIPTION OF THE COMPLAINT: \n" + most_similar_complaint["CDESCR"].iloc[i])
    #expander = col_1.expander("Doc 1")
    #expander.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC"]].iloc[0])
    #expander.write("DESCRIPTION OF THE COMPLAINT: " + most_similar_complaint["CDESCR"].iloc[0])#top_complaints_n-1])
    
    # find the user input complaint classification
    cluster_kmeans_pred, cluster_RFC_pred, query_vectorized = text_classifier.predict_cluster(complaint_query)
    # output the classification of the query to the right of the input text box
    #col_2.header("Kmeans")
    #col_2.subheader("Classification Prediction:")
    #col_2.write(cluster_kmeans_pred[0])
    

    # create a Matplotlib figure and axes
    #fig1, ax1 = plt.subplots()
    #kmeans_fig = text_classifier.plot_clusters(text_classifier.classifier_kmeans.labels_, 'KMeans Clusters of the Training Data', query_vectorized, fig1, ax1, most_similar_complaint.head(top_complaints_n))
    #col_2.pyplot(kmeans_fig, use_container_width=True)

    # inbed a plot with a cluster visualization onto the streamlit page by calling the plot_clusters methodright_col.pyplot(kmeans_fig)
    #col_2.write("---")
    #col_2.header("Random Forest")
    #col_2.subheader("Classification Prediction:")
    #col_2.write(cluster_RFC_pred[0])
    st.write("---")
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

    tab1, tab2 = st.tabs([kmeans_pred_tab_text, RFC_pred_tab_text])

    with tab1:
        st.subheader("Kmeans")
        st.write(kmeans_pred)
        # This is the default. So you can also omit the theme argument.
        kmeans_fig = text_classifier.plot_clusters_alt(text_classifier.classifier_kmeans.labels_, 'KMeans Clusters of the Training Data', query_vectorized, most_similar_complaint.head(top_complaints_n))
        #col_2.altair_chart(kmeans_fig, use_container_width=False)
        st.altair_chart(kmeans_fig, use_container_width=None, theme=None, selection_mode=None)
    with tab2:
        st.header("Random Forest")
        st.write(RFC_pred)
        # Use the native Altair theme.
        RFC_fig = text_classifier.plot_clusters_alt(text_classifier.classifier_RFC.predict(text_classifier.complaints_vectorized_train), 'Random Forest Classifier Clusters of the Training Data', query_vectorized, most_similar_complaint.head(top_complaints_n))
        st.altair_chart(RFC_fig, use_container_width=None, theme=None, selection_mode=None)
# run the streamlit app by running the below command in the terminal
# streamlit run streamlit_GUI.py
