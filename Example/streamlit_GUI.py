# import Text_Query.py to use the functions
from Text_Query import *
import streamlit as st

# https://www.nhtsa.gov/nhtsa-datasets-and-apis#recalls
# read in C:\Repo\SIADs_Audio_Text_SRS\Example\COMPLAINTS_RECEIVED_2025-2025.txt into a pandas dataframe, where the columns are RCL
df_complaints = pd.read_csv("C:\\Repo\\SIADs_Audio_Text_SRS\\Datasets\\COMPLAINTS_RECEIVED_2025-2025.txt", sep='\t', header=None, index_col=0)
df_complaints.columns = ['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'CRASH', 'FAILDATE', 'FIRE', 'INJURED', 'DEATHS', 'COMPDESC', 'CITY', 'STATE', 'VIN', 'DATEA', 'LDATE', 'MILES', 'OCCURENCES', 'CDESCR', 'CMPL_TYPE', 'POLICE_RPT_YN', 'PURCH_DT', 'ORIG_OWNER_YN', 'ANTI_BRAKES_YN', 'CRUISE_CONT_YN', 'NUM_CYLS', 'DRIVE_TRAIN', 'FUEL_SYS', 'FUEL_TYPE',
            'TRANS_TYPE', 'VEH_SPEED', 'DOT', 'TIRE_SIZE', 'LOC_OF_TIRE', 'TIRE_FAIL_TYPE', 'ORIG_EQUIP_YN', 'MANUF_DT', 'SEAT_TYPE', 'RESTRAINT_TYPE', 'DEALER_NAME', 'DEALER_TEL', 'DEALER_CITY', 'DEALER_STATE', 'DEALER_ZIP', 'PROD_TYPE', 'REPAIRED_YN', 'MEDICAL_ATTN', 'VEHICLES_TOWED_YN']

# create a list of unique manufacturers in the "MFR_NAME" column
list_of_manufacturers = df_complaints["MFR_NAME"].unique()

# testing out the functionilzed code

# call the TextClassifier class and create an instance of it as text_classifier
# pass in the df_complaints dataframe and the "CDESCR" column
text_classifier = TextClassifier(df_complaints, "CDESCR")
# process the text in the "CDESCR" column
text_classifier.process_dataframe()

# use one of the complaints in the test set as a query to find the most similar complaint in the training set
#complaint_test_query = text_classifier.df_test["CDESCR"].iloc[5]
#complaint_test_query = text_classifier.df_test["CDESCR"].iloc[4]
#complaint_test_query = "Car won't start and makes a clicking noise"
complaint_test_query = "Battery dies after a few days of not driving the car"

print(complaint_test_query)
# find the most similar complaint to the complaint test
most_similar_complaint = text_classifier.find_similar_complaint(complaint_test_query)
# print the most similar complaint with the below columns
print(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CDESCR"]])
print(most_similar_complaint["CDESCR"])

# Title of the web app
st.title('Text Query')

# Create a text box for the user to input a complaint
complaint_query = st.text_area("Enter a complaint", "Battery dies after a few days of not driving the car")
# Create a button for the user to click to find the most similar complaint
if st.button("Find most similar complaint"):
    # find the most similar complaint to the complaint test
    most_similar_complaint = text_classifier.find_similar_complaint(complaint_query)
    # print the most similar complaint with the below columns
    st.write(most_similar_complaint[["ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CDESCR"]])
    st.write(most_similar_complaint["CDESCR"])

# run the streamlit app by running the below command in the terminal
# streamlit run streamlit_GUI.py

