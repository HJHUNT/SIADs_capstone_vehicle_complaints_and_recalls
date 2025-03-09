#!/usr/bin/env python
##############################################################
#
# # Combining Complaints and Recall for Information Retrieval
# 
# **Author:** Harris Zheng
# 
# **Date:** March 2nd, 2025
#
###############################################################

import pandas as pd
import pprint
import numpy as np
import matplotlib.pyplot as plt
import os
import string
import re

class DatasetMerger:
    def __init__(self):
        PARENT_DIR = os.getcwd().rsplit("\\", maxsplit=1)[0]
        DATASET_DIR = os.path.join(PARENT_DIR, "Datasets")
        df_recall = pd.read_csv(f"{DATASET_DIR}/FLAT_RCL.txt", sep='\t', header=None, on_bad_lines='skip')
        # use the column names listed above
        df_recall.columns = ['RECORD_ID', 'CAMPNO', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'MFGCAMPNO', 'COMPNAME', 'MFGNAME', 'BGMAN', 'ENDMAN', 'RCLTYPECD', 'POTAFF', 'ODATE', 'INFLUENCED_BY', 'MFGTXT', 'RCDATE', 'DATEA', 'RPNO', 'FMVSS', 'DESC_DEFECT', 'CONSEQUENCE_DEFECT', 'CORRECTIVE_ACTION', 'NOTES', 'RCL_CMPT_ID', 'MFR_COMP_NAME', 'MFR_COMP_DESC', 'MFR_COMP_PTNO']
        df_complaints = pd.read_csv(f"{DATASET_DIR}/COMPLAINTS_RECEIVED_2025-2025.txt", 
                                    sep='\t', 
                                    header=None, 
                                    index_col=0)
        df_complaints.columns = ['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'CRASH', 'FAILDATE', 'FIRE', 'INJURED', 'DEATHS', 'COMPDESC', 'CITY', 'STATE', 'VIN', 'DATEA', 'LDATE', 'MILES', 'OCCURENCES', 'CDESCR', 'CMPL_TYPE', 'POLICE_RPT_YN', 'PURCH_DT', 'ORIG_OWNER_YN', 'ANTI_BRAKES_YN', 'CRUISE_CONT_YN', 'NUM_CYLS', 'DRIVE_TRAIN', 'FUEL_SYS', 'FUEL_TYPE',
                    'TRANS_TYPE', 'VEH_SPEED', 'DOT', 'TIRE_SIZE', 'LOC_OF_TIRE', 'TIRE_FAIL_TYPE', 'ORIG_EQUIP_YN', 'MANUF_DT', 'SEAT_TYPE', 'RESTRAINT_TYPE', 'DEALER_NAME', 'DEALER_TEL', 'DEALER_CITY', 'DEALER_STATE', 'DEALER_ZIP', 'PROD_TYPE', 'REPAIRED_YN', 'MEDICAL_ATTN', 'VEHICLES_TOWED_YN']
        self.fill_string_nulls(df_complaints)
        self.fill_string_nulls(df_recall)
    
    def preprocess_strings(self):
        df_complaints["MMYTXT"] = (
            df_complaints["MAKETXT"] + " " + df_complaints["MODELTXT"] + " " + df_complaints["YEARTXT"].astype(str).fillna("")
        )
        df_recall["MMYTXT"] = (
            df_recall["MAKETXT"] + " " + df_recall["MODELTXT"] + " " + df_recall["YEARTXT"].astype(str).fillna("")
        )
        df_recall["CDESCR"] = df_recall["DESC_DEFECT"].str.cat(
            df_recall[["CONSEQUENCE_DEFECT", "CORRECTIVE_ACTION"]],
            sep="\n\n "
        )
        df_complaints["YEARTXT"] = df_complaints["YEARTXT"].astype(str) # None entries get converted to literal string 'None'
        df_recall["YEARTXT"] = df_recall["YEARTXT"].astype(str)

    def fill_string_nulls(self, df : pd.DataFrame):
        # Fill null string columns in DataFrame
        for column in df.columns:
            if df[column].dtype == object:
                df[column] = df[column].fillna("").str.replace("\s+", " ", regex=True)
    
    def aggregate_on_cdescr(self):
        '''
            Removing Duplicates
        '''
        df_complaints_grouped = df_complaints.groupby("CDESCR").agg(
            {
                "COMPDESC" : lambda x: ', '.join(set(x)),
                "MMYTXT" : lambda x: ', '.join(set(x)),
                "ODINO" : lambda x: ','.join(set(x.astype(str))),
            }
        )
        df_complaints_size = df_complaints.groupby("CDESCR").size()
        df_complaints_size.name = "NUMCOMPLAINTS"
        df_complaints_new = pd.merge(
            df_complaints_grouped,
            df_complaints_size,
            left_index=True,
            right_index=True
        ).reset_index()

        df_complaints_new.rename(
            {"ODINO":"RECORDID"},
            axis=1, inplace=True
        )

        df_recall_agg = df_recall.groupby("CDESCR").agg(
            {
                "COMPNAME" : lambda x: ', '.join(set(x)),
                "MMYTXT" : lambda x: ', '.join(set(x)),
                "RECORD_ID" : lambda x: ', '.join(set(x.astype(str)))
            }
        ).reset_index()



        df_recall_new = df_recall_agg[[
            "COMPNAME", "MMYTXT", "CDESCR", "RECORD_ID"
        ]].rename(
            {"RECORD_ID":"RECORDID",
            "COMPNAME":"COMPDESC"},
            axis=1
        )


        df_complaints_new["IS_COMPLAINT"] = True
        df_recall_new["IS_COMPLAINT"] = False

        df_final = pd.concat(
            [df_complaints_new,
            df_recall_new]
        ).reset_index().rename({"index":"INDEX"})
# In[27]:


        df_final.to_csv(
            f"{DATASET_DIR}/complaints_and_recalls.csv",
            index=False
        )





