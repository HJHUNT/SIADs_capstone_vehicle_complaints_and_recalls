#!/usr/bin/env python
################################################################
# coding: utf-8
# Combining Complaints and Recall for Information Retrieval
# 
# **Author:** Harris Zheng
# 
# **Date:** March 2nd, 2025
# 
# TODO: Add remainder columns
##################################################################
import pandas as pd
import logging
from helpers import (
    fill_string_nulls,
    fill_string_spaces,
    trim_strings,
    catchtime
)
import argparse
import time
import os
import string
import re

class DatasetCombiner:
    def __init__(
        self,
        output_csv_name : str,
        dataset_dir=None,
        is_agg=True,
    ):
        '''
            specify is_agg is False if you want to keep duplicate
            CDESCRs.
        '''
        self.is_agg = is_agg
        self.output_csv_name = output_csv_name
        if dataset_dir is None:
            PARENT_DIR = os.getcwd().rsplit("\\", maxsplit=1)[0]
            self.dataset_dir = os.path.join(PARENT_DIR, "Datasets")
        else:
            self.dataset_dir = dataset_dir
        
    def read_csvs(
        self
    ):  
        # Just process 2025 data for now 
        
        # first_complaints = pd.read_csv(f"{self.dataset_dir}/COMPLAINTS_RECEIVED_2020-2024.txt", 
        #                             sep='\t', 
        #                             header=None, 
        #                             index_col=0)
        # first_complaints.columns = ['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'CRASH', 'FAILDATE', 'FIRE', 'INJURED', 'DEATHS', 'COMPDESC', 'CITY', 'STATE', 'VIN', 'DATEA', 'LDATE', 'MILES', 'OCCURENCES', 'CDESCR', 'CMPL_TYPE', 'POLICE_RPT_YN', 'PURCH_DT', 'ORIG_OWNER_YN', 'ANTI_BRAKES_YN', 'CRUISE_CONT_YN', 'NUM_CYLS', 'DRIVE_TRAIN', 'FUEL_SYS', 'FUEL_TYPE',
        #             'TRANS_TYPE', 'VEH_SPEED', 'DOT', 'TIRE_SIZE', 'LOC_OF_TIRE', 'TIRE_FAIL_TYPE', 'ORIG_EQUIP_YN', 'MANUF_DT', 'SEAT_TYPE', 'RESTRAINT_TYPE', 'DEALER_NAME', 'DEALER_TEL', 'DEALER_CITY', 'DEALER_STATE', 'DEALER_ZIP', 'PROD_TYPE', 'REPAIRED_YN', 'MEDICAL_ATTN', 'VEHICLES_TOWED_YN']
        # second_complaints = pd.read_csv(f"{self.dataset_dir}/COMPLAINTS_RECEIVED_2025-2025.txt", 
        #                             sep='\t', 
        #                             header=None, 
        #                             index_col=0)
        # second_complaints.columns = ['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'CRASH', 'FAILDATE', 'FIRE', 'INJURED', 'DEATHS', 'COMPDESC', 'CITY', 'STATE', 'VIN', 'DATEA', 'LDATE', 'MILES', 'OCCURENCES', 'CDESCR', 'CMPL_TYPE', 'POLICE_RPT_YN', 'PURCH_DT', 'ORIG_OWNER_YN', 'ANTI_BRAKES_YN', 'CRUISE_CONT_YN', 'NUM_CYLS', 'DRIVE_TRAIN', 'FUEL_SYS', 'FUEL_TYPE',
        #             'TRANS_TYPE', 'VEH_SPEED', 'DOT', 'TIRE_SIZE', 'LOC_OF_TIRE', 'TIRE_FAIL_TYPE', 'ORIG_EQUIP_YN', 'MANUF_DT', 'SEAT_TYPE', 'RESTRAINT_TYPE', 'DEALER_NAME', 'DEALER_TEL', 'DEALER_CITY', 'DEALER_STATE', 'DEALER_ZIP', 'PROD_TYPE', 'REPAIRED_YN', 'MEDICAL_ATTN', 'VEHICLES_TOWED_YN']
        # self.df_complaints = pd.concat(
        #     [
        #         first_complaints,
        #         second_complaints
        #     ],
        #     axis=0
        # )

        self.df_complaints = pd.read_csv(f"{self.dataset_dir}/COMPLAINTS_RECEIVED_2025-2025.txt", 
                                    sep='\t', 
                                    header=None, 
                                    index_col=0)
        self.df_complaints.columns = ['ODINO', 'MFR_NAME', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'CRASH', 'FAILDATE', 'FIRE', 'INJURED', 'DEATHS', 'COMPDESC', 'CITY', 'STATE', 'VIN', 'DATEA', 'LDATE', 'MILES', 'OCCURENCES', 'CDESCR', 'CMPL_TYPE', 'POLICE_RPT_YN', 'PURCH_DT', 'ORIG_OWNER_YN', 'ANTI_BRAKES_YN', 'CRUISE_CONT_YN', 'NUM_CYLS', 'DRIVE_TRAIN', 'FUEL_SYS', 'FUEL_TYPE',
                    'TRANS_TYPE', 'VEH_SPEED', 'DOT', 'TIRE_SIZE', 'LOC_OF_TIRE', 'TIRE_FAIL_TYPE', 'ORIG_EQUIP_YN', 'MANUF_DT', 'SEAT_TYPE', 'RESTRAINT_TYPE', 'DEALER_NAME', 'DEALER_TEL', 'DEALER_CITY', 'DEALER_STATE', 'DEALER_ZIP', 'PROD_TYPE', 'REPAIRED_YN', 'MEDICAL_ATTN', 'VEHICLES_TOWED_YN']
        self.df_recall = pd.read_csv(f"{self.dataset_dir}/FLAT_RCL.txt", sep='\t', header=None, on_bad_lines='skip')
        # use the column names listed above
        self.df_recall.columns = ['RECORD_ID', 'CAMPNO', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'MFGCAMPNO', 'COMPNAME', 'MFGNAME', 'BGMAN', 'ENDMAN', 'RCLTYPECD', 'POTAFF', 'ODATE', 'INFLUENCED_BY', 'MFGTXT', 'RCDATE', 'DATEA', 'RPNO', 'FMVSS', 'DESC_DEFECT', 'CONSEQUENCE_DEFECT', 'CORRECTIVE_ACTION', 'NOTES', 'RCL_CMPT_ID', 'MFR_COMP_NAME', 'MFR_COMP_DESC', 'MFR_COMP_PTNO']

        
    def preprocessing(
        self      
    ):
        # Fill nulls before concatenating strings
        fill_string_nulls(self.df_complaints)
        fill_string_nulls(self.df_recall)

        # Normalize strings for better deduplication
        trim_strings(self.df_complaints)
        trim_strings(self.df_recall)
        
        # Time consuming regex, don't run
        # fill_string_spaces(self.df_complaints)
        # fill_string_spaces(self.df_recall)

        self.df_complaints["MMYTXT"] = (
            self.df_complaints["MAKETXT"] + " " + self.df_complaints["MODELTXT"] + " " + self.df_complaints["YEARTXT"].astype(str).fillna("")
        )
        self.df_recall["MMYTXT"] = (
            self.df_recall["MAKETXT"] + " " + self.df_recall["MODELTXT"] + " " + self.df_recall["YEARTXT"].astype(str).fillna("")
        )
        self.df_complaints["CDESCR_CODE"] = pd.factorize(self.df_complaints['CDESCR'])[0]
        self.df_recall["CDESCR"] = (
            self.df_recall["DESC_DEFECT"]
            .str.cat(
                self.df_recall[["CONSEQUENCE_DEFECT"]],
                sep="\r\n"
            )
        )
        assert max(self.df_recall["CDESCR"].str.split("\r\n").str.len()) == 2, "Split is not clean"
        self.df_recall["CDESCR_CODE"] = pd.factorize(self.df_recall['CDESCR'])[0]

        if self.is_agg:
            self.df_complaints_new = DatasetCombiner.process_columns_accordingly(
                self.df_complaints, "CDESCR_CODE"
            )
            self.df_recall_new = DatasetCombiner.process_columns_accordingly(
                self.df_recall, "CDESCR_CODE"
            )
        else:
            ## If we just want to use the original dataframe, specify is_agg = False
            self.df_complaints_new = self.df_complaints
            self.df_recall_new = self.df_recall
        
        # We want component name and manufacture name
        # On one column
        self.df_recall_new = self.df_recall_new.rename(
            {
                "COMPNAME":"COMPDESC",
                "MFGNAME":"MFR_NAME"
            },
            axis=1
        )

    def combine(self):
        self.df_complaints_new["IS_COMPLAINT"] = True
        self.df_recall_new["IS_COMPLAINT"] = False

        self.df_final = pd.concat(
            [self.df_complaints_new,
            self.df_recall_new]
        ).drop("CDESCR_CODE", axis=1).reset_index(drop=True)

        self.df_final.to_csv(
            f"{self.dataset_dir}/{self.output_csv_name}"
        )

    def run_pipeline(self):
        print("Reading CSVs")
        with catchtime() as t:
            self.read_csvs()

        print("Preprocessing")
        with catchtime() as t:
            self.preprocessing()

        print("Combining")
        with catchtime() as t:            
            self.combine()

    @staticmethod
    def find_duplicate_and_non_duplicate_columns(df : pd.DataFrame, 
                                                column_defining_uniqueness : str):
        '''
        input: dataframe, and unique column identifier
        returns: duplicated columns and non-duplicated columns
        '''
        column_uniqueness = (
            df.groupby(column_defining_uniqueness)
            .nunique().sum(axis=0) 
            - 
            len(df[column_defining_uniqueness].unique())
        )
        duplicated_columns = column_uniqueness.loc[column_uniqueness > 0].index
        non_duplicated_columns = set(df.columns) - set(duplicated_columns)

        return list(duplicated_columns), list(non_duplicated_columns)
    
    @staticmethod
    def process_columns_accordingly(df : pd.DataFrame, 
                                    column_defining_uniqueness : str):
        '''
            Sorry the naming scheme doesn't make too munch sense here.
            Duplicate columns is supposed to be columns that have multiple values for
            one CDESCR value, so it really should be called one_to_many_columns,
            while non_duplicate_columns have one-to-one relationship with CDESCR (only one unique value per CDESCR).
        '''
        duplicate_columns, non_duplicate_columns = DatasetCombiner.find_duplicate_and_non_duplicate_columns(
            df, 
            column_defining_uniqueness
        )
        df[duplicate_columns] = df[duplicate_columns].fillna("").astype(str) # Preprocess ahead of time
        grouper = df.groupby(column_defining_uniqueness)
        df_dup = grouper.agg(
            {
                duplicate_column : lambda x: ', '.join(pd.unique(x))
                for duplicate_column in duplicate_columns
            }
        )
        df_no_dup = grouper.agg(
            {
                non_duplicate_column : "first"
                for non_duplicate_column in non_duplicate_columns
            }
        )
        df_size = grouper.size()
        df_size.name = "NUMRECORDS"
        df_new = pd.concat(
            [
                df_dup,
                df_no_dup,
                df_size
            ],
            axis=1
        )
        return df_new
    
if __name__ == "__main__":
    logging.basicConfig(filename='combine_dataset.log', level=logging.INFO)
    logger = logging.getLogger()

    start_time = time.time()
    parser = argparse.ArgumentParser(description='Combine Complaints and Recall')
    parser.add_argument('--dataset_dir', default="./Datasets", help='Dataset directory')
    parser.add_argument('--output_csv_name', default="test_agg.csv", help='Output CSV name')

    ## use --no-agg for no aggregation, --agg for aggregation
    parser.add_argument('--agg', default=True, action=argparse.BooleanOptionalAction, help='Use aggregation')
    args = parser.parse_args()
    dc = DatasetCombiner(
        output_csv_name=args.output_csv_name,
        dataset_dir=args.dataset_dir,
        is_agg=args.agg
    )
    dc.run_pipeline()
    end_time = time.time()
    logger.info(
            " -- ".join([
                f" Aggregate: {args.agg}",
                f"Output csv path: {os.path.join(args.dataset_dir, args.output_csv_name)}",
                f"Process took {end_time - start_time:.4f} seconds"
            ])
    )