from sentence_transformers import SentenceTransformer
import torch
from typing import Union, Literal
import os
import pandas as pd
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curdir)
sys.path.insert(0, parent_dir)
from helpers.utilities import get_dataset_dir
from helpers.pickle_decorator import pickle_io
# import numpy as np

class HuggingFaceClassifier:
    def __init__(self, model_name, column_name,
                 experiment_name,
                #  embedding_path=None,
                 similarity_fn_name : Union[
                            "cosine", "dot"
                ] = "cosine",
                rerun=False):
        self.column_name = column_name
        self.model_name = model_name
        self.rerun = rerun
        self.experiment_name = experiment_name
        self.dataset_dir = get_dataset_dir()

        # embeddings and models saved here
        self.model_dir = os.path.join(self.dataset_dir, f"huggingface", self.column_name)

        # Load sentence transformer if it has already been saved
        if os.path.exists(os.path.join(self.model_dir, self.model_name)):
            self.model = SentenceTransformer(os.path.join(self.model_dir, self.model_name),
                                             similarity_fn_name=similarity_fn_name)
        else:
            os.makedirs(self.model_dir, exist_ok=True)
            self.model = SentenceTransformer(self.model_name,
                                             similarity_fn_name=similarity_fn_name)
            self.model.save(os.path.join(self.model_dir, self.model_name))
    
    def fit_transform(self, series, experiment_name):
        '''
        TODO: Consider GPU acceleration
        '''
        # Encoding here
        return pickle_io(
            self.model.encode(series, show_progress_bar=True),
            file_path=os.path.join(self.model_dir, experiment_name, f"{experiment_name}.pkl"),
            rerun=self.rerun
        )
    
    def transform(self, series, experiment_name):
        '''
        TODO: Consider GPU acceleration
        '''
        # Encoding here
        return pickle_io(
            self.model.encode(series, show_progress_bar=True),
            file_path=os.path.join(self.model_dir, experiment_name, f"{experiment_name}.pkl"),
            rerun=self.rerun
        )
    
    def find_similar(self, df, query, embeddings,
                         keep : Literal[
                            "complaint",
                            "recall",
                            "all"
                        ] = "all", top=20):
        '''
            Find most similar recalls, complaints or both
        '''
        query_embedding = self.model.encode(query)
        if keep == "recall":
            filter_indices = df[df["IS_COMPLAINT"] == 0].index
        elif keep == "complaint":
            filter_indices = df[df["IS_COMPLAINT"] == 1].index
        else:
            filter_indices = df.index

        model_similarities = self.model.similarity(
            query_embedding,
            embeddings[filter_indices]
        )
        values, most_similar_indices = torch.topk(model_similarities, largest=True, k=top, dim=1)
        docs_to_return = df.loc[filter_indices].iloc[most_similar_indices[0].tolist()]
        docs_to_return["similarity"] = values.squeeze()
        return docs_to_return
    # def split_text_by_period(text, max_length=200):
    #     '''
    #         Chunking not in use currently.
    #     '''
    #     sentences = re.split(r'\.\s+', text)  # Split by period + space
    #     chunks = []
    #     current_chunk = ""

    #     for sentence in sentences:
    #         if len(current_chunk) + len(sentence) < max_length:
    #             current_chunk += sentence + ". "
    #         else:
    #             current_chunk += sentence + "."
    #             chunks.append(current_chunk.strip())
    #             current_chunk = "" # Reset current chunk

    #     if current_chunk:
    #         chunks.append(current_chunk.strip())

    #     return chunks

    # def dataset_processing(self):
    #     '''
    #         Not Used Yet
    #     '''
    #     self.read_dataset()
    #     self.df[self.column_name + "_CLEANED"] = (
    #         self.df[self.column_name]
    #         .apply(lambda x: self.split_text_by_period(x))
    #     )
    #     self.df = self.df[
    #         ["index", self.column_name + "_CLEANED"]
    #     ].explode(self.column_name + "_CLEANED")