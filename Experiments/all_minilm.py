from sentence_transformers import SentenceTransformer
import torch
from typing import Union, Literal
import os
import re
import pickle
import pandas as pd
# import numpy as np

class HuggingFaceClassifier:
    def __init__(self, dataset_path, model_name, column_name,
                #  embedding_path=None,
                 similarity_fn_name : Union[
                            "cosine", "dot"
                ] = "cosine"):
        self.column_name = column_name
        self.model_name = model_name
        self.dataset_path = os.path.abspath(dataset_path)
        dataset_dir, basename = self.dataset_path.rsplit(
            os.sep, maxsplit=1
        )
        self.df = pd.read_csv(self.dataset_path)

        # embeddings and models saved here
        self.model_path = os.path.join(dataset_dir, f"huggingface", self.column_name)

        # Load sentence transformer if it has already been saved
        if os.path.exists(os.path.join(self.model_path, self.model_name)):
            self.model = SentenceTransformer(os.path.join(self.model_path, self.model_name),
                                             similarity_fn_name=similarity_fn_name)
        else:
            os.makedirs(self.model_path, exist_ok=True)
            self.model = SentenceTransformer(self.model_name,
                                             similarity_fn_name=similarity_fn_name)
            self.model.save(os.path.join(self.model_path, self.model_name))

    def split_text_by_period(text, max_length=200):
        '''
            Chunking not in use currently.
        '''
        sentences = re.split(r'\.\s+', text)  # Split by period + space
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                current_chunk += sentence + "."
                chunks.append(current_chunk.strip())
                current_chunk = "" # Reset current chunk

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def dataset_processing(self):
        '''
            Not Used Yet
        '''
        self.df[self.column_name + "_CLEANED"] = (
            self.df[self.column_name]
            .apply(lambda x: self.split_text_by_period(x))
        )
        self.df = self.df[
            ["index", self.column_name + "_CLEANED"]
        ].explode(self.column_name + "_CLEANED")

    def encode_embeddings(self):
        '''
        TODO: Consider GPU acceleration
        '''
        
        # Put preprocessing Here

        # Encoding here
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            # Load model, dataframe, and encode text.
            self.embeddings = self.model.encode(self.df[self.column_name], show_progress_bar=True)
            with open(os.path.join(self.model_path, f"{self.model_name}-embeddings.pkl"), "wb") as f:
                pickle.dump(self.embeddings, f)
        else:
            print("Embeddings found. Loading embeddings.")
            with open(os.path.join(self.model_path, f"{self.model_name}-embeddings.pkl"), "rb") as f:
                self.embeddings = pickle.load(f)
    
    def query_embeddings(self, query, 
                         keep : Literal[
                            "complaint",
                            "recall",
                            "all"
                        ] = "all", top=20):
        query_embedding = self.model.encode(query)
        if keep == "recall":
            filter_indices = self.df[self.df["IS_COMPLAINT"] == 0].index
        elif keep == "complaint":
            filter_indices = self.df[self.df["IS_COMPLAINT"] == 1].index
        else:
            filter_indices = self.df.index

        model_similarities = self.model.similarity(
            query_embedding,
            self.embeddings[filter_indices]
        )
        values, most_similar_indices = torch.topk(model_similarities, largest=True, k=top, dim=1)
        docs_to_return = self.df.loc[filter_indices].iloc[most_similar_indices[0].tolist()]
        docs_to_return["similarity"] = values.squeeze()
        return docs_to_return

    def run_training_pipeline(self):
        '''
            Technically no training is done.
            Strictly speaking, this is simply a processing pipeline
        '''
        self.encode_embeddings()