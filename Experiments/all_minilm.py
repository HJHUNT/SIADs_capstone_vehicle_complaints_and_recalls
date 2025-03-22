from sentence_transformers import SentenceTransformer
import torch
from typing import Union

class HuggingFaceEmbeddings:
    def __init__(self, dataset_path, model_name, column_name,
                 similarity_fn_name : Union[
                            "cosine", "dot"
                ] = "cosine"):
        self.column_name = column_name
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.df = pd.read_csv(os.path.join(self.dataset_path, "complaints_and_recalls.csv"))
        self.model_path = os.path.join(self.dataset_path, "huggingface", 
                                      self.column_name)

        # Load sentence transformer if it has already been saved
        if os.path.exists(os.path.join(self.model_path, "model")):
            self.model = SentenceTransformer(os.path.join(self.model_path, "model"),
                                             similarity_fn_name=similarity_fn_name)
        else:
            self.model = SentenceTransformer(self.model_name,
                                             similarity_fn_name=similarity_fn_name)

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
            os.makedirs(self.model_path, exists_ok=True)
            # Load model, dataframe, and encode text.
            self.embeddings = self.model.encode(self.df[self.column_name], show_progress_bar=True)
            with open(os.path.join(self.model_path, f"{self.model_name}-embeddings.pkl"), "wb") as f:
                pickle.dump(self.embeddings, f)
        else:
            print("Embeddings found. Loading embeddings.")
            with open(os.path.join(self.model_path, f"{self.model_name}-embeddings.pkl"), "rb") as f:
                self.embeddings = pickle.load(f)
    
    def query_embeddings(self, query, top=20):
        query_embedding = self.model.encode(query)
        model_similarities = self.model.similarity(
            query_embedding,
            self.embeddings
        )
        values, indices = torch.topk(model_similarities, largest=True, k=top, dim=1)
        return self.df.iloc[indices[0].tolist()], values

    def run_training_pipeline(self):
        '''
            Technically no training is done.
            Strictly speaking, this is simply a processing pipeline
        '''
        self.encode_embeddings()

