from preprocessing import Preprocesser
from typing import Literal, Union
class FullExperimentPipeline:
    def __init__(self, preprocesser : Preprocesser, classifier):
        self.preprocesser = preprocesser
        self.classifier = classifier

    def run_IR(self, query="", 
        keep : Literal[
            "complaint",
            "recall",
            "all"
        ] = "all",
        dataset_to_check : Literal[
            "train",
            "val",
            "test"
        ] = "train",
        top=5):
        self.preprocesser.preprocess()
        if dataset_to_check == "train":
            vecs = self.classifier.fit_transform(self.preprocessor.x_train_vect)
        elif dataset_to_check == "val":
            vecs = self.classifier.transform(self.preprocessor.x_validation_vect)
        else:
            vecs = self.classifier.transform(self.preprocessor.x_test_vect)

        return self.classifier.find_similar(
            self.preprocessor.df_train,
            query,
            vecs
            keep=keep
            top=top
        )
    
