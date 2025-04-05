# In[1]
from sklearn.metrics import (
    f1_score, roc_curve, auc, confusion_matrix,
    classification_report
)
from helpers.pickle_decorator import pickle_io
from helpers.utilities import get_dataset_dir
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Union
import os
import sys


sys.path.insert(0, os.path.abspath(".."))
from Experiments.preprocessing import Preprocesser

class Classifier:
    def __init__(self, X_train, y_train,
                 X_test, y_test, classifier,
                 custom_classifier_name,
                 classifier_params=dict(
                     random_state=42
                 ),
                 rerun=True
                 ):
        self.classifier = classifier(**classifier_params)
        self.classifier_params = classifier_params
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test 
        self.custom_classifier_name = custom_classifier_name
        self.dataset_dir = get_dataset_dir()
        self.base_path = os.path.join(
            self.dataset_dir, 
            "classifier",
            type(self.classifier).__name__
        )
        self.rerun = rerun


    def fit(self):
        self.classifier = pickle_io(
            lambda: self.classifier.fit(self.X_train, self.y_train),
            file_path=os.path.join(
                self.base_path, 
                f"{self.custom_classifier_name}.pkl"
            ),
            metadata=self.classifier_params,
            rerun=self.rerun
        )()
    
    def predict(self, query : str, vectorizer,
                text_process_function=lambda x: x,
                text_process_params=dict(),
                is_proba=False
                ):
        query = text_process_function(query, **text_process_params)
        query_vector = vectorizer.transform([query])

        if is_proba is False:
            return self.classifier.predict(query_vector)
        else:
            return self.classifier.predict_proba(query_vector)

    def evaluate(self):
        self.y_pred = self.classifier.predict(self.X_test)
        return classification_report(self.y_test, self.y_pred)
    
    def get_top_stopwords(
        self,
        vectorizer,
        type="largest",
    ):
        df_feature_importance = self.get_feature_importance(
            vectorizer,
            type=type,
            top=50
        )
        top_50_words = df_feature_importance["feature_names"].tolist()
        stopwords_top = [stopword for stopword in top_50_words if stopword.find(" ") == -1]
        return stopwords_top
                
    def get_feature_importance(
        self,
        vectorizer,
        type : Union["smallest", "largest"],
        top=10
    ):
        self.lr_coef = pd.DataFrame(
            {
                "feature_names" : vectorizer.get_feature_names_out(),
                "weights" : self.classifier.coef_.flatten()
            }
        ).sort_values("weights", ascending=False)
        if type == "largest":
            local_lr_coef = self.lr_coef.head(top)
        else:
            local_lr_coef = self.lr_coef.tail(top).sort_values("weights", ascending=True)
        
        return local_lr_coef
    
    def plot_feature_importance(
        self,
        vectorizer,
        type : Union["smallest", "largest"],
        top=10
    ):
        df_feature_importance = self.get_feature_importance(
            vectorizer,
            type=type,
            top=top
        )
        fig, ax = plt.subplots(1, 1, figsize=(10,8), dpi=100)
        ax.bar(
            x=df_feature_importance["feature_names"],
            height=df_feature_importance["weights"],
            color="green" if type == "largest" else "red"
        )
        _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel("LR Coefficients")
        ax.set_xlabel("Feature Names")
        _ = ax.set_title(f"{top} {type} Coefficients",
                    ha="left", x=0, fontsize=17, fontweight=900,
                    va="top", y=1.05)
        return ax

        
        
    def plot_heatmap(self):
        self.y_pred = self.classifier.predict(self.X_test)
        c_matrix = confusion_matrix(self.y_test, self.y_pred)
        ax = sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues')
        _ = ax.set_title("Confusion Matrix")
        _ = ax.set_xticklabels(["Predicted Complaint", "Predicted Recall"])
        _ = ax.set_yticklabels(["Actual Complaint", "Actual Recall"])
        return ax

    def plot_roc_auc_score(self):
        self.y_pred_proba = self.classifier.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        self.roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {self.roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        return ax
# %%
