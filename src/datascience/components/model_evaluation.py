import os
import pandas as pd
from sklearn.metrics import (roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score)
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib

from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.constants import *
from src.datascience.utils.common import read_yaml, create_directories,save_json

#import os
#os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/dzima22/data_science_project.mlflow"
#os.environ["MLFLOW_TRACKING_USERNAME"]="dzima22"
#os.environ["MLFLOW_TRACKING_PASSWORD"]="47c9b2068d6b56a23dc8a7a9903055f99d55f61a"


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def eval_metrics(self, y_true, y_pred, y_proba):
        auc = roc_auc_score(y_true, y_proba)
        gini = 2 * auc - 1
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ks = np.max(tpr - fpr)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return auc,gini,ks,acc,prec,rec,f1
    
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]


        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)
            y_likelihood_of_1 = model.predict_proba(test_x)[:, 1]

            (auc,gini,ks,acc,prec,rec,f1) = self.eval_metrics(test_y, predicted_qualities,y_likelihood_of_1)
            
            # Saving metrics as local
            scores = {"auc": auc, "gini": gini, "ks": ks,"acc":acc,"prec":prec,"rec":rec,"f1":f1}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("auc", auc)
            mlflow.log_metric("gini", gini)
            mlflow.log_metric("ks", ks)
            mlflow.log_metric("acc", acc)
            mlflow.log_metric("prec", prec)
            mlflow.log_metric("rec", rec)
            mlflow.log_metric("f1", f1)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="XGBClassifier")
            else:
                mlflow.sklearn.log_model(model, "model")
    