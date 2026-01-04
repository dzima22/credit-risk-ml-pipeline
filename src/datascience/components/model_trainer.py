import pandas as pd
import os
from src.datascience import logger
from xgboost import XGBClassifier
import joblib

from src.datascience.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        lr = XGBClassifier( n_estimators=self.config.n_estimators, learning_rate=self.config.learning_rate,max_depth=self.config.max_depth,min_child_weight=self.config.min_child_weight,
                            subsample=self.config.subsample,colsample_bytree=self.config.colsample_bytree,gamma=self.config.gamma,reg_alpha=self.config.reg_alpha,reg_lambda=self.config.reg_lambda,
                            scale_pos_weight=self.config.scale_pos_weight,objective=self.config.objective,eval_metric=self.config.eval_metric,random_state=self.config.random_state)
        lr.fit(train_x, train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))

    