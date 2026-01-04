import os
from src.datascience import logger
from src.datascience.entity.config_entity import DataCleaningConfig
import pandas as pd
import numpy as np

class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config
    
    def clean_data(self):
        data = pd.read_csv(self.config.data_path)
        logger.info(data.shape)
        data=pd.get_dummies(data,columns=self.config.object_columns,drop_first=True)
        data[self.config.target_column]=np.where(data[self.config.target_column]=="Y",1,0)
        data=data.dropna()
        
        data.to_csv(os.path.join(self.config.root_dir, "credit_risk_dataset.csv"),index = False)
        logger.info("Data cleaning completed successfully")
        logger.info(data.shape)
        
        print(data.shape)