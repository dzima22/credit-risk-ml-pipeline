from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.data_cleaning import DataCleaning
from src.datascience import logger


STAGE_NAME="Data Cleaning Stage"
class DataCleaningTrainingPipeline:
    def __init__(self):
        pass
    def initiate_data_cleaning(self):
            config=ConfigurationManager()
            config_cleaning=config.get_data_cleaning_config()
            data_cleaning=DataCleaning(config=config_cleaning)
            data_cleaning.clean_data()
            