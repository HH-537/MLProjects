import os
import sys
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')
    logging.info('Assigned Preprocessor Path')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        logging.info('Create Data transformation config')

    @staticmethod
    def construct_preprocessor():
        try:
            numeric_cols = ["writing score", "reading score"]
            categorical_cols = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            numeric_pipeline = Pipeline(
                steps=[('SimpleImputer', SimpleImputer(strategy='median')),
                       ('Scaler', StandardScaler())
                       ]
            )

            categoric_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy="most_frequent")),
                       ('OneHotEncoder', OneHotEncoder()),
                       ('Scaler', StandardScaler(with_mean=False))]
            )

            preprocessor = ColumnTransformer(
                [('NumericPipeline', numeric_pipeline, numeric_cols),
                 ('CategoricPipeline', categoric_pipeline, categorical_cols)]
            )
            logging.info('Defined Pipeline and Preprocessor')
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            preprocessor = self.construct_preprocessor()
            logging.info('Created Preprocessor Instance')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read Train/Test Files')
            target_feat = 'math score'
            train_input_df = train_df.drop(columns=[target_feat], axis=1)
            train_target_df = train_df[target_feat]
            test_input_df = test_df.drop(columns=[target_feat], axis=1)
            test_target_df = test_df[target_feat]

            train_input_arr = preprocessor.fit_transform(train_input_df)
            test_input_arr = preprocessor.fit_transform(test_input_df)

            train_arr = np.c_[train_input_arr, np.array(train_target_df)]
            test_arr = np.c_[test_input_arr, np.array(test_target_df)]
            logging.info('Transformed Train/Test Data')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor
            )
            logging.info('Saved Preprocessor Object and Path')
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_path,
        except Exception as e:
            raise CustomException(e, sys)
