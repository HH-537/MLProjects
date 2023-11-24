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
    logging.INFO('Assigned Preprocessor Path')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        logging.INFO('Create Data transformation config')

    def construct_preprocessor(self):
        try:
            numeric_cols = ["writing_score", "reading_score"]
            categorical_cols = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numeric_pipeline = Pipeline(
                steps=[('SimpleImputer', SimpleImputer(strategy='median')),
                       ('Scaler', StandardScaler())
                       ]
            )

            categoric_pipeline = Pipeline(
                steps=[('OneHotEncoder', OneHotEncoder()),
                       ('Scaler', StandardScaler())]
            )

            preprocessor = ColumnTransformer(
                [('NumericPipeline', numeric_pipeline, numeric_cols),
                 ('CategoricPipeline', categoric_pipeline, categorical_cols)]
            )
            logging.INFO('Defined Pipeline and Preprocessor')
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            preprocessor = self.construct_preprocessor()
            logging.INFO('Created Preprocessor Instance')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.INFO('Read Train/Test Files')
            target_feat = 'math_score'
            train_input_df = train_df.drop(cols=[target_feat], axis=1)
            train_target_df = train_df[target_feat]
            test_input_df = test_df.drop(cols=[target_feat], axis=1)
            test_target_df = test_df[target_feat]

            train_input_arr = preprocessor.fit_transform(train_input_df)
            test_input_arr = preprocessor.fit_transform(test_input_df)

            train_arr = np.c_[train_input_arr, np.array(train_target_df)]
            test_arr = np.c_[test_input_arr, np.array(test_target_df)]
            logging.INFO('Transformed Train/Test Data')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor
            )
            logging.INFO('Saved Preprocessor Object and Path')
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path,
        except Exception as e:
            raise CustomException(e, sys)
