import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer


from src.exception import CustomException
from src.logger import logging
from src.utils import Save_Object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config =DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['price_per_sqft','bhk','bath']
            categorical_columns = ['total_sqft']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(

                 steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore")),
                    ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Numerical column {numerical_columns}")

            logging.info(f"Categorical column {categorical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            logging.info("preprocessor is completed")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print(train_df.head(2))
            print(test_df.head(2))
            print("train_df shape:", train_df.shape)
            print("test_df shape:", test_df.shape)


            logging.info ("read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj =self.get_data_transformer_object()
            
          
            target_column_name="price"
            numerical_columns = ['bath', 'balcony']
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=test_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            

            logging.info("Applying Preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            print("train_arr shape:", input_feature_train_arr.shape)
            print("test_arr shape:", input_feature_test_arr.shape)

            

        
             

            target_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            print("input_feature_train_arr shape:", input_feature_train_arr.shape)
            print("input_feature_test_arr shape:", input_feature_test_arr.shape)

            #test_arr = np.concatenate((input_feature_test_arr, target_test_arr))

           # print("test_arr shape:", test_arr.shape)
            

            
            print("target_train_arr shape:", target_train_arr.shape)
            print("target_test_arr shape:", target_test_arr.shape)

           # df.to_csv('data/02_final_bhpp_model.csv')
            
                       
            #logging.info(f" target_feature_train_df {target_feature_train_df}")
            #logging.info(f" target_feature_train_df {target_feature_test_df}")

            #test_arr = np.concatenate((input_feature_test_arr, target_test_arr), axis=1)
            #print("test_arr",test_arr.shape)
                        
            #train_arr = np.concatenate((input_feature_train_arr, target_train_arr), axis=1)
            #print("train_arr",train_arr.shape)


            train_arr = np.c_[input_feature_train_arr, target_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_test_arr]

            #print("train_arr",train_arr.shape)
            #print("test_arr",test_arr.shape)

        

            #train_arr = np.concatenate([np.concatenate(input_feature_train_arr), np.concatenate(target_feature_train_arr)], axis=1)
            #test_arr = np.concatenate([np.concatenate(input_feature_test_arr), np.concatenate(target_feature_test_arr)], axis=1)
            #test_arr = np.concatenate([input_feature_test_arr, np.array(target_feature_test_df)], axis=1)

            logging.info("saved preprocessing object")

            Save_Object (
                
                filepath=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                
                train_arr,
                test_arr,

                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
            
