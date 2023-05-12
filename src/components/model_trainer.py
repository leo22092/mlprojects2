import os
import sys
import pandas
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','Model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer=ModelTrainerConfig()
        

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
            "LinearRegression":LinearRegression(),
            # "Lasso":Lasso(),
            "Random Forest": RandomForestRegressor(),
            # "Ridge":Ridge(),
            "Gradient Boosting": GradientBoostingRegressor(),
            # "KNeighborsRegressor":KNeighborsRegressor(),
            "Decision Tree":DecisionTreeRegressor(),
            "XGBRegressor":XGBRegressor(),
            "CatBoostRegressor":CatBoostRegressor(verbose=False),
            "AdaBoostRegressor":AdaBoostRegressor()
                     }
            
            params={
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson']
                                },
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                                },
                "Gradient Boosting":{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256]
                                    },
                "CatBoostRegressor":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,50,100]
                                        },
                'LinearRegression':{},
                "XGBRegressor":{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                                },
                "AdaBoostRegressor":{
                    'learning_rate':[0.1,0.01,0.001],
                    'n_estimators':[8,16,32,64,256]
                                    }

                    }


            logging.info("Entering Evaluate function")
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)
            logging.info("evaluate function excecuted")

            # TO get the best model score

            best_model_score=max(sorted(model_report.values()))
            # To get the model name
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException (" No best model was found")
            logging.info(f"Best model found for both training and testing dataset")

            save_object(
                file_path=self.model_trainer.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            r2_squre=r2_score(y_test,predicted)
            return r2_squre

        except Exception as e:
            raise CustomException(e,sys)
        
