import os 
import sys 
import numpy as np 
import pandas as pd 
from src.exception import CustomException
import dill 
from src.logger import logging
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,root_mean_squared_error
from sklearn.model_selection import GridSearchCV

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb' )as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluation(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mae, mse, rmse, r2
def evaluate_models(X_train, y_train, X_test, y_test, models,params):
    """
    Trains all models, evaluates them and returns a report
    containing the test R² score for each model.
    """

    try:

        report = {}

        for name, model in models.items():

            logging.info(f"Training {name}")
            para=params[name]

            gs=GridSearchCV(estimator=model,param_grid=para,cv=3,scoring='r2',n_jobs=-1)

            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_mae, train_mse, train_rmse, train_r2 = evaluation(
                y_train,
                y_train_pred
            )

            test_mae, test_mse, test_rmse, test_r2 = evaluation(
                y_test,
                y_test_pred
            )

            logging.info(
                f"{name} | "
                f"Train R2: {train_r2:.4f} | "
                f"Test R2: {test_r2:.4f}"
            )

            print(f"\n{name}")
            print("-" * 40)

            print("TRAIN")
            print(
                f"MAE : {train_mae:.4f}\n"
                f"MSE : {train_mse:.4f}\n"
                f"RMSE: {train_rmse:.4f}\n"
                f"R2  : {train_r2:.4f}"
            )

            print("\nTEST")
            print(
                f"MAE : {test_mae:.4f}\n"
                f"MSE : {test_mse:.4f}\n"
                f"RMSE: {test_rmse:.4f}\n"
                f"R2  : {test_r2:.4f}"
            )

            report[name] = test_r2


        return report

    except Exception as e:
        raise CustomException(e, sys)