import torch
import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import globals
from optimization_processor import OptimizationProcessor

class ModelTrainer:
    def __init__(self, data: pd.DataFrame, feature_columns: list, y_column: str, bo_points_dict: dict):

        self.data            = data
        self.y_column        = y_column
        self.feature_columns = feature_columns
        self.bo_points_dict  = bo_points_dict

        self.xgb_model       = XGBRegressor(
                                            n_estimators=800,
                                            max_depth=5,
                                            eta=0.02,
                                            subsample=0.75,
                                            colsample_bytree=0.7,
                                            reg_lambda=0.6,
                                            reg_alpha=0.15,
                                            random_state=61
                                        )

        self.X_full                 = data[feature_columns].values
        self.y_full                 = data[y_column].values
        self.actual_top_100_indices = np.argsort(self.y_full)[-globals.TOP:]


    def run_bo_xgboost(self, acquired_cof_ids, results_df):
        
        acquired_data = self.data.iloc[acquired_cof_ids]
        X_train = acquired_data[self.feature_columns].values
        y_train = acquired_data[self.y_column].values
        
        self.xgb_model.fit(X_train, y_train)
        
        y_bo_xgb_pred_full = self.xgb_model.predict(self.X_full)
        top_100_bo_xgb_pred_indices = np.argsort(y_bo_xgb_pred_full)[-globals.TOP:]
        actual_top_100_in_bo = np.intersect1d(self.actual_top_100_indices, top_100_bo_xgb_pred_indices)
        
        results_df['BO_XGBoost'] = pd.Series([np.nan]*len(results_df))
        if len(top_100_bo_xgb_pred_indices) > 0:
            for idx in top_100_bo_xgb_pred_indices:
                results_df.loc[results_df['COF_ID'] == idx, 'BO_XGBoost'] = y_bo_xgb_pred_full[idx]


    def run_bo_gp(self, acquired_cof_ids, results_df, X):

        # Train a surrogate GP model on the BO-acquired points
        acquired_data = self.data.iloc[acquired_cof_ids]
        X_train = torch.tensor(acquired_data[self.feature_columns].values, dtype=globals.PRECISION, device=globals.device)
        y_train = torch.tensor(acquired_data[self.y_column].values, dtype=globals.PRECISION, device=globals.device).unsqueeze(-1)
        
        model = OptimizationProcessor.train_surrogate_model(X_train, y_train)        
        y_bo_gp_pred_full, _ = OptimizationProcessor.mu_sigma(model, X)

        y_bo_gp_pred_full = y_bo_gp_pred_full.cpu().numpy()
        top_100_bo_gp_pred_indices = np.argsort(y_bo_gp_pred_full)[-globals.TOP:]

        results_df['BO_GP'] = pd.Series([np.nan]*len(results_df))
        if len(top_100_bo_gp_pred_indices) > 0:
            for idx in top_100_bo_gp_pred_indices:
                results_df.loc[results_df['COF_ID'] == idx, 'BO_GP'] = y_bo_gp_pred_full[idx]

    def run_random_xgboost(self, num_acquired_cofs, results_df):

        # Loop through 20 different random states and run a random selection
        for random_state in range(20):
            random.seed(random_state)
            # Randomly select the same number of COFs as acquired by BO
            random_cof_ids = random.sample(range(len(self.data)), num_acquired_cofs)
            random_data = self.data.iloc[random_cof_ids]
            X_random_train = random_data[self.feature_columns].values
            y_random_train = random_data[self.y_column].values

            self.xgb_model.fit(X_random_train, y_random_train)
            y_random_pred_full = self.xgb_model.predict(self.X_full)

            top_100_random_pred_indices = np.argsort(y_random_pred_full)[-globals.TOP:]
            actual_top_100_in_random = np.intersect1d(self.actual_top_100_indices, top_100_random_pred_indices)
            
            col_name = f'Random_State_{random_state}'
            # For each of the actual top 100 indices, if the random model predicted it among the top 100, record the prediction
            results_df[col_name] = pd.Series([np.nan]*len(results_df))
            if len(actual_top_100_in_random) > 0:
                for idx in actual_top_100_in_random:
                    results_df.loc[results_df['COF_ID'] == idx, col_name] = y_random_pred_full[idx]