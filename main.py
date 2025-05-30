import os
import gc
import torch
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import globals
from model_trainer import ModelTrainer
from cof_processor import COFProcessor
from optimization_processor import OptimizationProcessor
from utility_functions import select_dataset, calculate_r2_score, explained_variance

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',   '--target',  help='Define the name of the desired target property.', default="nch4")
    
    parsed_args    = parser.parse_args()
    target_property   = parsed_args.target

    print(f"Using device: {globals.device}")

    # Set default tensor type to PRECISION
    torch.set_default_dtype(globals.PRECISION)

    np.random.seed(11)

    df, feature_columns, y_column = select_dataset(target_property)

    feature_columns_df = df[feature_columns]

    # Update the target variable used in the rest of the pipeline
    y = torch.from_numpy(df[y_column].values).double().to(globals.device)

    # Select features for X
    X = df[feature_columns]

    # Remove columns with constant values
    X = X.loc[:, X.nunique() > 1]  # Keeps only columns with more than one unique value

    # Convert each row to a tensor and stack them
    tensor_list = [COFProcessor.row_to_tensor(row) for _, row in X.iterrows()]
    X = torch.stack(tensor_list).double().to(globals.device)
    nb_COFs = X.shape[0]

    # Normalize X
    X_normalized = COFProcessor.min_max_normalize(X).to(globals.device).to(globals.PRECISION)

    # Delete `tensor_list` to free memory
    del tensor_list

    # Assertions to ensure proper normalization
    tolerance = 1e-6 if globals.PRECISION == torch.float64 else 1e-6
    min_vals = torch.min(X_normalized, dim=0).values
    max_vals = torch.max(X_normalized, dim=0).values
    assert torch.allclose(min_vals, torch.zeros(X.shape[1], dtype=globals.PRECISION, device=globals.device), atol=tolerance), "Minimum values after normalization are not close to zeros."
    assert torch.allclose(max_vals, torch.ones(X.shape[1], dtype=globals.PRECISION, device=globals.device), atol=tolerance), "Maximum values after normalization are not close to ones."
    print("Normalization successful.")

    X = X_normalized
    # Delete `X_normalized` to free memory
    del X_normalized, min_vals, max_vals
    torch.cuda.empty_cache()
    gc.collect()

    nb_runs = 50
    nb_init = globals.INITS

    cof_ids_to_sample = list(range(nb_COFs))
    central_cof = COFProcessor.get_initializing_COF(X)
    cof_ids_to_sample.pop(central_cof)
    seed_cofs = random.sample(cof_ids_to_sample, nb_runs-1)
    init_cof_ids = [COFProcessor.diverse_set(X, central_cof, nb_init)]
    init_cof_ids += [COFProcessor.diverse_set(X, seed_cof, nb_init) for seed_cof in seed_cofs]

    nb_COFs_initialization = 3
    nb_iterations = 2 * nb_COFs

    df['HF'] = np.random.uniform(low=20, high=800, size=len(df))
    cost_HF = df['HF'].values
    # cost = cost_HF
    cost = torch.tensor(cost_HF, dtype=globals.PRECISION, device=globals.device)

    print("\nraw data - \n\tX:", X.shape)
    print("\tfidelity: 1")
    print("\t\ty:", y.shape)
    print("\t\tcost: ", cost.shape)

    # Use updated BO run (set nb_iterations appropriately)
    acquired_set, bo_points_dict = OptimizationProcessor.run_Bayesian_optimization(nb_iterations, init_cof_ids[0], verbose=False, X=X, y=y)

    cof_ids = [int(acquired_set[i][0].item()) for i in range(len(acquired_set))]

    # cof_id_with_max_hi_fid_selectivity = np.argmax(y).item()
    cof_id_with_max_hi_fid_selectivity = np.argmax(y.cpu().numpy()).item()

    n_iter_top_cof_found = len(cof_ids)

    hi_fid_cofs = [cof_ids[i] for i in range(len(cof_ids))]

    # Prepare the training data from the acquired set
    X_train = COFProcessor.build_X_train(acquired_set[:n_iter_top_cof_found], X)
    y_train = COFProcessor.build_y_train(acquired_set[:n_iter_top_cof_found], y)

    # Train the surrogate model
    model = OptimizationProcessor.train_surrogate_model(X_train, y_train)

    # Make predictions for the entire dataset
    y_pred, sigma = OptimizationProcessor.mu_sigma(model, X)

    # Calculate evaluation metrics
    y_true = [y[c].item() for c in range(nb_COFs)]
    r2 = r2_score(y_true, y_pred)
    abserr = mean_absolute_error(y_true, y_pred)

    print("R² score:", r2)
    print("Mean Absolute Error:", abserr)

    # Identify the top 100 COFs based on the predicted values
    predicted_top_100_indices = np.argsort(y_pred)[-globals.TOP:]

    # Identify the actual top 100 COFs based on true values
    actual_top_100_indices = np.argsort(y.cpu().numpy())[-globals.TOP:]

    # Calculate how many of the top 100 predicted COFs are actually in the top 100 actual COFs
    top_100_pred_in_true = np.intersect1d(predicted_top_100_indices, actual_top_100_indices)
    count_top_100_pred_in_true = len(top_100_pred_in_true)

    # Standardize the entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.cpu().numpy())

    # Perform PCA on the standardized dataset
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Convert y_true to a NumPy array before calling the function
    y_true_np = np.array(y_true)
    y_pred_np = y_pred.cpu().numpy()  # Assuming y_pred is a tensor on CUDA

    r2_1 = calculate_r2_score(y_true_np, y_pred_np)
    print("R² score:", r2_1)

    r2_2 = explained_variance(y_true_np, y_pred_np)
    print("Explained Variance R² score:", r2_2)

    r2_sklearn = r2_score(y_true_np, y_pred_np)
    print("scikit-learn R² score:", r2_sklearn)

    # After Bayesian Optimization, we have acquired_set
    acquired_cof_ids = [int(acquired_set[i][0].item()) for i in range(len(acquired_set))]
    num_acquired_cofs = len(acquired_cof_ids)

    # Retrieve the rows from the original DataFrame
    acquired_data = df.iloc[acquired_cof_ids]

    # Prepare the data for XGBoost
    X_train = acquired_data[feature_columns].values
    y_train = acquired_data[y_column].values

    # Define the XGBoost regressor with the specified parameters
    model = globals.model

    # Train the XGBoost model
    model.fit(X_train, y_train)
    torch.cuda.empty_cache()
    gc.collect()

    # Prepare the entire dataset for prediction
    X_full = df[feature_columns].values
    y_full = df[y_column].values

    # Predict on the entire dataset
    y_pred_full = model.predict(X_full)

    # Calculate evaluation metrics for the entire dataset
    r2_full = r2_score(y_full, y_pred_full)
    mae_full = mean_absolute_error(y_full, y_pred_full)

    # Identify the top 100 COFs based on true methane uptake values
    top_100_true_values = y_full[actual_top_100_indices]

    # Identify the top 100 COFs based on predicted methane uptake values
    top_100_pred_indices = np.argsort(y_pred_full)[-globals.TOP:]
    top_100_pred_values = y_full[top_100_pred_indices]

    # Calculate how many of the actual top 100 COFs were predicted by XGBoost
    actual_top_100_in_predicted = np.intersect1d(actual_top_100_indices, top_100_pred_indices)
    count_actual_top_100_in_predicted = len(actual_top_100_in_predicted)

    # Calculate R^2 for the predicted values of the true top 100 COFs
    true_top_100_pred_values = y_pred_full[actual_top_100_indices]
    r2_top_100 = r2_score(top_100_true_values, true_top_100_pred_values)

    print(f'Number of actual top 100 COFs predicted by XGBoost: {count_actual_top_100_in_predicted}')
    print(f'R^2 for the predicted values of the true top 100 COFs: {r2_top_100:.2f}')

    # Extract the true and predicted values for COFs that are actually in the top 100
    true_top_100_pred_indices = np.intersect1d(actual_top_100_indices, actual_top_100_in_predicted)
    predicted_top_100_true_values = y_full[true_top_100_pred_indices]
    predicted_top_100_values = y_pred_full[true_top_100_pred_indices]

    # Check if predicted_top_100_true_values is empty
    if len(predicted_top_100_true_values) == 0:
        print("No predictions available for the top 100 COFs.")

    # Standardize the entire dataset
    # scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    # Extract the data points picked by Bayesian Optimization
    bo_picked_indices = [int(acquired_set[i][0].item()) for i in range(len(acquired_set))]

    # Calculate the actual top 100 COFs based on true values
    top_100_actual_indices = np.argsort(y.cpu().numpy())[-globals.TOP:]

    # Extract the COFs picked by Bayesian Optimization
    bo_picked_indices = [int(acquired_set[i][0].item()) for i in range(len(acquired_set))]

    # Calculate how many of the B.O. picked COFs are in the top 100 actual COFs
    bo_picked_in_top_100_actual = np.intersect1d(bo_picked_indices, top_100_actual_indices)
    count_bo_picked_in_top_100_actual = len(bo_picked_in_top_100_actual) 

    print(f'Number of B.O. picked COFs in top 100 actual COFs: {count_bo_picked_in_top_100_actual}')

    # save the dictionary of BO checkpoints
    with open("bo_points_dict.pkl", "wb") as f:
        pickle.dump(bo_points_dict, f)
    print("All BO checkpoint sets saved.")

    # Define output directory for results (one file per checkpoint)
    output_dir = 'random_sampling_plots_methane'
    os.makedirs(output_dir, exist_ok=True)

    model_trainer = ModelTrainer(df, feature_columns, y_column, bo_points_dict)

    # Now loop over each saved BO checkpoint (sample size)
    for sample_size in sorted(bo_points_dict.keys()):
        bo_sample = bo_points_dict[sample_size]  # This is a NumPy array of shape (n_points, 1)
        acquired_cof_ids = bo_sample[:, 0].astype(int).tolist()  # Convert to list of indices
        num_acquired_cofs = len(acquired_cof_ids)
        
        print(f"Processing BO checkpoint with {num_acquired_cofs} samples.")

        # DataFrame to store results; start with COF IDs and true target values for the actual top 100
        results_df = pd.DataFrame({'COF_ID': actual_top_100_indices,
                                'True_Methane_Uptake': y_full[actual_top_100_indices]})
        
        # Train xgboost model with random data points.
        model_trainer.run_random_xgboost(num_acquired_cofs, results_df)
        
        # Train xgboost model with bo selected data points.
        model_trainer.run_bo_xgboost(acquired_cof_ids, results_df)
        
        # Train gaussian processes model with bo selected data points
        model_trainer.run_bo_gp(acquired_cof_ids, results_df, X)

        # Optionally, compute ranking for each predictor column
        predictor_cols = [col for col in results_df.columns if col.startswith('Random_State_')]
        predictor_cols.append('BO_XGBoost')
        predictor_cols.append('BO_GP')
        for col in predictor_cols:
            results_df[col + '_Rank'] = results_df[col].rank(ascending=False, method='min')
        
        # Save the results DataFrame to an Excel file named with the current sample size
        output_file_path = os.path.join(output_dir, f'results_with_bo_predictions_{num_acquired_cofs}_samples.csv')
        results_df.to_csv(output_file_path, index=False)
        print(f"Results for sample size {num_acquired_cofs} saved to {output_file_path}")
        