import gc
import torch
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import globals
from cof_processor import COFProcessor
from optimization_processor import OptimizationProcessor
from utility_functions import calculate_r2_score, explained_variance

if __name__ == "__main__":
    
    print(f"Using device: {globals.device}")

    # Set default tensor type to PRECISION
    torch.set_default_dtype(globals.PRECISION)

    np.random.seed(11)

    df = pd.read_csv('HypoCOF-CH4H2-CH4-1bar-TPOT-Input-B - Original.csv')
    # df = pd.read_csv('HypoCOF-CH4H2-H2-1bar-TPOT-Input-B - Original.csv')

    # Drop the 'MOF_no' column as it is no longer needed
    # df.drop(columns=['number'], inplace=True)
    row_count = df.shape[0]
    print("Number of rows in the dataset:", row_count)
    df.dropna(inplace=True)

    row_count = df.shape[0]
    print("Number of rows in the dataset:", row_count)

    # Display the column names
    print("Column names in the file:")
    print(df.columns.tolist())

    feature_columns = [  'PLD (Å)', 'LCD (Å)', 'Sacc (m2/gr)', 'Porosity', 'Density (gr/cm3)',
            # 'Qst-CH4 (kJ/mol)',
            '%C', '%F', '%H', '%N', '%O', '%S', '%Si'
            ]

    feature_columns_df = df[feature_columns]

    y_column = 'NCH4 - 1 bar (mol/kg)'
    # y_column = 'NH2 - 1 bar (mol/kg)'


    y = torch.from_numpy(df[y_column].values).double().to(globals.device)  # Update the target variable used in the rest of the pipeline

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
    acquired_set, bo_points_dict = OptimizationProcessor.run_Bayesian_optimization(nb_iterations, init_cof_ids[0], verbose=False, target_variable=y)

    cof_ids = [int(acquired_set[i][0].item()) for i in range(len(acquired_set))]

    # cof_id_with_max_hi_fid_selectivity = np.argmax(y).item()
    cof_id_with_max_hi_fid_selectivity = np.argmax(y.cpu().numpy()).item()

    n_iter_top_cof_found = len(cof_ids)

    hi_fid_cofs = [cof_ids[i] for i in range(len(cof_ids))]

    # Prepare the training data from the acquired set
    X_train = COFProcessor.build_X_train(acquired_set[:n_iter_top_cof_found])
    y_train = COFProcessor.build_y_train(acquired_set[:n_iter_top_cof_found])

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

    gridspec_kw = {'width_ratios': [6, 2], 'height_ratios': [2, 6]}
    fig, ax = plt.subplots(nrows=2, ncols=2, gridspec_kw=gridspec_kw, figsize=(12, 12))  # Increased figure size
    ax[0, 1].axis("off")

    ax[1, 0].plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle="--", color="k", linewidth=1)
    ax[1, 0].text(min(y_true) + (max(y_true) - min(y_true)) * 0.05, max(y_true) * 0.9, "R$^2$=%.2f\nMAE=%.2f" % (r2, abserr))
    ax[1, 0].scatter(y_true, y_pred, fc='none', ec="k")
    ax[1, 0].set_xlabel("True Methane Uptake High P")
    ax[1, 0].set_ylabel("Predicted Methane Uptake High P")

    hist_color = sns.color_palette("husl", 8)[7]
    ax[0, 0].hist(y_true, color=hist_color, alpha=0.5)
    ax[0, 0].sharex(ax[1, 0])
    ax[0, 0].set_ylabel('# COFs')
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)

    ax[1, 1].hist(y_pred, color=hist_color, alpha=0.5, orientation="horizontal")
    ax[1, 1].sharey(ax[1, 0])
    ax[1, 1].set_xlabel('# COFs')
    plt.setp(ax[1, 1].get_yticklabels(), visible=False)

    sns.despine()
    plt.tight_layout()
    plt.show()

    # Standardize the entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.cpu().numpy())

    # Perform PCA on the standardized dataset
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot the PCA results
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, label='All COFs')

    # Highlight the top 100 predicted COFs with blue X
    plt.scatter(X_pca[predicted_top_100_indices, 0], X_pca[predicted_top_100_indices, 1], color='blue', marker='x', label='Top 100 Predicted COFs')

    # Highlight the actual top 100 COFs with red +
    plt.scatter(X_pca[actual_top_100_indices, 0], X_pca[actual_top_100_indices, 1], color='red', marker='+', label='Top 100 Actual COFs')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'PCA of COFs with Top 100 Predicted and Actual COFs Highlighted\n'
            f'Number of top 100 predicted COFs in top 100 actual: {count_top_100_pred_in_true}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
    model = XGBRegressor(
        n_estimators=800,
        max_depth=5,
        eta=0.02,
        subsample=0.75,
        colsample_bytree=0.7,
        reg_lambda=0.6,
        reg_alpha=0.15,
        random_state=61
    )

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

    # Plot predicted vs true values for the entire dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(y_full, y_pred_full, edgecolor='k', alpha=0.7)
    plt.plot([min(y_full), max(y_full)], [min(y_full), max(y_full)], '--', color='gray')
    plt.xlabel('True Methane Uptake (High P)')
    plt.ylabel('Predicted Methane Uptake (High P)')
    plt.title(f'Predicted vs True Methane Uptake on Entire Dataset\nR^2: {r2_full:.2f}, MAE: {mae_full:.2f}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
    else:
        # Plot predicted vs true values for the COFs that are actually in the top 100
        plt.figure(figsize=(8, 6))
        plt.scatter(predicted_top_100_true_values, predicted_top_100_values, edgecolor='k', alpha=0.7)
        plt.plot([min(predicted_top_100_true_values), max(predicted_top_100_true_values)],
                [min(predicted_top_100_true_values), max(predicted_top_100_true_values)], '--', color='gray')
        plt.xlabel('True Methane Uptake (High P) - Top 100 COFs')
        plt.ylabel('Predicted Methane Uptake (High P) - Top 100 COFs')
        plt.title(f'Predicted vs True Methane Uptake for Top 100 COFs\nR^2: {r2_top_100:.2f}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Standardize the entire dataset
    # scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    # Plot the PCA results
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, label='All COFs')

    # Highlight the top 100 predicted COFs with blue X
    plt.scatter(X_pca[top_100_pred_indices, 0], X_pca[top_100_pred_indices, 1], color='blue', marker='x', label='Top 100 Predicted COFs')

    # Highlight the actual top 100 COFs with red +
    plt.scatter(X_pca[actual_top_100_indices, 0], X_pca[actual_top_100_indices, 1], color='red', marker='+', label='Top 100 Actual COFs')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'PCA of COFs with Top 100 Predicted and Actual COFs Highlighted, Number of actual top 100 COFs predicted by XGBoost: {count_actual_top_100_in_predicted}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Extract the data points picked by Bayesian Optimization
    bo_picked_indices = [int(acquired_set[i][0].item()) for i in range(len(acquired_set))]

    # Plot the PCA results
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, label='All COFs')

    # Highlight the top 100 predicted COFs with blue X
    plt.scatter(X_pca[predicted_top_100_indices, 0], X_pca[predicted_top_100_indices, 1], color='blue', marker='x', label='Top 100 Predicted COFs')

    # Highlight the actual top 100 COFs with red +
    plt.scatter(X_pca[actual_top_100_indices, 0], X_pca[actual_top_100_indices, 1], color='red', marker='+', label='Top 100 Actual COFs')

    # Highlight the B.O. picked data points with green X
    plt.scatter(X_pca[bo_picked_indices, 0], X_pca[bo_picked_indices, 1], color='green', marker='x', label='B.O. Picked COFs')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'PCA of COFs with Top 100 Predicted and Actual COFs Highlighted By G.P.\n'
            f'Number of top 100 predicted COFs in top 100 actual: {count_top_100_pred_in_true}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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