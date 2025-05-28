import math
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import ndcg_score

# -------------------------------
# Function to compute evaluation metrics from the first two cells
# -------------------------------
def compute_basic_metrics(df):
    true_value = "True_Methane_Uptake"
    results = []
    # Loop for random states (iterations)
    for i in range(20):
        column = f"Random_State_{i}"
        valid_rows = df[df[column].notna()]
        top_mofs_count = valid_rows.shape[0]
        # Calculate errors
        errors = np.abs(((valid_rows[column] - valid_rows[true_value]) / valid_rows[true_value]) * 100)
        # Defaults in case no MOFs are found
        max_predicted_value = np.nan
        corresponding_true_value = np.nan
        max_error = np.nan
        average_error_top100 = np.nan

        if top_mofs_count > 0:
            max_predicted_value = valid_rows[column].max()
            max_row = valid_rows.loc[valid_rows[column].idxmax()]
            corresponding_true_value = max_row[true_value]
            max_error = np.abs(((max_predicted_value - corresponding_true_value) / corresponding_true_value) * 100)
            average_error_top100 = errors.mean()

        results.append({
            'Iteration': i,
            'MOFs Found': top_mofs_count,
            'Max Predicted Value': max_predicted_value,
            'Corresponding True Value': corresponding_true_value,
            'Error for Top Predicted MOF (%)': max_error,
            'Average Error for Top-100 (%)': average_error_top100
        })

    res_df = pd.DataFrame(results)
    
    # Compute overall statistics for the random states
    overall_stats = {
        'avg_top100MOFs_found': res_df['MOFs Found'].mean(),
        'std_top100MOFs_found': res_df['MOFs Found'].std(),
        'overall_avg_error_top100': res_df['Average Error for Top-100 (%)'].mean(),
        'std_error_top100': res_df['Average Error for Top-100 (%)'].std(),
        'avg_max_actual_value': res_df['Corresponding True Value'].mean(),
        'std_max_actual_value': res_df['Corresponding True Value'].std(),
        'avg_error_max_predicted': res_df['Error for Top Predicted MOF (%)'].mean(),
        'std_error_max_predicted': res_df['Error for Top Predicted MOF (%)'].std()
    }
    return overall_stats

# -------------------------------
# Function to compute BO_XGBoost metrics (second cell)
# -------------------------------
def compute_bo_metrics(df):
    true_value = "True_Methane_Uptake"
    column = 'BO_XGBoost'
    valid_rows = df[df[column].notna()]
    top_mofs_count = valid_rows.shape[0]
    if top_mofs_count > 0:
        max_predicted_value = valid_rows[column].max()
        max_row = valid_rows.loc[valid_rows[column].idxmax()]
        corresponding_true_value = max_row[true_value]
        Max_Possible_Value = df["True_Methane_Uptake"].max()
        max_error = np.abs(((max_predicted_value - corresponding_true_value) / corresponding_true_value) * 100)
        errors = np.abs(((valid_rows[column] - valid_rows[true_value]) / valid_rows[true_value]) * 100)
        average_error_top100 = errors.mean()
    else:
        max_predicted_value = np.nan
        corresponding_true_value = np.nan
        Max_Possible_Value = np.nan
        max_error = np.nan
        average_error_top100 = np.nan

    return {
        'BO_top_mofs_found': top_mofs_count,
        'BO_max_predicted_value': max_predicted_value,
        'BO_corresponding_true_value': corresponding_true_value,
        'BO_max_error': max_error,
        'BO_avg_error_top100': average_error_top100,
        'BO_actual_max_value': Max_Possible_Value
    }

# -------------------------------
# Function to compute a ranking metric (e.g., NDCG) from the third cell
# -------------------------------
def compute_ndcg_bo(df):
    # Ensure the expected ranking column exists
    if 'BO_XGBoost_Rank' not in df.columns:
        return np.nan

    true_value = "True_Methane_Uptake"
    # Sort the COFs by true value descending (as in your original cell)
    true_data_sorted = df[['COF_ID', true_value]].sort_values(by=true_value, ascending=False)
    top_k = true_data_sorted.shape[0]
    upper_bound = top_k + 1
    # Create the true relevance: highest true value gets highest relevance
    true_relevance = np.asarray([list(range(top_k, 0, -1))])
    
    # For predicted relevance, iterate in the sorted order
    predicted_relevance = []
    for cof in true_data_sorted['COF_ID']:
        val = df.loc[df['COF_ID'] == cof, 'BO_XGBoost_Rank'].values
        if len(val) > 0 and not math.isnan(val[0]):
            rank = int(val[0])
        else:
            rank = upper_bound  # worst rank if not found
        # Convert rank to relevance (higher relevance for lower rank)
        predicted_relevance.append(upper_bound - rank)
    
    predicted_relevance = np.array([predicted_relevance])
    ndcg = ndcg_score(true_relevance, predicted_relevance)
    return ndcg

def compute_ndcg_random_all(df):
    """Compute the average iNDCG over all 20 random runs using the same logic as for BO."""
    true_value = "True_Methane_Uptake"
    # Sort COFs by true value descending
    true_data_sorted = df[['COF_ID', true_value]].sort_values(by=true_value, ascending=False)
    top_k = true_data_sorted.shape[0]
    upper_bound = top_k + 1
    # True relevance: highest true value gets the highest relevance
    true_relevance = np.asarray([list(range(top_k, 0, -1))])
    
    total_ndcg = 0
    count = 0
    for i in range(20):
        col_rank = f"Random_State_{i}_Rank"
        if col_rank in df.columns:
            predicted_relevance = []
            # For each COF in the sorted order, get its predicted rank
            for cof in true_data_sorted['COF_ID']:
                val = df.loc[df['COF_ID'] == cof, col_rank].values
                if len(val) > 0 and not math.isnan(val[0]):
                    rank = int(val[0])
                else:
                    rank = upper_bound  # assign worst rank if missing
                # Convert rank to relevance
                predicted_relevance.append(upper_bound - rank)
            predicted_relevance = np.array([predicted_relevance])
            ndcg = ndcg_score(true_relevance, predicted_relevance)
            total_ndcg += ndcg
            count += 1
    if count > 0:
        return total_ndcg / count
    else:
        return np.nan

# -------------------------------
# Function to combine all metric computations from a given CSV file
# -------------------------------
def compute_metrics_from_file(filepath):
    df = pd.read_csv(filepath)
    metrics = {}
    metrics.update(compute_basic_metrics(df))
    metrics.update(compute_bo_metrics(df))
    metrics['ndcg_bo'] = compute_ndcg_bo(df)
    metrics['ndcg_random_all'] = compute_ndcg_random_all(df)
    return metrics