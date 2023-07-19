import pandas as pd
import numpy as np
from utils import clustering_dbscan
from sklearn.neighbors import NearestNeighbors


def from_list_to_stats(metrics_dict: dict):
    """
    Calculate the descriptive statistics of single day clustering metrics for the quarter
    """
    metrics = ['noise_m_d', 'n_clusters', 'n_noise']
    for metric in metrics:
        list_metric = pd.Series(list(metrics_dict[metric]))
        m = (round(list_metric.mean(), 2), list_metric.max(), list_metric.min())
        metrics_dict[metric] = m
    return metrics_dict


def combine_keys_and_values(collections_features: dict):
    """
    Add key with combination of avaliable collection. 
    -Collection names;
    -Collection features.
    """
    key = f'{list(collections_features.items())[0][0]}_{list(collections_features.items())[1][0]}'
    value = list(collections_features.values())[0] + list(collections_features.values())[1]
    collections_features[key] = value
    return collections_features


def from_dict_to_df(results: dict):
    """
    Convert nested dictionary to multiindex dataframe.
    """
    df_dict = {}
    for outerKey, innerDict in results.items():
        for innerKey, values in innerDict.items():
            df_dict[(outerKey, innerKey)] = values
    dataframe = pd.DataFrame(df_dict)
    return dataframe


def transpose_df(dataframe: pd.DataFrame):
    """
    Transpose and rearrange metrics dataframe.
    """
    dataframe_ = dataframe.transpose().reset_index()
    dataframe_.columns = ['param', 'metric', 'mean', 'max', 'min']
    return dataframe_


def noise_mean_distance(dbscan_model_, scaled):
    """
    Clustering metric: Mean distance between noise points and its 6 Nearest Neighbours (6-NN)'.
    
    Parameters
    ----------
    dbscan_model_: Fitted DBSCAN Model.
    scaled: np.Array with standardized features.
    
    Returns
    ----------
    noise_m_distance: float score, single value per fitted model.
    """
    noise_indices = dbscan_model_.labels_ == -1
    if True in noise_indices:
        neighboors = NearestNeighbors(n_neighbors=6).fit(scaled)
        distances, indices = neighboors.kneighbors(scaled)
        noise_distances = distances[noise_indices, 1:]
        noise_m_distance = round(noise_distances.mean(), 3)
    else:
        noise_m_distance = None
    return noise_m_distance


def dbscan_grid_search(oneday_table: pd.DataFrame, features: list, eps, min_samples):
    """
    - Fit DBSCan model standard scaler + model fit;
    - Calculate mean distance between noise points and its 6 Nearest Neighbours (6-NN)';
    
    Parameters
    ----------
    oneday_table: single day Dataframe to be scaled;
    features: List of columns names (features) from dataframe to be standardized.
    eps: Epsilon parameter DBSCAN, controls the local neighborhood of the points.
    min_samples: Minimum sample parameter DBSCAN, sets the minimum number of points per cluster,
    controls how tolerant the algorithm is towards noise.
    
    Returns
    ----------
    dbscan_model_: Fitted DBSCAN Model.
    noise_m_distance: float score, single value per fitted model.
    """
    dbscan_model_, scaled = clustering_dbscan.dbscan_model(oneday_table, features, eps, min_samples)
    noise_m_distance = noise_mean_distance(dbscan_model_, scaled)
    return dbscan_model_, noise_m_distance


def cluster_collections_grid_search(oneday_table: pd.DataFrame, collections_features: dict,
                                    eps: float, min_samples: int, metrics_dict: dict):
    """
    - Select only the last collection avaliable at collections_features;
    - Fit DBSCan model standard scaler + model fit;
    - Calculate clustering metrics:
        Mean distance between noise points and its 6 Nearest Neighbours (6-NN)';
        Number of clusters;
        Number of noise points.
    -Append clustering metrics to dictionary.
    
    Parameters
    ----------
    oneday_table: single day Dataframe to be scaled;
    collections_features: Dictionary with avaliable collections and columns names (features)
    from dataframe to be standardized.
    eps: Epsilon parameter DBSCAN, controls the local neighborhood of the points.
    min_samples: Minimum sample parameter DBSCAN, sets the minimum number of points per cluster,
    controls how tolerant the algorithm is towards noise.
    
    Returns
    ----------
    metrics_dict: Clustering metrics to dictionary.
    metrics_dict = {'noise_m_d': [], 'n_clusters': [], 'n_noise': []}
    """
    features = list(collections_features.values())[-1]
    dbscan_model_, noise_m_distance = dbscan_grid_search(oneday_table, features, eps, min_samples)
    metrics_dict['noise_m_d'].append(noise_m_distance)
    metrics_dict['n_clusters'].append(len(set(dbscan_model_.labels_[dbscan_model_.labels_ >= 0])))
    metrics_dict['n_noise'].append(list(dbscan_model_.labels_).count(-1))
    return metrics_dict


def cluster_quarter_grid_search(collections_table: pd.DataFrame, collections_features: dict,
                                eps: float, min_samples: int):
    """
    -Update collections_features with combination of avaliable collections field;
    -Create metrics dictionary;
    -List all dates in quarter;
    -For all dates fit DBSCan model and calculate clustering metrics.
    -Append clustering metrics to dictionary.
    
    Parameters
    ----------
    collections_table: Quarter Dataframe to be scaled;
    collections_features: Dictionary with avaliable collections and columns names (features)
    from dataframe to be standardized.
    eps: Epsilon parameter DBSCAN, controls the local neighborhood of the points.
    min_samples: Minimum sample parameter DBSCAN, sets the minimum number of points per cluster,
    controls how tolerant the algorithm is towards noise.
    
    Returns
    ----------
    metrics_dict: Clustering metrics to dictionary.
    metrics_dict = {'noise_m_d': [], 'n_clusters': [], 'n_noise': []}
    """
    collections_features = combine_keys_and_values(collections_features)
    metrics_dict = {'noise_m_d': [], 'n_clusters': [], 'n_noise': []}
    date_list = list(collections_table.data.unique())
    for date in date_list:
        oneday_table = collections_table[collections_table.data == date]
        metrics_dict = cluster_collections_grid_search(oneday_table, collections_features, eps,
                                                       min_samples, metrics_dict)
    return metrics_dict


def grid_search(collections_table, collections_features, min_test, max_test):
    """
    -Calculate for all combinations of Hyperparameters the respective clustering metrics;
    -Calculate the descriptive statistics of the clustering metrics for the quarter;
    -Append clustering metrics descriptive statistics to dictionary.
    
    Parameters
    ----------
    collections_table: Quarter Dataframe to be scaled;
    collections_features: Dictionary with avaliable collections and columns names (features)
    from dataframe to be standardized;
    min_test: Minimum value of min_samples to be tested;
    max_test: Maximum value of min_samples to be tested.
    
    Returns
    ----------
    results: Clustering metrics descriptive statistics dictionary for every combination of
    hyperparameters.
    results = {(eps, min_samples): {'noise_m_d': (mean, max, min), 'n_clusters': (mean, max, min),
    'n_noise': (mean, max, min)},
    """
    eps_to_test = [round(eps, 1) for eps in np.arange(0.1, 2, 0.1)]
    min_samples_to_test = range(min_test, max_test + 1, 1)
    results = {}
    for eps in eps_to_test:
        for min_samples in min_samples_to_test:
            metrics_dict = cluster_quarter_grid_search(collections_table, collections_features,
                                                       eps, min_samples)
            metrics_dict = from_list_to_stats(metrics_dict)
            key = (eps, min_samples)
            results[key] = metrics_dict
    return results


def select_best_hyparameters(results: dict, max_noise_percent: float, total_points=257):
    """
    -Convert clustering metrics descriptive statistics dictionary to dataframe;
    -Transpose and rearrange dataframe.
    -Select the best hyperparameter combination:
        First Condition: The maximum number of noise points should not be greater than {max_noise_percent};
        Second Condition: The minimum number of clusters per model should be 2;
        Third Condition: From the remaining hyperparameter combinations choose the one with the lowest
        noise_m_distance score.
    
    Parameters
    ----------
    results: Clustering metrics descriptive statistics dictionary for every combination of
    hyperparameters.
    max_noise_percent: Maximum percentage of noise points allowed per model.
    total_points: Number of rows on single day Dataframe to be scaled.
    
    Returns
    ----------
    best_hp: best combination of hyperparameters tuple, (eps, min_samples).
    """
    backup = (1.4, 2)
    df = from_dict_to_df(results)
    table = transpose_df(df)
    first_cond = list(
        table[(table['metric'] == 'n_noise') & (table['max'] <= total_points * max_noise_percent)].iloc[:, 0])
    table_1 = table[table['param'].isin(first_cond)]
    if len(table_1) > 1:
        second_cond = list(table_1[(table_1['metric'] == 'n_clusters') & (table_1['min'] > 2)].iloc[:, 0])
        table_2 = table_1[table_1['param'].isin(second_cond)]
        if len(table_2) > 1:
            best_hp = table_2[table_2['metric'] == 'noise_m_d'].sort_values(by='mean').iloc[0, 0]
        elif len(table_2) == 1:
            best_hp = table_2.iloc[0, 0]
        else:
            best_hp = backup
    elif len(table_1) == 1:
        best_hp = table_1.iloc[0, 0]
    else:
        best_hp = backup
    return best_hp


def best_hyperparameters(collections_table, collections_features, min_test=2, max_test=5, max_noise_percent=0.33):
    """
    Grid search and select best Hyperparameters combination for DBSCAN clustering.
    
    Parameters
    ----------
    collections_table: Quarter Dataframe to be scaled;
    collections_features: Dictionary with avaliable collections and columns names (features)
    from dataframe to be standardized.
    min_test: Minimum value of min_samples to be tested;
    max_test: Maximum value of min_samples to be tested.
    max_noise_percent: Maximum percentage of noise points allowed per model.
    
    Returns
    ----------
    eps: Best epsilon parameter DBSCAN clustering.
    min_samples: Best minimum sample parameter DBSCAN clustering.
    """
    results = grid_search(collections_table, collections_features, min_test, max_test)
    best_hp = select_best_hyparameters(results, max_noise_percent)
    eps = best_hp[0]
    min_samples = best_hp[1]
    return eps, min_samples
