import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def standard_scaler(dataframe: pd.DataFrame, features_list: list) -> np.ndarray:
    """
    Scale standard selected columns from dataframe;

    Parameters
    ----------
    dataframe: Dataframe to be scaled;
    features_list: List of columns names (features) from dataframe to be standardized.

    Returns
    ----------
    scaled_features: Dataframe with standardized features.
    """
    scaler = StandardScaler()
    scaled_table = scaler.fit_transform(dataframe[features_list])
    return scaled_table


def dbscan_model(dataframe: pd.DataFrame, features_list: list, eps, min_samples):
    """
    Fit DBSCan model standard scaler + model fit
    
    Parameters
    ----------
    dataframe: Dataframe to be scaled;
    features_list: List of columns names (features) from dataframe to be standardized.
    eps: Epsilon parameter DBSCAN, controls the local neighborhood of the points.
    min_samples: Minimum sample parameter DBSCAN, sets the minimum number of points per cluster,
    controls how tolerant the algorithm is towards noise.
    
    Returns
    ----------
    dbscan_model_: Fitted DBSCAN Model.
    scaled: np.Array with standardized features.
    """
    scaled = standard_scaler(dataframe, features_list)
    dbscan_model_ = DBSCAN( eps = eps, min_samples = min_samples)
    dbscan_model_.fit(scaled)
    return dbscan_model_, scaled


def dbscan(oneday_table: pd.DataFrame, features: list, eps: float, min_samples: int):
    """
    - Fit DBSCan model standard scaler + model fit;
    - Add label column to original single day dataframe'.
    
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
    label_dataframe: Labeled column 'dbscan'.
    
    """
    dbscan_model_, scaled = dbscan_model(oneday_table, features, eps, min_samples)
    label_dataframe = oneday_table.assign(dbscan = dbscan_model_.labels_)
    return label_dataframe, dbscan_model_