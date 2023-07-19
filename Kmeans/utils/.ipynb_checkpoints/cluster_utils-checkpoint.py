import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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
    scaled_features = scaler.fit_transform(dataframe[features_list])
    return scaled_features


def kmeans(scaled_features: np.ndarray, n_clusters_: int) -> np.ndarray:
    """
    Calculate Kmeans clusters;

    Parameters
    ----------
    scaled_features: Standardized features
    n_clusters_: Number of clusters

    Returns
    -------
    clusters_list: Vector with cluster classification for each row.
    """
    clusters = KMeans(n_clusters=n_clusters_, random_state=0, n_init="auto")
    clusters_list = clusters.fit(scaled_features).labels_
    return clusters_list


def scores(scaled_features: np.ndarray, clusters_list: np.ndarray) -> dict:
    """
    Calculate metric scores from cluster classification.
    Metric scores: Silhouette, Calinski Habasz and Davies Bouldin.

    Parameters
    ----------
    scaled_features: Standardized features.
    clusters_list: Vector with cluster classification.

    Returns
    -------
    metrics_scores: Dictionary with the metric scores.
    """
    s_score = metrics.silhouette_score(scaled_features, clusters_list, metric='euclidean')
    ch_score = metrics.calinski_harabasz_score(scaled_features, clusters_list)
    db_score = metrics.davies_bouldin_score(scaled_features, clusters_list)
    metric_scores = {'silhouette': s_score, 'calinski_harabasz': ch_score, 'davies_bouldin': db_score}
    return metric_scores


def best_n_clusters(scaled_features: np.ndarray, n_min=2, n_max=15) -> int:
    """
    Sets the number of clusters that maximizes the Kmeans silhouette score.

    Parameters
    ----------
    scaled_features: Standardized features.
    n_min: Minimum number of clusters
    n_max: Maximum number of clusters

    Returns
    -------
    n_clusters: Number of clusters that maximizes the Kmeans silhouette score
    """
    metrics_dict = {'n_clusters': [], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': []}
    n_clusters_range = list(range(n_min, n_max + 1, 1))
    for n_clusters_ in n_clusters_range:
        clusters_list = kmeans(scaled_features, n_clusters_)
        ms = scores(scaled_features, clusters_list)
        metrics_dict['n_clusters'].append(n_clusters_)
        metrics_dict['silhouette'].append(ms['silhouette'])
        metrics_dict['calinski_harabasz'].append(ms['calinski_harabasz'])
        metrics_dict['davies_bouldin'].append(ms['davies_bouldin'])

    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    max_s = max(metrics_df.silhouette)
    n_clusters = metrics_df.loc[metrics_df.silhouette == max_s].values.flatten().tolist()[0]
    n_clusters = int(n_clusters)
    return n_clusters


def best_kmeans(dataframe: pd.DataFrame, features_list: list) -> pd.DataFrame:
    """
    Calculate Kmeans clusters with the number of clusters that maximizes the Kmeans silhouette score.

    Parameters
    ----------
    dataframe: Dataframe to be clustered;
    features_list: List of columns names (features) from dataframe considered on the clustering.

    Returns
    -------
    label_dataframe: Dataframe with Kmeans clustering labels with the maximum silhouette score.
    """
    scaled_features = standard_scaler(dataframe, features_list)
    n_clusters = best_n_clusters(scaled_features)
    label_dataframe = dataframe.assign(kcls_std=kmeans(scaled_features, n_clusters))
    return label_dataframe
