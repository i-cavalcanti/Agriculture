from utils import database_utils
from utils import geo_utils
from utils import clustering_dbscan
from utils import store_results
from utils import data_import 
from functools import reduce
import similarity
import pandas as pd


def import_table(mongo_client, import_database: str, collection: str, 
                 query_type: bool, year: int, quarter: str):
    """
    -Import data from Mongo DB collection;
    -Convert latitude and longitude coordinates to cities.
    
    Parameters
    ----------
    mongo_client: MongoDB connection address;
    import_database: MongoDB database name;
    collection: MongoDB collection name;
    query_type: {True: Climate_data, False: Atmosphere_data}
    year: period of time to query;
    quarter: quarter code to query from ['q1', 'q2', 'q3', 'q4'].
    
    Returns
    ----------
    table: Single day Dataframe with parameter values per city and date;
    features: Parameters column names list.
    """
    imported = database_utils.database_import(mongo_client, import_database, collection)
    start_day, end_day = data_import.quarter_dates(year, quarter)
    
    if query_type:
        cursor = data_import.query_db_climate(imported, start_day, end_day)
        arr_table, features = data_import.get_dataframe_climate(cursor)
        
    else:
        cursor = data_import.query_db_atmosphere(imported, start_day, end_day)
        arr_table, features = data_import.get_dataframe_atmosphere(cursor)
        
    table = geo_utils.from_coord_to_city_mean(arr_table, features)
    return table, features


def import_collections(mongo_client, import_database: str, collections: dict, year: int, quarter: str):
    """
    For all collections:
    -Import data from Mongo DB;
    -Convert latitude and longitude coordinates to cities.
    -Merge collections to a single dataframe;
    -Create dictionary with feature names from each collection.
    
    Parameters
    ----------
    mongo_client: MongoDB connection address;
    import_database: MongoDB database name;
    collections: MongoDB collection name;
    year: period of time to query;
    quarter: quarter code to query from ['q1', 'q2', 'q3', 'q4'].
    
    Returns
    ----------
    collections_table: Quarter Dataframe with parameter values per city and date for all collections;
    collections_features: Dictionary with avaliable collections and columns names (features)
    from dataframe to be standardized.
    """
    collections_features = {}
    table_list = []
    for collection, query_type in collections.items():
        table, features = import_table(mongo_client, import_database, collection, query_type, year, quarter)
        collections_features[collection] = features
        table_list.append(table)
    collections_table = reduce(lambda left,
                               right: pd.merge(left, right, on=['data', 'COMUNE'], how='outer'), table_list)
    return collections_table, collections_features


def cluster_collections(oneday_table: pd.DataFrame, collections_features: dict, date: str, 
                        eps: float, min_samples: int):
    """
    For each collection:
        -Fit DBSCan model standard scaler + model fit;
        -Add collection group column for reference;
        -Store model as pickle file.

    Parameters
    ----------
    oneday_table: single day Dataframe to be scaled;
    collections_features: Dictionary with avaliable collections and columns names (features);
    from dataframe to be standardized;
    date: oneday_table date;
    eps: Epsilon parameter DBSCAN;
    min_samples: Minimum sample parameter DBSCAN.
    
    Returns
    ----------
    oneday_labeled: Single day Dataframe with DBSCAN clustering labels.
    """
    labeled_list = []
    for collection, features in collections_features.items():
        labeled, dbscan_model_ = clustering_dbscan.dbscan(oneday_table, features, eps, min_samples)
        labeled = labeled[['COMUNE', 'dbscan']]
        labeled['collection'] = collection
        labeled_list.append(labeled)
        store_results.save_pkl(dbscan_model_, date, collection)
    oneday_labeled = pd.concat(labeled_list)
    return oneday_labeled


def combine_keys_and_values(collections_features: dict):
    """
    Add key with combination of avaliable collection. 
    -Collection names;
    -Collection features.
    """
    key = f'{list(collections_features.items())[0][0]}_{list(collections_features.items())[1][0]}'
    value = list(collections_features.values())[0]+list(collections_features.values())[1] 
    collections_features[key] = value
    return collections_features


def cluster_quarter(collections_table: pd.DataFrame, collections_features: dict, eps: float, min_samples: int):
    """
    -Update collections_features with combination of avaliable collections field;
    -List all dates in quarter;
    -For each date in the quarter:
        Get single day Dataframe with DBSCAN clustering labels.
        Append labeled single day Dataframe to quarter labeled dataframe.
        
    Parameters
    ----------
    collections_table: Quarter Dataframe with parameter values per city and date for all collections;
    collections_features: Dictionary with available collections and columns names (features)
    from dataframe to be standardized;
    eps: Epsilon parameter DBSCAN;
    min_samples: Minimum sample parameter DBSCAN.
    
    Returns
    ----------   
    quarter_labeled: Quarter Dataframe with DBSCAN clustering labels per day for all collections;
    date_list: Days string labels for the quarter;
    n_dates: lenth of the quarter in days.
    """
    collections_features = combine_keys_and_values(collections_features)
    date_list = list(collections_table.data.unique())
    n_dates = len(date_list)
    oneday_list = []
    for date in date_list:
        oneday_table = collections_table[collections_table.data == date]
        oneday_labeled = cluster_collections(oneday_table, collections_features, date, eps, min_samples)
        oneday_labeled.rename(columns={'dbscan': date}, inplace=True)
        oneday_list.append(oneday_labeled)
    quarter_labeled = reduce(lambda left, right: pd.merge(left, right, on=['COMUNE', 'collection'], how='outer'),
                             oneday_list)
    quarter_labeled = quarter_labeled.set_index(['COMUNE', 'collection']).reset_index() 
    return quarter_labeled, n_dates, date_list


def frequency_collection(collection_labeled: pd.DataFrame, n_dates: int):
    """
    Calculate the similarity of each city and every other city in the dataset
    by the frequency (days) each city pair is classified in the same cluster
    in a quarter.
    
    Parameters
    ----------
    collection_labeled: Quarter Dataframe with DBSCAN clustering labels per day for one collection;
    n_dates: lenth of the quarter in days;
    
    Returns
    ----------
    collection_freq: Frequency Dataframe per quarter for one collection.
    """
    collection_labeled = collection_labeled.reset_index(drop=True)
    cities = list(collection_labeled.COMUNE.unique())
    similarity_dict = similarity.similar_cities_dict(collection_labeled)
    collection_freq = similarity.cities_similarity_df(cities, similarity_dict, n_dates)
    return collection_freq


def frequency_to_dict(collection_freq, col: str, q: str, year: str):
    """
    -Add reference columns;
    -Convert DataFrame to dictionary organized as records.
    
    Parameters
    ----------
    collection_freq: Frequency Dataframe per quarter for one collection.
    col: Collection name;
    q: Quarter code;
    year: Year as string.

    Returns
    ----------
    frequency_dict: Frequency dictionary per quarter for one collection.
    """
    collection_freq['ref_collection'] = col
    collection_freq['ref_quarter'] = q
    collection_freq['ref_year'] = year
    collection_freq['_id'] = collection_freq[['city', 'ref_collection', 'ref_quarter',
                                              'ref_year']].agg('_'.join, axis=1)
    collection_freq = collection_freq.drop('city', axis=1)
    frequency_dict = collection_freq.to_dict('records')
    return frequency_dict


def frequency_quarter(insert_collection, quarter_labeled: pd.DataFrame, n_dates: int, q: str, year: str):
    """
    For each collection group calculate the similarity of each city and every
    other city in the dataset by the frequency (days) each city pair is
    classified in the same cluster in a quarter.
    
    Parameters
    ----------
    insert_collection: Destination frequency season collection MongoDB;
    quarter_labeled: Quarter Dataframe with DBSCAN clustering labels per day for all collections;
    n_dates: length of the quarter in days;
    q: quarter code to query from ['q1', 'q2', 'q3', 'q4'];
    year: period of time to query.

    
    Returns
    ----------
    quarter_list: binary list with success inserts to MongoDB, (0=failure, 1=success).
    """
    year = str(year)
    cols = list(quarter_labeled.collection.unique())
    quarter_list = []
    for col in cols:
        collection_labeled = quarter_labeled[quarter_labeled.collection == col]
        collection_freq = frequency_collection(collection_labeled, n_dates)
        frequency_dict = frequency_to_dict(collection_freq, col, q, year)
        ack = database_utils.try_mongo_insert(frequency_dict, insert_collection)
        quarter_list.append(ack)
    return quarter_list
