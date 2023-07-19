from utils import database_utils
from utils import geo_utils
from utils import cluster_utils
from utils import mongo_handler
from functools import reduce
import similarity
import pandas as pd
import geopandas as gpd


################################
# ImportAndArrangement

def import_table(mongo_client, database, collection: str, query_aggregate: bool, year: str):
    """
    - Import parameter from Mongo DB collection;
    - Add Geopandas point geometry column.
    """
    imported = database_utils.database_import(mongo_client, database, collection)
    #imported = mongo_handler.MongoHandler().get_mongo_collection(collection)
    
    if query_aggregate:
        table = database_utils.query_db_aggregate(imported, year)
        arr_table = database_utils.rearrange(table)
        features = arr_table.iloc[:, 4:].columns.to_list()
    else:
        table = database_utils.query_db_find(imported, year)
        arr_table = database_utils.day_average(table)
        features = arr_table.iloc[:, 3:].columns.to_list()
    geo_table = geo_utils.add_geo_point(arr_table)
    return features, geo_table


def create_tables(mongo_client, database: str, collections: dict, year: str, region_shape: gpd.geopandas.GeoDataFrame):
    """
    For all collections in collections dict:
    - Import parameter from Mongo DB collection;
    - Assign latitude and longitude coordinates to municipalities;
    - Merge collections to a single dataframe;
    - Create dictionary with feature names from each collection.
    """
    collections_features = {}
    table_list = []
    for collection, query_aggregate in collections.items():
        features, geo_table = import_table(mongo_client, database, collection, query_aggregate, year)
        collections_features[collection] = features
        table = geo_utils.set_coord(geo_table, region_shape)
        s_table = geo_utils.calc_city_average(table)
        table_list.append(s_table)
    shape_table = reduce(lambda left, right: pd.merge(left, right, on=['data', 'COMUNE'], how='outer'), table_list)
    return shape_table, collections_features


################################
# QuarterKmeansClustering


def cluster_collections(oneday_table: pd.DataFrame, collections_features: dict):
    """
    -Kmeans clusters with features from all collections;
    -Kmeans clusters with features from each collection individually;
    -Add collection group column for reference.
    """
    # Architecture allows only 2 collections, to be expanded to all n collections.
    labeled_list = []
    all_features = list(oneday_table.iloc[:, 2:].columns)
    labeled = cluster_utils.best_kmeans(oneday_table, all_features)
    labeled = labeled[['COMUNE', 'kcls_std']]
    labeled['collection'] = f'{list(collections_features.items())[0][0]}_{list(collections_features.items())[1][0]}'
    labeled_list.append(labeled)
    for collection, features in collections_features.items():
        labeled = cluster_utils.best_kmeans(oneday_table, features)
        labeled = labeled[['COMUNE', 'kcls_std']]
        labeled['collection'] = collection
        labeled_list.append(labeled)
    oneday_labeled = pd.concat(labeled_list)
    return oneday_labeled


def cluster_quarter(quarter_table: pd.DataFrame, collections_features: dict):
    """
    -List all dates in quarter;
    -Fit Kmeans clusters for each date.
    """
    date_list = list(quarter_table.data.unique())
    n_dates = len(date_list)
    oneday_list = []
    for date in date_list:
        oneday_table = quarter_table[quarter_table.data == date]
        oneday_labeled = cluster_collections(oneday_table, collections_features)
        oneday_labeled.rename(columns={'kcls_std': date}, inplace=True)
        oneday_list.append(oneday_labeled)
    quarter_labeled = reduce(lambda left, right: pd.merge(left, right, on=['COMUNE', 'collection'], how='outer'),
                             oneday_list)
    quarter_labeled = quarter_labeled.set_index(['COMUNE', 'collection']).reset_index()  # To be reset
    return quarter_labeled, n_dates


################################
# FrequencySameCluster


def frequency_collection(collection_labeled: pd.DataFrame, n_dates: int):
    """
    Calculate the similarity of each city and every other city in the dataset
    by the frequency (days) each city pair is classified in the same cluster
    in a quarter.
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
    """
    collection_freq['ref_collection'] = col
    collection_freq['ref_quarter'] = q
    collection_freq['ref_year'] = year
    collection_freq['_id'] = collection_freq[['city', 'ref_collection', 'ref_quarter',
                                              'ref_year', 'ref_year']].agg('_'.join, axis=1)
    collection_freq.drop('city')
    frequency_dict = collection_freq.to_dict('records')
    return frequency_dict


def frequency_quarter(insert_collection, quarter_labeled: pd.DataFrame, n_dates: int, q: str, year: str):
    """
    For each collection group calculate the similarity of each city and every
    other city in the dataset by the frequency (days) each city pair is
    classified in the same cluster in a quarter.
    """

    cols = list(quarter_labeled.collection.unique())
    quarter_list = []
    for col in cols:
        collection_labeled = quarter_labeled[quarter_labeled.collection == col]
        collection_freq = frequency_collection(collection_labeled, n_dates)
        frequency_dict = frequency_to_dict(collection_freq, col, q, year)
        ack = database_utils.try_mongo_insert(frequency_dict, insert_collection)
        quarter_list.append(ack)
    return quarter_list


def select_quarter(shape_table: pd.DataFrame, quarter: dict, q: str):
    """
    Filter rows from one quarter.
    """
    row_list = []
    for i, row in shape_table.iterrows():
        if row['data'][5:7] in quarter[q]:
            row_list.append(i)
    quarter_table = shape_table.iloc[row_list]
    return quarter_table


def frequency_year(shape_table: pd.DataFrame, collections_features: dict, quarter: dict, year: str, insert_collection):
    """
    For each quarter in a year calculate the similarity of each city
    and every other city in the dataset by the frequency (days) each
    city pair is classified in the same cluster in the quarter.
    """
    q_list = ['q1', 'q2', 'q3', 'q4']
    year_list = []
    for q in q_list:
        quarter_table = select_quarter(shape_table, quarter, q)
        quarter_labeled, n_dates = cluster_quarter(quarter_table, collections_features)
        quarter_list = frequency_quarter(insert_collection, quarter_labeled, n_dates, q, year)
        year_list.append(quarter_list)
    return year_list
