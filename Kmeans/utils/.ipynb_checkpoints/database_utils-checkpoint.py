import logging
from pymongo import MongoClient
import pandas as pd


def calculate_parametro(x):
    x = str(x)
    parametro = x.split(sep=', ')[0].split(sep=': ')[1]
    return parametro


def calculate_lon(x):
    x = str(x)
    lon = x.split(sep=', ')[1].split(sep=': ')[1]
    return lon


def calculate_lat(x):
    x = str(x)
    lat = x.split(sep=', ')[2].split(sep=': ')[1]
    return lat


def calculate_data(x):
    x = str(x)
    data = x.split(sep=', ')[3].split(sep=': ')[1][1:][:-2]
    return data


def connect_mongodb(db_url, port, username, password):
    """
    Connect to Mongo DB database server.
    """
    client = MongoClient(f'mongodb://{username}:{password}@{db_url}:{port}')
    return client


def database_import(client, database_name: str, collection_name: str):
    """
    Select dataset and collection from Mongo DB.
    """
    database = client[database_name]
    collection = database[collection_name]
    return collection


def try_mongo_insert(records: dict, collection):
    """
    Insert dictionary to MongoDB.
    """
    try:
        collection.insert_many(records)
        return 1
    except:
        return 0


def day_average(table: pd.DataFrame):
    """
    Aggregate table by day taking the mean.
    """
    table_day = table.groupby(by=['latitudine', 'longitudine', 'data']).mean().reset_index()
    return table_day


def query_db_find(collection, year: str):
    df_month_list = []
    month_list = [f'{i:>02}' for i in range(1, 2)]  # Add all months
    for month in month_list:
        try:
            m = collection.find({"data": {"$regex": f".*{year}-{month}.*"}})
            df_month = pd.DataFrame(list(m))
            df_month_list.append(df_month)
        except Exception as e:
            logging.warning(f"Mongo DB query error at: {year}-{month}")
    table = pd.concat(df_month_list)
    table = table.drop(['_id', '@timestamp', '@topic', '@version', 'id', 'orario'], axis=1)
    table = table.set_index(['latitudine', 'longitudine', 'data'])
    table = table.reset_index()
    return table


def split_id(table):
    df = table
    df['Parametro'] = df['_id'].apply(calculate_parametro)
    df['Parametro'] = df['Parametro'].str.replace("'", "")
    df['lon'] = df['_id'].apply(calculate_lon)
    df['lat'] = df['_id'].apply(calculate_lat)
    df['data'] = df['_id'].apply(calculate_data)
    df.drop('_id', axis=1, inplace=True)
    df.set_index(['Parametro', 'lon', 'lat', 'data'])
    return df


def pivot(table_splitted):
    table = pd.pivot_table(table_splitted, values='valore', index=['lat', 'lon', 'data'], columns='Parametro')
    table = table.reset_index()
    table = table.rename(columns={"lat": "latitudine", "lon": "longitudine"})
    return table


def rearrange(table):
    table_splitted = split_id(table)
    table_pivot = pivot(table_splitted)
    return table_pivot


def query_db_aggregate(collection, year: str):
    df_month_list = []
    month_list = [f'{i:>02}' for i in range(1, 2)]  # Add all months
    for month in month_list:
        try:
            m = collection.aggregate([
                {"$match": {"data": {"$regex": f".*{year}-{month}.*"}}},
                {"$project": {"data": 1, "valore": 1, "latitudine": 1, "longitudine": 1, "parametro": 1}},
                {"$group": {
                    "_id": {"parametro": "$parametro", "longitudine": "$longitudine", "latitudine": "$latitudine",
                            "data": "$data"}, "valore": {"$avg": "$valore"}}}])
            df_month = pd.DataFrame(list(m))
            df_month_list.append(df_month)
        except Exception as e:
            logging.warning(f"Mongo DB query error at: {year}-{month}")
    table = pd.concat(df_month_list)
    return table
