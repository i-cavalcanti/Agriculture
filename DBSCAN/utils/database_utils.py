import logging
from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd


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