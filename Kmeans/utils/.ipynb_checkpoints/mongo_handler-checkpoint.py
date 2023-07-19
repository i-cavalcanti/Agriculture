import pandas as pd
from pymongo import MongoClient
from utils import settings


def _connect_mongo(host=settings.MONGO_HOST, port=settings.MONGO_PORT,
                   username=settings.MONGO_USER, password=settings.MONGO_PASSWORD, db=settings.MONGO_DB):
    """ A util for making a connection to mongo """
    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)
    return conn[db]


def get_collection(collection, host=settings.MONGO_HOST, port=settings.MONGO_PORT,
                   username=settings.MONGO_USER, password=settings.MONGO_PASSWORD, db=settings.MONGO_DB):
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)
    return db[collection]
