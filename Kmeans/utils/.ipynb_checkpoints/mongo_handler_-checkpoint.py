import os
from urllib.parse import quote_plus
from pymongo import MongoClient


# usr, psw, host, database_name, collection_name
def retrieve_env():
    mongo_auth = {
        'usr': os.getenv("MONGO_SARA_USERNAME"),
        'psw': os.getenv("MONGO_SARA_PASSWORD"),
        'host': os.getenv('MONGO_SARA_IP'),
        'database_name': os.getenv('DATABASE')
    }
    return mongo_auth


class MongoHandler:
    def __init__(self):
        mongo_auth = retrieve_env()
        self.database = self.get_mongo_database(**mongo_auth)

    def get_mongo_database(self, usr, psw, host, database_name):
        port = 40512
        uri = f"mongodb://{usr}:{psw}@{host}:{port}/?authSource={database_name}"
        client = MongoClient(uri, replicaSet="rs0", directConnection=False)
        database = client[f"{database_name}"]
        return database

    def get_mongo_collection(self, collection_name):
        collection = self.database[collection_name]
        return collection
    
