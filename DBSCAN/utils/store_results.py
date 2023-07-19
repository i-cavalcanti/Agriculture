from utils import database_utils
import pandas as pd
import pickle
import logging
import joblib


def from_df_to_dict(quarter_labeled, date_list):
    """
    Convert clusters dataframe to dictionary.
    """
    cluster_df = pd.melt(quarter_labeled, id_vars=['COMUNE', 'collection'], value_vars=date_list)
    cluster_df['variable'] = cluster_df['variable'].astype(str)
    cluster_df['_id'] = cluster_df[['COMUNE', 'collection', 'variable']].agg('_'.join, axis=1)
    cluster_df.columns = ['city', 'collection', 'date', 'dbscan_cluster', '_id']
    cluster_dict = cluster_df.to_dict('records')
    return cluster_dict
    
    
def save_db(insert_collection, quarter_labeled, date_list):
    """
    Convert dataframe to dictionary and insert data to Mongo DB.
    """
    cluster_dict = from_df_to_dict(quarter_labeled, date_list)
    ack = database_utils.try_mongo_insert(cluster_dict, insert_collection)
    return ack


def save_pkl(model, date, collection):
    """
    Save models to directory as pickle file
    """
    from pdb import set_trace
    try:
        directory = f'C:\\Users\\i.cavalcanti\\Progetti_condivisi\\clustering_copernicus\\Clustering\\DBSCAN\\saved_models\\dbscan_{collection}_{date}.pickle'
        set_trace()
        joblib.dump(model,directory,)
        # with open(f'{directory}/saved_models/dbscan_{collection}_{date}', 'wb') as file:
        #     pickle.dump(model, file)
    except: 
        logging.warning(f'Unable to save pickle model for {collection} on {date}')
        pass