from utils import store_results
from utils import grid_search_dbscan
import execute


def frequency_by_quarter_calculator(import_client, cluster_client, frequency_client, years):
    """
    -Import data from Mongo DB collection;
    -Convert latitude and longitude coordinates to cities;
    -Grid search and select best Hyperparameters combination for DBSCAN clustering for each quarter;
    -Fit DBSCAN model for the best Hyperparameters combination for each day;
    -Store model as pickle file;
    -Save DBSCAN labels to MongoDB;
    -Calculate the similarity of each city and every other city in the dataset, by the percentual of days in a quarter 
    each city is classified in the same cluster as every other city in the dataset.
    -Save DBSCAN percentual results to MongoDB;
    
    Parameters
    ----------
    import_client: MongoDB client credentials for input;
    cluster_client: MongoDB database and collection address for DBSCAN labels output;
    frequency_client: MongoDB database and collection address for DBSCAN percentual results output;
    years: list of years integers.
    
    Returns
    ----------
    result: Error log.
    """
    import_database = 'copernicus_datastore'
    collections = {'atmosphere_data': False, 'climate_data': True}
    quarters = ['q1', 'q2', 'q3', 'q4']
    upload_success_count = []
    for year in years:
        for quarter in quarters:
            collections_table, collections_features = execute.import_collections(import_client, import_database,
                                                                                 collections, year, quarter)
            eps, min_samples = grid_search_dbscan.best_hyperparameters(collections_table, collections_features)
            quarter_labeled, n_dates, date_list = execute.cluster_quarter(collections_table, collections_features,
                                                                          eps, min_samples)
            store_results.save_db(cluster_client, quarter_labeled, date_list)
            quarter_list = execute.frequency_quarter(frequency_client, quarter_labeled, n_dates, quarter, year)
            upload_success_count.append(quarter_list)
            
    if len(upload_success_count) == sum(upload_success_count):
        result = 'All frequency results saved with success'
    else:
        result = 'Check for frequency upload errors'
    return result
