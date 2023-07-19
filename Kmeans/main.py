import execute


def frequency_by_quarter_calculator(import_client, insert_collection, region_shape, years):
    """
    
    """
    import_database = 'copernicus_datastore'
    collections = {'atmosphere_data': False, 'climate_data_old': True}
    quarter = {'q1': [f'{i:>02}' for i in range(1, 4)],
               'q2': [f'{i:>02}' for i in range(4, 7)],
               'q3': [f'{i:>02}' for i in range(7, 10)],
               'q4': [f'{i:>02}' for i in range(10, 13)]}
    n_errors_list = []
    for year in years:
        shape_table, collections_features = execute.create_tables(import_client, import_database, collections, year,
                                                                  region_shape)
        insert_success_list = execute.frequency_year(shape_table, collections_features, quarter, year,
                                                     insert_collection)
        n_errors_list.append(insert_success_list)
    n_errors = len(n_errors_list) - sum(n_errors_list)
    return n_errors
