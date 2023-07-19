from datetime import datetime
import pandas as pd
import numpy as np



def quarter_dates(year: int, quarter: str):
    """
    Convert quarter labels to start and end dates.
    
    """
    assert quarter in ['q1', 'q2', 'q3', 'q4'], f"{quarter} is not in the right format."
    if quarter == 'q1':
        start_day, end_day = datetime(year, 1, 1), datetime(year, 3, 31)
    elif quarter == 'q2':
        start_day, end_day = datetime(year, 4, 1), datetime(year, 6, 30)
    elif quarter == 'q3':
        start_day, end_day = datetime(year, 7, 1), datetime(year, 9, 30)
    else:
        start_day, end_day = datetime(year, 10, 1), datetime(year, 12, 31)
    return start_day, end_day


def query_db_climate(collection, start_day, end_day):
    """
    Query Mongo DB by start and end dates for the Climate collection.
    """
    pipeline = \
            [
                {
                    '$match': {
                        'data': {
                            '$gte': start_day,
                            '$lte': end_day
                            }
                    }
                }, 
                
                {
                    '$addFields': {
                        'year_': {
                            '$year': '$data'
                            }, 
                        'month_': {
                            '$month': '$data'
                            }, 
                        'day_': {
                            '$dayOfMonth': '$data'
                            }
                    }
                }, 
                
                {
                    '$group': {
                        '_id': {
                            'Year': '$year_', 
                            'Month': '$month_', 
                            'Day': '$day_', 
                            'latitudine': '$latitudine', 
                            'longitudine': '$longitudine', 
                            'parametro': '$parametro'
                            }, 
                        'valore': {
                            '$avg': '$valore'
                            }
                    }
                }
            ]
    cursor = list(collection.aggregate(pipeline=pipeline, allowDiskUse=True))
    return cursor


def get_dataframe_climate(cursor):
    """
    Dataframe transformations for the Climate collection.
    """
    table = pd.DataFrame(cursor)
    table = pd.concat([table._id.apply(pd.Series), table.drop(columns=['_id'])], axis=1)
    table['data'] = pd.to_datetime(table[['Year', 'Month', 'Day']])
    table.drop(columns=['Year', 'Month', 'Day'], inplace=True)
    pivot_table = table.pivot_table(index=table[['data', 'latitudine', 'longitudine']], columns='parametro', 
                                   values='valore', aggfunc='mean').reset_index().interpolate('ffill')
    features = pivot_table.iloc[:, 3:].columns.to_list()
    return pivot_table, features
    
    
def query_db_atmosphere(collection, start_day, end_day):
    """
    Query Mongo DB by start and end dates for the Climate collection.
    """
    pipeline = [
            {
                '$match': {
                    'data': {
                        '$gte': start_day,
                        '$lte': end_day
                    }
                }
            }, {
                '$addFields': {
                    'year_': {
                        '$year': '$data'
                    },
                    'month_': {
                        '$month': '$data'
                    },
                    'day_': {
                        '$dayOfMonth': '$data'
                    }
                }
            }, {
                '$group': {
                    '_id': {
                        'Year': '$year_',
                        'Month': '$month_',
                        'Day': '$day_',
                        'latitudine': '$latitudine',
                        'longitudine': '$longitudine'
                    },
                    'Dust': {
                        '$avg': '$Dust'
                    },
                    'PM10 Aerosol': {
                        '$avg': '$PM10 Aerosol'
                    },
                    'PM2_5 Aerosol': {
                        '$avg': {
                            '$getField': 'PM2.5 Aerosol'
                        }
                    },
                    'Nitrogen Monoxide': {
                        '$avg': '$Nitrogen Monoxide'
                    },
                    'Nitrogen Dioxide': {
                        '$avg': '$Nitrogen Dioxide'
                    },
                    'Sulphur Dioxide': {
                        '$avg': '$Sulphur Dioxide'
                    },
                    'Ozone': {
                        '$avg': '$Ozone'
                    }
                }
            }
        ]
    cursor = collection.aggregate(pipeline=pipeline)
    return cursor


def get_dataframe_atmosphere(cursor):
    """
    Dataframe transformations for the Atmosphere collection
    """
    table = pd.DataFrame(list(cursor))
    table.rename(columns={'PM2_5 Aerosol': 'PM2.5 Aerosol'}, inplace=True)
    table = pd.concat([table._id.apply(pd.Series), table.drop(columns=['_id'])], axis=1)
    table['data'] = pd.to_datetime(table[['Year', 'Month', 'Day']])
    table.drop(columns=['Year', 'Month', 'Day'], inplace=True)
    features = table.iloc[:, 2:-1].columns.to_list()
    return table, features