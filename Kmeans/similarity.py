import pandas as pd
from pandas import DataFrame


def similarity_counter(dataframe: pd.DataFrame):
    """
    Count number of occurrences: Two columns in a dataframe
    have equal values.
    """
    values = []
    for i, row in dataframe.iterrows():
        if row[0] == row[1]:
            values.append(True)
        else:
            values.append(False)
    return values.count(True)


def similar_cities_dict(labeled_time: pd.DataFrame):
    """
    Create dictionary with the number of days each city
    is classified in the same cluster as every other 
    city in the dataset in the same quarter.
    """
    df = labeled_time
    similarity_dict = {}
    for i, row in df.iterrows():
        ref_city = df.iloc[i, 0]
        city_list = df[df.columns[0]].tolist()
        city_list.remove(ref_city)
        city_dict = {}
        for city in city_list:
            two_df = df[(df.COMUNE == ref_city) | (df.COMUNE == city)]
            t_two_df = two_df.transpose()
            t_two_df.columns = t_two_df.iloc[0]
            t_two_df = t_two_df.iloc[1:]
            city_dict[city] = similarity_counter(t_two_df)
        similarity_dict[ref_city] = city_dict
    return similarity_dict   
    
                                           
def cities_dict_to_df(city: str, similarity_dict: dict, n_dates: int):
    """
    Create dataframe with the percentual of days each city
    is classified in the same cluster as every other 
    city in the dataset in the same quarter.
    """
    city_dict = similarity_dict[city]
    city_freq = pd.DataFrame.from_dict(city_dict, orient='index')
    city_freq['COMUNE'] = city_freq.index
    city_freq['COMUNE'] = city_freq['COMUNE'].str.replace(' ', '_')
    city_freq['perc_sim'] = round((city_freq[0] / n_dates) * 100, 2)
    city_freq = city_freq.drop(0, axis=1)
    city_freq.reset_index(inplace=True)
    city_freq = city_freq.iloc[:, 1:]
    city_freq['ref_COMUNE'] = city.replace(' ', '_')
    city_freq['city'] = city_freq[['COMUNE', 'ref_COMUNE']].agg('_'.join, axis=1)
    return city_freq


def cities_similarity_df(cities: list, similarity_dict: dict, n_dates: int):
    """
    Create a dataframe for all cities in the dataset with the 
    percentual of days each city is classified in the same cluster 
    as every other city in the dataset in the same quarter.
    """
    city_list = []
    for city in cities:
        city_freq = cities_dict_to_df(city, similarity_dict, n_dates)
        city_list.append(city_freq)
    collection_freq = pd.concat(city_list)
    return collection_freq
