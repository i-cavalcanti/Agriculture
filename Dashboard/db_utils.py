import pandas as pd
import geopandas as gpd
import plotly.express as px
import pyproj


#List data functions:

def list_avaliable_collections(client):
    """
    List avaliable collections in Mongo DB dataset.
    """
    return list(client.distinct('ref_collection'))


def list_avaliable_years(client):
    """
    List avaliable years in Mongo DB dataset.
    """
    return list(client.distinct('ref_year'))


def list_avaliable_cities_ref(client):
    """
    List avaliable cities in Mongo DB dataset.
    """
    city_list = list(client.distinct('ref_COMUNE'))
    converter = lambda x: x.replace('_', ' ')
    city_list = list(map(converter, city_list))
    return city_list


def list_avaliable_cities(client):
    """
    List avaliable cities in Mongo DB dataset.
    """
    city_list = list(client.distinct('COMUNE'))
    converter = lambda x: x.replace('_', ' ')
    city_list = list(map(converter, city_list))
    return city_list


#Data transformation:

def collection_label_to_key(label):
    """
    Convert collection dashboard labels to database keys.
    """
    dic = {"['Atmosphere']": "atmosphere_data",
      "['Climate']": "climate_data",
      "['Atmosphere', 'Climate']": "atmosphere_data_climate_data",
      "['Climate', 'Atmosphere']": "atmosphere_data_climate_data"}
    return dic[str(label)]


def season_to_quarter(season):
    """
    Convert season label to quarter key 
    """
    dic = {'Winter': 'q1',
           'Spring': 'q2',
           'Summer': 'q3',
           'Autumn': 'q4'}
    return dic[(season)]


def quarter_to_season(quarter):
    """
    Convert quarter label to season value 
    """
    dic = {'Winter': 'q1',
           'Spring': 'q2',
           'Summer': 'q3',
           'Autumn': 'q4'}
    return list(dic.keys())[list(dic.values()).index(quarter)]


def quarter_year_to_season(elements):
    """
    Convert quarter and year columns to a single Season, year format column.
    """
    period_list = []
    for i, row in elements.iterrows():
        quarter = elements.iloc[i,5]
        season = quarter_to_season(elements.iloc[i,5])
        year = elements.iloc[i,6]
        period = f'{season} {year}'
        period_list.append(period)
    elements['Season'] = period_list
    return elements


def quarter_to_month(quarter: str):
    """
    Convert database quarter code sting to the resepective calendar months 
    as strings.
    --------------
    Example:
    
    'q1' = [01, 02, 03]
    .
    'q4' = ['10', '11', '12']
    """
    quarter_dict = {'q1': [f'{i:>02}' for i in range(1, 4)],
               'q2': [f'{i:>02}' for i in range(4, 7)],
               'q3': [f'{i:>02}' for i in range(7, 10)],
               'q4': [f'{i:>02}' for i in range(10, 13)]}
    month_list = quarter_dict[quarter]
    return month_list


#Dashboard graphs:

def query_db_similarity(client, collections, year, season, ref_city):
    """
    -Access the Frequency_season collection;
    -Query data for the Similarity graph;
    -Format the dataframe.
    
    Parameters
    ----------
    client: MongoClient with specified database and collection;
    collections: list of collections "['Atmosphere', 'Climate']";
    year: str;
    season: str in ['Winter', 'Spring', 'Summer', 'Autumn'];
    ref_city: str reference city in list(client.distinct('ref_COMUNE')).
    
    Returns
    ----------
    labeled_dataframe: similarity among cities plot dataframe.
    """
    selected_collection = collection_label_to_key(collections)
    selected_year = year
    selected_quarter = season_to_quarter(season)
    selected_city = ref_city.replace(" ", "_")
    elements = pd.DataFrame(list(client.find({"$and": [{"ref_collection": selected_collection},
                                                       {"ref_year": selected_year},
                                                       {"ref_quarter": selected_quarter},
                                                       {"ref_COMUNE": selected_city}]})))
    labeled_dataframe = elements.iloc[:,1:3]
    labeled_dataframe['COMUNE'] = labeled_dataframe['COMUNE'].str.replace("_", " ")
    return labeled_dataframe


def similarity_plot(labeled_dataframe : pd.DataFrame, city_geo : gpd.geodataframe.GeoDataFrame):
    '''
    -Add georeferenced poligon to each municipality;
    -Plot cluster map with plotly using categorical scale labels.
    
    Parameters
    ----------
    
    labeled_dataframe: Similarity among cities plot dataframe;
    city_geo: Shapefile with the geometry poligon for each city in Puglia.
    
    Returns
    ----------
    fig: Plotly figure.
    '''
    labeled_geo = city_geo.merge(labeled_dataframe, on= 'COMUNE')
    labeled_geo = labeled_geo[['COMUNE','geometry', 'perc_sim']].set_index('COMUNE')
    labeled_geo.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    fig = px.choropleth(labeled_geo,
               geojson=labeled_geo.geometry,
               locations=labeled_geo.index,
               color=labeled_geo.perc_sim,
               labels={'perc_sim':'Percentual of days (%)'},
               projection="mercator")
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=500, width=600, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def plot_values_similarity(collections, year, season, city, client, city_geo):
    """
    Streamlit implementation function for the Similarity graph.
    
    Combine the two previous functions:
        - query_db_similarity()
        - similarity_plot()    
    """
    labeled_dataframe = query_db_similarity(client, collections, year, season, city)
    fig = similarity_plot(labeled_dataframe, city_geo)
    return fig


def query_db_comparison(client, collections, ref_city, cities):
    """
    -Access the Frequency_season collection;
    -Query data for the Comparison graph in all avaliable time periods;
    -Format the dataframe.
    
    Parameters
    ----------
    client: MongoClient with specified database and collection;
    collections: list of collections "['Atmosphere', 'Climate']";
    ref_city: str reference city in list(client.distinct('ref_COMUNE'));
    cities: list of cities from list(client.distinct('COMUNE')).
    
    Returns
    ----------
    labeled_dataframe: Comparison among cities in time plot dataframe.
    """
    selected_collection = collection_label_to_key(collections)
    selected_ref_city = ref_city.replace(" ", "_")
    cities_l = []
    for city in cities:
        selected_city = city.replace(" ", "_")
        elements = pd.DataFrame(list(client.find({"$and": [{"ref_collection": selected_collection},
                                                       {"ref_COMUNE": selected_ref_city},
                                                       {"COMUNE": selected_city}]})))
        elements = quarter_year_to_season(elements)
        elements = elements.sort_values(['ref_year', 'ref_quarter'])
        elements.rename(columns = {'perc_sim':'Percentual of days (%)'}, inplace = True)
        elements['COMUNE'] = elements['COMUNE'].str.replace("_", " ")
        elements = elements[['COMUNE', 'Percentual of days (%)', 'Season']]
        cities_l.append(elements)
    labeled_dataframe = pd.concat(cities_l, axis=0)
    return labeled_dataframe


def comparison_plot(labeled_dataframe: pd.DataFrame):
    """
    -Line plot percentual of days each city is classified in the same group as the ref_city per season.
    
    Parameters
    ----------

    labeled_dataframe: Comparison among cities in time plot dataframe.;
    
    Returns
    ----------
    fig: Plotly figure.
    """
    fig = px.line(labeled_dataframe, x='Season', y='Percentual of days (%)', color= 'COMUNE', markers=True)
    return fig


def plot_values_comparison(collections, ref_city, cities, client):
    """
    Streamlit implementation function for the Comparison graph.
    
    Combine the two previous functions:
        - query_db_comparison()
        - comparison_plot()   
    """
    labeled_dataframe = query_db_comparison(client, collections, ref_city, cities)
    fig = comparison_plot(labeled_dataframe)
    return fig
    

def calc_outliers(clusters_results):
    """
    Count the percentual of days each city is classified as an outlier for a given quarter.
    
    Parameters
    ----------
    clusters_results: Raw dataframe containing clustering labels for each city and time period.
    
    Returns
    ----------
    labeled_dataframe: Dataframe with a column n_noise: percentual of days each city is classified as an outlier.
    """    
    n_dates = len(clusters_results.date.unique())
    noise_city = {'n_noise': [], 'city': []}
    for i in clusters_results.city.unique():
        city_t = clusters_results[clusters_results.city == i]
        noise_city['n_noise'].append(round((len(city_t[city_t.dbscan_cluster == -1])/n_dates)*100, 2))
        noise_city['city'].append(i)
    labeled_dataframe = pd.DataFrame.from_dict(noise_city)
    labeled_dataframe = labeled_dataframe.rename(columns={"city": "COMUNE"})
    return labeled_dataframe

    
def query_db_clusters(client, collections, year, season):
    """
    -Access the clusters_daily collection;
    -Query data for the Outliers graph;
    -Format the dataframe.
    
    Parameters
    ----------
    client: MongoClient with specified database and collection;
    collections: list of collections "['Atmosphere', 'Climate']";
    year: str;
    season: str in ['Winter', 'Spring', 'Summer', 'Autumn'].
    
    Returns
    ----------
    labeled_dataframe: Dataframe with a column n_noise: percentual of days each city is classified as an outlier.
    """
    selected_collection = collection_label_to_key(collections)
    selected_year = year
    selected_months = quarter_to_month(season_to_quarter(season))
    elements = pd.DataFrame(list(client.find({"$and": [{"collection": selected_collection},
                                    {"$or": [{"date": {"$regex": f".*{selected_year}-{selected_months[0]}.*"}},
                                             {"date": {"$regex": f".*{selected_year}-{selected_months[1]}.*"}},
                                             {"date": {"$regex": f".*{selected_year}-{selected_months[2]}.*"}}]}]})))
    clusters_results = elements.iloc[:,1:]
    labeled_dataframe = calc_outliers(clusters_results)
    return labeled_dataframe


def clusters_plot(labeled_dataframe: pd.DataFrame, city_geo: gpd.geodataframe.GeoDataFrame):
    """    
    -Add georeferenced poligon to each city;
    -Plot cluster map with plotly.
    
    Parameters
    ----------
    
    labeled_dataframe: Dataframe with a column n_noise: percentual of days each city is classified as an outlier;
    city_geo: Shapefile with the geometry poligon for each city in Puglia.
    
    Returns
    ----------
    fig: Plotly figure.
    """
    labeled_geo = city_geo.merge(labeled_dataframe, on= 'COMUNE')
    labeled_geo = labeled_geo[['COMUNE','geometry', 'n_noise']].set_index('COMUNE')
    labeled_geo.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    fig = px.choropleth(labeled_geo,
               geojson=labeled_geo.geometry,
               locations=labeled_geo.index,
               color=labeled_geo.n_noise,
               labels={'n_noise':'Percentual of days (%)'},
               projection="mercator")
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=500, width=600, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def plot_values_clusters(collections, year, season, client, city_geo):
    """
     Streamlit implementation function for the Outliers graph.
    
    Combine the two previous functions:
        - query_db_clusters()
        - clusters_plot() 
    """
    labeled_dataframe = query_db_clusters(client, collections, year, season)
    fig = clusters_plot(labeled_dataframe, city_geo)
    return fig

