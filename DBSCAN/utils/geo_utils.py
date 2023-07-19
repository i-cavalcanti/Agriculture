import pandas as pd
import geopandas as gpd
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)

shp_region = gpd.read_file('C:/Users/i.cavalcanti/Progetti_condivisi/clustering_copernicus/Clustering/DBSCAN/puglia_shape.shp')


def add_geo_point(dataframe: pd.DataFrame):
    """
    Convert longitudine and latitudine coordinates to GeoPandas geometry.

    Parameters
    ----------
    
    dataframe: DataFrame with longitudine and latitudine coordinates as columns.

    Returns
    ----------
    
    shp_coord: Geopandas Dataframe with a geometry point for each coordinate.
    """
    shp_coord = gpd.geopandas.GeoDataFrame(
        dataframe, geometry=gpd.geopandas.points_from_xy(dataframe.longitudine, dataframe.latitudine))
    shp_coord.set_crs("epsg:4326", inplace=True)
    return shp_coord


def set_coord(shp_coord: gpd.geopandas.GeoDataFrame, shp_region: gpd.geopandas.GeoDataFrame, interval=1000):
    """
    Aggregate latitude and longitude coordinates by city.
    
    Parameters
    ----------
    shp_coord: Geopandas Dataframe with a geometry point for each coordinate;
    shp_region: Geopandas shapefile with the georeferenced poligon geometry from each city;
    interval: Lenth of city radious increase at each iteration. interval = 1000 as defaut.

    Returns
    ----------
    comuni_with_point: Dataframe with one assigned city for each coordinate point.
    
    ------------------

    For cities with coordinates in the perimeter area:
    -Assign the average values of all coordinates in the city perimeter area.
    For cities without coordinates in the perimeter area:
    -Set a circle with radius {interval} centered at the city's centroid as
    the new city perimeter area.
    -At each iteration, assign the average values of all coordinates in the
    new city perimeter area and increase the radius by {interval}.
    -Stop when coordinates are assigned to all cities.
    """

    r_list = list(range(2000, 100000, interval))
    comuni_cv_list = shp_region
    cv_list = comuni_cv_list['COMUNE'].unique()
    comuni_with_point = gpd.GeoDataFrame()

    for r in r_list:
        merged = gpd.sjoin(shp_coord.to_crs(epsg=4326), comuni_cv_list.to_crs(epsg=4326), how="inner",
                           op='intersects')
        comuni_with_point = pd.concat([comuni_with_point, merged])
        cp_list = merged['COMUNE'].unique()
        cv_list = list(set(cv_list) - set(cp_list))

        if len(cv_list) > 0:
            # Expand geometry poligon
            comuni_cv_list = comuni_cv_list[comuni_cv_list['COMUNE'].isin(cv_list)]
            comuni_cv_list['geometry'] = comuni_cv_list.centroid
            comuni_cv_list['geometry'] = comuni_cv_list.buffer(r)
        else:
            break
            
    return comuni_with_point


def calc_city_mean(comuni_with_point, features: list):
    """
    Calculate the average features values by city.
    
    Parameters
    ----------
    
    comuni_with_point: Dataframe with one assigned city for each coordinate point.
    features: List with features names.
    
    Returns
    ----------
    
    shape_table: Dataframe with the average result for each city on each date.
    """
    comuni_with_point_ = comuni_with_point[['data', 'COMUNE'] + features]
    shape_table = comuni_with_point_.groupby(by=['data', 'COMUNE']).mean().reset_index()
    return shape_table
    
  
def from_coord_to_city_mean(dataframe: pd.DataFrame, features: list):
    """
    - Convert longitudine and latitudine coordinates to GeoPandas geometry;
    - Aggregate latitude and longitude coordinates by city;
    - Calculate average values by city.
    
    Parameters
    ----------
    dataframe: DataFrame with longitudine and latitudine coordinates as columns.
    features: List with features names.
    
    Returns
    ----------
    shape_table: Dataframe with the average result for each city on each date.
    """
    shp_coord = add_geo_point(dataframe)
    comuni_with_point = set_coord(shp_coord, shp_region)
    shape_table = calc_city_mean(comuni_with_point, features)
    return shape_table
