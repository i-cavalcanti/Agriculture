import pandas as pd
import geopandas as gpd


def add_geo_point(dataframe: pd.DataFrame):
    """
    Convert longitudine and latitudine coordinates to GeoPandas geometry.

    Parameters
    ----------
    dataframe: DataFrame with columns longitudine and latitudine.

    Returns
    -------
    table_geo: Geopandas Dataframe with geometry column

    """
    table_geo = gpd.geopandas.GeoDataFrame(
        dataframe, geometry=gpd.geopandas.points_from_xy(dataframe.longitudine, dataframe.latitudine))
    table_geo.set_crs("epsg:4326", inplace=True)
    table_geo = table_geo.iloc[:, 2:]
    return table_geo


def set_coord(shp_coord: gpd.geopandas.GeoDataFrame, shp_region: gpd.geopandas.GeoDataFrame, interval=1000):
    """
    Aggregate latitude and longitude coordinates by city.
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


def calc_city_average(comuni_with_point: gpd.geopandas.GeoDataFrame):
    """
    Calculate the average features values by city.
    """
   
    comuni_with_point_ = comuni_with_point.drop(['geometry', 'index_right', 'PROVINCE'], axis=1)
    shape_table = comuni_with_point_.groupby(by=['data', 'COMUNE']).mean().reset_index()
    return shape_table
    
    
    
