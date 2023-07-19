import main
import geopandas as gpd
from utils import database_utils
from pymongo import MongoClient

geo_comuni = gpd.read_file('C:/Users/i.cavalcanti/Progetti/clustering_copernicus/data/coord_comuni_puglia.shp')
import_client = database_utils.connect_mongodb("10.8.0.89", 40512, "ivan.cavalcanti", "IVoIDnkA48O8J37R")
export_client = MongoClient('mongodb://localhost:27017')['copernicus_similarity_comuni_puglia']['frequency']
year_list = ['2021', '2022']
main.frequency_by_quarter_calculator(import_client, export_client, geo_comuni, year_list)