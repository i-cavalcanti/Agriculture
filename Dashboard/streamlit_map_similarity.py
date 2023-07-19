import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
from pymongo import MongoClient

from utils import db_utils
import sys
sys.path.append('C:/Users/i.cavalcanti/Progetti_condivisi/clustering_copernicus/Dashboard')

st.set_page_config(layout='wide') # wide, centered

frequency_client = MongoClient('mongodb://localhost:27017')['copernicus_similarity_comuni_puglia']['frequency_1']
cluster_client = MongoClient('mongodb://localhost:27017')['copernicus_similarity_comuni_puglia']['clusters_1']
shp_region = gpd.read_file('C:/Users/i.cavalcanti/Progetti_condivisi/clustering_copernicus/Clustering/DBSCAN/puglia_shape.shp')

def main_clustering():
    #Title
    st.write("""
            ## Similarity among cities in Puglia
            ### In terms of climate and atmosphere parameters
             
            """)
    st.divider()
    
    #Input
    collections = ('Atmosphere', 'Climate')
    city_list = db_utils.list_avaliable_cities(frequency_client)
    ref_city_list = db_utils.list_avaliable_cities_ref(frequency_client)
    year_list = db_utils.list_avaliable_years(frequency_client)
    
    #Sidebar - variables
    
    st.sidebar.multiselect("**Choose the data domain**", key='selected_collections', options=collections, default=collections[0])
    st.sidebar.selectbox("**Select year**", key='selected_year', options=year_list)
    st.sidebar.radio("**Select season**", key='selected_season', options=['Winter', 'Spring', 'Summer', 'Autumn'])
    
    if collections[0] in st.session_state.selected_collections:
        st.sidebar.write("Atmosphere variables:")
        st.sidebar.markdown("- Dust")
        st.sidebar.markdown("- PM10 Aerosol")
        st.sidebar.markdown("- PM2.5 Aerosol")
        st.sidebar.markdown("- Nitrogen Monoxide")
        st.sidebar.markdown("- Nitrogen Dioxide")
        st.sidebar.markdown("- Sulphur Dioxide")
        st.sidebar.markdown("- Ozone")
    if collections[1] in st.session_state.selected_collections:
        st.sidebar.write("Climate variables:")
        st.sidebar.markdown("- Cloud Cover")
        st.sidebar.markdown("- Precipitation")
        st.sidebar.markdown("- Precipitation Duration")
        st.sidebar.markdown("- Relative Humidity")
        st.sidebar.markdown("- Solar Radiation")
        st.sidebar.markdown("- Temperature Air")
        st.sidebar.markdown("- Vapour Pressure")
        st.sidebar.markdown("- Wind Speed")
    

    #Main - variables 
    st.write('#### Choose one city of reference in Puglia')
    st.selectbox("", key='selected_ref_city', options=ref_city_list, index=18)  
    st.divider()
     
    with st.expander("### **:blue[Similarity among cities]**"):
        st.markdown('Here you can:\
                    \n - From a single city of reference and visualize a **geographical map** showing its similarity with all other cities in Puglia for each season.')
        
        #Similarity graph
        if len(st.session_state.selected_collections) == 0:
            st.write('Please select at least one collection')
        else:
            fig = db_utils.plot_values_similarity(st.session_state.selected_collections,
                                st.session_state.selected_year,
                                st.session_state.selected_season,
                                st.session_state.selected_ref_city,
                                frequency_client,
                                shp_region)
            st.write(f'### Percentual of days in which all other cities in Puglia are classified in the same cluster as {st.session_state.selected_ref_city}, {st.session_state.selected_season} {st.session_state.selected_year}')
            st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    
    st.divider()
    
    with st.expander("### **:blue[Similarity among cities in time]**"):
        st.markdown('Here you can:\
                    \n - Visualize the variation of similarity among the city of reference and (_up to_) five other cities in Puglia for all avaliable seasons')
        
        st.write(f'### Percentual of days in which cities are classified in the same cluster as {st.session_state.selected_ref_city} per season')
        
        st.write('##### Choose additioanal comparison cities')
        st.multiselect("**Up to five additioanl cities**", key='selected_cities', options=city_list, 
                       default=[city_list[82], city_list[102], city_list[227]])

        #Comparison graph
        if st.button('Get graph'):
            if st.session_state.selected_ref_city in st.session_state.selected_cities:
                    st.session_state.selected_cities.remove(st.session_state.selected_ref_city)
            if ((len(st.session_state.selected_collections) == 0) and (len(st.session_state.selected_cities) == 0)):
                st.write(f'Please select at least one collection and one comparison city different than {st.session_state.selected_ref_city}')
            elif len(st.session_state.selected_collections) == 0:
                st.write('Please select at least one collection')
            elif len(st.session_state.selected_cities) == 0:
                st.write(f'Please select at least one comparison city different than {st.session_state.selected_ref_city}')
            elif len(st.session_state.selected_cities) > 5:
                st.write(f'Please select only up to five comparison cities different than {st.session_state.selected_ref_city}')
            else:
                fig = db_utils.plot_values_comparison(st.session_state.selected_collections,
                                                    st.session_state.selected_ref_city,
                                                    st.session_state.selected_cities,
                                                    frequency_client)
                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    
    st.divider()
    
    with st.expander("**:blue[Outliers map]**"):
        st.markdown('Here you can:\
                    \n - Visualize the cities most classified as outliers in Puglia for each season.')
        
        #Outliers Graph  
        if len(st.session_state.selected_collections) == 0:
            st.write('Please select at least one collection')
        else:
            fig = db_utils.plot_values_clusters(st.session_state.selected_collections, 
                                            st.session_state.selected_year,
                                            st.session_state.selected_season,
                                            cluster_client,
                                            shp_region)
            st.write(f'### Percentual of days in which all cities in Puglia are classified as outlier, {st.session_state.selected_season} {st.session_state.selected_year}')
            st.plotly_chart(fig, theme='streamlit', use_container_width=True)
        
    
if __name__ == '__main__':
    main()