from pymongo import MongoClient
import db_utils
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import pickle
import os
import glob
import calendar
import sys

st.set_page_config(layout='wide') # wide, centered
sys.path.append('../Clustering/SOM/')

frequency_client = MongoClient('mongodb://localhost:27017')['copernicus_similarity_comuni_puglia']['frequency_1']
cluster_client = MongoClient('mongodb://localhost:27017')['copernicus_similarity_comuni_puglia']['clusters_1']
shp_region = gpd.read_file('../Clustering/DBSCAN/puglia_shape.shp')


pages = ['Clustering', 'Self-Organizing Maps (SOM)']
choose_page = st.sidebar.selectbox('**Pages**', pages)

variables_atmosphere = ['Dust',
                        'PM10 Aerosol',
                        'PM2.5 Aerosol',
                        'Nitrogen Monoxide',
                        'Nitrogen Dioxide',
                        'Sulphur Dioxide',
                        'Ozone']
short_variables_atmosphere = ['Dust', 'PM10', 'PM2.5', 'NO', 'NO2', 'SO2', 'O3']
variables_dict_atmosphere = dict(zip(short_variables_atmosphere, variables_atmosphere))
variables_dict_reversed_atmosphere = dict(zip(variables_atmosphere, short_variables_atmosphere))
uom_atmosphere = dict(zip(variables_atmosphere, ['\u03BCg/m\u00b3'] * len(variables_atmosphere)))

variables_climate = ['Cloud_Cover',
                     'Precipitation',
                     'Precipitation_Duration',
                     'Relative_Humidity',
                     'Solar_Radiation',
                     'Temperature_Air',
                     'Vapour_Pressure',
                     'Wind_Speed']
variables_climate_spaced = {k: " ".join(k.split('_')) for k in variables_climate}
uom_climate = dict(zip(variables_climate, ['', 'mm/day', '', '%', '', '°C', 'hPa', 'm/s']))

uoms = {}
uoms.update(uom_atmosphere)
uoms.update(uom_climate)

dict_trimester = dict(Winter=1, Spring=2, Summer=3, Autumn=4)

def load_model(collection, year, season, variable):
    path = '../Clustering/SOM/TrainingPipeline/saved_models/SOM_' + f'{collection}_' + f'{variable}_' + f'{season}_' + f'{year}.pkl'
    with open(path, 'rb') as m:
        model = pickle.load(m)
    return model
    
def visualize_data(model):
    dataset = model.data_().copy()
    dataset.columns = ["-".join(c.split('-')[:-1]) for c in dataset.columns]
    st.dataframe(dataset.style.highlight_max(axis=0, color='violet'))

def plot_distance_map(model, user_ts=None):
    fig = model.plot_distance_3d(predict_vector=user_ts)
    return st.plotly_chart(fig, theme='streamlit')

def plot_heatmap(model, variable, user_ts=None):
    fig = model.plot_heatmap_3d(variable=variable, predict_vector=user_ts)
    return st.plotly_chart(fig, theme='streamlit')

def plot_time_series(model, cities):
    df = model.data_().loc[cities].T
    dates = ["-".join(c.split('-')[:-1]) for c in df.index]
    fig = px.line(df, x=dates, y=cities)
    fig.update_layout(yaxis_title=uoms[df.index[0].split('-')[-1]],
                      xaxis_title='Dates',
                      width=900,
                      height=500,
                      legend=dict(title_text='<b>Cities</b>'),
                      hovermode='x')
    fig.update_traces(hovertemplate='<b>Value</b>: %{y}')
    return st.plotly_chart(fig, theme=None)
    
def plot_user_time_series(model, ts):
    predicted_position = model.predict(ts)[0]
    predicted_ts = model.weights()[predicted_position]
    df = pd.DataFrame({'Your time series': ts,
                       'Predicted time series': predicted_ts},
                       index = model.data_().columns)
    dates = ["-".join(c.split('-')[:-1]) for c in df.index]
    fig = px.line(df, x=dates, y=df.columns)
    fig.update_layout(yaxis_title=uoms[df.index[0].split('-')[-1]],
                      xaxis_title='Dates',
                      width=900,
                      height=500,
                      legend=dict(title_text='<b>Time series</b>'),
                      hovermode='x')
    fig.update_traces(hovertemplate='<b>Value</b>: %{y}')
    return st.plotly_chart(fig, theme=None)

def plot_map(model, city):
    fig = model.plot_geomap(city)
    return st.pyplot(fig)

def plot_user_map(model, user_ts):
    fig = model.plot_user_geomap(user_ts)
    return st.pyplot(fig)

def postprocess_user_ts(model, year, season, ts):
    trimester = dict_trimester[season]
    total_days = sum([calendar.monthrange(int(year), m)[-1] for m in range((trimester-1)*3+1, trimester*3+1)])
    if len(ts) < total_days:
        add_elements = total_days - len(ts)
        new_ts = np.concatenate([ts, [np.nan] * add_elements]).reshape(-1)
    elif len(ts) > total_days:
        subtract_elements = len(ts) - total_days
        new_ts = ts[:-subtract_elements]
    else:
        new_ts = ts
    rolled_ts = model.predict(new_ts)[1]
    return rolled_ts

def return_correct_name(collection, variable):
    if collection == 'Atmosphere':
        variable = variables_dict_atmosphere[variable] # This is st.session_state.selected_variable
    elif collection == 'Climate':
        variable = "_".join(variable.split()) # same as above
    else:
        pass
    return variable
   

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
        
   
def main():
    if choose_page == 'Clustering':
        main_clustering()
    else:
        st.title('Self-Organizing Maps (SOM)')
        folder_path = "../Clustering/SOM/TrainingPipeline/saved_models/"
        extension = "*.pkl"
        search_path = os.path.join(folder_path, extension)
        _files = glob.glob(search_path)
        list_collections = sorted(list(set([coll.split('_')[2].title() for coll in _files])))
        st.sidebar.divider()
        st.sidebar.selectbox("**Choose the data domain**", key='selected_collection', options=list_collections)
        list_years = sorted(list(set([name.split('_')[-1][:-4] for name in _files])))
        st.sidebar.selectbox("**Select year**", key='selected_year', options=list_years)
        st.sidebar.radio("**Select season**", key='selected_season', options=['Winter', 'Spring', 'Summer', 'Autumn'])
        # 'model_variable' will store the name of the variable used to save the pkl files to access them.
        if st.session_state.selected_collection == 'Atmosphere':
            st.sidebar.radio("**Select variable**", key='selected_variable', options=variables_atmosphere)
            model_variable = variables_dict_reversed_atmosphere[st.session_state.selected_variable]
        elif st.session_state.selected_collection == 'Climate':
            st.sidebar.radio("**Select variable**", key='selected_variable', options=variables_climate_spaced.values())
            model_variable = "_".join(st.session_state.selected_variable.split())
        # Load requested model
        model_ = load_model(st.session_state.selected_collection.lower(),
                            st.session_state.selected_year, 
                            st.session_state.selected_season,
                            model_variable)
        with st.expander("**:blue[Visualize data]**"):
            st.write('Here you can see the dataset used to train the model. The highest\
                     value for each day is highlighted in :violet[purple]. You can press CTRL+F \
                     to search for a particular city.')
            if st.session_state.selected_collection == 'Atmosphere':
                st.subheader(f'Dataframe: {variables_dict_atmosphere[model_variable]} - {st.session_state.selected_season} {st.session_state.selected_year}')
            elif st.session_state.selected_collection == 'Climate':
                st.subheader(f'Dataframe: {" ".join(model_variable.split("_"))} - {st.session_state.selected_season} {st.session_state.selected_year}')
            else:
                pass
            visualize_data(model_)
        with st.expander("**:blue[Model results]**"):
            st.markdown("Here you can visualize the results of the selected model. We have two plots:\
                        \n - **_Distance Map_**: darker red areas correspond to more isolated (groups of) cities\
                              while lighter areas correspond to denser groups of cities, i.e. cities whose time\
                              series exhibit similar patterns and values; \
                        \n - **_Feature Heatmap_**: it shows the seasonal average values of the SOM-predicted\
                             time series.")
            st.write('')
            if st.session_state.selected_collection == 'Atmosphere':
                st.subheader(f'Graphs: {variables_dict_atmosphere[model_variable]} - {st.session_state.selected_season} {st.session_state.selected_year}')
            elif st.session_state.selected_collection == 'Climate':
                st.subheader(f'Graphs: {" ".join(model_variable.split("_"))} - {st.session_state.selected_season} {st.session_state.selected_year}')
            else:
                pass
            plot_distance_map(model_)
            plot_heatmap(model_, return_correct_name(st.session_state.selected_collection, model_variable))
        with st.expander("**:blue[Geographical map]**"):
            st.markdown('Here you can:\
                        \n - Visualize (_up to_) 3 **time series** for the requested cities for the given year, season and variable; \
                        \n - Select a single city and visualize a **geographical map** showing the distances of its predicted time series with those of the other cities in Puglia. \
                             In this way you can have an idea of the geographical similarity between the cities.')
            st.multiselect("Select cities",
                           key='selected_city',
                           options=model_.data_().index,
                           max_selections=3)
            st.subheader(f':blue[Time series]')
            if st.session_state.selected_collection == 'Atmosphere':
                st.subheader(f'{variables_dict_atmosphere[model_variable]}')
            elif st.session_state.selected_collection == 'Climate':
                st.subheader(f'{" ".join(model_variable.split("_"))}')
            else:
                pass
            plot_time_series(model_, st.session_state.selected_city)
            st.write('')
            st.subheader(f':blue[Geographical map]')
            st.selectbox("Select a single city",
                         key='selected_city_single',
                         options=model_.data_().index.values)
            plot_map(model_, st.session_state.selected_city_single)
        with st.expander("**:blue[Predictions]**"):
            st.markdown('Here you can upload a time series of your own and visualize:\
                        \n - Its lineplot for the requested season compared with that of the most similar time series according to the SOM model;\
                        \n - Its position in the distance map;\
                        \n - Its position in the feature heatmap;\
                        \n - Its _hypothetical_ position on the Puglia geographical map\
                             (based on the most similar SOM-predicted time series).')
            st.write("")
            st.markdown(':heavy_exclamation_mark:' + '**Attention**' + ':heavy_exclamation_mark:')
            st.markdown("If your time series doesn't contain the exact same number of days as the season you requested\
                        would expect, then a form of automatic interpolation process will be applied to it,\
                        thus resulting in a less accurate prediction.\
                        \n In addition, *if your time series contains more than the 25% of missing values, you won't\
                        be able to perform any kind of prediction on it*, so make sure your data meet this requirement\
                        before uploading it.")
            uploaded_ts = st.file_uploader(label='Upload your time series here:',
                                           type=['csv', 'xlsx'],
                                           label_visibility='visible')
            if uploaded_ts is not None:
                file_extension = uploaded_ts.name.split('.')[-1]
                if file_extension == 'csv':
                    ts = pd.read_csv(uploaded_ts).to_numpy().reshape(-1)
                elif file_extension == 'xlsx':
                    ts = pd.read_excel(uploaded_ts).to_numpy().reshape(-1)
                new_ts = postprocess_user_ts(model_,
                                             st.session_state.selected_year,
                                             st.session_state.selected_season,
                                             ts)
                if st.session_state.selected_collection == 'Atmosphere':
                    st.subheader(f'{variables_dict_atmosphere[model_variable]} - Your time series')
                elif st.session_state.selected_collection == 'Climate':
                    st.subheader(f'{" ".join(model_variable.split("_"))} - Your time series')
                else:
                    pass
                plot_user_time_series(model_, new_ts)
                plot_distance_map(model_, new_ts)
                plot_heatmap(model_, 
                             return_correct_name(st.session_state.selected_collection, model_variable), 
                             new_ts)
                plot_user_map(model_, new_ts)

if __name__ == '__main__':
    main()
