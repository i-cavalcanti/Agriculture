# Clustering Agricuture

#### Import data from Mongo DB collection;
#### Convert latitude and longitude coordinates to cities;
#### Grid search and select best Hyperparameters combination for DBSCAN and Kmeans clustering for each quarter;
#### Fit DBSCAN and Kmeans model for the best Hyperparameters combination for each day;
#### Store model as pickle file;
#### Save DBSCAN and Kmeans labels to MongoDB;
#### Calculate the similarity of each city and every other city in the dataset, by the percentual of days in a quarter each city is classified in the same cluster as every other city in the dataset.
#### Save DBSCAN and Kmeans percentual results to MongoDB.
#### Update and deploy dashboard with results.