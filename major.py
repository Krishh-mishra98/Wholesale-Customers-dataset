#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:55:21 2019

@author: km
"""
import pandas as pd
import numpy as np
# Load the wholesale customers dataset
#check whether any dataset missing ?
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print( "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print ("Dataset could not be loaded. Is the dataset missing?")


stats = data.describe()
print(stats)
# Using data.loc to filter a pandas DataFrame
data.loc[[100, 200, 300],:]
data.columns
# Select three indices of your choice you wish to sample from the dataset
indices = [43, 12, 39]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.columns).reset_index(drop = True)
print ("Chosen samples of wholesale customers dataset:")
# Get the means 
mean_data = data.describe().loc['mean', :]
# Append means to the samples' data
samples_bar = samples.append(mean_data)
# Construct indices
samples_bar.index = indices + ['mean']
# Plot bar plot
samples_bar.plot(kind='bar', figsize=(14,8))
# First, calculate the percentile ranks of the whole dataset.
percentiles = data.rank(pct=True)
# Then, round it up, and multiply by 100
percentiles = 100*percentiles.round(decimals=3)
# Select the indices you chose from the percentiles dataframe
percentiles = percentiles.iloc[indices]
data.columns

# Importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

# Create list to loop through
dep_vars = list(data.columns)


# Create loop to test each feature as a dependent variable
for var in dep_vars:
    #Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    new_data = data.drop([var], axis = 1)
    new_feature = pd.DataFrame(data.loc[:, var]) # feature series
    # Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(new_data, new_feature, test_size=0.25, random_state=42)
    # Creating a decision tree regressor and fitting it to the training set
    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X_train, y_train) # fitting the data
    y_pred = dtr.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("confusion matrix of decision regression\n",cm)
    print('Accuracy : %.8f'%accuracy_score(y_test,y_pred))
    

pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde'); #scatter matrix
import matplotlib.pyplot as plt

def plot_corr(df,size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(df, interpolation='nearest')
    ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);

plot_corr(data)



#Implementation: Feature Scaling

#Scale the data using the natural logarithm
log_data = np.log(data)

#Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# This is how np.percentile would work
# np.percentile[series, percentile]
np.percentile(data.loc[:, 'Milk'], 25)


#Modified Code


import itertools
# Select the indices for data points you wish to remove
outliers_lst  = []

# For each feature find the data points with extreme high or low values
for feature in log_data.columns:
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data.loc[:, feature], 25)

    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data.loc[:, feature], 75)

    #  Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)

    # Display the outliers
    print ("Data points considered outliers for the feature '{}':".format(feature))

    # The tilde sign ~ means not
    # So here, we're finding any points outside of Q1 - step and Q3 + step
    outliers_rows = log_data.loc[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step)), :]
    # display(outliers_rows)

    outliers_lst.append(list(outliers_rows.index))

outliers = list(itertools.chain.from_iterable(outliers_lst))

# List of unique outliers
# We use set()
# Sets are lists with no duplicate entries
uniq_outliers = list(set(outliers))

# List of duplicate outliers
dup_outliers = list(set([x for x in outliers if outliers.count(x) > 1]))
print ('Outliers list:\n', uniq_outliers)
print ('Length of outliers list:\n', len(uniq_outliers))

print ('Duplicate list:\n', dup_outliers)
print ('Length of duplicates list:\n', len(dup_outliers))

# Remove duplicate outliers
# Only 5 specified
good_data = log_data.drop(log_data.index[dup_outliers]).reset_index(drop = True)

# Original Data 
print ('Original shape of data:\n', data.shape)
# Processed Data
print ('New shape of data:\n', good_data.shape)


from sklearn.decomposition import PCA

#Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=6)
# Fit
pca.fit(good_data)

# Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results 
#Apply PCA by fitting the good data with only two dimensions

pca = PCA(n_components=2)
pca.fit(good_data)
#Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)
#Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(log_samples)
# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# Importing libraries

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Create range of clusters 
range_n_clusters = list(range(2,11))
print(range_n_clusters)

#GMM Implementation


    
lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']


#KNN Implementation
for n_clusters in range_n_clusters:
    # Apply your clustering algorithm of choice to the reduced data 
    clusterer = KMeans(n_clusters=n_clusters).fit(reduced_data)

    # Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    #Find the cluster centers
    centers = clusterer.cluster_centers_

    #Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    #Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds, metric='euclidean')

    print ("For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score))


for var in dep_vars:

    #Make a copy of the DataFrame, using the 'drop' function to drop the given feature
    new_data = data.drop([var], axis = 1)

    # Create feature Series (Vector)
    new_feature = pd.DataFrame(data.loc[:, var])

    # Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = train_test_split(new_data, new_feature, test_size=0.25, random_state=42)
    dtr = SVC(kernel='linear', C=1,random_state = 42)
    dtr.fit(X_train, y_train)
    y_pred = dtr.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("confusion matrix of SVM\n",cm)
    print('Accuracy : %.8f'%accuracy_score(y_test,y_pred))




