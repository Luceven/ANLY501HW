# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 18:32:56 2018

ANLY-501 HW4

@author: Kate Zeng
"""

# Load all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn import decomposition

# preprocessing
def create_test_data(df):
    # print to screen the first 10 row of the data
    print('The first 10 rows of data is:\n', file=open("output.txt", "a"))
    print(df.head(10), file=open("output.txt", "a"))
    print(df.head(10))
    # creat a test dataframe with first 1000 entries
    test_data = df.head(100)
    return test_data

def bin_age(df):
    # Compute min and max values, then add 1 to each side.
    minAge = df["Age"].min()
    maxAge = df["Age"].max()
    minAge = minAge - 1
    maxAge = maxAge + 1
    # Create even spaced bins using min and max
    bins =  np.arange(minAge,maxAge, 10)
    bins1 = pd.cut(df['Age'], 5, retbins=True)
    bins1=np.asarray(bins1)
    # Create a new variable AgeGroups that groups users into bins, e.g. < 18, 28-44, etc.
    # For this example, I use the bins created above
    df['AgeGroups'] = np.digitize(df['Age'],bins)
    return df

# histogram of Impression based on gender

def impression_by_gender(df):
    male = df[df['Gender'] == 0]['Impressions']
    female = df[df['Gender'] == 1]['Impressions']
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.hist([male, female], label = ['M', 'F'])
    ax.set_title("Histogram of Impressions by Gender")
    ax.set_xlabel("Number of Imporessions")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right')
    return fig

# histogram of Impressions based on age group
def impression_by_age(df):
    ag1 = df[df['AgeGroups'] == 1]['Impressions']
    ag2 = df[df['AgeGroups'] == 2]['Impressions']
    ag3 = df[df['AgeGroups'] == 3]['Impressions']
    ag4 = df[df['AgeGroups'] == 4]['Impressions']
    ag5 = df[df['AgeGroups'] == 5]['Impressions']
    ag6 = df[df['AgeGroups'] == 6]['Impressions']
    ag7 = df[df['AgeGroups'] == 7]['Impressions']
    ag8 = df[df['AgeGroups'] == 8]['Impressions']
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.hist([ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8], label = 
            ['group1', 'group2', 'group3', 'group4', 'group5', 'group6', 'group7', 'group8'])
    ax.set_title("Histogram of Impressions by Age Group")
    ax.set_xlabel("Number of Imporessions")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right')
    return fig

# Plot the distribution of the number of impressions for the age categories created.
def hists_age_impressions(df):
    # Get unique Age groups. Iterate through list and plot each histogram
    ageSeries = df['AgeGroups'].unique()
    ageSeries.sort()
    # Iterate through each age and generate the plot
    counter = 1
    for age in ageSeries:
        # We need to select rows containing a particular age
        queryString = "AgeGroups == " + str(age)
        ageGroupImpressions = df[['AgeGroups', 'Impressions']].query(queryString)

        # Create histogram and label it
        ageGroupImpressions['Impressions'].hist()
        titleLabel = "Distribution of Impressions for Age Group " + str(age)
        plt.title(titleLabel)
        plt.xlabel("Number of Impressions")
        plt.ylabel("Frequency")

        # Write to file
        fileName = 'age' + str(counter) + '.png'
        plt.savefig(fileName)

        # Clear plot
        plt.clf()
        counter += 1
        
# fist function for k-means clustering, n is number of k
def kmeans_clustering(df, n):
    myData = pd.concat([df['Age'], df['Gender'], df['Impressions'], df['Clicks'], df['Signed_In'], df['AgeGroups'], df['ClickThrougRate']], 
                 axis=1, keys=['Age', 'Gender', 'Impressions', 'Clicks', 'Signed_In','AgeGroups', 'ClickThrougRate' ])
    myData = myData.dropna()
    x = myData.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    k = n
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg, file=open("output.txt", "a"))
    #####
    # PCA
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    pca2D = decomposition.PCA(2)
    
    # Turn the NY Times data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("Clustering w/ K="+str(k))
    plt.savefig('clustering' + str(k) + '.png')
    plt.show()


# CATEGORICAL VARIABLES
# Create a new variable that categorizes behavior based on click-thru rate.
# Create categories for click thru behavior. We can do this on the original data
def click_behavior(row):
   if row['Clicks'] > 0:
      return 'Clicks'
   if row['Impressions'] == 0:
      return 'noImpressions'
   if row['Impressions'] > 0:
      return 'Impressions'

   return 'Other'

# second function for k-means clustering after adding clickBehavior
def kmeans_clustering_new(df, n):
    myData = pd.concat([df['Age'], df['Gender'], df['Impressions'], df['Clicks'], df['Signed_In'], df['AgeGroups'], df['ClickThrougRate'], df['clickBehavior']], 
                 axis=1, keys=['Age', 'Gender', 'Impressions', 'Clicks', 'Signed_In','AgeGroups', 'ClickThrougRate', 'clickBehavior'])
    codes = {'Clicks': 1, 'Impressions': 2, 'noImpressions': 3, 'Other': 4}
    myData['clickBehavior'] = myData['clickBehavior'].map(codes)
    myData = myData.dropna()
    x = myData.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    k = n
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg, file=open("output.txt", "a"))
    #####
    # PCA
    # Let's convert our high dimensional data to 2 dimensions
    # using PCA
    pca2D = decomposition.PCA(2)
    
    # Turn the NY Times data into two columns with PCA
    plot_columns = pca2D.fit_transform(normalizedDataFrame)
    
    # Plot using a scatter plot and shade by cluster label
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=cluster_labels)
    plt.title("Clustering with clickBehavior w/ K="+str(k))
    plt.savefig('clustering_click' + str(k) + '.png')
    plt.show()


def main():
    # load data
    data = pd.read_csv('NY_Times_LARGE.csv')
    
    # data summary
    print("Data Summary using info method\n")
    print(data.info())

    print("Data summary using describe method (stats about each column)\n", file=open("output.txt", "a"))
    print(data.describe(), file=open("output.txt", "a"))
    
    # preprocess data set, print to screen the first 10 and create test dataframe w/ first 1000
    test_data = create_test_data(data)
    
    # try two different binning strategies
    test_data = bin_age(test_data)
    
    # a column that represents the ratio between Impressions and Clicks is also good to be binned, therefore we can group
    # people to groups that shows how likely they would click an advertisement
    # Create a variable called ClickThrougRate (# of clicks/# of impressions)
    test_data['ClickThrougRate'] = test_data['Clicks'] / test_data['Impressions']
    
    # output histogram of Impression based on gender
    hist_gender_impressions = impression_by_gender(test_data)
    hist_gender_impressions.savefig('hist_gender_impressions.png')
    plt.close(hist_gender_impressions)
    
    # output histogram of Impression based on age groups
    hist_age_impressions = impression_by_age(test_data)
    hist_age_impressions.savefig('hist_age_impressions.png')
    plt.close(hist_age_impressions)
    
    # output histograms of Impression based on age groups
    hists_age_impressions(test_data)
    
    # try 3 different k value for k-means clustering
    kmeans_clustering(test_data, 2)
    kmeans_clustering(test_data, 3)
    kmeans_clustering(test_data, 4)
    
    
    # Create a new variable that categorizes behavior based on click-thru rate.
    # Create categories for click thru behavior. We can do this on the original data
    test_data['clickBehavior']  = test_data.apply(lambda row: click_behavior(row), axis=1)
    
    print("\n", file=open("output.txt", "a"))
    
    # try 3 different k values for k-means clustering after adding clickBehavior
    kmeans_clustering_new(test_data, 2)
    kmeans_clustering_new(test_data, 3)
    kmeans_clustering_new(test_data, 4)
    
if __name__ == "__main__":
	main()