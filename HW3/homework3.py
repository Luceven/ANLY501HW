# -*- coding: utf-8 -*-
"""
ANLY-501 HW3
Author: Kate Zeng
"""

import pandas as pd
import sys

## functions
# Part 1
# determine the number of unique values in each column
#data.apply(pd.Series.nunique)
def print_unique(df):
    for cols in df:
        count = df[cols].nunique(dropna=False)
        print("The number of unique values in", cols, "is", count)
        
# Find the number of rows that have a user rating that is not 5

def rating_not5(df):
    rating_not5 = df.loc[df['User Ratings'] != '5']
    row_count = rating_not5.shape[0]
    print("The number of rows that have user rating is not 5: ", row_count)
    
# Find the number of missing values in each column
#data.isnull().sum()

def print_missing(df):
    for cols in df:
        count = df[cols].isnull().sum()
        print("The number of missing values in", cols, "is", count)
        
# Part 2
# make City, State column to lowercase, then seperate city and state

def city_state(df):
    df['City, State'] = df['City, State'].str.lower()
    df['City'] = df['City, State'].str.rsplit(',', 1, expand = True)[0]
    df['State'] = df['City, State'].str.rsplit(',', 1, expand = True)[1]
    df = df.drop(['City, State'], axis=1)
    return df

# print number of rows with null data, and the city and state if available, and drop those rows

def drop_null(df):
    null_data = df[df.isnull().any(axis=1)]
    row_count = null_data.shape[0]
    print("The number of rows that will be dropped:", row_count)
    print("The cities of the rows with missing value:", null_data['City'].tolist())
    print("The states of the rows with missing value:", null_data['State'].tolist())
    df.dropna()
    return df

# Remove rows with invalid user rating. 
# Print to the screen the number of rows you dropped and the City and State for that row (if available).

def drop_bad_rating(df):
    bad_rating = df.loc[df['User Ratings'] == 'a']
    row_count = bad_rating.shape[0]
    print("The number of rows that will be dropped:", row_count)
    print("The cities of the rows with missing value:", bad_rating['City'].tolist())
    print("The states of the rows with missing value:", bad_rating['State'].tolist())
    df = df[df['User Ratings'] != 'a']
    return df

# Sort data by latitude and longitude
def sort_by_latlong(df):
    sorted_data = df.sort_values(['Latitude', 'Longitude'], ascending=True)
    return sorted_data
        
# change delimiter to |
def change_delimiter(df):
    df.to_csv('cleaned_data.csv', sep = '|', index = False)
        

## main function
def main():
    # loading data
    data = pd.read_csv('data.csv')
    
    ## Part 1
    # print number of unique values in each column
    print("The number of unique values in each column is: ")
    print_unique(data)
    
    print("\n\n")
    
    # find the number of rows with user_ratings not 5
    print("The number of rows with user ratings not equal to 5: ")
    rating_not5(data)
    
    print("\n\n")
    
    # find number of missing values in each column
    print("The number of missing values in each column: ")
    print_missing(data)
    
    print("\n\n")

    ## Part 2
    # Make City, State column to lowercase, then seperate city and state
    data = city_state(data)

    # print number of rows with null data, and the city and state if available, and drop those rows
    print("The number of the rows dropped: ")
    data = drop_null(data)
    
    print("\n\n")

    # Remove rows with invalid user rating. 
    # Print to the screen the number of rows you dropped and the City and State for that row (if available).
    print("The number of rows dropped: ")
    data = drop_bad_rating(data)
    
    print("\n\n")

    # check unique value
    #data['User Ratings'].unique()
    
    # Sort data by latitude and longitude
    sorted_data = sort_by_latlong(data)
    
    # print the first ten of each sorted dataframes
    print("The first 10 of dataframe that sort by lantitude and longitude: ")
    print(sorted_data.head(n=10))
    
    # save to file and change delimiter
    change_delimiter(sorted_data)
    
if __name__ == "__main__":
    # execute only if run as a script
    main()