# env: py3
# Author: Kate Zeng

import pandas as pd
import datetime
import urllib
from urllib.request import urlopen

def AirNow():
	baseURL = "http://www.airnowapi.org/aq/forecast/"
	api_key = '22AAC340-751E-4CDF-A907-17C3F98922B1'
	#date = '2018-08-04'
	# get the current date as input
	now = datetime.datetime.now()
	date = str(now)
	miles = 25
	dfs = list()

	text_file = open("INPUT.txt", "r")
	zipcodes = text_file.read().split(' ')
	text_file.close()

	for zipcode in zipcodes:
	    zipURL = baseURL + "zipCode/?" + urllib.parse.urlencode({
	        'format': 'application/json',
	        'zipCode': zipcode,
	        'date': date[:10],
	        'distance': miles,
	        'API_KEY': api_key
	        })
	    response = urlopen(zipURL).read().decode('utf-8')
	    df = pd.read_json(response)
	    df = df.assign(Zipcode=zipcode)
	    dfs.append(df)

	results = pd.concat(dfs)
	#columns = ['ActionDay', 'Category', 'DateIssue', 'Discussion', 'Latitude', 'Longitude']
	#results.drop(columns, inplace=True, axis=1)
	return results

def main():
	results = AirNow()
	print("\nAQI data collected:\n\n", results)
	results.to_csv('AQI_output.csv', index=False)

if __name__ == "__main__":
	main()