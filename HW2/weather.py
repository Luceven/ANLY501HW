# env: py3
# part 4 of HW2
# Author: Kate Zeng


import pandas as pd
import datetime
import urllib
import json
from urllib.request import urlopen

def Weather():
	### for open weather map
	API_KEY = '###YOUR_API_KEY###'
	cnt = 5
	base_url = 'http://api.openweathermap.org/data/2.5/find?'
	#city_name = 'Washington D.C.'
	latitude = '38.9'
	longitude = '-77.0'

	all_url = base_url + urllib.parse.urlencode({
		'APPID': API_KEY,
		'lat': latitude,
		'lon': longitude,
		'cnt': cnt
		})

	response = urlopen(all_url).read().decode('utf-8')

	return response

def main():
	response = Weather()

	with open("weather_data_file.json", "w") as write_file:
		json.dump(response, write_file)

	json_data = json.loads(response)

	city_list = []
	temp_list = []
	temp_max_list = []
	temp_min_list = []
	humidity_list = []

	for each in json_data['list']:
		city_list.append(each['name'])
		temp_list.append(str(each['main']['temp']))
		temp_max_list.append(str(each['main']['temp_max']))
		temp_min_list.append(str(each['main']['temp_min']))
		humidity_list.append(str(each['main']['humidity']))

	with open("output.txt", "w") as outfile:
		outfile.write(city_list[0] + '|' + str(json_data['count']) + '\n')
		for tmp, tmp_mi, tmp_mx, hum in zip(temp_list, temp_min_list, temp_max_list, humidity_list):
			outfile.write(tmp + '|' + tmp_mi + '|' + tmp_mx + '|' + hum + '\n')

if __name__ == "__main__":
	main()
