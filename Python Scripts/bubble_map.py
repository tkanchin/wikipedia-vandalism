import numpy as np
import pandas as pd
import csv

##Code to generate flare.csv for bubble map visualization

#pass the data directory here


def process_data(datadir):
    new_dict = {}
    for i in range(1,13):
        df = pd.read_csv(datadir + '/vandal_2013_' + str(i)+ '.csv', usecols=['pagetitle'])
        df = df.as_matrix().flatten().tolist()
        for i in df:
            if i not in new_dict:
                new_dict[i] = 1
            else:
                val = new_dict[i]
                val += 1
                new_dict[i] = val
    return new_dict
	
def main():
	datadir_file = "input_data_directory_here"
	dictionary = process_data(datadir_file)
	keys = dictionary.keys()
	c = csv.writer(open("flare.csv", "wb"))
	c.writerow(["id","value"])
	for i in keys:
		c.writerow([i,dictionary[i]])