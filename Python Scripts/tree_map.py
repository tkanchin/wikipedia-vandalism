import numpy as np
import pandas as pd
import csv

##Code to generate csv for the tree map which was used later as inpput to create flare.json

def process_data_treeMap(datadir):
    new_dict = {}
    for i in range(1,13):
        df = pd.read_csv(datadir + '/vandal_2013_' + str(i)+ '.csv', usecols=['pagetitle', 'username', 'stiki_score', 'stiki_REP_USER'])
        df_pagetitle = df['pagetitle'].as_matrix().flatten().tolist()
        df_username = df['username'].as_matrix().flatten().tolist()
        df_stiki_score = df['stiki_score'].as_matrix().flatten().tolist()
        df_stiki_REP_USER = df['stiki_REP_USER'].as_matrix().flatten().tolist()
        for i in range(len(df_pagetitle)):
            if df_pagetitle[i] not in new_dict:
                new_dict[df_pagetitle[i]] = [df_username[i],df_stiki_score[i], df_stiki_REP_USER[i], 1]
            else:
                val = new_dict[df_pagetitle[i]]
                val[1] += df_stiki_score[i]
                val[2] += df_stiki_REP_USER[i]
                val[3] += 1
                new_dict[df_pagetitle[i]] = val
    return new_dict
	
def main():
	datadir_file = "inoput_datadirectory_here"
	dictionary = process_data_treeMap(datadir_file)
	keys = dictionary.keys()
	c = csv.writer(open("tree_map.csv", "wb"))
	c.writerow(["pagetitle","username", "stiki_score", "user_rep", "no_of_edits"])
	for i in keys:
		list_value = dictionary[i]
		c.writerow([i,list_value[0], list_value[1], list_value[2], list_value[3]])
