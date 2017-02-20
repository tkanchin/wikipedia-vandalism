import numpy as np
import pandas as pd
import csv

##Code to generate statistics for statistics visualization

def main():
    new_dict = {}
    for i in range(1, 13):
        f = open('benign_2013_'+ str(i) + '.csv', 'r')
        f1 = open('vandal_2013_'+ str(i) + '.csv', 'r')
        benign = vandal = 0
        lines = f.readlines() #array
        lines1 = f1.readlines() #array
        for line in lines:
            if line in ['\n', '\r\n']:
                benign += 1
        for line in lines1:
            if line in ['\n', '\r\n']:
                vandal += 1
        new_dict[i] = (benign, vandal)
    new_dict = sorted(new_dict.items(), key=operator.itemgetter(0))
    freq_data = []
    for i in new_dict:
        k , val = i
        benign, vandal = val
        freq_data.append("{State:'" + str(k) + "', freq: { Vandal: " + str(benign) + ", Benign: " + str(vandal) + "}}")
    print ','.join(freq_data)
	
	
