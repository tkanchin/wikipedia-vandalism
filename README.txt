*** This project works on Mozilla Browser ****
*** Sometimes the d3.js library is not working on Google Chrome ***

This folder contains the following files:


-- index.html 
	The main page for the Wikipedia Vandalism visualization. Using this page we can navigate to other 
	visualizations.


-- indexStats.html
	The data was extracted, cleaned using a Python script. The data was inputted in this file itself.
	It describes various statistics about benign and vandal users and more is explained in the report.

-- BubbleMap.html
	The data file for this visualization is flare.csv. 
	See report for more information

-- TreeMap.html
	The data file for this visualization is flare.json
	See report for more information.

-- clusterPurity.html
	The data file for this visualization is clusterPriorityDataSet and the JS file is clusterpurityChart.
	See report for more information.


****************************************************************************************************************

The Scripts folder contains three script files which generate data required for the three visualizations. Namely:

-- stats.py : to generate data for indexStats.html

-- bubble_map.py : to generate data for BubbleMap.html

-- tree_map.py : to generate data for TreeMap.html

-- We used excel to normalize the data and process the data for clusterPurity.html

******************************************************************************************************************


