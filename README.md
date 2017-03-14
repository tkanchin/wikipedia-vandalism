
# Visualizing Vandal users on Wikipedia using d3.js

## File Descriptions
* #### Index.html 
The main page for the Wikipedia Vandalism visualization. Use this page to navigate to other pages.

* #### IndexStats.html
It describes various statistics about benign and vandal users for each month.

* #### BubbleMap.html
The badly edited pages are placed in bubbles. The more the number of bad edits, the bigger is the bubble. For simplicity and for better visualization, only first 50 pages are taken into consideration.

* #### TreeMap.html
The users responsible for bad edits are placed in trees along with thier reputation, the page information and the average sticky score (scored by a bot). For better visualization, only first 50 users were taken into consideration.

* #### clusterPurity.html
The users reponsbile for bad edits are visualized based on parameters such as reputation, bot score and intensity are visualized using clusters for each respective month.

## Data
* The data folder contains the csv files of both Benign and Vandal Users responsible for edits on Wikipedia.

## Python Scripts
All the data was cleaned, extracted and processed using Python. The script folder contains the script files.

* #### stats.py
To generate data for indexStats.html

* #### bubble_map.py
To generate data for BubbleMap.html

* #### tree_map.py  
To generate data for TreeMap.html

* Excel was used to normalize the data and process the data for clusterPurity.html

### Known Issues
The code works well only on Mozilla Browser (It may not work so well on Google Chrome)
