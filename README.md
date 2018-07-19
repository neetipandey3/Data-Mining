# Spark-Scala-and-Python

## Implementation of Data Mining Algorithms

pyspark 2.3.0 
python 2.7.14


### Dataset: 
MovieLens
ml-latest-small
ml-20m


This repository comprises of source codes of implementations of several Data Mining algorithms, in Python, 
and in Scala, leveraging parallel computing feature of PySpark library. Spark library helped to accomplish in-main memory, parallel and faster data processing in some of these algorithms:
 
### - Finding Frequent Itemsets (Basket analysis)
1. A-priori
2. SON - Which internally executes A-priori in one of its passes

### - Movie Recommendation Engine
1. User Based Collaborative Filtering - Using Pearson's coefficient. Also implemented content based item recommendation for cold start scenarios
2. Item Based Collaborative Filtering
3. Model Based Collaborative Filtering - using MlLib library
4. Comparison of Item Based recommendation using Pearson's and Jaccard Similarity

### - Social Network Girvan-Newman Community Detection Algorithm
1. Betweenness - Credit Calculation for edges and nodes
2. Community-Detection - Use Betweenness (1), to eliminate edges with highest betweenness, and detect
the communities with the highest modularity. (i.e. Higher within community edge density and lower inter-community density) 

