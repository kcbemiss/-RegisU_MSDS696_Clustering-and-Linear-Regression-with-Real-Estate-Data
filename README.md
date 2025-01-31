# Regis Data Science Practicum II (MSDS696): Clustering and Linear Regression with Real Estate Data

![ClusterMap](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/SFR_Cluster_Map1.jpg)

## Abstract

Finding a home with the features you want can be challenging, especially with a market that has low volume. My family is in the process of looking for a new home in an area we are not familiar with, and we could use some help narrowing down the options!  The goal of the project was to use publicly available assessor's data and data from an online real estate site to identify properties that met specific home feature criteria and predict an actual assessed home value.

The data used for this project came from the Douglas County Assessor Office (Douglas County Assessor Office, 2021).  Seven (7) data files were downloaded from the assessors website as text files and contained both categorical and continuous data. Using unsupervised learning this project created clusters of properties with geocoded locations to identify properties and their locations with similarities over a large number of features.  PCA was used for feature reduction, and unsupervised clustering was performed using k-means, hierarchical agglomerative clustering (HCA) and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) using scikit-learn libraries.  

The project is broken into 4 different Juypter Notebooks:
* Data Collection, Cleaning and EDA (Jupyter Notebook: MSDS696_Practicum2_1b_AssessorsData_PCA_Clustering_BemissKimberly)
* Clustering 1 - Records with Built Year >= 2000 (Jupyter Notebook:  MSDS696_Practicum2_1c_AssessorsData_PCA_Clustering_BemissKimberly)
* Clustering 2 - Decreased features, Built Year >= 2000 & Garage Size >= 690 SF (Jupyter Notebook: MSDS696_Practicum2_1d_AssessorsData_PCA_Clustering_BemissKimberly)
* Geocoding and Regression- Deacreased features, Built Year >= 2000 & Garage Size >= 690 SF (Jupyter Notebook: MSDS696_Practicum2_1e_AssessorsData_PCA_Clustering_BemissKimberly)

A 5th notebook (MSDS696_Practicum2_1a_AssessorsData_PCA_Clustering_BemissKimberly) contains code for web scraping Real Estate website Realtor.com.  Due to time constraints I was not able to complete this piece of the project.  

The conclusion of the project is publicly available assessor's data can be clustered into groups that are defined by specific features and value ranges.  Multiple Linear Regression in this project was able to somewhat predict an assessors "actual value" for a property, but not at a level that I would use for any real life application.  While I was able to identify two clusters of properties that met my specific criteria, unsupervised clustering didn't perform as well as I had hoped.  Another clustering method, perhaps supervised, might have better performance and identify clusters that are better defined and have less overlap between points.  

## Data Source

The Douglas County Assessor Office provides text files from it's database for public consumption.  There are 7 text files without headers available.  The header information is listed with each file and must be copied manually from the text on the website.

###### 7 Data Files (Built Year 1825 to 2021):

    1) Property Improvements
    2) Property Location (address)
    3) Property Ownership
    4) Property Subdivision
    5) Property Filing
    6) Property Sales
    7) Property Values

## Data Preparation
The data preparation code can be found in the Jupyter Notebook file:  MSDS696_Practicum2_1b_AssessorsData_Collection_EDA_Cleaning_BemissKimberly.ipynb

The steps that were taken to prepare the data for the machine learning tasks were:

    1) read the file into a pandas dataframe and set the headers.
    2) review and describe the features of the dataframe.
    3) review records for duplicates.
    4) identify the level of granularity for each row of the dataframe
    5) review the unique values in each column
    6) identify the null/nan values in each column
    7) filter the data to the appropriate rows and columns


File No |           File Name         |   Rows   | Columns |                  Data Types                 |       Level of Granularity        |    Unique Identifiers     | 
--------|-----------------------------|----------|---------|---------------------------------------------|-----------------------------------|---------------------------|
1       | Property_Improvement.txt    | 133,865  |   35    | 'object','float64','int64','datetime64[ns]  | 1 row per property building       | Account_No, Building_ID   |  
2       | Property_Location.txt       | 159,058  |   21    | 'object','float64','int64'                  | 1 row per account/address         | Account_No, address fields|
3       | Property_Ownership.txt      | 156,734  |    7    | 'object'                                    | 1 row per account                 | Account_No                |
4       | Property_Subdivision.txt    | 152,280  |    6    | 'object','int64'                            | 1 row per account                 | Account_No                | 
5       | Property_Sales.txt          | 587,643  |    9    | 'object','float64'                          | 1 row per account/sale/recording  | Account_No, Recording_No  |
6       | Property_Filing.txt         |   4,871  |    4    | 'object'                                    | 1 row per filing record #         | Sub_Filing_Recording_No   |
7       | Property_Values.txt         | 281,735  |    9    | 'object','float64'                          | 1 row per account code/asses value| Account_No, Valuation_Class_Code, Assessed_Value

##### File Details:

(1) Property_Improvement:  
* contains multiple types of property (residential, commercial, land, etc..) and built types (homes, outbuildings, etc...).
* a property can have multiple buildings (multiple rows per property)
* filtered to only single family residential properties.
* final cleaned dataframe (df_PI_sfres_det):
        rows:102,136
        columns: 32

(2) Property_Location:
* contains detailed location information (including address).
* multiple rows per account (property)
* create a new field for Address (combine 6 fields into 1)
* filtered to only the residential property addresses
* final cleaned dataframe (df_loc_res):
        rows: 116,131
        columns: 22

(3) Property_Ownership:
* contains 1 row per account with owner's name and address.
* final dataframe (df_own) - no changes or manipulations.
        rows: 156,734
        columns: 7

(4) Property_Subdivision:
* contains the subdivision information for each account
* final dataframe (df_sub) - no changes or manipulations.
        rows: 152,280
        columns: 6

(5) Property_Sales:
* contains sale information for each property
* pivot and summarize each sale to create 1 row per account - selecting min/max sale price, min/max sale date and the associate prices (min/max).
* final dataframe (df_sale_det):
        rows: 139,197
        columns: 10

(6) Property_Filing:
* contains filing (subdivision/amendments) information for each property
* final dataframe (df_file) - no changes or manipulations:
        rows: 4,871
        columns: 4

(7) Property_Values:
* contains sale information for each property, 1 row per account/value code/assessed value.
* filter to only Valuation_Type_Code = "I" (Improvement), Account Type of Residential, Description Single Family Residential - Improvements
* final dataframe (df_val_res):
        rows: 102,016
        columns: 9


#### Prepared Data Output:

The prepared data is a pandas data frame that merges the seven data files into a single dataframe - **df_properties**.

Data Frame Output:  Pandas Dataframe (df_properties)
Info      | Description
--------- | ------------
Rows:     |  101,867
Columns:  |  83
Unique By:|  Account_No, Building_ID


## Exploratory Data Analysis (EDA)

##### File Details:
The details of the transformations for NaN values, Encoding and Removal of fields can be found in the file: MSDS696_Practicum2_Bemiss_EDA_Transformations.txt https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/MSDS696_Practicum2_Bemiss_EDA_Transformations.txt

##### Nulls and NaN Values: (see file for details)

There were a large number of Nan values for some of the columns in the data set. Some of these values could be updated to a value that represented a lack of information. Others were imputed using KNNImputer.  The end result was a dataframe with no null or NaN values. 

##### Encode Data: (see file for details)

For some of the data that was Categorical, it was encoded so that the data could be used in the unsupervised machine learning methods.  Using One Hot Encoding, Ordinal Encoding, and Label Encoding, the categorical data was transformed to numeric.  

##### Remove Fields: (see file for details)

Some of the data columns were removed (dropped) as the decision was made that they did not offer any information for our problem.


#### EDA Question 1 - How many Accounts have more than one building on the property?

193 Properties (accounts) have more than one building on the property.

![MultiBuilding](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/eda_bar_multibuilding.png)

#### EDA Question 2 - How many New Property Improvements (buildings) and Remodels were done each year?

The number of new buildings in Douglas County started to grow around 1970 and peaked in the Year 2000 with over 5220 new property improvements.
![BuildingsPerYear](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/eda_line_buildyear.png)

#### EDA Question 3 - What are the top Cities, Subdivisions and Build Types for Single Family Residential Homes in Douglas County?

Top Cities for properties:  Parker, Highlands Ranch, Castle Rock

Top Subdivisions for properties:  Highlands Ranch, The Meadows, Stonegate

Top Build Types for properties:  2 Story, 1 Story Ranch, Bi-Level
![MultipleBarCharts](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/eda_bar_multibar.png)


#### Dataframes for Analysis:

At the end of the Data Processing and Exploratory Data Analysis, there are 5 dataframes that we will use in the next steps of the project.

ID|      Dataframe                    |     Rows     | Columns |        Description                |
 -|-----------------------------------|--------------|---------|--------------------------------   |
1 |df_properties_onehot_imputed       |    101,817   |   788   |  One Hot Encoded                  |
2 |df_properties_onehot_imputed_stan  |    101,817   |   788   |  One Hot Encoded - Standardized   |
3 |df_prop_analy_prim_imputed         |    101,817   |    25   |  Limited Features                 |
4 |df_prop_analy_prim_imputed_stan    |    101,817   |    25   |  Limited Features - Standardized  |
5 |df_prop_analy_loc_feat             |    101,817   |     4   |  Location Data                    |


## Principal Component Analysis (PCA)

In this project PCA is used as an Unsupervised dimensionality reduction technique.  This method allows clustering of data based on the correlation between features.  The new features (or components) created are based on the original features.  Their importance in the dataset is given by the eigenvalues.
These principal components are based on our original features and their importance in terms of explaining the variability in the dataset is given by the explained variation percentage.

PCA was performed on 7 dataframes (one hot and primary).  This project used clustering with 6 of the dataframes.  A decision was made to continue with the 3rd dataframe of standardized data for the first Clustering attempt.  While the K-means clustering was successful, the HCA clustering was not able to complete on my system due to a lack of memory.  It was necessary to decrease the amount of data I was analyzing.  

DF # | Dataframe                         |   PCA Data       |    PCA DF   | # of Components | % Variability  | Details                       |
-----|-----------------------------------|------------------|-------------|-----------------|----------------|-------------------------------|
1    | df_properties_onehot_imputed_stan |  Cluster_df      | PCA_OH_ST   |   508           | 85%            | Original                      |
2    | df_properties_onehot_imputed      |  Cluster_df2     | PCA_OH      |     6           | 99%            | Original                      |
3    | df_prop_analy_prim_imputed_stan * |  Cluster_df3     | PCA_PR_ST   |    12           | 85%            | Original                      |
4    | df_prop_analy_prim_imputed        |  Cluster_df4     | PCA_PR      |     6           | 99%            | Original                      |

![EVP1](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_ExplainedVariance.JPG)


One of the features in a home that my family is looking for is the Built Year being after the year 2000.  I filtered the data to properties with a built year>= 2000 and standardized the data before performing PCA (PCA_PR_2000).  The 6th and 7th dataframes were additional runs of the clustering algorithm to determine how a reduction in the features and types of features used would affect the clusters.  I did note that the standardized data produced more components than the non-standardized data.   

DF # | Dataframe                          |   PCA Data       |    PCA DF   | # of Components | % Variability  | Details                       |
-----|------------------------------------|------------------|-------------|-----------------|----------------|-------------------------------|
5    | df_prop_analy_prim_imputed *       |  Cluster_df5     | PCA_PR_2000 |    11           | 85%            | Original Filtered - 2000      |
6    | df_prop_analy_prim_imputed *       |  Cluster_df      | PCA_PR      |     4           | 87%            | Features Filtered (8) - 2000  |
7    | df_prop_analy_prim_imputed *       |  Cluster_df      | PCA_PR      |     3           | 88%            | Features Filtered (6) - 2000  |

( * - dataframe with clustering techniques performed).

### PCA Analysis for Single Family Residential Properties - Built in or after 2000
The graph below is the Explained Variance Plot for the Primary Features data filtered to the build year 2000 or greater (PCA_PR_2000).  This plot shows the optimal number of components, where the % variability is greater than (>) 85%.

![PCA_Explained Variance_Plots](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_PR_2000_ExplainedVariance.png)

##### Pair Plot for PCA Components: Primary (PR) Data
The pairs plot below shows the 11 components graphed against each other.  There are no readily noticeable clusters, but there are quite a few linear relationships that were noted.

![PCA_Pair_Plots](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_Pair_PR_2000_ST_11.png)

##### Heat Maps:  PCA Components to Primary (PR) Data features

The next visualizitation was a HEAT map to view the components against the features, and to identify which features had the most impact on each of the PCA Components. Below are listed the features that had the most impact on the PCA components. 
* Built_As_Code
* No_of_Fireplaces
* Total_Garage_SF*
* Total_Porch_SF
* Total_Finished_Basement_SF*
* Total_Unfinished_Basement_SF
* Built_as_SF*
* No_of_Story
* No_of_Bedrooms*
* No_of_Bathrooms*
* Built_Year
* Total_Net_Acres
* Actual_Value*
* Assessed_Value
* No_of_Sales
* Quality_ord*
* Condition_ord*

**PCA Components 1 through 5 to Primary (PR_2000) Data features**

![PCA_HeatMaps_Plots](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_Heat_PR_2000_ST_12_1.png)

**PCA Components 6 through 11 to Primary (PR_2000) Data features**

![PCA_HeatMaps_Plots](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_Heat_PR_2000_ST_12_2.png)

**PCA Components 1 through 5 to Primary (PR_2000) Data features**

![PCA_HeatMaps_Plots](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_Heat_PR_2000_ST_12_3.png)

**PCA Components 6 through 11 to Primary (PR_2000) Data features**

![PCA_HeatMaps_Plots](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_Heat_PR_2000_ST_12_4.png)


## Unsupervised Clustering
 
The clustering analysis is performed on the filtered data for the built year 2000, as presented above in the PCA analysis. Three different clustering methods were performed and compared using three different performance metrics to identify the best method.

##### Clustering Methods:
* K-Means 
* HCA - hierarchical
* DBSCAN

##### Performance Evaluation Methods

***The Silhouette Coefficient***

This method can be used for clustering without ground truth labels. A higher Silhouette Coefficient score identifies a model with better defined clusters.

    - Close to -1:incorrect clustering
    - Close to 0: overlapping clusters
    - Close to 1:highly dense clustering

***Calinski-Harabasz Index***

This method can be used for clustering without ground truth labels. A higher Calinski-Harabasz score identifies a model with better defined clusters.

***Davies-Bouldin Index***

This method can be used for clustering without ground truth labels. A lower Davies-Bouldin index identifies a model with better separation between the clusters

### K-Means Clustering

The objective of K-means clustering is to group similar data points together to find underlying patterns in the data.  Using the sklearn library for K-Means, I first identify the "n" or number of clusters using the Elbow Method, and then evaluate the performance of the algorithm.

#### Determine the "n" # of clusters (Elbow Method and Kneed Library)
Using the Elbow method of identifying "n", and the kneed library in python, I choose an "n" # of clusters of 8 for the K-Means algorithm.

![KMeans_Elbow1](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/KMeans_Elbow_Clustering_PR2000_1.png)

#### K-Means Clustering (sklearn)

The K-Means algorithm with n=8 ran with 32 iterations.  The results of the algorithm (8 clusters) were plotted by components.

##### Visualize the clusters

The clusters are not separated or delineated, and overlap greatly.  In the first and second visualization you can see the individual groupings in the large "blob" of data points.  There is some delineation of the points into clusters, but you can also see the overlap between the clusters. A 3 dimensional graph could have given us a visual of the clusters to take the extra dimensions into account.  

![Clusters1](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/KMeans_Clustering1_PR2000_Comp1_Comp2.png)

![Clusters2](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/KMeans_Clustering1_PR2000_Comp1_Comp8.png)

#### Visualize the clusters individually

Looking at the clusters individually helped to see where they were unique and where they overlapped.  Clusters 5, 6 and 7 are very dispersed and smaller than the rest of the clusters.  Clusters 1,2,3,4 and 5 are more dense and overlap with other clusters at the edges.

![Clusters3](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PR_2000_Indv_Clusters_2.jpg)

#### Performance Evaluation

The K-Means algorithm had a Silhouette Coefficient performance score of 0.204 is closer to 0 than to 1.  Values of 0 mean clusters are indifferent (the distance between clusters is not significant). Values of 1 mean clusters are well distinguished and the distance is significant.

The Calinski-Harabasz performance score is 7357.44.  This method is better used to compare between clusters or methods, and the higher the score the better.

The Davies-Bouldin performance score is 1.1277, and the lower the value the better the clustering.  Like the calinski-harabasz, a comparison to clusters or methods will tell us more.

![KmeansPerf1](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PerformanceEval2_PR2000.JPG)

### Hierarchical Agglomerative Clustering (HCA)

This method of clustering iterates through data points, merging data points with clusters until one cluster is formed.  Using a dendrogram (linkage, ward method) the optimal number of clusters is identified.  Using sklearn's Agglomerative Clustering algorithm with the identified number of clusters, 4 clusters are produced using ward method and euclidean affinity.

#### Dendrogram - identify 4 clusters

The dendrogram clearly shows 4 clusters - identified in color but also by the height of the branches of the tree.

![HACDend](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/HAC_n_Hist_Pr2000.png)


#### Visualize Clusters

The clusters are not separated or delineated and overlap greatly, similar to the K-Means Clustering method.  In the first visualization you can see the 4 individual groupings in the large "blob" of data points.  There is some delineation of the points into clusters (especially cluster 4), but you can also see the overlap between the clusters. The second visualization, the truncated dendrogram of the 4 clusters is another method to visualize clusters from HAC. 

![HACscat](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/HCA_Clustering2_Comp1_Comp2_11.png)


![HACTree](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/HCA_Dendrogram_11.png)

#### Performance Evaluation

The HAC algorithm had a Silhouette Coefficient performance score of 0.245982 is closer to 0 than to 1.  It is closer to 1 than K-Means showing the 4 clusters are better distinguished.

The Calinski-Harabasz performance score is 6964.999812.  This score is smaller than K-means, indicating the K-Means algorithm performed better.

The Davies-Bouldin performance score is 1.625696, and is larger than K-means.  This indicates the K-Means algorithm performed better.  

![HCAPerf1](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/HCA_PerfEval_PCA_PR_2000_1.JPG)


### DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a method of clustering that can create clusters of arbitrary shapes and varying densities.  It is also a robust method for data with outliers, and doesn't require a number of clusters specified beforehand.   It requires 2 parameters, epsilon and minPoints.

* Epsilon:  "the radius of the circle to be created around each data point to check the density." (analyticsvihya.com)
* minPoints: "the minimum number of data points required inside that circle for that data point to be classified as a Core point." (analyticsvihya.com)

#### 1st Attempt:   eps = 0.5 and min_samples = 25

The first attempt at DBSCAN resulted in 43 clusters.  min_samles was set at 25 (11 dimensions x2 + 3) using a recommended method from Analytics Vidhya.  This first attempt was not a great result.  

![DBSCAN1](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/DBSCAN_scatter_1.png)


Using KNN and a K-Distance Graph, the algorithm was tuned to the right Epsilon. An Epsilon value of 2 was identified from the graph.

![KNNGraph](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/K_DistanceGraph.png)


#### 2nd Attempt:   eps = 2 and min_samples = 25

The second attempt uses an epsilon value of 2 identified from the K-Distance graph.  The second attempt was much better, Identifying 8 clusters.  The 8 clusters are mostly dispersed and have few points, with only 2 having any density.

![DBSCAN2](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/DBSCAN_scatter_2.png)

![DBSCAN_all](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PR_2000_Indv_Clusters_DBSCAN.jpg)


### Choosing a Clustering Method

The performance metrics evaluate the comparative performance of models against each other, because we are doing unsupervised clustering and do not have a target variable.

 Silhouette:  Closer to 1 the better
 * Best Method:  DBSCAN_PCA 

 Calinski-Harabasz:  The larger the score the better
 * Best Method: K-Means_PCA

 Davies-Bouldin:  The lower the score the better
 * Best MEthod:  K-Means_PCA

The DBSCAN and the K-Means Clustering Algorithms both identified 8 Clusters. The K-Means Algorithm has the best performance score for 2 of the 3 methods used, so I am going to choose that method to continue my Analysis and to perform some further clustering analysis based on specific features.

![MethodCompare](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PerformanceEval_all_PR2000.JPG)



## Using K-Means Clustering - Additional Feature Configurations

Having identified K-Means as the clustering method that worked best with this dataset, I want to look specifically at identifying a cluster of homes that meet the criteria my family is looking for.  There are features that are important to us that also had a high impact on the PCA components.

List of features I am interested in with high impact on PCA Clusters.
* Total_Garage_SF
* Total_Finished_Basement_SF
* Built_as_SF
* No_of_Bedrooms
* No_of_Bathrooms
* Total_Net_Acres
* Quality_ord
* Condition_ord
* Actual_Value

![CorrMatrix](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/CoorelationMatrix.png)

The PCA for this smaller data set resulted in 9 components.
![ExpVar](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/small_pca_exvar.png)

Running each of the clustering methods (K-Means, Hierarchical, and DBSCAN) identified that K-Means was the best method for clustering with 6 clusters.

Silhouette:  Closer to 1 the better
* Best Method:  K-Means_PCA 

Calinski-Harabasz:  The larger the score the better
* Best Method: K-Means_PCA

Davies-Bouldin:  The lower the score the better
* Best MEthod:  K-Means_PCA

![ExpVar](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/small_performance_all.JPG)


![smkmeanscl](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/small_clusterplot.png)


## Finding Homes with Big Garages

One of the goals of the project was to use publicly available assessor's data and data from an online real estate site to identify properties that met specific home feature criteria. The clustering method (K-Means) has been identified with the specific home features.  One of the specific features we are looking for is homes with greater than (>) 690 square feet.  Using the methodology and K-Means algorithm that have been identified to this point, a data set of properties with garage square footage greater than or equal to 690 sqft was analyzed to identify clusters of properties using the set of features identified.  

Data:  Properties built 2000 and later with garage sq ft >= 690, standardized.
* Columns:  9
* Rows: 15815

List of features:
* Total_Garage_SF
* Total_Finished_Basement_SF
* Built_as_SF
* No_of_Bedrooms
* No_of_Bathrooms
* Total_Net_Acres
* Quality_ord
* Condition_ord
* Actual_Value

PCA of the data produced 9 components with explained variation 

Explained variation per principal component: 
* Component 1: 0.37075446 
* Component 2: 0.23364882
* Component 3: 0.11335468
* Component 4: 0.10306531
* Component 5: 0.06378425
* Component 6: 0.04751985
* Component 7: 0.02676451
* Component 8: 0.02291783
* Component 9: 0.01819028

![expvargrg](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/garage_expvar.png)

###### Heat map of how component are influenced by each feature

![grgheat](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/garagesf_heatmap.png)

##### K-Means Unsupervised Clustering

Using the elbow method - the "n" # of clusters was identified at 5.

![grgelbow](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/garagf_elbowmethod.png)

The K-Means algorithm (from sklearn) was applied with an n = 5 through 12 iterations. The clusters have dense centers with some dispersion around the edges and overlap each other.

![grgcluster](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/Garage_Clusters.jpg)

The Performance measures of the clustering algorithm were in line with the other clustering sessions (with slight improvement).

![grgperfeval](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/GaragePerfEval.JPG)

The individual plots of the clusters help to visualize where each cluster shares space (overlaps) and where they are individual (unique space).

![grgindivclust](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/garage_clusters_indiv.jpg)


## Geo-Coding and Mapping Property Clusters

To better visualize the clusters, a feature of location was added by geocoding the address of the property and using Tableau to map each property.

The guidance for this section of the project was found from towardsdatascience.com and shanelynn.ie.
* https://towardsdatascience.com/pythons-geocoding-convert-a-list-of-addresses-into-a-map-f522ef513fd6
* https://towardsdatascience.com/geocode-with-python-161ec1e62b89
* https://www.shanelynn.ie/batch-geocoding-in-python-with-google-geocoding-api/

### Data Prep:

Taking the data output from clustering the property data (garage SF >= 690), the location data is joined using the row index that was tracked through each step.

**DataFrame** (df_geo_start):  Rows (15815), Columns (34)

**GeoCoding**

An list of addresses (Loc_Full_Address, index) was created and formatted to send to the Google Maps GeoCoding API
* Base address: https://maps.googleapis.com/maps/api/geocode/json?
* AUTH_KEY = API key assigned by Google Maps

After testing a few addresses to make sure that the connection to the google API was successful and returned the data that was expected.
* Lat (latitude)
* Lng (longitude)
* formatted_address

A function was defined to combine each of the steps:
* using the library urllib, create a base URL to send to the API.
* using the requests library, request the geocode information from google.
* using the json library, extract the lat, lng and formatted address information from the returned information.

Using a for loop - the full list of 15,815 properties is passed to the google API and a list of the results is created.

A dataframe was created from the list of geocode information and compared to the original list.  A few (7) of the addresses did not return a geocode.  Using the index value, the missing index values were identified from the original list. The missing values were then passed to the google API and the geocode information was obtained for the missing records, which were added to the dataframe of geocoded information.

To create a map, Tableau was used.  The dataframe was written to a csv file, which was then used as the datasource for Tableau.

![sfr_map2](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/SFR_Cluster_Map2.jpg)


## Multi-Linear Regression
The last piece of the project, and one of our goals was to predict an actual home value using multi-linear regression.  Using sklearn and the statsmodel libraries, multi-linear regression was completed and an actual home value was predicted.

The data was plotted using a correlation plot and a pairs plot to determine which features to use in MLR.

![MLRcoorplot](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/CoorMatrix_MLR.png)

Selection of pairs:
![MLRscatters](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/MLR_scatters.png)


4 features were selected - and set as "X"
* Built_as_SF (red)
* Total_Garage_SF (blue)
* Total_Finished_Basement_SF (magenta)
* Quality_ord (light blue)

The target Y variable to be predicted is "Actual Value".

#### Multiple Linear Regression with SKLearn

The data was split into test and train datasets with a test size of 0.2.  Using the linear_model.LinearRegression() from sklearn, the model was fit using the training data set. 

#### prediction with sklearn - Test

Using the sklearn model to predict Actual Value resulted in the following results:

Measure | Train Data  |  Test Data      |  Full Data      | Measure Description                               |
--------|-------------|-----------------|-----------------|-------------------------------------------------- |
R2      | 0.686       |  0.72           | 0.70            | how well the model fits the dependent (y) variable |
RMSE    |             |  150949.72      | 164848.67       | absolute measure of goodness of fit               |
MSE     |             |  22785818757.88 |  27175083255.38 | absolute measure of goodness of fit               |

![PredvsAbs](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/Actual_vs_Predicted.png)

![PredvsAb_scatter](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/MLR_ScatterCompare.png)



## Conclusions and Findings
The goal of the project was to use publicly available assessor's data and data from an online real estate site to identify properties that met specific home feature criteria and predict an actual assessed home value.  The unsupervised clustering using K-Means from sklearn performed with an Silhouette Score of 0.33 and an Davies-Boudin of 1.08.  The Silhouette score was above 0 telling me that there were overlapping clusters that had some density to them.  I was pleased that the score was above 0 and not negative.  The clustering was able to identify properties by square footage (built and finished basement), bedrooms, bathrooms, and quality of build.  Using Multi-linear regression, the R2 value of 0.70 (for 4 features) for the full data set was lower than I had hoped and further analysis of features to identify those best for predicting an assessed value would be an additional step I would take.  

The conclusion of the project is publicly availabe assessor's data can be clustered into groups that are defined by specific features and value ranges.  Multiple Linear Regression in this project was able to somewhat predict an assessors "actual value" for a property, but not at a level that I would use for any real life application.  While I was able to identify two clusters of properties that met my specific criteria, unsupervised clustering didn't perform as well as I had hoped.  Another clustering method, perhaps supervised, might have better performance and identify clusters that are better defined and have less overlap between points. 

For our family - Cluster 2 and 4 are where we will be looking for homes - these homes have the 4 bedroom 3 bathroom or larger crieria that are on smaller lots.  

Tableau was used to create the maps for this project, and to create visualizations that help to describe each of the clusters.
https://public.tableau.com/profile/kimberly.bemiss#!/vizhome/ClusteringandLinearRegressionwithRealEstateData/AssessorsData_Story?publish=yes

#### Description of Clusters:

**Cluster 1 (First):  Smaller Homes with Bigger Finished Basements**
* Medium Acreage: These properties have a more medium Total Net Acres (Average of 2 Net Acres)
* Small Square Footage:  These properties have an Average Built as square footage under 2800 SF
* Large Finished Basement: This cluster has the largest Average Square Footage of finished basements.
* Bathrooms: This cluster has home with 3 or less Bathrooms
* Bedrooms: This cluster has homes with 4 or Less Bedrooms
* Quality: Medium to High (High Average Quality)

**Cluster 2 (Second): Medium sized Homes**
* Small Acreage:  These properties are under a half acre on average (Average < 0.5 Net Acres)
* Medium Square Footage:  These properties have an Average Built as square footage around 3000 SF, and Average Total SF of 3500.
* Small Finished Basement: This cluster has Average Square Footage of finished basements around 500 SF.
* Bathrooms:  This cluster has home with 2 to 5 Bathrooms
* Bedroom:  This cluster has bedrooms ranging from 1 to 6
* Quality: Medium to Low (Medium Average Quality)

**Cluster 3 (Third): Medium to Larger sized homes**
* Medium Acreage: These properties have a more medium Total Net Acres (Average of 1.3 Net Acres)
* Large Square Footage:  These properties have a total square footage > 5000 SF.
* Large Finished Basement: This cluster has Average Square Footage of finished basements over 1600 SF.
* Bathrooms:  This cluster has home with 3 or more Bathrooms
* Bedroom:  This cluster has bedrooms ranging from 1 to 6
* Quality: Medium to High (Highest Average Quality)

**Cluster 4 (Fourth): Medium sized Homes**
* Small Acreage:  These properties are under a half acre on average (Average < 0.5 Net Acres)
* Medium Square Footage:  These properties have an Average Built as square footage around 3000 SF, and Average Total SF of 3500.
* Small Finished Basement: This cluster has Average Square Footage of finished basements around 500 SF.
* Bathrooms:  This cluster has home with 2 to 6 Bathrooms
* Bedroom:  This cluster has bedrooms ranging from 1 to 6
* Quality: Medium to Low (Medium Average Quality)

**Cluster 5 (Fifth):  Large Acreage Homes**
* Large Acreage:  These properties have Much larger Total Net Acres (Average of 36 Net Acres).
* Large Square Footage:  These properties have an Average Total square footage > 5000 SF.
* Large Finished Basement: This cluster has Average Square Footage of finished basements over 1800 SF.
* Bathrooms:  This cluster has home with 2 to 7 Bathrooms
* Bedrooms: This cluster has homes with 4 or Less Bedrooms
* Quality: Medium to High (High Average Quality)

## Reference


data4Help. (2009, May 5). Clustering Real Estate Data - Becoming Human: Artificial Intelligence Magazine. Medium. https://becominghuman.ai/clustering-real-estate-data-594894e24484

data4Help. (2020, May 7). Introduction — End-to-End Machine Learning for Real Estate Price Prediction. Medium. https://data4help.medium.com/introduction-end-to-end-machine-learning-for-real-estate-price-prediction-556beb2d0475

Douglas County Assessor Office. (2021). Douglas County Assessors Database. [Data file]. Retrieved from https://www.douglas.co.us/assessor/data-downloads/

Lau, N. (2020, June 19). 5 Ways to Apply Data Science to Real Estate - Towards Data Science. Medium. https://towardsdatascience.com/5-ways-to-apply-data-science-to-real-estate-e18cdcd0c1a6

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support



## Sources

##### Python Libraries
    Pandas: https://pandas.pydata.org/docs/reference/index.html
    
    scikit-learn (sklearn): https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
 
##### Python Code and Guidance
   
    Unsupervised Clustering (K-means and Hierarchical):
         https://medium.datadriveninvestor.com/unsupervised-learning-with-python-k-means-and-hierarchical-clustering-f36ceeec919c 
         http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/117-hcpc-hierarchical-clustering-on-principal-components-essentials/#:~:text=The%20PCA%20step%20can%20be,as%20in%20gene%20expression%20data.
         https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a
         https://medium.com/pursuitnotes/k-means-clustering-model-in-6-steps-with-python-35b532cfa8ad

    DBSCAN: https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
 
    Performance Evaluation:  
        https://medium.com/@haataa/how-to-measure-clustering-performances-when-there-are-no-ground-truth-db027e9a871c
        https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c
        https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b
 
    Geocoding: 
        https://towardsdatascience.com/pythons-geocoding-convert-a-list-of-addresses-into-a-map-f522ef513fd6
        https://www.shanelynn.ie/batch-geocoding-in-python-with-google-geocoding-api/
        https://developers.google.com/maps/documentation/geocoding/usage-and-billing?hl=en_US
        
    Multiple Linear Regression:
        https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
        https://scikit-learn.org/0.18/auto_examples/plot_cv_predict.html
        https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
