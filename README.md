# Regis Data Science Practicum II (MSDS696): Clustering and Linear Regression with Real Estate Data

![ClusterMap](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/SFR_Cluster_Map1.jpg)

## Abstract

Finding a homes with the features you want can be challenging, especially with a market that has low volume. My family is in the process of looking for a new home in an area we are not familiar with, and we could use some help narrowing down the options!  The goal of the project was to use publicly available assessor's data and data from an online real estate site to identify properties that met specific home feature criteria and predict an actual home value.

The data used for this project came from the Douglas County Assessor Office (Douglas County Assessor Office, 2021).  Seven (7) data files were downloaded from the assessors website as text files and contained both categorical and continuous data. Using unsupervised learning this project created clusters of properties with geocoded locations to identify properties and their locations with similarities over a large number of features.  PCA was used for feature reduction, and unsupervised clustering was performed using k-means, hierarchical agglomerative clustering (HCA) and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) using scikit-learn libraries.  

* Clustering 1 - Primary DS/Primary 2000 - MSDS696_Practicum2_1c_AssessorsData_PCA_Clustering_BemissKimberly
* Clustering 2 - Primary 2000 w/o dates
* Clustering 3 - Primary filtered DS
* Clustering 4 - Primary filtered for Garage Size

The conclusion of the project is .....................

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
* contains detailed location information (incuding address).
* multiple rows per account (property)
* create a new field for Address (combine 6 fields into 1)
* filered to only the residential property addresses
* final cleaned dataframe (df_loc_res):
        rows: 116,131
        columns: 22

(3) Property_Ownership:
* contains 1 row per account with owners name and address.
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
* contains filing (subdivision/ammendments) information for each property
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

Data Frame Output:  Pandas Data Frame (df_properties)
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

Some of the data columns were removed (droped) as the decsion was made that they did not offer any information for our problem.


#### EDA Question 1 - How many Accounts have more than one building on the property?

193 Properties (accounts) have more than one building on the property.

![MultiBuilding](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/eda_bar_multibuilding.png)

#### EDA Question 2 - How many New Property Improvements (buildings) and Remodels were done each year?

The number of new bulidings in Douglas County started to grow around 1970 and peaked in the Year 2000 with over 5220 new property improvements.
![BuildingsPerYear](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/eda_line_buildyear.png)

#### EDA Question 3 - What are the top Cities, Subdivisions and Build Types for Single Family Residential Homes in Douglas County?

Top Cities for properties:  Parker, Highlands Ranch, Castle Rock

Top Subdivisions for properties:  Highlands Ranch, The Meadows, Stonegate

Top Build Types for properties:  2 Story, 1 Story Ranch, Bi-Level
![MultipleBarCharts](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/eda_bar_multibar.png)


#### Dataframes for Analysis:

At the end of the Data Processsing and Exploratory Data Analysis, there are 5 dataframes that we will use in the next steps of the project.

ID|      Dataframe                    |     Rows     | Columns |        Description                |
 -|-----------------------------------|--------------|---------|--------------------------------   |
1 |df_properties_onehot_imputed       |    101,817   |   788   |  One Hot Encoded                  |
2 |df_properties_onehot_imputed_stan  |    101,817   |   788   |  One Hot Encoded - Standardized   |
3 |df_prop_analy_prim_imputed         |    101,817   |    25   |  Limited Features                 |
4 |df_prop_analy_prim_imputed_stan    |    101,817   |    25   |  Limited Features - Standardized  |
5 |df_prop_analy_loc_feat             |    101,817   |     4   |  Location Data                    |


## Principal Component Analysis (PCA)

In this project PCA is used as an Unsupervised dimensionality reduction technique.  This method allows clustering of data based on the coorelation between features.  The new features (or components) created are based on the original features.  Their importance in the dataset is given by the eigenvalues.
These principal components are based on our original features and their importance in terms of explaining the variability in the dataset is given by the explained variation percentage.

PCA was performed on 7 dataframes (onehot and primary).  This project used clustering with 6 of the dataframes.  A decsion was made to continue with the 3rd dataframe of standardized data for the first Clustering attempt.  While the K-means clustering was successful, the HCA clustering was not able to complete on my sytem due to a lack of memory.  It was necessary to decrease the amount of data I was analyzing.  

DF # | Dataframe                         |   PCA Data       |    PCA DF   | # of Components | % Variability  | Details                       |
-----|-----------------------------------|------------------|-------------|-----------------|----------------|-------------------------------|
1    | df_properties_onehot_imputed_stan |  Cluster_df      | PCA_OH_ST   |   508           | 85%            | Original                      |
2    | df_properties_onehot_imputed      |  Cluster_df2     | PCA_OH      |     6           | 99%            | Original                      |
3    | df_prop_analy_prim_imputed_stan * |  Cluster_df3     | PCA_PR_ST   |    12           | 85%            | Original                      |
4    | df_prop_analy_prim_imputed        |  Cluster_df4     | PCA_PR      |     6           | 99%            | Original                      |

![EVP1](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_ExplainedVariance.JPG)


One of the features in a home that my family is looking for is the Built Year being after the year 2000.  I filtered the data to properties with a built year>= 2000 and standardized the data before performing PCA (PCA_PR_2000).  The 6th and 7th dataframes were additional runs of the clustering algorithim to determine how a reduction in the features and types of features used would affect the clusters.  I did note that the standardized data produced more components than the non-standardized data.   

DF # | Dataframe                          |   PCA Data       |    PCA DF   | # of Components | % Variability  | Details                       |
-----|------------------------------------|------------------|-------------|-----------------|----------------|-------------------------------|
5    | df_prop_analy_prim_imputed *       |  Cluster_df5     | PCA_PR_2000 |    11           | 85%            | Original Filtered - 2000      |
6    | df_prop_analy_prim_imputed *       |  Cluster_df      | PCA_PR      |     4           | 87%            | Features Filtered (8) - 2000  |
7    | df_prop_analy_prim_imputed *       |  Cluster_df      | PCA_PR      |     3           | 88%            | Features Filtered (6) - 2000  |

( * - dataframe with clustering techinques performed).

The graph below is the Explained Variance Plot for the Primary Features data filtered to the build year 2000 or greater (PCA_PR_2000).  This plot shows the optimal number of components, where the % variability is greater than (>) 85%.

![PCA_Explained Variance_Plots](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_PR_2000_ExplainedVariance.png)


The pairs plot below shows the 11 components graphed against eachother.  There are no readily noticable clusters, but there are quite a few linear relationships that were noted.

###### Pair Plot for PCA Components: Primary (PR) Data
![PCA_Pair_Plots](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_Pair_PR_2000_ST_11.png)


###### Heat Maps:  PCA Components to Primary (PR) Data features

The next visaluzation was a HEAT map to view the components against the features, and to identify which features had the most impact on each of the PCA Components. Below are listed the features that had the most impact on the PCA components. 
* Built_As_Code
* No_of_Story
* No_of_Bedrooms*
* No_of_Bathrooms*
* Built_Year
* No_of_Fireplaces
* Total_Garage_SF*
* Total_Porch_SF
* Total_Finished_Basement_SF*
* Built_as_SF*
* Actual_Value*
* Assessed_Value
* No_of_Sales
* Total_Net_Acres
* Total_Unfinished_Basement_SF
* Quality_ord*
* Condition_ord*

**PCA Components 1 through 5 to Primary (PR) Data features**

![PCA_HeatMaps_Plots](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_Heat_PR_2000_ST_12_1.png)

**PCA Components 6 through 10 to Primary (PR) Data features**

![PCA_HeatMaps_Plots](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_Heat_PR_2000_ST_12_2.png)

**PCA Components 11 through 12 to Primary (PR) Data features**

![PCA_HeatMaps_Plots](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_Heat_PR_2000_ST_12_3.png)

![PCA_HeatMaps_Plots](https://github.com/kcbemiss/RegisU_MSDS696_ClusteringAndLinearRegressionWithRealEstateData/blob/main/Images/PCA_Heat_PR_2000_ST_12_4.png)

## Unsupervised Clustering
 
 Ran on the full dataset and found that my memory wouldn't handle the amount of data for HCA.  Filtered the dataset to the year 2000.

### K-Means Clustering

#### Determine the "n" # of clusters (Elbow Method and Kneed Library)

#### K-Means Clustering (sklearn)

#### Visualize the clusters

#### Visualize the clusters individually

#### Performance Evaluation

### Hierarchical Agglomarative Clustering (HCA)

### DBSCAN

### Choosing a Clustering Method


## Using K-Means Clustering - Additional Feature Configurations

#### Finding Homes with Big Garages


## Multi-Linear Regression


## Geo-Coding and Mapping Property Clusters


## Conclusions and Findings






## Reference

Douglas County Assessor Office. (2021). Douglas County Assessors Database. [Data file]. Retrieved from https://www.douglas.co.us/assessor/data-downloads/
