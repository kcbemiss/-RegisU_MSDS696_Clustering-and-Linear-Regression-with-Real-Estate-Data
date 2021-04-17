# Regis Data Science Practicum II (MSDS696): Clustering and Linear Regression with Real Estate Data
Practicum Project for Masters in Data Science Degree at Regis University.  Using County Assessors Data to apply Unsupervised Learning Clustering Algorithms and Linear Regression for value prediction.

## Abstract

Finding a homes with the features you want can be challenging, especially with a market that has low volume. My family is in the process of looking for a new home in an area we are not familiar with, and we could use some help narrowing down the options!  The goal of the project was to use publicly available assessor's data and data from an online real estate site to identify properties that met specific home feature criteria and predict an actual home value.

The data used for this project came from the Douglas County Assessor Office (Douglas County Assessor Office, 2021).  Seven (7) data files were downloaded from the assessors website as text files and contained both categorical and continuous data. Using unsupervised learning this project created clusters of properties with geocoded locations to identify properties and their locations with similarities over a large number of features.  PCA was used for feature reduction, and unsupervised clustering was performed using k-means, hierarchical agglomerative clustering (HCA) and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) using scikit-learn libraries.  

The conclusion of the project is .....................

PICTURE:   ![Tree_DT](https://github.com/kcbemiss/PredictingHospitalClaimDenials/blob/main/Images/Tree_dtc.svg)

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






## Reference

Douglas County Assessor Office. (2021). Douglas County Assessors Database. [Data file]. Retrieved from https://www.douglas.co.us/assessor/data-downloads/
