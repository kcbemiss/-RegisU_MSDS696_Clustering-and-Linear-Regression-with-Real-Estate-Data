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
    6) 


File No |           File Name         |   Rows   | Columns |        Data Types         |     Level of Granularity      |    Unique Identifiers   | Details
--------|-----------------------------|----------|---------|---------------------------|-------------------------------|-------------------------|---------
1       | Property_Improvement.txt    | 133,865  |   35    | 'object','float64','int64 | 1 row per property building   | Account_No, Building_ID |  - contains multiple types of property (residential, commercial, land, etc..) and built types (homes, outbuildings, etc...).- 1 property can have multiple buildings
2       | Property_Location.txt       | 
3       | Property_Ownership.txt      |
4       | Property_Subdivision.txt    |
5       | Property_Sales.txt          |
6       | Property_Filing.txt         |
7       | Property_Values.txt         |






## Reference

Douglas County Assessor Office. (2021). Douglas County Assessors Database. [Data file]. Retrieved from https://www.douglas.co.us/assessor/data-downloads/
