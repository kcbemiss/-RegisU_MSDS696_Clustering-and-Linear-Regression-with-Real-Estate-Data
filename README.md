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




## Reference

Douglas County Assessor Office. (2021). Douglas County Assessors Database. [Data file]. Retrieved from https://www.douglas.co.us/assessor/data-downloads/
