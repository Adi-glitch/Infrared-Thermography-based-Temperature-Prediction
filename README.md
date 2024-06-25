# Infrared Thermography based Temperature Prediction

## Abstract

The dataset was a regression problem, where ’aveOralM’, that is the average oral temperature measured in monitor mode was the target. Extensive preprocessing, feature selection, dimensionality reduction in the form of PCA, and model selection were conducted. The machine learning methods used were: A trivial and two baseline systems (Linear Regression and 1 Nearest Neighbor), Support Vector Regression, Polynomial Regression, Random Forest Regression and RBF Network Methods (RBF Sampler, KMeans Clustering for finding Basis Function centers and an ANN). Comparisons were made on Validation Root Mean Squared Error (RMSE) and Mean Squared Error (MSE) values over 5 fold (in most), 10 and 20 fold (in few) cross-validated data. The best results were obtained on the Support Vector Regressor with RBF Kernel, on normalized data along with PCA and Feature Selection.

## Problem Assessment and Goals

The Infrared Thermography Temperature Dataset contains temperatures read from various locations of infrared images about patients, with the addition of oral temperatures measured for each individual. The 33 features consist of gender, age, ethnicity, ambient temperature, humidity, distance, and other temperature readings from the thermal images. The goal is to predict the oral temperature using the environment information as well as the thermal image readings.

## Approach and Implementation

The given dataset had some empty values for few rounds for several patients. The preprocessing helped replacing the blank values with mean of the entire column and then to get one set of 27 features, we took mean over 4 rounds of patients data. This data was then combined with one hot encoded data on categorical values and individual columns to create the final dataset. Then we standardized the data with MinMaxScaler to range the data between 0 and 1 due to one hot
encoded values. For the ’With normalization case’ we did not apply PCA and feature selection and ran models of Trivial System, linear regression, nearest neigbhors, polynomial regression, support vector regressor, RBF neural network and random forest from sklearn and pytorch. Similar models were apply for ’With normalization, PCA and feature selection’ case as well. The feature selelction method was forward sequential feature selection. We used cross validation with 5 folds to train the data and get MSE, RMSE, and MAE. The final model was selected based on lowest MSE among the models which came to be for SVR. This model was then applied to testing data, for which we got the best performance.

## Dataset Usage

This dataset was taken from: https://archive.ics.uci.edu/dataset/925/infrared+thermography+temperature+dataset

In the training dataset there were total 710 data points. During training, I used KFold cross validation with folds as 5. So, at every fold 142 data points were validation data points and the remaining
568 data points were for training. For every model, I used cross validation process with one single
for loop inside which I calculated Mean squared error (MSE), root mean squared error, and mean
absolute error for each fold and took the mean as the final output. The decision of the best model
was made on the lowest mean of MSE received from the validation sets for each fold.
The training data was used during preprocessing where I took the mean of the 4 rounds and
converted them into respective single columns. Also, one hot encoding was applied to the categorical data in the dataset separately and then concatenated into single X train. Normalization,
Principal Component Analysis and feature selection was applied on this preprocessed training data.
This training data was then used to train the models and select the best model based on lowest MSE.
The final model selected was applied to the test data. This time the training data were entire
710 data points and the testing data were 300 data points. Test tim
