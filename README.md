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
To know more about each feature, please refer: https://physionet.org/content/face-oral-temp-data/1.0.0/ and https://www.mdpi.com/1424-8220/22/1/215

In the training dataset there were total 710 data points. During training, I used KFold cross validation with folds as 5. So, at every fold 142 data points were validation data points and the remaining 568 data points were for training. For every model, I used cross validation process with one single for loop inside which I calculated Mean squared error (MSE), root mean squared error, and mean absolute error for each fold and took the mean as the final output. The decision of the best model was made on the lowest mean of MSE received from the validation sets for each fold. The training data was used during preprocessing where I took the mean of the 4 rounds and converted them into respective single columns. Also, one hot encoding was applied to the categorical data in the dataset separately and then concatenated into single X train. Normalization, Principal Component Analysis and feature selection was applied on this preprocessed training data. This training data was then used to train the models and select the best model based on lowest MSE. The final model selected was applied to the test data. This time the training data were entire 710 data points and the testing data were 300 data points. Test time was used only once for the final prediction on my selected model of Support vector regressor.

## Preprocessing

The dataset given had 4 rounds of temperature reads from different location for each patient. First, we filled the empty NaN values of each column by their mean. Then to convert these columns in 4 rounds, I took the mean value of same column in each round, giving me 27 columns where each row is the mean value of the 4 rounds of that patient. For the categorical data, we one hot encoded the ethnicity, gender columns, and for the range based data of Age, we took the median value for the given age range for each row. At the end, we concatenated all the temperature columns, one hot encoded columns and the age column, giving me the final training dataset. The dataset was normalized using MinMaxScaler because we the one hot encoded values are either 0 or 1, so the model will perform well and would not bias towards a specific feature.

## Feature engineering

We implemented a Forward Sequential feature selection method to get the first 15 best features that gave the lowest mean squared error for our training data. To implement this algorithm, we used the greedy approach, wherein we ran an outside loop ranging from 1 to total number of best features required and created an empty set variable to store all the best features one by one. For each iteration, one more for loop was used in which each feature was picked and trained on Support vector regressor and tested based on mean squared error (MSE) on validation set. The one with the lowest MSE was picked as the best feature and appended to the set of best features. Now, this best feature is appended with the other features to get the best MSE on validation together and second best feature. This process goes till we get total number of features required. In this process, categorical data was not included because during experimentation, we were getting some ethnicities with best features, if chosen, that would have biased our models towards those ethnicities. The final set of best features were aveAllL13 mean, T offset mean, T FH Max mean, T Max mean, T OR Max mean, T RC mean, T atm, Max1R13 mean, Max1L13 mean, T FHLC mean, T LC Dry mean, canthi4Max mean, T FHC Max mean, Distance, T FHBC mean.

## Feature dimensionality adjustment

We applied principal component analysis to our 15 best features that we got from sequential feature selection. After multiple runs for different values of principal components, we took principal component as 5 keeping K-fold cross-validation in mind during training as well. PCA significantly reduced the training time of multiple models compared to running the entire X train with all the features together.

## Training, Classification or Regression, and Model Selection

After the data preprocessing, feature engineering and dimensionality reduction, our data was ready to be trained on different models. Chosen models were the trivial system, baseline systems 1 and 2 (Linear regression and 1 Nearest Neighbor), Support Vector regressor with linear and RBF kernels, Polynomial Regression with linear regression model, Decision Trees and Radial Basis Function Network. Most of the models are trained on K-Fold cross validation with folds = 5. Few are trained on 10-fold and 20-fold cross validation.

## Final System

The lowest mean squared error on validation set was for Support Vector Regressor, so we chose that as our final system to test on the test set.

# Table: Performance Metrics for Final System (SVR with RBF) with Normalization only

|   | MSE | RMSE | MAE |
| ------------- | ------------- | ------------- | ------------- |
| Training Data  | 0.0133  | 0.1153 | 0.0948 |
| Testing Data | 0.2232  | 0.4725 | 0.3205 |

   
   
# Table: Performance Metrics for Final System (SVR with RBF) with Normalization, PCA, and Feature Selection

|   | MSE | RMSE | MAE |
| ------------- | ------------- | ------------- | ------------- |
| Training Data  | 0.0579  | 0.2407 | 0.1831 |
| Testing Data | 0.0982  | 0.3134 | 0.2342 |

All of our models are performing better than the trivial system model in terms of MSE, RMSE
and MAE on training as well as validation data.
