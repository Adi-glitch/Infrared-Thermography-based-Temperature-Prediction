//Abstract

The dataset was a regression problem, where ’aveOralM’, that is the average oral temperature
measured in monitor mode was the target. Extensive preprocessing, feature selection, dimensionality
reduction in the form of PCA, and model selection were conducted. The machine learning methods
used were: A trivial and two baseline systems (Linear Regression and 1 Nearest Neighbor), Support
Vector Regression, Polynomial Regression, Random Forest Regression and RBF Network Methods
(RBF Sampler, KMeans Clustering for finding Basis Function centers and an ANN). Comparisons
were made on Validation Root Mean Squared Error (RMSE) and Mean Squared Error (MSE) values
over 5 fold (in most), 10 and 20 fold (in few) cross-validated data. The best results were obtained on
the Support Vector Regressor with RBF Kernel, on normalized data along with PCA and Feature
Selection.
