# A-Machine-Learning-Approach-towards-the-Prediction-and-Compression-of-WRF-Datasets

The collection, storage and retrieval of Weather Research and Forecasting(WRF) datasets is a barrier to the researchers and decision
makers for fast manipulation and analysis of weather data. It is also difficult to work with them in local computing machines with
limited memory and without internet access. A machine learning model capable of predicting important weather parameters on the
fly could minimize these challenges as well as eliminate the need of adding the predictable weather parameters in the datasets leading
to an efficient data compression approach. The objective is to produce accurate synthetic data that can simulate the overall geographic
process to be used ubiquitously by all stakeholders. 

In this project, we implemented 9 different regression models to predict Soil Moisture data, learning from a set of 14 million WRF data
samples all over the western Canada. A set of 32 datasets from 32 months over the years of 2013, 2014 and 2015 has been sampled
and 25 out of 36 potential features have been extracted into CSV files. The features were then normalized and fed into Principal
Component Analysis for dimensionality reduction which gave 2 components representing 99.5% of the total variance. The components
went through a special feature engineering process producing 3 approaches. 

Each approach reshaped the datasets in different ways where 2 of them used neighbourhood information while training.
We choose Bayesian Ridge, Partial Least Square and Neural Network algorithms to implement all 3 approaches creating 9
different models. The training datasets were used to Cross Validate each model using a 10-fold cross validation. The models were then
evaluated using test sets split in an 80-20 fashion that contained equal number of summer and winter datasets for generalization.
The R2 score for all models is ranged between 0.94 (94%) to 0.98 (98%), MAE is between 0.022 and 0.054 and MSE is between 0.0023
and 0.01- which were quite good in this problem domain. Finally, the results were post-processed for further visualization, analysis
and decision-making purposes.
