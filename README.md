# Prediction-of-Car-Prices-using-ML
Prediction of Prices of Cars using  Machine Learning with the help of Linear as well as Lasso Regression for better comparison.

For better understanding of Regression in ML, we have taken the car price prediction as a result. Here ,we are not classifying data types rather we are using a regression analysis on the prices. Before starting our project, lets recall some of the regression algorithms.

Linear Regression: It is a type of linear model where we can make predictions for numerical variables such as sales,                   salary, prices, etc. Linear regression algorithm shows a linear relationship between a dependent                     (y)and one or more independent (x) variables, hence called as linear regression. Linear                               Regression can be applied if there is any direct correlation with the values i.e directly                             proportional.

Lasso Regression:  Lasso Regression is a popular type of regularized linear regression that includes an L1 penalty.                      This has the effect of shrinking the coefficients for those input variables that do not contribute                    much to the prediction task. This penalty allows some coefficient values to go to the value of                        zero, allowing input variables to be effectively removed from the model, providing a type of                          automatic feature selection.

Lasso Regression is an extension of linear regression that adds a regularization penalty to the loss function during training.

Libraries used: numpy, pandas, matplotlib.pyplot, sklearn.model_selection, sklearn.linear_model, metrics, seaborn

Summary of application done using Python.

First , we take the required dataset of car prices and by using pandas , we store it in a class. Then we analyse the data(class_name.info()), take its summary(class_name.head()) and check if there is any null value(class_name.isnull().sum()).

As ML can only detect and take numerical values, we transform the categorical data having strings into numerical input by encoding the strings after checking the distribution of categorical data . Taking X and Y as the target variables, we keep all the required input values in X and output values in Y.

Now,using a sub-library of sklearn.model_selection( i.e. train_test_split), we split the data into training data and testing data and in the ratio of 1/100(test_size = 0.1)( 90% data will be from training data and the other 10% in testing data). We split the target variables into two halves and the four variables will take the input from original target variables plus the test size of the dataset and its random state number.

We now, fit the variables using a sub-library of sklearn.linear_model i.e. LinearRegression and Lasso.After fitting the data values in the target variables( both training and testing variables of both X and Y), we then predict the result, present the error(the error value must be greater than 0.8 for a correct analysis of prediction and we have applied r squared analysis for error) and by plotting them(scatter plot) , we can analyse the outcomes between actual prices and prediction prices. We can even use Lasso Regression and XGBoost for suitable comparison.
