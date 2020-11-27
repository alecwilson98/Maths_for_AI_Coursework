# Maths for AI and ML Coursework

In this assignment, you will build linear regression models to predict admission likelihood of a Masters Programs candidate based on some independent variables such as GRE and TOEFL scores, University Rating, Undergraduate GPA, etc. The dataset is available at this Kaggle data repository  (Links to an external site.)and the paper describing the data  is (Acharya et al. 2019)

Requirements:

You are required to build your linear regression models from scratch, and you are not allowed to use any off-the-shelf linear regression source code or library. Specifically, you are required to implement the following gradient descent optimisation algorithms:

* Standard Gradient Descent 
* Stochastic Gradient Descent 
* Mini-batch Gradient Descent 

to minimise the Sum Squared Error (SSE). The aim is to obtain accurate predictive performance on the test set of 100 observations in the data. You can use any programming language, although Python is recommended. Please submit your source code with a detailed readme file to explain how to compile and run your code. You must also submit a report with the following:

* A detailed description of the linear regression model, Linear Least Squares and the three gradient descent algorithms
* Prediction performance of the linear regression models learned by the three gradient descent optimisation algorithms on the test set of 100 observations. 
* You should report the predictive performance using the following two metrics: R Square, Root Mean Square Error(RMSE)
* Analysis and discussion of the prediction performance of the three linear regression models and how those hyperparameters affect the performance.  

Marking criteria:

* Correct implementation of the linear regression models and the three gradient descent optimisation algorithms [40%]
* A detailed report that satisfies the requirements listed above [50%] 
* Data exploration, e.g., visualisation and in-depth analysis for selecting the optimisation algorithms and hyperparameters to improve the prediction performance [10%]

Reference:

Acharya, M. S., A. Armaan, and A. S. Antony. 2019. “A Comparison of Regression Models for Prediction of Graduate Admissions.” In 2019 International Conference on Computational Intelligence in Data Science (ICCIDS), 1–5.