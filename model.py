import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Defines the Batch Gradient Descent linear regression class
class BGD:

    def __init__(self, iterations=1000, learning_rate=0.1):
        # Hyper-parameters
        self.iterations = iterations
        self.learning_rate = learning_rate

    def fit(self, X, Y):
        # Define structure
        self.m, self.n = X.shape
        self.thetas = np.random.rand(self.n, 1)
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            # Difference between predicted and actual values
            error = np.dot(X, self.thetas) - Y
            # Compute gradient
            grad = np.dot(X.T, error) / self.m
            # Update coefficients
            self.thetas = self.thetas - (self.learning_rate * grad)
        return self.thetas

    # Function used to predict dependent variable of test set data
    def predict(self, X):
        return np.dot(X, self.thetas)

    # Function which defines the evaluation metrics; RMSE, R-Squared
    def eval(self, X, Y):
        RMSE = np.sqrt(((self.predict(X) - Y) ** 2).mean())
        r2 = 1 - (np.sum((Y - self.predict(X)) ** 2)) / (np.sum((Y - Y.mean()) ** 2))

        print("RMSE :", RMSE)
        print("R-Squared: ", r2)

# Defines the Stochastic Gradient Descent linear regression class
class SGD:
    def __init__(self, epochs=1000, t0=5, t1=50):
        # Hyper-parameters
        self.epochs = epochs
        self.t0 = t0
        self.t1 = t1

    def learning_schedule(self, t):
        return self.t0 / (t + self.t1)

    def fit(self, X, Y):
        # Define Structure
        self.m, self.n = X.shape
        self.thetas = np.random.rand(self.n, 1)

        # Iterates over the number of epochs chosen
        for epoch in range(self.epochs):
            # Iterates over the number of rows; in each iteration a random row of the data is selected
            for i in range(self.m):
                # Defines the random row number for the current iteration
                r = np.random.randint(self.m)
                # Picks out the random row number defined by r from the data
                xi = X.iloc[r: r+1]
                yi = Y.values[r: r+1]
                # Difference between predicted and actual values
                error = np.dot(xi, self.thetas) - yi
                # Compute gradient
                grad = 2 * np.dot(xi.T, error)
                # Define Learning rate
                lr = self.learning_schedule(epoch * self.m + i)
                # Update coefficients
                self.thetas = self.thetas - (lr * grad)
        return self.thetas

    # Function used to predict dependent variable of test set data
    def predict(self, X):
        return np.dot(X, self.thetas)

    # Function which defines the evaluation metrics; RMSE, R-Squared
    def eval(self, X, Y):
        RMSE = np.sqrt(((self.predict(X) - Y) ** 2).mean())
        r2 = 1 - (np.sum((Y - self.predict(X)) ** 2)) / (np.sum((Y - Y.mean()) ** 2))

        print("RMSE :", RMSE)
        print("R-Squared: ", r2)

# Defines the Mini-Batch Gradient Descent linear regression class
class MBGD:

    def __init__(self, epochs=1000, mini_batch_size=200, learning_rate=0.2):
        # Hyper-parameters
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate

    def batch_size(self, X, Y, b):
        # Define structure
        self.X = X
        self.Y = Y
        # Define batches
        new_size = b + self.mini_batch_size
        X_new = X.iloc[b:new_size]
        Y_new = Y.values[b:new_size]
        return X_new, Y_new

    def fit(self, X, Y):
        # Initialising weights
        self.m, self.n = X.shape
        self.thetas = np.random.rand(self.n, 1)
        num_batches = self.m/self.mini_batch_size

        # Iterates over the number of epochs chosen
        for epoch in range(self.epochs):
            # Iterates over the number of batches defined as: the number of rows / hyper-parameter mini_batch_size
            for b in range(int(num_batches)):
                # Define the batches of X and Y
                X_batch, Y_batch = self.batch_size(X, Y, b)
                # Difference between predicted and actual values
                error = np.dot(X_batch, self.thetas) - Y_batch
                # Compute gradients
                grad = 2 * np.dot(X_batch.T, error) / self.mini_batch_size
                # Update coefficients
                self.thetas = self.thetas - (self.learning_rate * grad)
        return self.thetas

    # Function used to predict dependent variable of test set data
    def predict(self, X):
        return np.dot(X, self.thetas)

    # Function which defines the evaluation metrics; RMSE, R-Squared
    def eval(self, X, Y):
        RMSE = np.sqrt(((self.predict(X) - Y) ** 2).mean())
        r2 = 1 - (np.sum((Y - self.predict(X)) ** 2)) / (np.sum((Y - Y.mean()) ** 2))

        print("RMSE :", RMSE)
        print("R-Squared: ", r2)

def main():
    gd = BGD()
    sgd = SGD()
    mgd = MBGD()

    # Reads the dataset and removes the serial number column
    df = pd.read_csv('./Admission_Predict_Ver1.csv')
    df = df.drop(columns='Serial No.')

    # Normalise the values
    column_max = df.max()
    df_norm = df / column_max
    df_norm['Chance of Admit '] = (df_norm[
        'Chance of Admit ']) * 0.97  # Reverses the normalisation of the probability values

    # Defines the independent and dependent variables
    X = df_norm.drop(columns='Chance of Admit ')
    Y = df_norm[['Chance of Admit ']]

    # Number of training examples and features
    X_train = X.iloc[:400]
    Y_train = Y.iloc[:400]
    X_test = X.iloc[400:]
    Y_test = Y.iloc[400:]
    '''
    gd.fit(X_train, Y_train)
    y_pred = gd.predict(X_test)
    print("Predicted values", y_pred[:5])
    print("Actual values", Y_test[:5])
    gd.eval(X_test, Y_test)
    '''
    
    sgd.fit(X_train, Y_train)
    y_pred = sgd.predict(X_test)
    print("Predicted values", y_pred[:5])
    print("Actual values", Y_test[:5])
    sgd.eval(X_test, Y_test)
    '''
    mgd.fit(X_train, Y_train)
    y_pred = mgd.predict(X_test)
    print("Predicted values", y_pred[:5])
    print("Actual values", Y_test[:5])
    mgd.eval(X_test, Y_test)'''

if __name__ == '__main__':
    main()

