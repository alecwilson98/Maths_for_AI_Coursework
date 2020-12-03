import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Defines the Batch Gradient Descent linear regression class
class BGD:

    def __init__(self, iterations=10000, learning_rate=0.01):
        # Hyper-parameters
        self.iterations = iterations
        self.learning_rate = learning_rate

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.thetas = np.random.rand(self.n, 1)
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            # Difference between predicted and actual values
            error = np.dot(X, self.thetas) - Y
            # Sum of squares function
            cost = np.sum(error ** 2) / (2 * self.m)
            # Compute gradient
            grad = np.dot(X.T, error) / self.m
            # Update coefficients
            self.thetas = self.thetas - self.learning_rate*grad

        return self.thetas

    def predict(self, X):
        return np.dot(X, self.thetas)

    def eval(self, X, Y):
        RMSE = np.sqrt(((self.predict(X) - Y) ** 2).mean())
        r2 = 1 - (np.sum((Y - self.predict(X)) ** 2)) / (np.sum((Y - Y.mean()) ** 2))

        print("RMSE :", RMSE)
        print("R-Squared: ", r2)

# Defines the Stochastic Gradient Descent linear regression class
class SGD:
    def __init__(self, epochs=1000, t0=5, t1=500):
        # Hyper-parameters
        self.epochs = epochs
        self.t0 = t0
        self.t1 = t1

    def learning_schedule(self, t):
        return self.t0 / (t + self.t1)

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.thetas = np.random.rand(self.n, 1)
        self.random = np.random.randint(self.m)
        self.xi = X[self.random:self.random + 1]
        self.yi = Y[self.random:self.random + 1]

        for epoch in range(self.epochs):
            for i in range(self.m):
                # Difference between predicted and actual values
                error = np.dot(self.xi, self.thetas) - self.yi
                # Sum of squares function
                cost = np.sum(error ** 2) / (2 * self.m)
                # Compute gradient
                grad = np.dot(self.xi.T, error) / self.m
                # Update coefficients
                self.thetas = self.thetas - (self.learning_schedule(self.epochs * self.m + i) * grad)
        return self.thetas

    def predict(self, X):
        return np.dot(self.xi, self.thetas)

    def eval(self, X, Y):
        RMSE = np.sqrt(((self.predict(X) - Y) ** 2).mean())
        r2 = 1 - (np.sum((Y - self.predict(X)) ** 2)) / (np.sum((Y - Y.mean()) ** 2))

        print("RMSE :", RMSE)
        print("R-Squared: ", r2)

# Defines the Mini-Batch Gradient Descent linear regression class
class MBGD:

    def __init__(self, iterations=10, learning_rate=1, mini_batch_size=50):
        # Hyper-parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size

        # Number of training examples and features
        self.m, self.n = X.shape

        # Initialising weights
        self.W = np.zeros(self.m)
        self.b = 0
        self.X = X
        self.Y = Y

    def mini_batch(self, learning_rate, batch_size, iterations):

        loss_log      = []
        test_acc_log  = []
        train_acc_log = []

        N = min(len(self.X_train), len(self.Y_train))
        num_batches = int(N/self.mini_batch_size)

        for t in range(self.iterations):
            permutation = np.random.permutation(N)

            for k in range(num_batches):
                # Reset buffer containing updates
                nabla_b = [np.zeros(b.shape) for b in self.b]
                nabla_w = [np.zeros(w.shape) for w in self.W]

                for i in range(self.mini_batch_size):

                    x = self.X[permutation[k * self.mini_batch_size + i]]
                    y = self.Y[permutation[k * self.mini_batch_size + i]]

                    # Differential of the loss function with resect to w
                    Y_pred = self.predict()
                    dw = np.dot(np.transpose(Y_pred-self.Y), (Y_pred-self.Y))
                    #db =
                nabla_b = [n_b + d_b for n_b, d_b in zip(nabla_b, db)]
                nabla_w = [n_w + d_w for n_w, d_w in zip(nabla_w, dw)]

            # Update weights
            self.W = self.W - (learning_rate * (nabla_w/self.mini_batch_size))
            self.b = self.b - (learning_rate * (nabla_b/self.mini_batch_size))

    def predict(self, X):
        return np.sum(X.dot(self.W) + self.b)


def main():
    gd = BGD()
    sgd = SGD()
    #mgd = MBGD()

    # Reads the dataset and removes the serial number column
    df = pd.read_csv('/home/alecwilson/PycharmProjects/Maths_for_AI_Coursework/Admission_Predict_Ver1.csv')
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

    gd.fit(X_train, Y_train)
    gd.eval(X_test, Y_test)

    sgd.fit(X_train, Y_train)
    sgd.eval(X_test, Y_test)



if __name__ == '__main__':
    main()





