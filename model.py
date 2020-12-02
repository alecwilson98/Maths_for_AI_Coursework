import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Defines the Batch Gradient Descent linear regression class
class BGD:

    def __init__(self, iterations=100000, learning_rate=0.0000001):
        # Hyper-parameters
        self.iterations = iterations
        self.learning_rate = learning_rate

    def fit(self, X, Y):
        self.m, self.n = X.shape
        # Initialising weights
        self.W = np.zeros([self.n])
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.iterate()
        return self

    def iterate(self):

        Y_pred = self.predict(self.X)

        # Differential of the loss function with resect to w and b
        dw = np.average(-2 * (self.X.T).dot(self.Y - Y_pred))
        db = np.average(-2 * np.sum(self.Y - Y_pred))

        # Update weights
        self.W = self.W - (self.learning_rate * dw)
        self.b = self.b - (self.learning_rate * db)
        return self

    def predict(self, X):
        return X.dot(self.W) - self.b

# Defines the Stochastic Gradient Descent linear regression class
class SGD:
    def __init__(self, epochs=100000, t0=5, t1=500):
        # Hyper-parameters
        self.epochs = epochs
        self.t0 = t0
        self.t1 = t1

    def learning_schedule(self, t):
        return self.t0 / (t + self.t1)

    def fit(self, X, Y):
        self.m, self.n = X.shape
        # Initialising weights
        self.W = np.random.randint(1, 2, self.n)
        self.b = 0
        random = np.random.randint(self.m)
        self.xi = X[random:random+1]
        self.yi = Y[random:random+1]

        for epoch in range(self.epochs):
            self.iterate()
        return self

    def iterate(self):
        for i in range(self.m):
            Y_pred = self.predict(self.xi)

            # Differential of the loss function with resect to w and b
            dw = np.average(-2 * (self.xi.T).dot(self.yi - Y_pred))
            db = np.average(-2 * np.sum(self.yi - Y_pred))

            # Update weights
            self.W = self.W - (self.learning_schedule(self.epochs * self.m + i) * dw)
            self.b = self.b - (self.learning_schedule(self.epochs * self.m + i) * db)
            return self

    def predict(self, xi):
        return xi.dot(self.W) - self.b

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
        return X.dot(self.W) + self.b


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

    # BGD
    '''
    gd.fit(X_train, Y_train)
    Y_pred = gd.predict(X_test)

    print("Predicted values ", Y_pred[:3])
    print("Real values      ", Y_test[:3])
    print("Trained W        ", gd.W)
    print("Trained b        ", gd.b)

    # Visualization on test set

    plt.scatter(X_test, Y_test, color='blue')
    plt.plot(X_test, Y_pred, color='orange')
    plt.show(Block=True)
'''
    sgd.fit(X_train, Y_train, 0)
    Y_pred = sgd.predict(X_test)

    print("Predicted values ", Y_pred[:3])
    print("Real values      ", Y_test[:3])
    print("Trained W        ", sgd.W)
    print("Trained b        ", sgd.b)

    # Visualization on test set

    plt.scatter(X_test, Y_test, color='blue')
    plt.plot(X_test, Y_pred, color='orange')
    plt.show(Block=True)

if __name__ == '__main__':
    main()





